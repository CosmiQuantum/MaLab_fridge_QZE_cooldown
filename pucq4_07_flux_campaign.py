"""Resumable PUCQ4 flux characterization through the normal round robin.

Only physical Q4/M5/Yoko3 and physical Q6/M6/Yoko4 are in scope. Currents are
limited to the nine points shown in the source characterization deck. Every
subprocess uses ``round_robin_benchmark.py`` with explicit environment overrides;
the section classes and their QICK programs are not modified.
"""

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from malab_yokogawa_gs200 import YokogawaGS200


DATA_ROOT = Path(r"M:\_Data\20250822 - Olivia\pucq4_run_started_Aug_3\PUCQ4\pucq4_first_light")
CAMPAIGN_ROOT = DATA_ROOT / "yoko_flux_campaign"
MANIFEST = CAMPAIGN_ROOT / "flux_campaign_manifest.json"
CURRENT_GRID_MA = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]
ACQUISITION_ORDER_MA = [0.0, -2.5, -5.0, -7.5, -10.0, 2.5, 5.0, 7.5, 10.0]
MAX_ABS_CURRENT_A = 0.010
RAMP_RATE_A_PER_S = 0.0005
PI_TARGET = 0.65
PI_WINDOW = (0.60, 0.70)

DEVICES = {
    "q4": {
        "physical": "Q4",
        "row": "M5",
        "index": 4,
        "ip": "192.168.1.73",
        "yoko": "Yoko3",
        "qspec_gain": 0.004,
        "sigma_us": 0.055,
        "qspec_gain_by_current": {-7.5: 0.010, -10.0: 0.015},
        "sigma_by_current_us": {-10.0: 0.04519688543145721},
        "pi_amp_by_current": {-10.0: 0.6985900018311665},
        "calibrate_rabi_before_readout": True,
        "locator_readout_offset_by_current_mhz": {
            -10.0: 0.0, 2.5: 0.0, 5.0: 0.0, 7.5: 0.0, 10.0: 0.0,
        },
        "readout_gain_by_current": {-10.0: 0.09},
        "readout_offset_by_current_mhz": {
            -10.0: -0.05, 2.5: 0.0, 5.0: 0.0, 7.5: 0.0, 10.0: 0.0,
        },
        "verified_seed_currents_mA": {-10.0, -7.5, 2.5, 5.0, 7.5, 10.0},
        "expected_mhz": {
            -10.0: 5984.071123039789, -7.5: 6423.24, -5.0: 6725.45, -2.5: 6945.5,
            0.0: 7084.78, 2.5: 7143.847, 5.0: 7121.549, 7.5: 7022.172, 10.0: 6841.385,
        },
    },
    "q6": {
        "physical": "Q6",
        "row": "M6",
        "index": 5,
        "ip": "192.168.1.77",
        "yoko": "Yoko4",
        "qspec_gain": 0.010,
        "sigma_us": 0.062,
        "readout_gain": 0.7967,
        "readout_length_us": 12.0,
        "readout_offset_mhz": -0.52,
        "readout_gain_by_current": {-10.0: 0.88, -7.5: 0.7945, -5.0: 0.8767},
        "readout_offset_by_current_mhz": {-10.0: -0.40, -7.5: -0.76, -5.0: -0.28},
        "qspec_gain_by_current": {
            -10.0: 0.004, -7.5: 0.006, -2.5: 0.002,
            5.0: 0.006, 7.5: 0.006, 10.0: 0.006,
        },
        # Current-indexed flux seeds are deliberately separate from the 0 mA
        # device sigma above. Promote a value to system_config only after its
        # measured pi amplitude is inside PI_WINDOW.
        "sigma_by_current_us": {
            -10.0: 0.09862908781821704,
            -7.5: 0.12225276618925736,
            -5.0: 0.06714287589206007,
            -2.5: 0.060540487104986655, 2.5: 0.058001,
            5.0: 0.055669293746329056, 7.5: 0.058001,
            10.0: 0.05115680589458381,
        },
        "pi_amp_by_current": {
            -10.0: 0.6886101446621498,
            -7.5: 0.6985900018311665,
            -5.0: 0.6586705731550998,
            5.0: 0.6287310016480498,
            7.5: 0.6387108588170665,
            10.0: 0.6287310016480498,
            -2.5: 0.6087712873100165,
        },
        "calibrate_rabi_before_readout": True,
        "verified_seed_currents_mA": {-2.5, 0.0, 2.5, 5.0, 7.5, 10.0},
        "expected_mhz": {
            -10.0: 4376.306486033511, -7.5: 4840.0, -5.0: 5587.9,
            -2.5: 6103.846686566928,
            0.0: 6522.634, 2.5: 6842.132, 5.0: 7062.765919638013,
            7.5: 7185.658452611533, 10.0: 7211.438328231458,
        },
    },
}


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        "created": datetime.now().isoformat(),
        "current_grid_mA": CURRENT_GRID_MA,
        "source_deck": r"C:\Users\Ma Quantum Lab\Downloads\PUCQ4 Initial Characterization (1).pptx",
        "devices": {},
    }


def save_manifest(data):
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(MANIFEST)


def activated_subprocess_env():
    """Return an environment with this Conda interpreter's DLL paths active."""
    env = os.environ.copy()
    prefix = Path(sys.prefix)
    conda_paths = [
        prefix,
        prefix / "Library" / "mingw-w64" / "bin",
        prefix / "Library" / "usr" / "bin",
        prefix / "Library" / "bin",
        prefix / "Scripts",
        prefix / "bin",
    ]
    env["PATH"] = os.pathsep.join(str(path) for path in conda_paths) + os.pathsep + env.get("PATH", "")
    env["CONDA_PREFIX"] = str(prefix)
    return env


def run_round_robin(device, current_ma, stage, overrides):
    tag = f"{device['yoko'].lower()}_{device['physical'].lower()}_{current_ma:+05.1f}mA_{stage}"
    env = activated_subprocess_env()
    env.update({
        "PUCQ4_QS": str(device["index"]),
        "PUCQ4_SUB_STUDY": f"flux_{tag}",
        **{key: str(value) for key, value in overrides.items()},
    })
    if device["physical"] == "Q4" and current_ma == -10.0:
        env["PUCQ4_RES_EXTREMUM"] = "peak"
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    log_path = CAMPAIGN_ROOT / f"{tag}.log"
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {tag}: starting", flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        completed = subprocess.run(
            [sys.executable, "round_robin_benchmark.py"],
            cwd=Path(__file__).resolve().parent,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError(f"{stage} failed with exit code {completed.returncode}; see {log_path}")
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {tag}: complete", flush=True)
    return text, str(log_path)


def run_optimizer(device, current_ma, stage, overrides):
    tag = f"{device['yoko'].lower()}_{device['physical'].lower()}_{current_ma:+05.1f}mA_{stage}"
    env = activated_subprocess_env()
    env.update({key: str(value) for key, value in overrides.items()})
    env["PUCQ4_OPT_QUBITS"] = str(device["index"])
    env["PUCQ4_OPT_ROOT_TAG"] = f"flux_{tag}"
    log_path = CAMPAIGN_ROOT / f"{tag}.log"
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {tag}: starting", flush=True)
    with log_path.open("w", encoding="utf-8", errors="replace") as stream:
        completed = subprocess.run(
            [sys.executable, "Combined_Optimization_Scriptsv2.py"],
            cwd=Path(__file__).resolve().parent,
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if completed.returncode:
        raise RuntimeError(f"{stage} failed with exit code {completed.returncode}; see {log_path}")
    print(f"[{datetime.now().isoformat(timespec='seconds')}] {tag}: complete", flush=True)
    return text, str(log_path)


def last_float(pattern, text, label):
    matches = re.findall(pattern, text)
    if not matches:
        raise RuntimeError(f"Could not parse {label} from round-robin log")
    return float(matches[-1])


def resonator_frequency(text):
    explicit = re.findall(r"g-e Resonator \d+ frequency: ([0-9.+-eE]+)", text)
    if explicit:
        return float(explicit[-1])
    legacy = re.findall(r"(?m)^([89]\d{3}\.\d+)\s*$", text)
    if not legacy:
        raise RuntimeError("Could not parse resonator frequency from round-robin log")
    return float(legacy[-1])


def validated_qspec(text, center_mhz, half_span_mhz, label, max_fwhm_mhz):
    if "Error during Lorentzian fit" in text:
        raise RuntimeError(f"{label} Lorentzian fit reported an error")
    qfreq = last_float(r"g-e Qubit \d+ frequency: ([0-9.+-eE]+)", text, f"{label} qfreq")
    fwhm = abs(last_float(r"g-e Qubit \d+ FWHM: ([0-9.+-eE]+) MHz", text, f"{label} FWHM"))
    if abs(qfreq - center_mhz) > half_span_mhz:
        raise RuntimeError(f"{label} fit {qfreq} MHz lies outside its search window")
    if not 0.02 <= fwhm <= max_fwhm_mhz:
        raise RuntimeError(f"{label} FWHM {fwhm} MHz is outside 0.02-{max_fwhm_mhz} MHz")
    return qfreq, fwhm


def optimize_readout(device, current_ma, qfreq, sigma_us, pi_amp, res_base_mhz):
    gain = device.get("readout_gain_by_current", {}).get(
        current_ma, device.get("readout_gain", 0.33 if device["index"] == 4 else 0.60)
    )
    offset = device.get("readout_offset_by_current_mhz", {}).get(
        current_ma, device.get("readout_offset_mhz", 0.44 if device["index"] == 4 else -0.04)
    )
    length = device.get("readout_length_us", 5.0 if device["index"] == 4 else 3.0)
    logs = []
    common = {
        "PUCQ4_OPT_RES_BASE_MHZ": res_base_mhz,
        "PUCQ4_OPT_QFREQ_MHZ": qfreq,
        "PUCQ4_OPT_SIGMA_US": sigma_us,
        "PUCQ4_OPT_PI_AMP": pi_amp,
    }
    if device["physical"] == "Q6" and current_ma == -10.0:
        common["PUCQ4_OPT_MAX_GAIN"] = 0.88
    previous_fidelity = None
    best = None
    for iteration in range(8):
        length_text, length_log = run_optimizer(device, current_ma, f"readout_length{iteration + 1}", {
            **common,
            "PUCQ4_OPT_STAGE": "length",
            "PUCQ4_OPT_READOUT_LENGTH_US": length,
            "PUCQ4_OPT_READOUT_GAIN": gain,
            "PUCQ4_OPT_READOUT_OFFSET_MHZ": offset,
        })
        match = re.findall(r"shortest plateau ([0-9.]+) us, max fidelity ([0-9.]+)", length_text)
        if not match:
            raise RuntimeError("Could not parse readout-length optimization result")
        length, length_fidelity = map(float, match[-1])
        logs.append(length_log)
        gain_text, gain_log = run_optimizer(device, current_ma, f"readout_gf{iteration + 1}", {
            **common,
            "PUCQ4_OPT_STAGE": "gain_frequency",
            "PUCQ4_OPT_READOUT_LENGTH_US": length,
            "PUCQ4_OPT_READOUT_GAIN": gain,
            "PUCQ4_OPT_READOUT_OFFSET_MHZ": offset,
        })
        match = re.findall(
            r"gain ([0-9.]+), offset ([+-]?[0-9.]+) MHz, fidelity ([0-9.]+)", gain_text
        )
        if not match:
            raise RuntimeError("Could not parse gain/frequency optimization result")
        gain, offset, fidelity = map(float, match[-1])
        logs.append(gain_log)
        candidate = {
            "gain": gain, "offset_mhz": offset, "length_us": length,
            "fidelity": max(fidelity, length_fidelity),
        }
        if best is None or candidate["fidelity"] > best["fidelity"]:
            best = candidate
        if previous_fidelity is not None and fidelity - previous_fidelity < 0.005:
            break
        previous_fidelity = fidelity
    if best["fidelity"] < 0.30:
        rescue_text, rescue_log = run_optimizer(device, current_ma, "readout_rescue", {
            **common,
            "PUCQ4_OPT_STAGE": "gain_frequency",
            "PUCQ4_OPT_READOUT_LENGTH_US": best["length_us"],
            "PUCQ4_OPT_READOUT_GAIN": best["gain"],
            "PUCQ4_OPT_READOUT_OFFSET_MHZ": best["offset_mhz"],
            "PUCQ4_OPT_GAIN_HALF_SPAN": 0.40,
            "PUCQ4_OPT_FREQ_HALF_SPAN_MHZ": 1.50,
            "PUCQ4_OPT_GAIN_STEPS": 11,
            "PUCQ4_OPT_FREQ_STEPS": 13,
        })
        match = re.findall(
            r"gain ([0-9.]+), offset ([+-]?[0-9.]+) MHz, fidelity ([0-9.]+)", rescue_text
        )
        if not match:
            raise RuntimeError("Could not parse rescue readout optimization result")
        rescue_gain, rescue_offset, rescue_fidelity = map(float, match[-1])
        logs.append(rescue_log)
        if rescue_fidelity > best["fidelity"]:
            best.update({
                "gain": rescue_gain,
                "offset_mhz": rescue_offset,
                "fidelity": rescue_fidelity,
            })
    if best["fidelity"] < 0.20:
        raise RuntimeError(
            f"Readout fidelity {best['fidelity']:.4f} remains below 0.20 after rescue sweep"
        )
    best["logs"] = logs
    return best


def calibrate_rabi_before_readout(device, current_ma, qfreq, sigma_us):
    """Break the Rabi/SSF dependency using averaged, unscaled I/Q first."""
    logs = []
    pi_amp = PI_TARGET
    for attempt in range(4):
        stage = "raw_iq_rabi" if attempt == 0 else f"raw_iq_rabi_retry{attempt}"
        text, log_path = run_round_robin(device, current_ma, stage, {
            "PUCQ4_RUN_FLAGS": "res_spec,rabi",
            "PUCQ4_QFREQ_MHZ": qfreq,
            "PUCQ4_SIGMA_US": sigma_us,
            "PUCQ4_PI_AMP": PI_TARGET,
            "PUCQ4_RABI_SCALING": "false",
            "PUCQ4_RABI_REPS": 1200,
            "PUCQ4_READOUT_LENGTH_US": device.get("readout_length_us", 3.0),
            "PUCQ4_READOUT_GAIN": device.get("readout_gain", 0.60),
            "PUCQ4_READOUT_OFFSET_MHZ": device.get("readout_offset_mhz", -0.04),
        })
        logs.append(log_path)
        pi_amp = last_float(
            r"Pi amplitude for qubit\s+\d+\s+is:\s+([0-9.+-eE]+)", text, "raw-I/Q pi amplitude"
        )
        if PI_WINDOW[0] <= pi_amp <= PI_WINDOW[1]:
            return sigma_us, pi_amp, logs
        if not 0.05 <= pi_amp <= 1.5:
            raise RuntimeError(f"Raw-I/Q Rabi returned nonphysical pi amplitude {pi_amp}")
        # Damped correction avoids jumping between Rabi-fit aliases near a branch boundary.
        sigma_us = sigma_us * (pi_amp / PI_TARGET) ** 0.5
    raise RuntimeError(
        f"Raw-I/Q pi amplitude {pi_amp} did not enter {PI_WINDOW} after bounded sigma retries"
    )


def characterize_point(device, current_ma, sigma_us):
    expected = device["expected_mhz"][current_ma]
    qspec_gain = device.get("qspec_gain_by_current", {}).get(current_ma, device["qspec_gain"])
    # Q6 at -10 mA has a comparatively broad line.  Preserve the 5 kHz point
    # spacing while including enough off-resonance baseline for a stable fit.
    qspec_half_span_mhz = 2.0 if device["physical"] == "Q6" and current_ma == -10.0 else 1.0
    qspec_steps = int(2 * qspec_half_span_mhz / 0.005) + 1
    coarse_log = None
    qfreq = expected
    if current_ma != 0.0 and current_ma not in device.get("verified_seed_currents_mA", set()):
        coarse_text, coarse_log = run_round_robin(device, current_ma, "locator", {
            "PUCQ4_RUN_FLAGS": "res_spec,q_spec",
            "PUCQ4_QFREQ_MHZ": expected,
            "PUCQ4_QSPEC_HALF_SPAN_MHZ": 25.0,
            "PUCQ4_QSPEC_STEPS": 501,
            "PUCQ4_QSPEC_REPS": 2000,
            "PUCQ4_QSPEC_GAIN": 0.03,
            "PUCQ4_QSPEC_LENGTH_US": 40.0,
            "PUCQ4_QSPEC_SCALING": "false",
            "PUCQ4_SIGMA_US": sigma_us,
            "PUCQ4_READOUT_LENGTH_US": device.get("readout_length_us", 3.0),
            "PUCQ4_READOUT_GAIN": device.get("readout_gain", 0.60),
            "PUCQ4_READOUT_OFFSET_MHZ": device.get("readout_offset_mhz", -0.04),
        })
        qfreq, _ = validated_qspec(coarse_text, expected, 25.0, "locator", 5.0)

    calibration_overrides = {
        "PUCQ4_RUN_FLAGS": "res_spec,q_spec",
        "PUCQ4_QFREQ_MHZ": qfreq,
        "PUCQ4_QSPEC_HALF_SPAN_MHZ": qspec_half_span_mhz,
        "PUCQ4_QSPEC_STEPS": qspec_steps,
        "PUCQ4_QSPEC_REPS": 5000,
        "PUCQ4_QSPEC_GAIN": qspec_gain,
        "PUCQ4_QSPEC_LENGTH_US": 20.0,
        "PUCQ4_QSPEC_SCALING": "false",
        "PUCQ4_SIGMA_US": sigma_us,
        "PUCQ4_READOUT_LENGTH_US": device.get("readout_length_us", 3.0),
        "PUCQ4_READOUT_GAIN": device.get("readout_gain_by_current", {}).get(
            current_ma, device.get("readout_gain", 0.60)
        ),
        "PUCQ4_READOUT_OFFSET_MHZ": device.get("readout_offset_mhz", -0.04),
    }
    initial_readout_offset = device.get("readout_offset_by_current_mhz", {}).get(current_ma)
    if initial_readout_offset is not None:
        calibration_overrides["PUCQ4_READOUT_OFFSET_MHZ"] = initial_readout_offset
    calibration_text, calibration_log = run_round_robin(
        device, current_ma, "calibration", calibration_overrides
    )
    res_base_mhz = resonator_frequency(calibration_text)
    raw_rabi_logs = []
    bootstrap_log = None
    raw_qspec_valid = True
    prep_pi_amp = PI_TARGET
    try:
        qfreq, _ = validated_qspec(
            calibration_text, qfreq, qspec_half_span_mhz, "narrow qspec", 0.8
        )
    except RuntimeError as raw_qspec_error:
        print(f"Raw narrow qspec rejected ({raw_qspec_error}); bootstrapping readout", flush=True)
        raw_qspec_valid = False
    if current_ma != 0.0 and device.get("calibrate_rabi_before_readout", False):
        saved_pi_amp = device.get("pi_amp_by_current", {}).get(current_ma)
        if saved_pi_amp is not None:
            prep_pi_amp = saved_pi_amp
        else:
            sigma_us, prep_pi_amp, raw_rabi_logs = calibrate_rabi_before_readout(
                device, current_ma, qfreq, sigma_us
            )
    try:
        readout = optimize_readout(
            device, current_ma, qfreq, sigma_us, prep_pi_amp, res_base_mhz
        )
    except RuntimeError as readout_error:
        print(
            f"Readout bootstrap rejected ({readout_error}); calibrating raw-I/Q Rabi first",
            flush=True,
        )
        sigma_us, prep_pi_amp, raw_rabi_logs = calibrate_rabi_before_readout(
            device, current_ma, qfreq, sigma_us
        )
        readout = optimize_readout(
            device, current_ma, qfreq, sigma_us, prep_pi_amp, res_base_mhz
        )
    if not raw_qspec_valid:
        bootstrap_overrides = {
            "PUCQ4_READOUT_LENGTH_US": readout["length_us"],
            "PUCQ4_READOUT_GAIN": readout["gain"],
            "PUCQ4_READOUT_OFFSET_MHZ": readout["offset_mhz"],
        }
        bootstrap_text, bootstrap_log = run_round_robin(device, current_ma, "readout_bootstrap_qspec", {
            "PUCQ4_RUN_FLAGS": "res_spec,q_spec",
            "PUCQ4_QFREQ_MHZ": qfreq,
            "PUCQ4_QSPEC_HALF_SPAN_MHZ": 5.0,
            "PUCQ4_QSPEC_STEPS": 501,
            "PUCQ4_QSPEC_REPS": 5000,
            "PUCQ4_QSPEC_GAIN": max(qspec_gain, 0.010),
            "PUCQ4_QSPEC_LENGTH_US": 20.0,
            "PUCQ4_QSPEC_SCALING": "true",
            "PUCQ4_SIGMA_US": sigma_us,
            "PUCQ4_PI_AMP": prep_pi_amp,
            **bootstrap_overrides,
        })
        qfreq, _ = validated_qspec(
            bootstrap_text, qfreq, 5.0, "readout-bootstrap qspec", 2.0
        )
    readout_overrides = {
        "PUCQ4_READOUT_LENGTH_US": readout["length_us"],
        "PUCQ4_READOUT_GAIN": readout["gain"],
        "PUCQ4_READOUT_OFFSET_MHZ": readout["offset_mhz"],
    }
    final_text, final_log = run_round_robin(device, current_ma, "post_readout_calibration", {
        "PUCQ4_RUN_FLAGS": "res_spec,q_spec,rabi",
        "PUCQ4_QFREQ_MHZ": qfreq,
        "PUCQ4_QSPEC_HALF_SPAN_MHZ": qspec_half_span_mhz,
        "PUCQ4_QSPEC_STEPS": qspec_steps,
        "PUCQ4_QSPEC_REPS": 5000,
        "PUCQ4_QSPEC_GAIN": qspec_gain,
        "PUCQ4_QSPEC_LENGTH_US": 20.0,
        "PUCQ4_SIGMA_US": sigma_us,
        "PUCQ4_RABI_REPS": 1200,
        **readout_overrides,
    })
    qfreq, _ = validated_qspec(
        final_text, qfreq, qspec_half_span_mhz, "final narrow qspec", 0.8
    )
    pi_amp = last_float(r"Pi amplitude for qubit\s+\d+\s+is:\s+([0-9.+-eE]+)", final_text, "pi amplitude")
    rabi_logs = raw_rabi_logs + [final_log]

    for attempt in range(3):
        if PI_WINDOW[0] <= pi_amp <= PI_WINDOW[1]:
            break
        # Population-scaled Rabi can be noisier than raw I/Q at difficult flux
        # points. Match the raw-I/Q averaging and damp corrections so one noisy
        # alias cannot move sigma by an order of magnitude.
        sigma_us = sigma_us * (pi_amp / PI_TARGET) ** 0.5
        rabi_text, rabi_log = run_round_robin(device, current_ma, f"rabi_retry{attempt + 1}", {
            "PUCQ4_RUN_FLAGS": "res_spec,rabi",
            "PUCQ4_QFREQ_MHZ": qfreq,
            "PUCQ4_SIGMA_US": sigma_us,
            "PUCQ4_RABI_REPS": 1200,
            **readout_overrides,
        })
        pi_amp = last_float(r"Pi amplitude for qubit\s+\d+\s+is:\s+([0-9.+-eE]+)", rabi_text, "pi amplitude")
        rabi_logs.append(rabi_log)
    if not PI_WINDOW[0] <= pi_amp <= PI_WINDOW[1]:
        raise RuntimeError(f"Pi amplitude {pi_amp} did not enter {PI_WINDOW} after bounded sigma retries")

    _, coherence_log = run_round_robin(device, current_ma, "coherence", {
        "PUCQ4_RUN_FLAGS": "res_spec,t1,t2r,t2e",
        "PUCQ4_QFREQ_MHZ": qfreq,
        "PUCQ4_SIGMA_US": sigma_us,
        "PUCQ4_PI_AMP": pi_amp,
        **readout_overrides,
    })
    return {
        "status": "complete",
        "current_mA": current_ma,
        "expected_qfreq_MHz": expected,
        "qfreq_MHz": qfreq,
        "sigma_us": sigma_us,
        "pi_amp": pi_amp,
        "res_base_MHz": res_base_mhz,
        "readout": readout,
        "logs": {
            "locator": coarse_log,
            "calibration": calibration_log,
            "readout_bootstrap_qspec": bootstrap_log,
            "post_readout_calibration": final_log,
            "readout_optimization": readout["logs"],
            "rabi": rabi_logs,
            "coherence": coherence_log,
        },
        "completed": datetime.now().isoformat(),
    }


def locator_only_point(device, current_ma, sigma_us):
    expected = device["expected_mhz"][current_ma]
    overrides = {
        "PUCQ4_RUN_FLAGS": "res_spec,q_spec",
        "PUCQ4_QFREQ_MHZ": expected,
        "PUCQ4_QSPEC_HALF_SPAN_MHZ": 100.0,
        "PUCQ4_QSPEC_STEPS": 2001,
        "PUCQ4_QSPEC_REPS": 2000,
        "PUCQ4_QSPEC_GAIN": 0.03,
        "PUCQ4_QSPEC_LENGTH_US": 40.0,
        "PUCQ4_QSPEC_SCALING": "false",
        "PUCQ4_SIGMA_US": sigma_us,
    }
    locator_offset = device.get("locator_readout_offset_by_current_mhz", {}).get(current_ma)
    if locator_offset is not None:
        overrides["PUCQ4_READOUT_OFFSET_MHZ"] = locator_offset
    text, log_path = run_round_robin(device, current_ma, "wide_locator_diagnostic", overrides)
    qfreq = last_float(r"g-e Qubit \d+ frequency: ([0-9.+-eE]+)", text, "wide locator qfreq")
    return {
        "status": "diagnostic_complete",
        "current_mA": current_ma,
        "search_center_MHz": expected,
        "search_half_span_MHz": 100.0,
        "fit_qfreq_MHz": qfreq,
        "log": log_path,
        "completed": datetime.now().isoformat(),
    }


def verify_inactive_source(active_key):
    inactive_key = "q6" if active_key == "q4" else "q4"
    inactive = DEVICES[inactive_key]
    with YokogawaGS200(inactive["ip"], max_current=MAX_ABS_CURRENT_A) as source:
        if abs(source.get_current()) > 1e-7 or source.get_output():
            raise RuntimeError(f"Inactive {inactive['yoko']} is not safely at 0 A with output off")


def run_device(key, requested_currents, locator_only=False):
    device = DEVICES[key]
    verify_inactive_source(key)
    manifest = load_manifest()
    device_data = manifest["devices"].setdefault(key, {
        "physical": device["physical"], "row": device["row"],
        "yoko": device["yoko"], "ip": device["ip"], "points": {},
    })
    sigma_us = device["sigma_us"]
    completed_sigmas = [point.get("sigma_us") for point in device_data["points"].values()
                        if point.get("status") == "complete"]
    if completed_sigmas:
        sigma_us = completed_sigmas[-1]

    with YokogawaGS200(device["ip"], max_current=MAX_ABS_CURRENT_A) as source:
        source.set_mode("current")
        source.set_range(0.01)
        if abs(source.get_current()) > 1e-7:
            source.set_output(True)
            source.ramp_current(0.0, sweeprate=RAMP_RATE_A_PER_S)
        source.set_current(0.0)
        source.set_output(True)
        try:
            for current_ma in requested_currents:
                sigma_us = device.get("sigma_by_current_us", {}).get(current_ma, sigma_us)
                key_ma = f"{current_ma:+.1f}"
                if device_data["points"].get(key_ma, {}).get("status") == "complete":
                    sigma_us = device_data["points"][key_ma]["sigma_us"]
                    continue
                source.ramp_current(current_ma * 1e-3, sweeprate=RAMP_RATE_A_PER_S)
                measured = source.get_current()
                if abs(measured - current_ma * 1e-3) > 1e-7:
                    raise RuntimeError(f"Yoko setpoint verification failed at {current_ma} mA")
                device_data["points"][key_ma] = {
                    "status": "running", "current_mA": current_ma,
                    "started": datetime.now().isoformat(),
                }
                save_manifest(manifest)
                print(
                    f"{device['yoko']} / {device['physical']}: ramp verified at {current_ma:+.1f} mA",
                    flush=True,
                )
                try:
                    if locator_only:
                        result = locator_only_point(device, current_ma, sigma_us)
                    else:
                        result = characterize_point(device, current_ma, sigma_us)
                except KeyboardInterrupt:
                    device_data["points"][key_ma].update({
                        "status": "aborted",
                        "failed": datetime.now().isoformat(),
                        "error": "Operator interrupted after quality rejection",
                    })
                    save_manifest(manifest)
                    raise
                except Exception as exc:
                    device_data["points"][key_ma].update({
                        "status": "failed",
                        "failed": datetime.now().isoformat(),
                        "error": str(exc),
                    })
                    save_manifest(manifest)
                    raise
                else:
                    if locator_only:
                        device_data.setdefault("diagnostics", {})[key_ma] = result
                        device_data["points"].pop(key_ma, None)
                    else:
                        device_data["points"][key_ma] = result
                        sigma_us = result["sigma_us"]
                    save_manifest(manifest)
            manifest["updated"] = datetime.now().isoformat()
            save_manifest(manifest)
        finally:
            print(f"Returning {device['yoko']} to 0 A and disabling output", flush=True)
            source.ramp_current(0.0, sweeprate=RAMP_RATE_A_PER_S)
            source.set_output(False)
            if abs(source.get_current()) > 1e-7 or source.get_output():
                raise RuntimeError(f"Failed to leave {device['yoko']} at 0 A with output off")


def run_rabi_only(key, requested_currents):
    """Recover pulse calibration at an established current-specific qfreq.

    This does not mark a full flux point complete; qspec, readout, and coherence
    must still pass the normal ``run_device`` chain.
    """
    device = DEVICES[key]
    verify_inactive_source(key)
    manifest = load_manifest()
    device_data = manifest["devices"].setdefault(key, {
        "physical": device["physical"], "row": device["row"],
        "yoko": device["yoko"], "ip": device["ip"], "points": {},
    })
    with YokogawaGS200(device["ip"], max_current=MAX_ABS_CURRENT_A) as source:
        source.set_mode("current")
        source.set_range(0.01)
        if abs(source.get_current()) > 1e-7:
            source.set_output(True)
            source.ramp_current(0.0, sweeprate=RAMP_RATE_A_PER_S)
        source.set_current(0.0)
        source.set_output(True)
        try:
            for current_ma in requested_currents:
                key_ma = f"{current_ma:+.1f}"
                qfreq = device["expected_mhz"][current_ma]
                sigma_us = device.get("sigma_by_current_us", {}).get(current_ma, device["sigma_us"])
                source.ramp_current(current_ma * 1e-3, sweeprate=RAMP_RATE_A_PER_S)
                if abs(source.get_current() - current_ma * 1e-3) > 1e-7:
                    raise RuntimeError(f"Yoko setpoint verification failed at {current_ma} mA")
                print(
                    f"{device['yoko']} / {device['physical']}: rabi-only ramp verified at {current_ma:+.1f} mA",
                    flush=True,
                )
                point = device_data["points"].setdefault(key_ma, {
                    "status": "pulse_only", "current_mA": current_ma,
                })
                try:
                    sigma_us, pi_amp, logs = calibrate_rabi_before_readout(
                        device, current_ma, qfreq, sigma_us,
                    )
                except Exception as exc:
                    point["pulse_calibration"] = {
                        "status": "failed", "failed": datetime.now().isoformat(),
                        "qfreq_MHz": qfreq, "starting_sigma_us": sigma_us,
                        "error": str(exc),
                    }
                    save_manifest(manifest)
                    raise
                else:
                    point["pulse_calibration"] = {
                        "status": "revalidation_required",
                        "completed": datetime.now().isoformat(),
                        "qfreq_MHz": qfreq, "sigma_us": sigma_us,
                        "raw_iq_pi_amp": pi_amp, "logs": logs,
                    }
                    save_manifest(manifest)
        finally:
            print(f"Returning {device['yoko']} to 0 A and disabling output", flush=True)
            source.ramp_current(0.0, sweeprate=RAMP_RATE_A_PER_S)
            source.set_output(False)
            if abs(source.get_current()) > 1e-7 or source.get_output():
                raise RuntimeError(f"Failed to leave {device['yoko']} at 0 A with output off")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", choices=("q4", "q6", "all"), default="all")
    parser.add_argument("--current-ma", type=float, action="append",
                        help="Run only listed deck-grid current(s); repeat option as needed")
    parser.add_argument("--locator-only", action="store_true",
                        help="Run a ±100 MHz high-power diagnostic locator only")
    parser.add_argument("--rabi-only", action="store_true",
                        help="Calibrate sigma/pi at an established current-specific qfreq only")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.locator_only and args.rabi_only:
        raise ValueError("--locator-only and --rabi-only are mutually exclusive")
    if args.current_ma:
        requested = args.current_ma
        if any(value not in CURRENT_GRID_MA for value in requested):
            raise ValueError(f"Currents must be from the deck grid {CURRENT_GRID_MA}")
    else:
        requested = ACQUISITION_ORDER_MA
    keys = ("q4", "q6") if args.device == "all" else (args.device,)
    for key in keys:
        if args.rabi_only:
            run_rabi_only(key, requested)
        else:
            run_device(key, requested, locator_only=args.locator_only)


if __name__ == "__main__":
    main()
