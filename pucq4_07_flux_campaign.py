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
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from malab_yokogawa_gs200 import YokogawaGS200
from system_config import PUCQ4_FLUX_CALIBRATION, PUCQ4_FLUX_CURRENT_GRID_MA


DATA_ROOT = Path(r"M:\_Data\20250822 - Olivia\pucq4_run_started_Aug_3\PUCQ4\pucq4_first_light")
CAMPAIGN_ROOT = DATA_ROOT / "yoko_flux_campaign"
MANIFEST = CAMPAIGN_ROOT / "flux_campaign_manifest.json"
CURRENT_GRID_MA = list(PUCQ4_FLUX_CURRENT_GRID_MA)
ACQUISITION_ORDER_MA = [0.0, -2.5, -5.0, -7.5, -10.0, 2.5, 5.0, 7.5, 10.0]
MAX_ABS_CURRENT_A = 0.010
RAMP_RATE_A_PER_S = 0.0005
PI_TARGET = 0.65
PI_WINDOW = (0.60, 0.70)
QSPEC_FWHM_WINDOW_MHZ = (0.099, 0.21)
UNIFORM_QSPEC_START_MHZ = 5950.0
UNIFORM_QSPEC_STOP_MHZ = 7200.0
UNIFORM_QSPEC_STEP_MHZ = 0.02
UNIFORM_QSPEC_REPS = 300
UNIFORM_QSPEC_SSF_SHOTS = 1000
UNIFORM_QSPEC_SSF_RELAX_DELAY_US = 250.0

# Lower-frequency follow-up maps requested after the original Q4-only
# 5950-7200 MHz map.  A 20 kHz / 300-rep endpoint pilot on 2026-09-03 resolved
# Q4 at 6037.559 MHz (local SNR 23.0) and Q6 at 4375.950 MHz (local SNR 5.34).
# The production grid uses 40 kHz / 100 reps: this still supplies several
# samples across the expected 0.1-0.2 MHz qspec line while avoiding hours of
# redundant averaging on the featureless multi-GHz baseline.  Each band stays
# below the successfully exercised 62,501-point program size.  Adjacent bands
# overlap by 2 MHz because QICK's realized DDS span is slightly shorter than
# its requested span.  Analysis cuts each overlap at the next requested band
# start, preserving complete coverage.
WIDE_UNIFORM_QSPEC_SETTINGS = {
    "q4": {
        "frequency_start_MHz": 5000.0,
        "frequency_stop_MHz": 7300.0,
        "frequency_step_MHz": 0.04,
        "reps": 100,
        "segments_MHz": [(5000.0, 5802.0), (5800.0, 6552.0), (6550.0, 7300.0)],
    },
    "q6": {
        "frequency_start_MHz": 4000.0,
        "frequency_stop_MHz": 7300.0,
        "frequency_step_MHz": 0.04,
        "reps": 100,
        "segments_MHz": [(4000.0, 5102.0), (5100.0, 6202.0), (6200.0, 7300.0)],
    },
}

# The literal keeps campaign-only rescue/locator settings. Any duplicated
# calibration seeds inside it are replaced immediately after construction from
# system_config.PUCQ4_FLUX_CALIBRATION, the sole runtime source of truth.
DEVICES = {
    "q4": {
        "physical": "Q4",
        "row": "M5",
        "index": 4,
        "ip": "192.168.1.73",
        "yoko": "Yoko3",
        "qspec_gain": 0.002,
        "sigma_us": 0.055,
        "readout_gain": 0.33,
        "readout_length_us": 5.0,
        "readout_offset_mhz": 0.44,
        "qspec_gain_by_current": {
            -10.0: 0.001649267441664, -7.5: 0.000839,
            -5.0: 0.002, -2.5: 0.0025, 0.0: 0.003,
            2.5: 0.0045, 5.0: 0.03, 7.5: 0.002, 10.0: 0.002,
        },
        # The common-grid map has only 300 averages per one of its 62,501
        # frequencies.  At -10 mA the accepted narrow-line gain is too weak at
        # that averaging depth, while the completed 0.03 locator scan cleanly
        # resolves the same 6037.6 MHz transition.  Keep this map-only drive
        # separate from the final local-linewidth calibration.
        "uniform_qspec_gain_by_current": {-10.0: 0.03},
        "qspec_length_by_current_us": {-10.0: 40.0, 5.0: 40.0},
        # Fresh population-calibrated scans at +5 mA resolve a reproducible
        # 0.69-0.82 MHz power-broadened line at the wide-scan/Rabi center.  The
        # low-power 0.104 MHz raw-I candidate excites only ~0.04 population;
        # gain 0.03 is required to reach the 0.3 population target.  Retain the
        # measured tradeoff explicitly instead of reporting a narrow noise fit.
        "qspec_fwhm_window_by_current_mhz": {5.0: (0.60, 1.20)},
        "sigma_by_current_us": {
            -10.0: 0.0165397052076522,
            -7.5: 0.0617065954067694,
            -5.0: 0.07923262394176339,
            -2.5: 0.05911146169340639,
            0.0: 0.05911146169340639,
            2.5: 0.0549176813895992,
            5.0: 0.08084894313680056,
            7.5: 0.05873041582061609,
            10.0: 0.059226039930878716,
        },
        "pi_amp_by_current": {
            -10.0: 0.9949337728132821,
            -7.5: 0.6287310016480498,
            -5.0: 0.6886101446621498,
            -2.5: 0.6486907159860831,
            0.0: 0.6287310016480498,
            2.5: 0.6287310016480498,
            5.0: 0.6786302874931331,
            7.5: 0.6087712873100165,
            10.0: 0.6187511444790331,
        },
        "res_base_by_current_mhz": {
            -10.0: 9010.49, -7.5: 9011.54, -5.0: 9012.79,
            -2.5: 9013.74, 0.0: 9014.74, 2.5: 9015.04,
            5.0: 9014.59, 7.5: 9014.04, 10.0: 9013.09,
        },
        "rabi_steps_by_current": {-10.0: 201},
        "rabi_reps_by_current": {-10.0: 1200},
        "rabi_stop_by_current": {-10.0: 1.0},
        "settle_seconds_by_current": {-10.0: 600},
        "allow_provisional_rabi_by_current": {-10.0},
        "locator_half_span_by_current_mhz": {-10.0: 250.0},
        "locator_step_by_current_mhz": {-10.0: 0.10},
        "locator_reps_by_current": {-10.0: 2000},
        "calibrate_rabi_before_readout": True,
        "locator_readout_offset_by_current_mhz": {
            -10.0: -0.05, 2.5: 0.0, 5.0: 0.0, 7.5: 0.0, 10.0: 0.0,
        },
        "readout_length_by_current_us": {
            -10.0: 5.0, -7.5: 7.0, -5.0: 6.0, -2.5: 6.0, 0.0: 6.0,
            2.5: 5.0, 5.0: 6.0, 7.5: 10.0, 10.0: 6.0,
        },
        "readout_gain_by_current": {
            -10.0: 0.95, -7.5: 0.95, -5.0: 0.95, -2.5: 0.73,
            0.0: 0.65, 2.5: 0.93, 5.0: 0.92, 7.5: 0.95, 10.0: 0.95,
        },
        "readout_offset_by_current_mhz": {
            -10.0: 0.95, -7.5: 0.68, -5.0: 0.08, -2.5: 0.32,
            0.0: 0.20, 2.5: -0.48, 5.0: -0.48, 7.5: -0.24, 10.0: -0.12,
        },
        "readout_fidelity_by_current": {
            -10.0: 0.124, -7.5: 0.6825, -5.0: 0.725, -2.5: 0.6183,
            0.0: 0.581, 2.5: 0.615, 5.0: 0.512, 7.5: 0.773, 10.0: 0.6773,
        },
        # The broad, repeated rescue grid at -10 mA has an interior optimum
        # near 0.1 fidelity. Preserve it as a measured low-fidelity readout so
        # averaged population/coherence data can be acquired; do not label it
        # as a high-fidelity single-shot discriminator.
        "readout_min_fidelity_by_current": {-10.0: 0.10},
        "verified_seed_currents_mA": set(CURRENT_GRID_MA),
        "expected_mhz": {
            -10.0: 6037.585407899529, -7.5: 6422.620534808272,
            -5.0: 6724.886451375884, -2.5: 6945.111991749794,
            0.0: 7084.565677917129, 2.5: 7143.8595708799985,
            5.0: 7121.580621960356, 7.5: 7022.121082210489,
            10.0: 6841.20173951502,
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
        "readout_length_by_current_us": {
            -10.0: 12.0, -7.5: 12.0, -5.0: 12.0, -2.5: 12.0,
            0.0: 10.0, 2.5: 12.0, 5.0: 12.0, 7.5: 11.0, 10.0: 12.0,
        },
        "readout_gain_by_current": {
            -10.0: 0.88, -7.5: 0.7945, -5.0: 0.8767, -2.5: 0.7967,
            0.0: 0.95, 2.5: 0.7967, 5.0: 0.9167, 7.5: 0.8334, 10.0: 0.8722,
        },
        "readout_offset_by_current_mhz": {
            -10.0: -0.40, -7.5: -0.76, -5.0: -0.28, -2.5: 0.08,
            0.0: -0.40, 2.5: -0.52, 5.0: -0.76, 7.5: -0.04, 10.0: -0.16,
        },
        "qspec_gain_by_current": {
            -10.0: 0.004, -7.5: 0.006, -5.0: 0.006, -2.5: 0.002,
            0.0: 0.006, 2.5: 0.006, 5.0: 0.006, 7.5: 0.006, 10.0: 0.006,
        },
        # Map-only high-power gains preserve visibility at the three negative
        # biases whose optimized single-shot separation is poor.  These values
        # do not replace the accepted local-linewidth gains above.
        "uniform_qspec_gain_by_current": {
            -10.0: 0.03, -7.5: 0.03, -5.0: 0.03,
        },
        "res_base_by_current_mhz": {
            -10.0: 9055.31, -7.5: 9056.01, -5.0: 9056.91,
            -2.5: 9057.78, 0.0: 9059.26, 2.5: 9060.46,
            5.0: 9060.98, 7.5: 9061.53, 10.0: 9061.68,
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
            -7.5: 0.6786302874931331,
            -5.0: 0.6586705731550998,
            5.0: 0.6287310016480498,
            7.5: 0.6387108588170665,
            10.0: 0.6287310016480498,
            -2.5: 0.6087712873100165,
        },
        "calibrate_rabi_before_readout": True,
        "verified_seed_currents_mA": {-10.0, -7.5, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0},
        "expected_mhz": {
            -10.0: 4376.306486033511, -7.5: 4839.162419284267, -5.0: 5587.9,
            -2.5: 6103.846686566928,
            0.0: 6522.634, 2.5: 6842.132, 5.0: 7062.765919638013,
            7.5: 7185.658452611533, 10.0: 7211.438328231458,
        },
    },
}


def apply_final_runtime_calibrations():
    """Make every future campaign stage seed from the final shared table."""
    calibration_keys = {
        "q4": "Yoko3_Q4_M5",
        "q6": "Yoko4_Q6_M6",
    }
    for device_name, calibration_key in calibration_keys.items():
        device = DEVICES[device_name]
        table = PUCQ4_FLUX_CALIBRATION[calibration_key]
        if set(table) != set(CURRENT_GRID_MA):
            raise RuntimeError("Shared %s calibration does not cover the full current grid" % device_name)
        device["qspec_gain_by_current"] = {
            current: row["qspec_gain"] for current, row in table.items()
        }
        device["qspec_length_by_current_us"] = {
            current: row["qspec_length_us"] for current, row in table.items()
        }
        device["sigma_by_current_us"] = {
            current: row["sigma_us"] for current, row in table.items()
        }
        device["pi_amp_by_current"] = {
            current: row["pi_amp"] for current, row in table.items()
        }
        device["res_base_by_current_mhz"] = {
            current: row["res_base_MHz"] for current, row in table.items()
        }
        device["readout_frequency_by_current_mhz"] = {
            current: row["readout_frequency_MHz"] for current, row in table.items()
        }
        device["readout_length_by_current_us"] = {
            current: row["readout_length_us"] for current, row in table.items()
        }
        device["readout_gain_by_current"] = {
            current: row["readout_gain"] for current, row in table.items()
        }
        device["readout_offset_by_current_mhz"] = {
            current: row["readout_offset_MHz"] for current, row in table.items()
        }
        device["expected_mhz"] = {
            current: row["qfreq_MHz"] for current, row in table.items()
        }
        device["verified_seed_currents_mA"] = set(CURRENT_GRID_MA)
        device["allow_provisional_rabi_by_current"] = set()

        zero_row = table[0.0]
        device["qspec_gain"] = zero_row["qspec_gain"]
        device["sigma_us"] = zero_row["sigma_us"]
        device["readout_gain"] = zero_row["readout_gain"]
        device["readout_length_us"] = zero_row["readout_length_us"]
        device["readout_offset_mhz"] = zero_row["readout_offset_MHz"]


apply_final_runtime_calibrations()


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


def selected_flux_environment(device, current_ma):
    """Select the swept line and hold the other characterized line at 0 mA."""
    qubit_index = int(device["index"])
    if qubit_index == 4:
        yoko3_current_ma, yoko4_current_ma = float(current_ma), 0.0
    elif qubit_index == 5:
        yoko3_current_ma, yoko4_current_ma = 0.0, float(current_ma)
    else:
        raise ValueError("Flux campaign device must use Python index 4 or 5")
    return {
        "PUCQ4_YOKO3_CURRENT_MA": str(yoko3_current_ma),
        "PUCQ4_YOKO4_CURRENT_MA": str(yoko4_current_ma),
    }


def run_round_robin(device, current_ma, stage, overrides):
    tag = f"{device['yoko'].lower()}_{device['physical'].lower()}_{current_ma:+05.1f}mA_{stage}"
    env = activated_subprocess_env()
    env.update({
        "PUCQ4_QS": str(device["index"]),
        "PUCQ4_SUB_STUDY": f"flux_{tag}",
        **selected_flux_environment(device, current_ma),
        **{key: str(value) for key, value in overrides.items()},
    })
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
    env.update(selected_flux_environment(device, current_ma))
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
    length = device.get("readout_length_by_current_us", {}).get(
        current_ma, device.get("readout_length_us", 5.0 if device["index"] == 4 else 3.0)
    )
    logs = []
    common = {
        "PUCQ4_OPT_RES_BASE_MHZ": res_base_mhz,
        "PUCQ4_OPT_QFREQ_MHZ": qfreq,
        "PUCQ4_OPT_SIGMA_US": sigma_us,
        "PUCQ4_OPT_PI_AMP": pi_amp,
    }
    if device["physical"] == "Q6" and current_ma == -10.0:
        common["PUCQ4_OPT_MAX_GAIN"] = 0.95
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
    minimum_fidelity = device.get("readout_min_fidelity_by_current", {}).get(current_ma, 0.20)
    if best["fidelity"] < minimum_fidelity:
        raise RuntimeError(
            f"Readout fidelity {best['fidelity']:.4f} remains below "
            f"{minimum_fidelity:.2f} after rescue sweep"
        )
    best["logs"] = logs
    return best


def calibrate_rabi_before_readout(device, current_ma, qfreq, sigma_us, res_base_mhz=None):
    """Break the Rabi/SSF dependency using averaged, unscaled I/Q first."""
    logs = []
    pi_amp = PI_TARGET
    allow_provisional = current_ma in device.get("allow_provisional_rabi_by_current", set())
    attempts = 1 if allow_provisional else 5
    for attempt in range(attempts):
        stage = "raw_iq_rabi" if attempt == 0 else f"raw_iq_rabi_retry{attempt}"
        overrides = {
            "PUCQ4_RUN_FLAGS": "res_spec,rabi",
            "PUCQ4_QFREQ_MHZ": qfreq,
            "PUCQ4_SIGMA_US": sigma_us,
            "PUCQ4_PI_AMP": PI_TARGET,
            "PUCQ4_RABI_SCALING": "false",
            "PUCQ4_RABI_REPS": device.get("rabi_reps_by_current", {}).get(current_ma, 1200),
            "PUCQ4_RABI_STEPS": device.get("rabi_steps_by_current", {}).get(current_ma, 86),
            "PUCQ4_RABI_STOP": device.get("rabi_stop_by_current", {}).get(current_ma, 0.85),
            "PUCQ4_READOUT_LENGTH_US": device.get("readout_length_by_current_us", {}).get(
                current_ma, device.get("readout_length_us", 3.0)
            ),
            "PUCQ4_READOUT_GAIN": device.get("readout_gain_by_current", {}).get(
                current_ma, device.get("readout_gain", 0.60)
            ),
            "PUCQ4_READOUT_OFFSET_MHZ": device.get("readout_offset_by_current_mhz", {}).get(
                current_ma, device.get("readout_offset_mhz", -0.04)
            ),
        }
        if res_base_mhz is not None:
            overrides["PUCQ4_RES_FREQ_MHZ"] = res_base_mhz
        text, log_path = run_round_robin(device, current_ma, stage, overrides)
        logs.append(log_path)
        pi_amp = last_float(
            r"Pi amplitude for qubit\s+\d+\s+is:\s+([0-9.+-eE]+)", text, "raw-I/Q pi amplitude"
        )
        if PI_WINDOW[0] <= pi_amp <= PI_WINDOW[1]:
            return sigma_us, pi_amp, logs
        if not 0.05 <= pi_amp <= 1.5:
            raise RuntimeError(f"Raw-I/Q Rabi returned nonphysical pi amplitude {pi_amp}")
        if allow_provisional:
            print(
                f"Using provisional raw-I/Q pi amplitude {pi_amp}; sigma retuning is deferred "
                f"for {device['physical']} at {current_ma:+.1f} mA",
                flush=True,
            )
            return sigma_us, pi_amp, logs
        # Damped correction avoids jumping between Rabi-fit aliases near a branch boundary.
        sigma_us = sigma_us * (pi_amp / PI_TARGET) ** 0.5
    raise RuntimeError(
        f"Raw-I/Q pi amplitude {pi_amp} did not enter {PI_WINDOW} after bounded sigma retries"
    )


def characterize_point(device, current_ma, sigma_us, res_seed_mhz=None):
    expected = device["expected_mhz"][current_ma]
    qspec_gain = device.get("qspec_gain_by_current", {}).get(current_ma, device["qspec_gain"])
    qspec_length_us = device.get("qspec_length_by_current_us", {}).get(current_ma, 20.0)
    # The -10 mA Q4 transition drifts while the flux line settles, so retain a
    # wider local window for both raw and population-calibrated validation.
    q4_minus10 = device["physical"] == "Q4" and current_ma == -10.0
    qspec_accept_window_mhz = (
        (QSPEC_FWHM_WINDOW_MHZ[0], 0.22) if q4_minus10 else QSPEC_FWHM_WINDOW_MHZ
    )
    qspec_half_span_mhz = 25.0 if q4_minus10 else (
        2.0 if device["physical"] == "Q6" and current_ma == -10.0 else 1.0
    )
    # Keep the broad locator coarse, then resolve the accepted 0.1-0.2 MHz
    # line with at least five samples across its FWHM in the final sweeps.
    qspec_step_mhz = 0.02 if q4_minus10 else 0.005
    qspec_reps = 2000 if q4_minus10 else 5000
    qspec_steps = int(2 * qspec_half_span_mhz / qspec_step_mhz) + 1
    coarse_log = None
    qfreq = expected
    if current_ma != 0.0 and current_ma not in device.get("verified_seed_currents_mA", set()):
        coarse_overrides = {
            "PUCQ4_RUN_FLAGS": "res_spec,q_spec",
            "PUCQ4_QFREQ_MHZ": expected,
            "PUCQ4_QSPEC_HALF_SPAN_MHZ": 25.0,
            "PUCQ4_QSPEC_STEPS": 501,
            "PUCQ4_QSPEC_REPS": 2000,
            "PUCQ4_QSPEC_GAIN": 0.03,
            "PUCQ4_QSPEC_LENGTH_US": 40.0,
            "PUCQ4_QSPEC_SCALING": "false",
            "PUCQ4_SIGMA_US": sigma_us,
            "PUCQ4_READOUT_LENGTH_US": device.get("readout_length_by_current_us", {}).get(
                current_ma, device.get("readout_length_us", 3.0)
            ),
            "PUCQ4_READOUT_GAIN": device.get("readout_gain_by_current", {}).get(
                current_ma, device.get("readout_gain", 0.60)
            ),
            "PUCQ4_READOUT_OFFSET_MHZ": device.get("readout_offset_by_current_mhz", {}).get(
                current_ma, device.get("readout_offset_mhz", -0.04)
            ),
        }
        if res_seed_mhz is not None:
            coarse_overrides["PUCQ4_RES_FREQ_MHZ"] = res_seed_mhz
        coarse_text, coarse_log = run_round_robin(
            device, current_ma, "locator", coarse_overrides
        )
        qfreq, _ = validated_qspec(coarse_text, expected, 25.0, "locator", 5.0)

    calibration_overrides = {
        "PUCQ4_RUN_FLAGS": "res_spec,q_spec",
        "PUCQ4_QFREQ_MHZ": qfreq,
        "PUCQ4_QSPEC_HALF_SPAN_MHZ": qspec_half_span_mhz,
        "PUCQ4_QSPEC_STEPS": qspec_steps,
        "PUCQ4_QSPEC_REPS": qspec_reps,
        "PUCQ4_QSPEC_GAIN": qspec_gain,
        "PUCQ4_QSPEC_LENGTH_US": qspec_length_us,
        "PUCQ4_QSPEC_SCALING": "false",
        "PUCQ4_SIGMA_US": sigma_us,
        "PUCQ4_READOUT_LENGTH_US": device.get("readout_length_by_current_us", {}).get(
            current_ma, device.get("readout_length_us", 3.0)
        ),
        "PUCQ4_READOUT_GAIN": device.get("readout_gain_by_current", {}).get(
            current_ma, device.get("readout_gain", 0.60)
        ),
        "PUCQ4_READOUT_OFFSET_MHZ": device.get("readout_offset_mhz", -0.04),
    }
    if res_seed_mhz is not None:
        calibration_overrides["PUCQ4_RES_FREQ_MHZ"] = res_seed_mhz
    initial_readout_offset = device.get("readout_offset_by_current_mhz", {}).get(current_ma)
    if initial_readout_offset is not None:
        calibration_overrides["PUCQ4_READOUT_OFFSET_MHZ"] = initial_readout_offset
    calibration_text, calibration_log = run_round_robin(
        device, current_ma, "calibration", calibration_overrides
    )
    res_base_mhz = resonator_frequency(calibration_text)
    qspec_gain_logs = [calibration_log]
    try:
        qfreq, qspec_fwhm_mhz = validated_qspec(
            calibration_text, qfreq, qspec_half_span_mhz, "narrow qspec", 5.0
        )
    except RuntimeError as qspec_fit_error:
        print(f"Initial narrow qspec fit rejected ({qspec_fit_error}); increasing gain", flush=True)
        qspec_fwhm_mhz = 0.0
    for attempt in range(10 if q4_minus10 else 6):
        if qspec_accept_window_mhz[0] <= qspec_fwhm_mhz <= qspec_accept_window_mhz[1]:
            break
        coarse_q4_edge = device["physical"] == "Q4" and current_ma in {-10.0, -2.5, 5.0}
        if qspec_fwhm_mhz > 0.20:
            gain_scale = 0.80 if coarse_q4_edge else 0.99
        else:
            gain_scale = 1.20 if coarse_q4_edge else 1.01
        qspec_gain = max(0.00025, min(0.03, qspec_gain * gain_scale))
        calibration_overrides.update({
            "PUCQ4_QFREQ_MHZ": qfreq,
            "PUCQ4_QSPEC_GAIN": qspec_gain,
        })
        calibration_text, gain_log = run_round_robin(
            device, current_ma, f"qspec_gain_retry{attempt + 1}", calibration_overrides
        )
        qspec_gain_logs.append(gain_log)
        res_base_mhz = resonator_frequency(calibration_text)
        try:
            qfreq, qspec_fwhm_mhz = validated_qspec(
                calibration_text, qfreq, qspec_half_span_mhz, "gain-tuned qspec", 5.0
            )
        except RuntimeError as qspec_fit_error:
            print(f"Gain-tuned qspec fit rejected ({qspec_fit_error}); increasing gain", flush=True)
            qspec_fwhm_mhz = 0.0
    if not qspec_accept_window_mhz[0] <= qspec_fwhm_mhz <= qspec_accept_window_mhz[1]:
        raise RuntimeError(
            f"Qspec FWHM {qspec_fwhm_mhz} MHz did not enter {qspec_accept_window_mhz} MHz "
            f"after bounded gain retries (last gain {qspec_gain})"
        )
    raw_rabi_logs = []
    bootstrap_log = None
    raw_qspec_valid = True
    prep_pi_amp = PI_TARGET
    try:
        qfreq, _ = validated_qspec(
            calibration_text, qfreq, qspec_half_span_mhz, "narrow qspec", qspec_accept_window_mhz[1]
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
                device, current_ma, qfreq, sigma_us, res_base_mhz=res_base_mhz
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
            device, current_ma, qfreq, sigma_us, res_base_mhz=res_base_mhz
        )
        readout = optimize_readout(
            device, current_ma, qfreq, sigma_us, prep_pi_amp, res_base_mhz
        )
    if not raw_qspec_valid:
        bootstrap_overrides = {
            "PUCQ4_READOUT_LENGTH_US": readout["length_us"],
            "PUCQ4_READOUT_GAIN": readout["gain"],
            "PUCQ4_READOUT_OFFSET_MHZ": readout["offset_mhz"],
            "PUCQ4_RES_FREQ_MHZ": res_base_mhz,
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
        "PUCQ4_RES_FREQ_MHZ": res_base_mhz,
    }
    final_text, final_log = run_round_robin(device, current_ma, "post_readout_calibration", {
        "PUCQ4_RUN_FLAGS": "res_spec,q_spec,rabi",
        "PUCQ4_QFREQ_MHZ": qfreq,
        "PUCQ4_QSPEC_HALF_SPAN_MHZ": qspec_half_span_mhz,
        "PUCQ4_QSPEC_STEPS": qspec_steps,
        "PUCQ4_QSPEC_REPS": qspec_reps,
        "PUCQ4_QSPEC_GAIN": qspec_gain,
        "PUCQ4_QSPEC_LENGTH_US": qspec_length_us,
        "PUCQ4_SIGMA_US": sigma_us,
        # Qspec's SSF scaling prepares |e> with this pulse before projecting the
        # entire spectroscopy trace.  Reuse the current-specific raw-Rabi
        # result instead of silently falling back to the zero-current pi gain.
        "PUCQ4_PI_AMP": prep_pi_amp,
        "PUCQ4_RABI_REPS": device.get("rabi_reps_by_current", {}).get(current_ma, 1200),
        "PUCQ4_RABI_STEPS": device.get("rabi_steps_by_current", {}).get(current_ma, 86),
        "PUCQ4_RABI_STOP": device.get("rabi_stop_by_current", {}).get(current_ma, 0.85),
        **readout_overrides,
    })
    # The final round-robin res scan can move the local center by one or more
    # frequency steps. Record that measured base so the saved offset exactly
    # reproduces the readout tone used for qspec/Rabi/coherence and later maps.
    res_base_mhz = resonator_frequency(final_text)
    readout_overrides["PUCQ4_RES_FREQ_MHZ"] = res_base_mhz
    qfreq, final_qspec_fwhm_mhz = validated_qspec(
        final_text, qfreq, qspec_half_span_mhz, "final narrow qspec", qspec_accept_window_mhz[1]
    )
    pi_amp = last_float(r"Pi amplitude for qubit\s+\d+\s+is:\s+([0-9.+-eE]+)", final_text, "pi amplitude")
    rabi_logs = raw_rabi_logs + [final_log]

    allow_provisional_rabi = current_ma in device.get("allow_provisional_rabi_by_current", set())
    for attempt in range(0 if allow_provisional_rabi else 3):
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
            "PUCQ4_RABI_REPS": device.get("rabi_reps_by_current", {}).get(current_ma, 1200),
            "PUCQ4_RABI_STEPS": device.get("rabi_steps_by_current", {}).get(current_ma, 86),
            "PUCQ4_RABI_STOP": device.get("rabi_stop_by_current", {}).get(current_ma, 0.85),
            **readout_overrides,
        })
        pi_amp = last_float(r"Pi amplitude for qubit\s+\d+\s+is:\s+([0-9.+-eE]+)", rabi_text, "pi amplitude")
        rabi_logs.append(rabi_log)
    rabi_in_target = PI_WINDOW[0] <= pi_amp <= PI_WINDOW[1]
    if not rabi_in_target and not allow_provisional_rabi:
        raise RuntimeError(f"Pi amplitude {pi_amp} did not enter {PI_WINDOW} after bounded sigma retries")
    if not rabi_in_target:
        print(
            f"Final pi amplitude {pi_amp} is provisional; sigma retuning is deferred for "
            f"{device['physical']} at {current_ma:+.1f} mA",
            flush=True,
        )

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
        "qspec_gain": qspec_gain,
        "qspec_fwhm_MHz": final_qspec_fwhm_mhz,
        "sigma_us": sigma_us,
        "pi_amp": pi_amp,
        "rabi_status": "accepted" if rabi_in_target else "provisional_sigma_recalibration_required",
        "sigma_recalibration_required": not rabi_in_target,
        "res_base_MHz": res_base_mhz,
        "readout": readout,
        "logs": {
            "locator": coarse_log,
            "calibration": calibration_log,
            "qspec_gain_calibration": qspec_gain_logs,
            "readout_bootstrap_qspec": bootstrap_log,
            "post_readout_calibration": final_log,
            "readout_optimization": readout["logs"],
            "rabi": rabi_logs,
            "coherence": coherence_log,
        },
        "completed": datetime.now().isoformat(),
    }


def locator_only_point(device, current_ma, sigma_us, res_seed_mhz=None):
    expected = device["expected_mhz"][current_ma]
    half_span_mhz = device.get("locator_half_span_by_current_mhz", {}).get(current_ma, 100.0)
    step_mhz = device.get("locator_step_by_current_mhz", {}).get(current_ma, 0.10)
    steps = int(2 * half_span_mhz / step_mhz) + 1
    overrides = {
        "PUCQ4_RUN_FLAGS": "res_spec,q_spec",
        "PUCQ4_QFREQ_MHZ": expected,
        "PUCQ4_QSPEC_HALF_SPAN_MHZ": half_span_mhz,
        "PUCQ4_QSPEC_STEPS": steps,
        "PUCQ4_QSPEC_REPS": device.get("locator_reps_by_current", {}).get(current_ma, 2000),
        "PUCQ4_QSPEC_GAIN": 0.03,
        "PUCQ4_QSPEC_LENGTH_US": 40.0,
        "PUCQ4_QSPEC_SCALING": "false",
        "PUCQ4_SIGMA_US": sigma_us,
        "PUCQ4_READOUT_LENGTH_US": device.get("readout_length_by_current_us", {}).get(
            current_ma, device.get("readout_length_us", 3.0)
        ),
        "PUCQ4_READOUT_GAIN": device.get("readout_gain_by_current", {}).get(
            current_ma, device.get("readout_gain", 0.60)
        ),
    }
    if res_seed_mhz is not None:
        overrides["PUCQ4_RES_FREQ_MHZ"] = res_seed_mhz
    locator_offset = device.get("locator_readout_offset_by_current_mhz", {}).get(current_ma)
    if locator_offset is not None:
        overrides["PUCQ4_READOUT_OFFSET_MHZ"] = locator_offset
    text, log_path = run_round_robin(device, current_ma, "wide_locator_diagnostic", overrides)
    qfreq = last_float(r"g-e Qubit \d+ frequency: ([0-9.+-eE]+)", text, "wide locator qfreq")
    return {
        "status": "diagnostic_complete",
        "current_mA": current_ma,
        "search_center_MHz": expected,
        "search_half_span_MHz": half_span_mhz,
        "fit_qfreq_MHz": qfreq,
        "log": log_path,
        "completed": datetime.now().isoformat(),
    }


def wide_res_point(device, current_ma):
    text, log_path = run_round_robin(device, current_ma, "wide_res_survey", {
        "PUCQ4_RUN_FLAGS": "res_spec",
        "PUCQ4_RES_START_MHZ": -15.0,
        "PUCQ4_RES_STEP_MHZ": 0.1,
        "PUCQ4_RES_STEPS": 301,
        "PUCQ4_RES_REPS": 600,
        "PUCQ4_RES_EXTREMUM": "dip",
    })
    return {
        "status": "complete",
        "current_mA": current_ma,
        "res_freq_MHz": resonator_frequency(text),
        "sweep_start_offset_MHz": -15.0,
        "sweep_stop_offset_MHz": 15.0,
        "step_MHz": 0.1,
        "steps": 301,
        "reps": 600,
        "extremum": "minimum",
        "log": log_path,
        "completed": datetime.now().isoformat(),
    }


def decode_h5_numeric(dataset):
    value = dataset[0] if dataset.shape else dataset[()]
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).ravel()
    if isinstance(value, (bytes, np.bytes_)):
        text = value.decode("utf-8")
    else:
        text = str(value)
    fields = [field for field in re.split(r"[,\s\[\]]+", text.strip()) if field]
    return np.asarray([float(field) for field in fields], dtype=float)


def qspec_population_metrics(device, log_path, qfreq_mhz):
    """Read the canonical HDF5 and score absolute, non-normalized population."""
    run_dir = DATA_ROOT / f"flux_{Path(log_path).stem}"
    h5_files = sorted(
        run_dir.glob("*/study_data/Data_h5/qspec_ge/*.h5"),
        key=lambda path: path.stat().st_mtime,
    )
    if not h5_files:
        raise RuntimeError(f"No canonical qspec HDF5 file under {run_dir}")
    h5_path = h5_files[-1]
    with h5py.File(h5_path, "r") as handle:
        group = handle[f"Q{device['index'] + 1}"]
        frequency = decode_h5_numeric(group["Frequencies"])
        population = decode_h5_numeric(group["Population"])
    length = min(frequency.size, population.size)
    frequency = frequency[:length]
    population = population[:length]
    order = np.argsort(frequency)
    frequency = frequency[order]
    population = population[order]
    finite = np.isfinite(frequency) & np.isfinite(population)
    frequency = frequency[finite]
    population = population[finite]
    if frequency.size < 20:
        raise RuntimeError("Qspec population trace contains fewer than 20 finite samples")
    step_mhz = abs(float(np.median(np.diff(frequency))))
    smooth_points = max(3, int(round(0.025 / max(step_mhz, 1e-12))))
    if smooth_points % 2 == 0:
        smooth_points += 1
    padding = smooth_points // 2
    smoothed = np.convolve(
        np.pad(population, (padding, padding), mode="edge"),
        np.ones(smooth_points) / smooth_points,
        mode="valid",
    )
    distance = np.abs(frequency - qfreq_mhz)
    local = distance <= 0.50
    background = (distance >= 0.70) & (distance <= 0.95)
    if local.sum() < 5 or background.sum() < 10:
        raise RuntimeError("Qspec population trace lacks the required local/background samples")
    baseline = float(np.median(smoothed[background]))
    peak = float(np.max(smoothed[local]))
    mad = float(np.median(np.abs(smoothed[background] - baseline)))
    noise = max(1.4826 * mad, np.finfo(float).eps)
    return {
        "h5": str(h5_path),
        "baseline_population": baseline,
        "peak_population": peak,
        "population_excursion": peak - baseline,
        "population_snr": (peak - baseline) / noise,
        "smoothing_points": smooth_points,
        "frequency_step_MHz": step_mhz,
    }


def verify_inactive_source(active_key):
    inactive_key = "q6" if active_key == "q4" else "q4"
    inactive = DEVICES[inactive_key]
    with YokogawaGS200(inactive["ip"], max_current=MAX_ABS_CURRENT_A) as source:
        if abs(source.get_current()) > 1e-7 or source.get_output():
            raise RuntimeError(f"Inactive {inactive['yoko']} is not safely at 0 A with output off")


def uniform_qspec_point(device, current_ma, point, reference_h5=None):
    """Acquire one row of the common-grid, population-calibrated qspec map."""
    qfreq = float(point["qfreq_MHz"])
    readout = point["readout"]
    readout_frequency_mhz = (
        float(point["res_base_MHz"]) + float(readout["offset_mhz"])
    )
    qspec_gain = device.get("uniform_qspec_gain_by_current", {}).get(
        current_ma, float(point["qspec_gain"])
    )
    steps = int(round(
        (UNIFORM_QSPEC_STOP_MHZ - UNIFORM_QSPEC_START_MHZ) / UNIFORM_QSPEC_STEP_MHZ
    )) + 1
    text, log_path = run_round_robin(device, current_ma, "uniform_qspec_5950_7200", {
        "PUCQ4_RUN_FLAGS": "q_spec",
        # Keep the validated local line in the device configuration so the
        # excited-state SSF calibration pulse is resonant.  The explicit
        # start/stop variables below control only the spectroscopy axis.
        "PUCQ4_QFREQ_MHZ": qfreq,
        "PUCQ4_QSPEC_START_MHZ": UNIFORM_QSPEC_START_MHZ,
        "PUCQ4_QSPEC_STOP_MHZ": UNIFORM_QSPEC_STOP_MHZ,
        "PUCQ4_QSPEC_STEPS": steps,
        "PUCQ4_QSPEC_REPS": UNIFORM_QSPEC_REPS,
        "PUCQ4_QSPEC_SSF_SHOTS": UNIFORM_QSPEC_SSF_SHOTS,
        "PUCQ4_QSPEC_SSF_RELAX_DELAY_US": UNIFORM_QSPEC_SSF_RELAX_DELAY_US,
        "PUCQ4_QSPEC_GAIN": qspec_gain,
        "PUCQ4_QSPEC_LENGTH_US": device.get("qspec_length_by_current_us", {}).get(
            current_ma, 20.0
        ),
        "PUCQ4_QSPEC_SCALING": "true",
        "PUCQ4_QSPEC_COMMON_GRID_MODE": "true",
        "PUCQ4_QSPEC_KEEP_SEED_ON_FIT_FAILURE": "true",
        "PUCQ4_QSPEC_LOG_CALIBRATION": "true",
        # There is intentionally no res-spec stage in this 62,501-point run.
        # Supply the final physical readout tone explicitly; the resonator base
        # and its optimized offset remain separate in the manifest below.
        "PUCQ4_READOUT_FREQ_MHZ": readout_frequency_mhz,
        "PUCQ4_READOUT_LENGTH_US": float(readout["length_us"]),
        "PUCQ4_READOUT_GAIN": float(readout["gain"]),
        "PUCQ4_READOUT_OFFSET_MHZ": float(readout["offset_mhz"]),
        "PUCQ4_SIGMA_US": float(point["sigma_us"]),
        "PUCQ4_PI_AMP": float(point["pi_amp"]),
    })
    reported_readout_frequency_mhz = last_float(
        r"Qspec readout frequency: ([0-9.+-eE]+) MHz", text,
        "uniform-qspec readout frequency",
    )
    if abs(reported_readout_frequency_mhz - readout_frequency_mhz) > 1e-6:
        raise RuntimeError(
            "Uniform qspec used an unexpected readout tone: "
            f"{reported_readout_frequency_mhz} MHz, expected "
            f"{readout_frequency_mhz} MHz"
        )
    run_dir = DATA_ROOT / f"flux_{Path(log_path).stem}"
    h5_files = sorted(
        run_dir.glob("*/study_data/Data_h5/qspec_ge/*.h5"),
        key=lambda path: path.stat().st_mtime,
    )
    if not h5_files:
        raise RuntimeError(f"Uniform qspec completed without a canonical qspec HDF5 file: {run_dir}")
    with h5py.File(h5_files[-1], "r") as handle:
        frequency = decode_h5_numeric(
            handle[f"Q{device['index'] + 1}"]["Frequencies"]
        )
    frequency = np.asarray(frequency, dtype=float)
    frequency = frequency[np.isfinite(frequency)]
    if frequency.size != steps:
        raise RuntimeError(
            f"Uniform qspec saved {frequency.size} finite frequencies; expected {steps}"
        )
    frequency = np.sort(frequency)
    actual_step_mhz = float(np.median(np.diff(frequency)))
    if (abs(frequency[0] - UNIFORM_QSPEC_START_MHZ) > 0.01
            or frequency[0] > qfreq - 25.0
            or frequency[-1] < qfreq + 25.0
            or abs(actual_step_mhz - UNIFORM_QSPEC_STEP_MHZ) > 0.001):
        raise RuntimeError(
            "Uniform qspec saved an unexpected actual frequency axis: "
            f"{frequency[0]}-{frequency[-1]} MHz, median step {actual_step_mhz} MHz"
        )
    if reference_h5 is not None:
        with h5py.File(reference_h5, "r") as handle:
            reference_frequency = decode_h5_numeric(
                handle[f"Q{device['index'] + 1}"]["Frequencies"]
            )
        reference_frequency = np.sort(
            np.asarray(reference_frequency, dtype=float)[
                np.isfinite(reference_frequency)
            ]
        )
        if (reference_frequency.size != frequency.size
                or not np.allclose(reference_frequency, frequency, rtol=0, atol=1e-6)):
            raise RuntimeError(
                "Uniform qspec actual frequency axis does not match the completed reference row"
            )
    centroid_matches = re.findall(
        r"Qspec SSF centroids: g_I=([0-9.+-eE]+), g_Q=([0-9.+-eE]+), "
        r"e_I=([0-9.+-eE]+), e_Q=([0-9.+-eE]+)",
        text,
    )
    if not centroid_matches:
        raise RuntimeError("Uniform qspec did not report the SSF calibration centroids")
    g_i, g_q, e_i, e_q = (float(value) for value in centroid_matches[-1])
    return {
        "status": "complete",
        "completed": datetime.now().isoformat(),
        "current_mA": current_ma,
        "qfreq_seed_MHz": qfreq,
        "frequency_start_MHz": UNIFORM_QSPEC_START_MHZ,
        "frequency_stop_MHz": UNIFORM_QSPEC_STOP_MHZ,
        "frequency_step_MHz": UNIFORM_QSPEC_STEP_MHZ,
        "actual_frequency_start_MHz": float(frequency[0]),
        "actual_frequency_stop_MHz": float(frequency[-1]),
        "actual_frequency_median_step_MHz": actual_step_mhz,
        "steps": steps,
        "reps": UNIFORM_QSPEC_REPS,
        "ssf_shots_per_state": UNIFORM_QSPEC_SSF_SHOTS,
        "ssf_relax_delay_us": UNIFORM_QSPEC_SSF_RELAX_DELAY_US,
        "qspec_gain": qspec_gain,
        "qspec_length_us": device.get("qspec_length_by_current_us", {}).get(current_ma, 20.0),
        "sigma_us": float(point["sigma_us"]),
        "pi_amp": float(point["pi_amp"]),
        "res_base_MHz": float(point["res_base_MHz"]),
        "readout": {
            "length_us": float(readout["length_us"]),
            "gain": float(readout["gain"]),
            "offset_mhz": float(readout["offset_mhz"]),
            "frequency_MHz": reported_readout_frequency_mhz,
        },
        "population_calibration": (
            "one SSF g/e centroid pair for this bias, applied to every frequency; "
            "common physical scale g=0/e=1; no row normalization"
        ),
        "ssf_centroids": {
            "g_I": g_i, "g_Q": g_q, "e_I": e_i, "e_Q": e_q,
        },
        "log": log_path,
        "h5": str(h5_files[-1]),
    }


def run_uniform_qspec_map(key, requested_currents, force=False):
    """Acquire identical-axis qspec rows using completed point calibrations."""
    device = DEVICES[key]
    verify_inactive_source(key)
    manifest = load_manifest()
    device_data = manifest["devices"].setdefault(key, {
        "physical": device["physical"], "row": device["row"],
        "yoko": device["yoko"], "ip": device["ip"], "points": {},
    })
    missing = [
        current_ma for current_ma in requested_currents
        if device_data.get("points", {}).get(f"{current_ma:+.1f}", {}).get("status") != "complete"
    ]
    if missing:
        raise RuntimeError(
            f"Uniform qspec requires completed local calibration at every requested bias; missing {missing}"
        )
    uniform_settings = {
        "frequency_start_MHz": UNIFORM_QSPEC_START_MHZ,
        "frequency_stop_MHz": UNIFORM_QSPEC_STOP_MHZ,
        "frequency_step_MHz": UNIFORM_QSPEC_STEP_MHZ,
        "steps": int(round(
            (UNIFORM_QSPEC_STOP_MHZ - UNIFORM_QSPEC_START_MHZ) / UNIFORM_QSPEC_STEP_MHZ
        )) + 1,
        "reps": UNIFORM_QSPEC_REPS,
        "ssf_shots_per_state": UNIFORM_QSPEC_SSF_SHOTS,
        "ssf_relax_delay_us": UNIFORM_QSPEC_SSF_RELAX_DELAY_US,
        "attenuation_dB": {
            "resonator_DAC": [10, 15], "qubit_DAC": [5, 4], "ADC": 17,
        },
        "population_definition": (
            "identical affine SSF convention at every bias: ground=0, excited=1; "
            "no per-curve or per-row normalization"
        ),
    }
    uniform = device_data.setdefault("uniform_qspec_map", {"points": {}})
    uniform.update(uniform_settings)
    uniform.setdefault("points", {})

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
                if not force and uniform["points"].get(key_ma, {}).get("status") == "complete":
                    continue
                source.ramp_current(current_ma * 1e-3, sweeprate=RAMP_RATE_A_PER_S)
                measured = source.get_current()
                if abs(measured - current_ma * 1e-3) > 1e-7:
                    raise RuntimeError(f"Yoko setpoint verification failed at {current_ma} mA")
                uniform["points"][key_ma] = {
                    "status": "running", "current_mA": current_ma,
                    "started": datetime.now().isoformat(),
                }
                save_manifest(manifest)
                print(
                    f"{device['yoko']} / {device['physical']}: uniform-qspec ramp verified "
                    f"at {current_ma:+.1f} mA",
                    flush=True,
                )
                settle_seconds = device.get("settle_seconds_by_current", {}).get(current_ma, 0)
                if settle_seconds:
                    print(
                        f"Settling {device['physical']} at {current_ma:+.1f} mA for "
                        f"{settle_seconds} s before the common-grid scan",
                        flush=True,
                    )
                    time.sleep(settle_seconds)
                try:
                    reference_h5 = next((
                        existing.get("h5")
                        for existing in uniform["points"].values()
                        if existing.get("status") == "complete" and existing.get("h5")
                    ), None)
                    result = uniform_qspec_point(
                        device, current_ma, device_data["points"][key_ma],
                        reference_h5=reference_h5,
                    )
                except KeyboardInterrupt:
                    uniform["points"][key_ma].update({
                        "status": "aborted", "failed": datetime.now().isoformat(),
                        "error": "Operator interrupted the common-grid acquisition",
                    })
                    save_manifest(manifest)
                    raise
                except Exception as exc:
                    uniform["points"][key_ma].update({
                        "status": "failed", "failed": datetime.now().isoformat(),
                        "error": str(exc),
                    })
                    save_manifest(manifest)
                    raise
                else:
                    uniform["points"][key_ma] = result
                    uniform["actual_frequency_start_MHz"] = result[
                        "actual_frequency_start_MHz"
                    ]
                    uniform["actual_frequency_stop_MHz"] = result[
                        "actual_frequency_stop_MHz"
                    ]
                    uniform["actual_frequency_median_step_MHz"] = result[
                        "actual_frequency_median_step_MHz"
                    ]
                    manifest["updated"] = datetime.now().isoformat()
                    save_manifest(manifest)
        finally:
            print(f"Returning {device['yoko']} to 0 A and disabling output", flush=True)
            source.ramp_current(0.0, sweeprate=RAMP_RATE_A_PER_S)
            source.set_output(False)
            if abs(source.get_current()) > 1e-7 or source.get_output():
                raise RuntimeError(f"Failed to leave {device['yoko']} at 0 A with output off")


def wide_uniform_qspec_calibration(device, device_data, current_ma):
    """Resolve the current-specific pulse and shifted resonator readout.

    Completed flux points remain the first choice.  The three Q6 negative-bias
    points are spectroscopy-only rather than full-chain calibrations, so their
    independently measured resonator bases and pulse calibrations are retained
    as explicitly provisional map inputs instead of being promoted to the
    accepted point table.
    """
    key_ma = f"{current_ma:+.1f}"
    point = device_data.get("points", {}).get(key_ma, {})
    if point.get("status") == "complete":
        qfreq = point.get("qfreq_MHz")
        res_base = point.get("res_base_MHz")
        readout = point.get("readout", {})
        sigma_us = point.get("sigma_us")
        pi_amp = point.get("pi_amp")
        source = "completed_flux_point"
    else:
        qfreq = device.get("expected_mhz", {}).get(current_ma)
        res_base = device.get("res_base_by_current_mhz", {}).get(current_ma)
        if res_base is None:
            res_base = (device_data.get("wide_res_survey", {}).get(key_ma, {})
                        .get("res_freq_MHz"))
        readout = {
            "length_us": device.get("readout_length_by_current_us", {}).get(
                current_ma, device["readout_length_us"]
            ),
            "gain": device.get("readout_gain_by_current", {}).get(
                current_ma, device["readout_gain"]
            ),
            "offset_mhz": device.get("readout_offset_by_current_mhz", {}).get(
                current_ma, device["readout_offset_mhz"]
            ),
        }
        sigma_us = device.get("sigma_by_current_us", {}).get(current_ma)
        pi_amp = device.get("pi_amp_by_current", {}).get(current_ma)
        source = "spectroscopy_only_provisional_map_input"
    required = {
        "qfreq_MHz": qfreq,
        "res_base_MHz": res_base,
        "readout_length_us": readout.get("length_us"),
        "readout_gain": readout.get("gain"),
        "readout_offset_MHz": readout.get("offset_mhz"),
        "sigma_us": sigma_us,
        "pi_amp": pi_amp,
    }
    missing = [name for name, value in required.items() if value is None]
    if missing:
        raise RuntimeError(
            f"No complete wide-qspec calibration for {device['physical']} "
            f"at {current_ma:+.1f} mA; missing {missing}"
        )
    readout_frequency_mhz = float(res_base) + float(readout["offset_mhz"])
    return {
        "source": source,
        "qfreq_MHz": float(qfreq),
        "qspec_gain": float(
            device.get("uniform_qspec_gain_by_current", {}).get(
                current_ma,
                point.get(
                    "qspec_gain",
                    device.get("qspec_gain_by_current", {}).get(
                        current_ma, device["qspec_gain"]
                    ),
                ),
            )
        ),
        "qspec_length_us": float(
            point.get(
                "qspec_length_us",
                device.get("qspec_length_by_current_us", {}).get(current_ma, 20.0),
            )
        ),
        "sigma_us": float(sigma_us),
        "pi_amp": float(pi_amp),
        "res_base_MHz": float(res_base),
        "readout": {
            "length_us": float(readout["length_us"]),
            "gain": float(readout["gain"]),
            "offset_mhz": float(readout["offset_mhz"]),
            "frequency_MHz": readout_frequency_mhz,
            "fidelity": readout.get("fidelity"),
        },
    }


def wide_uniform_qspec_segment(device, current_ma, calibration, settings,
                               segment_index, bounds_mhz, reference_h5=None):
    """Acquire one bounded band with the built-in qspec SSF calibration."""
    start_mhz, stop_mhz = (float(value) for value in bounds_mhz)
    step_mhz = float(settings["frequency_step_MHz"])
    steps = int(round((stop_mhz - start_mhz) / step_mhz)) + 1
    readout = calibration["readout"]
    stage = f"wqs{segment_index + 1}_{start_mhz:.0f}_{stop_mhz:.0f}"
    text, log_path = run_round_robin(device, current_ma, stage, {
        "PUCQ4_RUN_FLAGS": "q_spec",
        "PUCQ4_QFREQ_MHZ": calibration["qfreq_MHz"],
        "PUCQ4_QSPEC_START_MHZ": start_mhz,
        "PUCQ4_QSPEC_STOP_MHZ": stop_mhz,
        "PUCQ4_QSPEC_STEPS": steps,
        "PUCQ4_QSPEC_REPS": settings["reps"],
        "PUCQ4_QSPEC_SSF_SHOTS": UNIFORM_QSPEC_SSF_SHOTS,
        "PUCQ4_QSPEC_SSF_RELAX_DELAY_US": UNIFORM_QSPEC_SSF_RELAX_DELAY_US,
        "PUCQ4_QSPEC_GAIN": calibration["qspec_gain"],
        "PUCQ4_QSPEC_LENGTH_US": calibration["qspec_length_us"],
        "PUCQ4_QSPEC_SCALING": "true",
        "PUCQ4_QSPEC_COMMON_GRID_MODE": "true",
        "PUCQ4_QSPEC_KEEP_SEED_ON_FIT_FAILURE": "true",
        "PUCQ4_QSPEC_LOG_CALIBRATION": "true",
        "PUCQ4_READOUT_FREQ_MHZ": readout["frequency_MHz"],
        "PUCQ4_READOUT_LENGTH_US": readout["length_us"],
        "PUCQ4_READOUT_GAIN": readout["gain"],
        "PUCQ4_READOUT_OFFSET_MHZ": readout["offset_mhz"],
        "PUCQ4_SIGMA_US": calibration["sigma_us"],
        "PUCQ4_PI_AMP": calibration["pi_amp"],
    })
    reported_readout_frequency_mhz = last_float(
        r"Qspec readout frequency: ([0-9.+-eE]+) MHz", text,
        "wide-qspec readout frequency",
    )
    if abs(reported_readout_frequency_mhz - readout["frequency_MHz"]) > 1e-6:
        raise RuntimeError(
            "Wide qspec used an unexpected readout tone: "
            f"{reported_readout_frequency_mhz} MHz, expected "
            f"{readout['frequency_MHz']} MHz"
        )
    run_dir = DATA_ROOT / f"flux_{Path(log_path).stem}"
    h5_files = sorted(
        run_dir.glob("*/study_data/Data_h5/qspec_ge/*.h5"),
        key=lambda path: path.stat().st_mtime,
    )
    if not h5_files:
        raise RuntimeError(
            f"Wide qspec completed without a canonical qspec HDF5 file: {run_dir}"
        )
    h5_path = h5_files[-1]
    with h5py.File(h5_path, "r") as handle:
        group = handle[f"Q{device['index'] + 1}"]
        frequency = decode_h5_numeric(group["Frequencies"])
        population = decode_h5_numeric(group["Population"])
    frequency = np.asarray(frequency, dtype=float)
    population = np.asarray(population, dtype=float)
    finite = np.isfinite(frequency) & np.isfinite(population)
    frequency = frequency[finite]
    if frequency.size != steps:
        raise RuntimeError(
            f"Wide qspec saved {frequency.size} finite points; expected {steps}"
        )
    frequency = np.sort(frequency)
    actual_step_mhz = float(np.median(np.diff(frequency)))
    if (abs(frequency[0] - start_mhz) > 0.01
            or abs(actual_step_mhz - step_mhz) > 0.001):
        raise RuntimeError(
            "Wide qspec saved an unexpected actual frequency axis: "
            f"{frequency[0]}-{frequency[-1]} MHz, median step "
            f"{actual_step_mhz} MHz"
        )
    if reference_h5 is not None:
        with h5py.File(reference_h5, "r") as handle:
            reference_frequency = decode_h5_numeric(
                handle[f"Q{device['index'] + 1}"]["Frequencies"]
            )
        reference_frequency = np.sort(
            np.asarray(reference_frequency, dtype=float)[
                np.isfinite(reference_frequency)
            ]
        )
        if (reference_frequency.size != frequency.size
                or not np.allclose(reference_frequency, frequency, rtol=0, atol=1e-6)):
            raise RuntimeError(
                "Wide qspec band axis does not match the completed reference band"
            )
    centroid_matches = re.findall(
        r"Qspec SSF centroids: g_I=([0-9.+-eE]+), g_Q=([0-9.+-eE]+), "
        r"e_I=([0-9.+-eE]+), e_Q=([0-9.+-eE]+)",
        text,
    )
    if not centroid_matches:
        raise RuntimeError("Wide qspec did not report its built-in SSF centroids")
    g_i, g_q, e_i, e_q = (float(value) for value in centroid_matches[-1])
    return {
        "status": "complete",
        "completed": datetime.now().isoformat(),
        "segment_index": segment_index,
        "requested_start_MHz": start_mhz,
        "requested_stop_MHz": stop_mhz,
        "requested_step_MHz": step_mhz,
        "actual_start_MHz": float(frequency[0]),
        "actual_stop_MHz": float(frequency[-1]),
        "actual_median_step_MHz": actual_step_mhz,
        "steps": steps,
        "reps": int(settings["reps"]),
        "ssf_shots_per_state": UNIFORM_QSPEC_SSF_SHOTS,
        "ssf_relax_delay_us": UNIFORM_QSPEC_SSF_RELAX_DELAY_US,
        "ssf_centroids": {"g_I": g_i, "g_Q": g_q, "e_I": e_i, "e_Q": e_q},
        "population_calibration": (
            "built-in qspec SSF calibration acquired for this current and band; "
            "signed affine projection g=0/e=1; no row normalization"
        ),
        "log": log_path,
        "h5": str(h5_path),
    }


def run_wide_uniform_qspec_map(key, requested_currents, force=False):
    """Acquire the lower-frequency segmented population map for one Yoko."""
    device = DEVICES[key]
    settings = WIDE_UNIFORM_QSPEC_SETTINGS[key]
    manifest = load_manifest()
    device_data = manifest["devices"].setdefault(key, {
        "physical": device["physical"], "row": device["row"],
        "yoko": device["yoko"], "ip": device["ip"], "points": {},
    })
    calibrations = {
        current_ma: wide_uniform_qspec_calibration(device, device_data, current_ma)
        for current_ma in requested_currents
    }
    verify_inactive_source(key)
    map_settings = {
        **settings,
        "segments_MHz": [list(bounds) for bounds in settings["segments_MHz"]],
        "combined_steps": int(round(
            (settings["frequency_stop_MHz"] - settings["frequency_start_MHz"])
            / settings["frequency_step_MHz"]
        )) + 1,
        "ssf_shots_per_state": UNIFORM_QSPEC_SSF_SHOTS,
        "ssf_relax_delay_us": UNIFORM_QSPEC_SSF_RELAX_DELAY_US,
        "attenuation_dB": {
            "resonator_DAC": [10, 15], "qubit_DAC": [5, 4], "ADC": 17,
        },
        "population_definition": (
            "built-in current-and-band-specific SSF affine calibration, ground=0 "
            "and excited=1; no per-curve or per-row normalization"
        ),
    }
    wide_map = device_data.get("wide_uniform_qspec_map")
    signature = (
        settings["frequency_start_MHz"], settings["frequency_stop_MHz"],
        settings["frequency_step_MHz"], settings["reps"],
        tuple(tuple(bounds) for bounds in settings["segments_MHz"]),
    )
    existing_signature = None
    if wide_map:
        existing_signature = (
            wide_map.get("frequency_start_MHz"), wide_map.get("frequency_stop_MHz"),
            wide_map.get("frequency_step_MHz"), wide_map.get("reps"),
            tuple(tuple(bounds) for bounds in wide_map.get("segments_MHz", [])),
        )
    if wide_map and existing_signature != signature:
        device_data.setdefault("superseded_wide_uniform_qspec_maps", []).append(wide_map)
        wide_map = None
    if wide_map is None:
        wide_map = {"points": {}}
        device_data["wide_uniform_qspec_map"] = wide_map
    wide_map.update(map_settings)
    wide_map.setdefault("points", {})
    save_manifest(manifest)

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
                existing = wide_map["points"].get(key_ma, {})
                if not force and existing.get("status") == "complete":
                    continue
                calibration = calibrations[current_ma]
                segments = [] if force else list(existing.get("segments", []))
                while len(segments) < len(settings["segments_MHz"]):
                    segments.append({"status": "pending", "segment_index": len(segments)})
                record = {
                    "status": "running",
                    "started": datetime.now().isoformat(),
                    "current_mA": current_ma,
                    **calibration,
                    "segments": segments,
                }
                wide_map["points"][key_ma] = record
                save_manifest(manifest)
                source.ramp_current(current_ma * 1e-3, sweeprate=RAMP_RATE_A_PER_S)
                measured = source.get_current()
                if abs(measured - current_ma * 1e-3) > 1e-7:
                    raise RuntimeError(f"Yoko setpoint verification failed at {current_ma} mA")
                print(
                    f"{device['yoko']} / {device['physical']}: wide-qspec ramp verified "
                    f"at {current_ma:+.1f} mA; readout "
                    f"{calibration['readout']['frequency_MHz']:.3f} MHz",
                    flush=True,
                )
                settle_seconds = device.get("settle_seconds_by_current", {}).get(current_ma, 0)
                if settle_seconds:
                    print(
                        f"Settling {device['physical']} at {current_ma:+.1f} mA for "
                        f"{settle_seconds} s before the lower-frequency map",
                        flush=True,
                    )
                    time.sleep(settle_seconds)
                try:
                    for segment_index, bounds in enumerate(settings["segments_MHz"]):
                        if (not force
                                and record["segments"][segment_index].get("status") == "complete"
                                and Path(record["segments"][segment_index].get("h5", "")).is_file()):
                            continue
                        reference_h5 = next((
                            other.get("segments", [])[segment_index].get("h5")
                            for other in wide_map["points"].values()
                            if len(other.get("segments", [])) > segment_index
                            and other["segments"][segment_index].get("status") == "complete"
                            and other["segments"][segment_index].get("h5")
                        ), None)
                        record["segments"][segment_index] = {
                            "status": "running", "segment_index": segment_index,
                            "requested_start_MHz": bounds[0],
                            "requested_stop_MHz": bounds[1],
                            "started": datetime.now().isoformat(),
                        }
                        save_manifest(manifest)
                        record["segments"][segment_index] = wide_uniform_qspec_segment(
                            device, current_ma, calibration, settings,
                            segment_index, bounds, reference_h5=reference_h5,
                        )
                        save_manifest(manifest)
                except KeyboardInterrupt:
                    record.update({
                        "status": "aborted", "failed": datetime.now().isoformat(),
                        "error": "Operator interrupted the lower-frequency qspec map",
                    })
                    save_manifest(manifest)
                    raise
                except Exception as exc:
                    record.update({
                        "status": "failed", "failed": datetime.now().isoformat(),
                        "error": str(exc),
                    })
                    save_manifest(manifest)
                    raise
                record.update({"status": "complete", "completed": datetime.now().isoformat()})
                manifest["updated"] = datetime.now().isoformat()
                save_manifest(manifest)
        finally:
            print(f"Returning {device['yoko']} to 0 A and disabling output", flush=True)
            source.ramp_current(0.0, sweeprate=RAMP_RATE_A_PER_S)
            source.set_output(False)
            if abs(source.get_current()) > 1e-7 or source.get_output():
                raise RuntimeError(f"Failed to leave {device['yoko']} at 0 A with output off")


def run_device(key, requested_currents, locator_only=False, wide_res_only=False, force=False):
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
                if (not force and not wide_res_only
                        and device_data["points"].get(key_ma, {}).get("status") == "complete"):
                    sigma_us = device_data["points"][key_ma]["sigma_us"]
                    continue
                source.ramp_current(current_ma * 1e-3, sweeprate=RAMP_RATE_A_PER_S)
                measured = source.get_current()
                if abs(measured - current_ma * 1e-3) > 1e-7:
                    raise RuntimeError(f"Yoko setpoint verification failed at {current_ma} mA")
                state_bucket = (device_data.setdefault("wide_res_survey", {})
                                if wide_res_only else device_data["points"])
                state_bucket[key_ma] = {
                    "status": "running", "current_mA": current_ma,
                    "started": datetime.now().isoformat(),
                }
                save_manifest(manifest)
                print(
                    f"{device['yoko']} / {device['physical']}: ramp verified at {current_ma:+.1f} mA",
                    flush=True,
                )
                settle_seconds = device.get("settle_seconds_by_current", {}).get(current_ma, 0)
                if settle_seconds:
                    print(
                        f"Settling {device['physical']} at {current_ma:+.1f} mA for "
                        f"{settle_seconds} s before spectroscopy",
                        flush=True,
                    )
                    time.sleep(settle_seconds)
                try:
                    if wide_res_only:
                        result = wide_res_point(device, current_ma)
                    elif locator_only:
                        res_seed = device_data.get("wide_res_survey", {}).get(key_ma, {}).get("res_freq_MHz")
                        result = locator_only_point(
                            device, current_ma, sigma_us, res_seed_mhz=res_seed,
                        )
                    else:
                        res_seed = device_data.get("wide_res_survey", {}).get(key_ma, {}).get("res_freq_MHz")
                        result = characterize_point(device, current_ma, sigma_us, res_seed_mhz=res_seed)
                except KeyboardInterrupt:
                    state_bucket[key_ma].update({
                        "status": "aborted",
                        "failed": datetime.now().isoformat(),
                        "error": "Operator interrupted after quality rejection",
                    })
                    save_manifest(manifest)
                    raise
                except Exception as exc:
                    state_bucket[key_ma].update({
                        "status": "failed",
                        "failed": datetime.now().isoformat(),
                        "error": str(exc),
                    })
                    save_manifest(manifest)
                    raise
                else:
                    if wide_res_only:
                        state_bucket[key_ma] = result
                    elif locator_only:
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
                        res_base_mhz=device_data.get("wide_res_survey", {}).get(key_ma, {}).get("res_freq_MHz"),
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


def calibration_seed(device, device_data, current_ma):
    """Return measured parameters for a bounded recovery/diagnostic run."""
    key_ma = f"{current_ma:+.1f}"
    point = device_data.get("points", {}).get(key_ma, {})
    qfreq = float(point.get("qfreq_MHz", device["expected_mhz"][current_ma]))
    sigma_us = float(point.get(
        "sigma_us", device.get("sigma_by_current_us", {}).get(current_ma, device["sigma_us"])
    ))
    pi_amp = point.get("pi_amp", device.get("pi_amp_by_current", {}).get(current_ma))
    if pi_amp is None:
        raise RuntimeError(
            f"No measured current-specific pi amplitude is available at {current_ma:+.1f} mA"
        )
    res_base_mhz = point.get(
        "res_base_MHz", device.get("res_base_by_current_mhz", {}).get(current_ma)
    )
    if res_base_mhz is None:
        res_base_mhz = (
            device_data.get("wide_res_survey", {}).get(key_ma, {}).get("res_freq_MHz")
        )
    if res_base_mhz is None:
        raise RuntimeError(
            f"No measured current-specific resonator frequency is available at {current_ma:+.1f} mA"
        )
    readout = point.get("readout") or {
        "length_us": device.get("readout_length_by_current_us", {}).get(
            current_ma, device["readout_length_us"]
        ),
        "gain": device.get("readout_gain_by_current", {}).get(
            current_ma, device["readout_gain"]
        ),
        "offset_mhz": device.get("readout_offset_by_current_mhz", {}).get(
            current_ma, device["readout_offset_mhz"]
        ),
        "fidelity": device.get("readout_fidelity_by_current", {}).get(current_ma),
    }
    result = {
        "qfreq_MHz": qfreq,
        "sigma_us": sigma_us,
        "pi_amp": float(pi_amp),
        "res_base_MHz": float(res_base_mhz),
        "readout": {
            "length_us": float(readout["length_us"]),
            "gain": float(readout["gain"]),
            "offset_mhz": float(readout["offset_mhz"]),
        },
    }
    if readout.get("fidelity") is not None:
        result["readout"]["fidelity"] = float(readout["fidelity"])
    return result


def population_qspec_gain_point(device, current_ma, gain, seed):
    gain_key = f"{gain:.9g}"
    gain_tag = gain_key.replace(".", "p")
    qspec_length_us = device.get("qspec_length_by_current_us", {}).get(
        current_ma, 20.0
    )
    text, log_path = run_round_robin(
        device, current_ma, f"population_qspec_gain_{gain_tag}", {
            "PUCQ4_RUN_FLAGS": "res_spec,q_spec",
            "PUCQ4_QFREQ_MHZ": seed["qfreq_MHz"],
            "PUCQ4_QSPEC_HALF_SPAN_MHZ": 1.0,
            "PUCQ4_QSPEC_STEPS": 401,
            "PUCQ4_QSPEC_REPS": 5000,
            "PUCQ4_QSPEC_GAIN": gain,
            "PUCQ4_QSPEC_LENGTH_US": qspec_length_us,
            "PUCQ4_QSPEC_SCALING": "true",
            "PUCQ4_QSPEC_LOG_CALIBRATION": "true",
            "PUCQ4_SIGMA_US": seed["sigma_us"],
            "PUCQ4_PI_AMP": seed["pi_amp"],
            "PUCQ4_RES_FREQ_MHZ": seed["res_base_MHz"],
            "PUCQ4_READOUT_LENGTH_US": seed["readout"]["length_us"],
            "PUCQ4_READOUT_GAIN": seed["readout"]["gain"],
            "PUCQ4_READOUT_OFFSET_MHZ": seed["readout"]["offset_mhz"],
        },
    )
    qfreq = last_float(
        r"g-e Qubit \d+ frequency: ([0-9.+-eE]+)", text,
        "population-qspec frequency",
    )
    if abs(qfreq - seed["qfreq_MHz"]) > 0.5:
        raise RuntimeError(
            f"Population-qspec fit {qfreq} MHz is more than 0.5 MHz from the "
            f"validated wide-scan seed {seed['qfreq_MHz']} MHz"
        )
    fwhm = abs(last_float(
        r"g-e Qubit \d+ FWHM: ([0-9.+-eE]+) MHz", text,
        "population-qspec FWHM",
    ))
    metrics = qspec_population_metrics(device, log_path, qfreq)
    return {
        "status": "complete", "completed": datetime.now().isoformat(),
        "gain": gain, "qfreq_MHz": qfreq, "fwhm_MHz": fwhm,
        "qspec_length_us": qspec_length_us,
        "res_base_MHz": resonator_frequency(text),
        "readout": seed["readout"], "log": log_path, **metrics,
    }


def run_qspec_population_checks(key, requested_currents, gains):
    """Measure local qspec power candidates on an absolute SSF population scale."""
    device = DEVICES[key]
    verify_inactive_source(key)
    manifest = load_manifest()
    device_data = manifest["devices"].setdefault(key, {
        "physical": device["physical"], "row": device["row"],
        "yoko": device["yoko"], "ip": device["ip"], "points": {},
    })
    diagnostics = device_data.setdefault("qspec_population_gain_checks", {})
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
                seed = calibration_seed(device, device_data, current_ma)
                source.ramp_current(current_ma * 1e-3, sweeprate=RAMP_RATE_A_PER_S)
                if abs(source.get_current() - current_ma * 1e-3) > 1e-7:
                    raise RuntimeError(f"Yoko setpoint verification failed at {current_ma} mA")
                print(
                    f"{device['yoko']} / {device['physical']}: population-qspec ramp verified "
                    f"at {current_ma:+.1f} mA",
                    flush=True,
                )
                settle_seconds = device.get("settle_seconds_by_current", {}).get(current_ma, 0)
                if settle_seconds:
                    time.sleep(settle_seconds)
                current_results = diagnostics.setdefault(key_ma, {})
                for gain in gains:
                    gain_key = f"{gain:.9g}"
                    current_results[gain_key] = {
                        "status": "running", "started": datetime.now().isoformat(),
                    }
                    save_manifest(manifest)
                    try:
                        result = population_qspec_gain_point(
                            device, current_ma, gain, seed
                        )
                    except Exception as exc:
                        current_results[gain_key].update({
                            "status": "failed", "failed": datetime.now().isoformat(),
                            "error": str(exc),
                        })
                        save_manifest(manifest)
                        raise
                    current_results[gain_key] = result
                    save_manifest(manifest)
                    print(
                        f"Population-qspec gain {gain:.9g}: "
                        f"f01={result['qfreq_MHz']:.9f} MHz, "
                        f"FWHM={result['fwhm_MHz']:.6f} MHz, "
                        f"peak={result['peak_population']:.4f}, "
                        f"excursion={result['population_excursion']:.4f}, "
                        f"SNR={result['population_snr']:.2f}",
                        flush=True,
                    )
        finally:
            print(f"Returning {device['yoko']} to 0 A and disabling output", flush=True)
            source.ramp_current(0.0, sweeprate=RAMP_RATE_A_PER_S)
            source.set_output(False)
            if abs(source.get_current()) > 1e-7 or source.get_output():
                raise RuntimeError(f"Failed to leave {device['yoko']} at 0 A with output off")


def finalize_points_from_population_check(key, requested_currents, gain):
    """Finish Rabi/coherence from an accepted absolute-population qspec row."""
    device = DEVICES[key]
    verify_inactive_source(key)
    manifest = load_manifest()
    device_data = manifest["devices"].setdefault(key, {
        "physical": device["physical"], "row": device["row"],
        "yoko": device["yoko"], "ip": device["ip"], "points": {},
    })
    gain_key = f"{gain:.9g}"
    checks = device_data.get("qspec_population_gain_checks", {})
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
                selected = checks.get(key_ma, {}).get(gain_key)
                if not selected or selected.get("status") != "complete":
                    raise RuntimeError(
                        f"No completed population-qspec check exists at {current_ma:+.1f} mA "
                        f"and gain {gain:.9g}"
                    )
                qspec_fwhm_window = device.get(
                    "qspec_fwhm_window_by_current_mhz", {}
                ).get(current_ma, QSPEC_FWHM_WINDOW_MHZ)
                if not qspec_fwhm_window[0] <= selected["fwhm_MHz"] <= qspec_fwhm_window[1]:
                    raise RuntimeError(
                        f"Selected qspec FWHM {selected['fwhm_MHz']} MHz is outside "
                        f"the measured acceptance window {qspec_fwhm_window}"
                    )
                if not 0.30 <= selected["peak_population"] <= 0.50:
                    raise RuntimeError(
                        f"Selected qspec peak population {selected['peak_population']} is outside 0.30-0.50"
                    )
                if selected["population_snr"] < 5.0:
                    raise RuntimeError(
                        f"Selected qspec population SNR {selected['population_snr']} is below 5"
                    )

                seed = calibration_seed(device, device_data, current_ma)
                qfreq = float(selected["qfreq_MHz"])
                res_base_mhz = float(selected["res_base_MHz"])
                sigma_us = seed["sigma_us"]
                pi_amp = seed["pi_amp"]
                readout = dict(seed["readout"])
                previous = device_data.get("points", {}).get(key_ma)
                if previous:
                    device_data.setdefault("superseded_point_attempts", {}).setdefault(
                        key_ma, []
                    ).append(previous)
                device_data.setdefault("points", {})[key_ma] = {
                    "status": "finalization_running", "current_mA": current_ma,
                    "started": datetime.now().isoformat(),
                    "selected_qspec_gain": gain,
                }
                save_manifest(manifest)
                source.ramp_current(current_ma * 1e-3, sweeprate=RAMP_RATE_A_PER_S)
                if abs(source.get_current() - current_ma * 1e-3) > 1e-7:
                    raise RuntimeError(f"Yoko setpoint verification failed at {current_ma} mA")
                print(
                    f"{device['yoko']} / {device['physical']}: finalization ramp verified "
                    f"at {current_ma:+.1f} mA",
                    flush=True,
                )
                settle_seconds = device.get("settle_seconds_by_current", {}).get(current_ma, 0)
                if settle_seconds:
                    time.sleep(settle_seconds)
                rabi_logs = []
                allow_provisional = current_ma in device.get(
                    "allow_provisional_rabi_by_current", set()
                )
                try:
                    for attempt in range(4):
                        stage = "final_population_rabi" if attempt == 0 else f"final_population_rabi_retry{attempt}"
                        rabi_text, rabi_log = run_round_robin(device, current_ma, stage, {
                            "PUCQ4_RUN_FLAGS": "res_spec,rabi",
                            "PUCQ4_QFREQ_MHZ": qfreq,
                            "PUCQ4_SIGMA_US": sigma_us,
                            "PUCQ4_PI_AMP": pi_amp,
                            "PUCQ4_RABI_SCALING": "true",
                            "PUCQ4_RABI_REPS": device.get(
                                "rabi_reps_by_current", {}
                            ).get(current_ma, 1200),
                            "PUCQ4_RABI_STEPS": device.get(
                                "rabi_steps_by_current", {}
                            ).get(current_ma, 86),
                            "PUCQ4_RABI_STOP": device.get(
                                "rabi_stop_by_current", {}
                            ).get(current_ma, 0.85),
                            "PUCQ4_RES_FREQ_MHZ": res_base_mhz,
                            "PUCQ4_READOUT_LENGTH_US": readout["length_us"],
                            "PUCQ4_READOUT_GAIN": readout["gain"],
                            "PUCQ4_READOUT_OFFSET_MHZ": readout["offset_mhz"],
                        })
                        rabi_logs.append(rabi_log)
                        res_base_mhz = resonator_frequency(rabi_text)
                        measured_pi = last_float(
                            r"Pi amplitude for qubit\s+\d+\s+is:\s+([0-9.+-eE]+)",
                            rabi_text, "final population pi amplitude",
                        )
                        pi_amp = measured_pi
                        if PI_WINDOW[0] <= pi_amp <= PI_WINDOW[1] or allow_provisional:
                            break
                        if not 0.05 <= pi_amp <= 1.5:
                            raise RuntimeError(f"Final Rabi returned nonphysical pi amplitude {pi_amp}")
                        sigma_us = sigma_us * (pi_amp / PI_TARGET) ** 0.5
                    if not allow_provisional and not PI_WINDOW[0] <= pi_amp <= PI_WINDOW[1]:
                        raise RuntimeError(
                            f"Final pi amplitude {pi_amp} did not enter {PI_WINDOW}"
                        )
                    coherence_text, coherence_log = run_round_robin(
                        device, current_ma, "coherence_final", {
                        "PUCQ4_RUN_FLAGS": "res_spec,t1,t2r,t2e",
                        "PUCQ4_QFREQ_MHZ": qfreq,
                        "PUCQ4_SIGMA_US": sigma_us,
                        "PUCQ4_PI_AMP": pi_amp,
                        "PUCQ4_RES_FREQ_MHZ": res_base_mhz,
                        "PUCQ4_READOUT_LENGTH_US": readout["length_us"],
                        "PUCQ4_READOUT_GAIN": readout["gain"],
                        "PUCQ4_READOUT_OFFSET_MHZ": readout["offset_mhz"],
                        },
                    )
                    res_base_mhz = resonator_frequency(coherence_text)
                except Exception as exc:
                    device_data["points"][key_ma].update({
                        "status": "failed", "failed": datetime.now().isoformat(),
                        "error": str(exc),
                    })
                    save_manifest(manifest)
                    raise

                prefix = (
                    f"{device['yoko'].lower()}_{device['physical'].lower()}_"
                    f"{current_ma:+05.1f}mA"
                )
                readout_logs = sorted(str(path) for path in CAMPAIGN_ROOT.glob(
                    f"{prefix}_readout_*.log"
                ))
                device_data["points"][key_ma] = {
                    "status": "complete", "completed": datetime.now().isoformat(),
                    "current_mA": current_ma,
                    "expected_qfreq_MHz": device["expected_mhz"][current_ma],
                    "qfreq_MHz": qfreq,
                    "qspec_gain": gain,
                    "qspec_length_us": selected["qspec_length_us"],
                    "qspec_fwhm_MHz": selected["fwhm_MHz"],
                    "qspec_fwhm_acceptance_MHz": list(qspec_fwhm_window),
                    "qspec_population": {
                        "baseline": selected["baseline_population"],
                        "peak": selected["peak_population"],
                        "excursion": selected["population_excursion"],
                        "snr": selected["population_snr"],
                        "definition": "absolute signed SSF projection, g=0/e=1; no trace normalization",
                    },
                    "sigma_us": sigma_us,
                    "pi_amp": pi_amp,
                    "rabi_status": (
                        "accepted" if PI_WINDOW[0] <= pi_amp <= PI_WINDOW[1]
                        else "provisional_sigma_recalibration_required"
                    ),
                    "sigma_recalibration_required": not PI_WINDOW[0] <= pi_amp <= PI_WINDOW[1],
                    "res_base_MHz": res_base_mhz,
                    "readout": readout,
                    "logs": {
                        "locator": None,
                        "calibration": selected["log"],
                        "qspec_gain_calibration": [selected["log"]],
                        "readout_bootstrap_qspec": None,
                        "post_readout_calibration": selected["log"],
                        "post_readout_qspec": selected["log"],
                        "readout_optimization": readout_logs,
                        "rabi": rabi_logs,
                        "coherence": coherence_log,
                    },
                    "finalization": {
                        "method": "accepted population-qspec check followed by fresh Rabi and coherence",
                        "selected_gain_key": gain_key,
                        "selected_qspec_h5": selected["h5"],
                    },
                }
                manifest["updated"] = datetime.now().isoformat()
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
                        help="Run the configured high-power diagnostic locator only")
    parser.add_argument("--rabi-only", action="store_true",
                        help="Calibrate sigma/pi at an established current-specific qfreq only")
    parser.add_argument("--wide-res-only", action="store_true",
                        help="Acquire a -15 to +15 MHz resonator survey at each selected current")
    parser.add_argument("--uniform-qspec-map", action="store_true",
                        help="Acquire 5950-7200 MHz qspec on one identical fine grid at every bias")
    parser.add_argument("--wide-uniform-qspec-map", action="store_true",
                        help="Acquire the lower-frequency segmented qspec map with per-band SSF calibration")
    parser.add_argument("--qspec-population-gain", type=float, action="append",
                        help="Run a guarded local SSF-population qspec check at this gain; repeat as needed")
    parser.add_argument("--finalize-qspec-gain", type=float,
                        help="Finalize Rabi/coherence from a completed accepted population-qspec gain check")
    parser.add_argument("--force", action="store_true",
                        help="Reacquire selected points even when the manifest marks them complete")
    return parser.parse_args()


def main():
    args = parse_args()
    population_check = bool(args.qspec_population_gain)
    finalize_check = args.finalize_qspec_gain is not None
    if population_check and any(gain <= 0.0 or gain > 0.03
                                for gain in args.qspec_population_gain):
        raise ValueError("Population-qspec gains must be in the bounded (0, 0.03] range")
    if sum((args.locator_only, args.rabi_only, args.wide_res_only,
            args.uniform_qspec_map, args.wide_uniform_qspec_map,
            population_check, finalize_check)) > 1:
        raise ValueError(
            "--locator-only, --rabi-only, --wide-res-only, --uniform-qspec-map, "
            "--wide-uniform-qspec-map, and "
            "--qspec-population-gain/--finalize-qspec-gain "
            "are mutually exclusive"
        )
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
        elif population_check:
            run_qspec_population_checks(key, requested, args.qspec_population_gain)
        elif finalize_check:
            finalize_points_from_population_check(
                key, requested, args.finalize_qspec_gain
            )
        elif args.uniform_qspec_map:
            run_uniform_qspec_map(key, requested, force=args.force)
        elif args.wide_uniform_qspec_map:
            run_wide_uniform_qspec_map(key, requested, force=args.force)
        else:
            run_device(key, requested, locator_only=args.locator_only,
                       wide_res_only=args.wide_res_only, force=args.force)


if __name__ == "__main__":
    main()
