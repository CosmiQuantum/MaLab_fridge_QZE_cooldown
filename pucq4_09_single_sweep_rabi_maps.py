"""Acquire fixed-sigma current-by-gain Rabi maps for the two PUCQ4 flux lines.

This controller runs only ``rabi`` through the canonical
``round_robin_benchmark.py`` path.  Every current uses the observed peak from
the completed continuous qspec sweep and the exact physical resonator readout
tone used for that qspec row.  Each device has one fixed Gaussian sigma and one
identical Rabi gain grid at all nine currents.  The round-robin Rabi wrapper
performs a current-specific built-in SSF calibration and saves physical qubit
population with the canonical HDF5 result.

Hardware is touched only from :func:`main`.  The inactive source is verified at
0 A/output off, and every active source is returned to that state in ``finally``.
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np

from malab_yokogawa_gs200 import YokogawaGS200
from pucq4_07_flux_campaign import (
    ACQUISITION_ORDER_MA,
    CAMPAIGN_ROOT,
    CURRENT_GRID_MA,
    DATA_ROOT,
    DEVICES,
    MAX_ABS_CURRENT_A,
    RAMP_RATE_A_PER_S,
    decode_h5_numeric,
    run_round_robin,
    verify_inactive_source,
)


QSPEC_MANIFEST = CAMPAIGN_ROOT / "single_sweep_qspec_manifest.json"
MANIFEST = CAMPAIGN_ROOT / "single_sweep_rabi_manifest.json"
DEFAULT_REPS = 1200
DEFAULT_ROUNDS = 1
GAIN_START = 0.0
GAIN_STOP = 1.0
GAIN_STEPS = 201
RABI_RELAX_DELAY_US = 250.0
SSF_SHOTS_PER_STATE = 2000
SSF_RELAX_DELAY_US = 250.0
MINIMUM_REVIEW_POPULATION_SPAN = 0.05

# These values are intentionally local to this map and do not modify the
# current-indexed or zero-current sigmas in shared configuration.  They were
# selected from the established sigma/pi products so the predicted first Rabi
# extremum stays on the common 0-1 gain axis at every current.
FIXED_SIGMA_US = {
    "q4": 0.060,
    "q6": 0.090,
}


def current_key(current_ma):
    return f"{current_ma:+.1f}"


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_manifest():
    if MANIFEST.exists():
        return load_json(MANIFEST)
    return {
        "created": datetime.now().isoformat(),
        "purpose": (
            "fixed-sigma current-by-gain Rabi maps using observed continuous-"
            "qspec peaks and the qspec row's physical readout tone"
        ),
        "source_qspec_manifest": str(QSPEC_MANIFEST),
        "current_grid_mA": CURRENT_GRID_MA,
        "devices": {},
    }


def save_manifest(manifest):
    CAMPAIGN_ROOT.mkdir(parents=True, exist_ok=True)
    temporary = MANIFEST.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    temporary.replace(MANIFEST)


def settings_for(device_key, reps):
    return {
        "gain_start": GAIN_START,
        "gain_stop": GAIN_STOP,
        "gain_steps": GAIN_STEPS,
        "reps": int(reps),
        "rounds": DEFAULT_ROUNDS,
        "rabi_relax_delay_us": RABI_RELAX_DELAY_US,
        "fixed_sigma_us": FIXED_SIGMA_US[device_key],
        "ssf_shots_per_state": SSF_SHOTS_PER_STATE,
        "ssf_relax_delay_us": SSF_RELAX_DELAY_US,
        "runner": "round_robin_benchmark.py",
        "run_flags": ["rabi"],
        "sweeps_per_current": 1,
        "population_definition": (
            "one current-specific built-in SSF affine calibration for the full "
            "gain row; signed ground=0/excited=1 projection; no row normalization"
        ),
        "attenuation_dB": {
            "resonator_DAC": [10, 15],
            "qubit_DAC": [5, 4],
            "ADC": 17,
        },
    }


def stage_name(settings):
    sigma_ns = 1000.0 * settings["fixed_sigma_us"]
    return (
        f"fixed_rabi_s{sigma_ns:.0f}ns_g0_1_"
        f"n{settings['gain_steps']}_r{settings['reps']}"
    )


def expected_log_path(device, current_ma, settings):
    tag = (
        f"{device['yoko'].lower()}_{device['physical'].lower()}_"
        f"{current_ma:+05.1f}mA_{stage_name(settings)}"
    )
    return CAMPAIGN_ROOT / f"{tag}.log"


def prospective_artifact_path(device, current_ma, settings):
    tag = (
        f"{device['yoko'].lower()}_{device['physical'].lower()}_"
        f"{current_ma:+05.1f}mA_{stage_name(settings)}"
    )
    filename = (
        "2099-12-31_23-59-59_rabi_ge_results_batch_1_"
        "Num_per_batch1.h5"
    )
    return (
        DATA_ROOT / f"flux_{tag}" / "2099-12-31_23-59-59" /
        "study_data" / "Data_h5" / "rabi_ge" / filename
    )


def verify_paths(device, currents, settings):
    candidates = [
        prospective_artifact_path(device, current_ma, settings)
        for current_ma in currents
    ]
    longest = max(candidates, key=lambda path: len(str(path)))
    if len(str(longest)) >= 260:
        raise RuntimeError(
            f"Prospective Windows artifact path is {len(str(longest))} "
            f"characters: {longest}"
        )
    return longest


def qspec_calibrations(device_key, currents):
    if not QSPEC_MANIFEST.is_file():
        raise FileNotFoundError(QSPEC_MANIFEST)
    qspec_manifest = load_json(QSPEC_MANIFEST)
    map_data = (
        qspec_manifest.get("devices", {}).get(device_key, {}).get("map", {})
    )
    points = map_data.get("points", {})
    calibrations = {}
    for current_ma in currents:
        key = current_key(current_ma)
        point = points.get(key, {})
        if point.get("status") != "complete":
            raise RuntimeError(
                f"Continuous qspec is not complete for {device_key} {key} mA"
            )
        source_h5 = Path(point.get("h5", ""))
        if not source_h5.is_file():
            raise FileNotFoundError(source_h5)
        qfreq_mhz = float(point["peak_metrics"]["frequency_MHz"])
        readout = point["readout"]
        values = [
            qfreq_mhz,
            float(point["res_base_MHz"]),
            float(readout["frequency_MHz"]),
            float(readout["length_us"]),
            float(readout["gain"]),
            float(readout["offset_mhz"]),
        ]
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"Non-finite qspec/readout calibration at {device_key} {key} mA"
            )
        calibrations[current_ma] = {
            "qfreq_MHz": qfreq_mhz,
            "qfreq_source": "continuous qspec observed positive peak",
            "qfreq_seed_MHz": float(point["qfreq_seed_MHz"]),
            "source_qspec_h5": str(source_h5),
            "source_qspec_log": point["log"],
            "source_qspec_local_snr": float(
                point["peak_metrics"]["positive_snr"]
            ),
            "source_sigma_us": float(point["sigma_us"]),
            "source_pi_amp": float(point["pi_amp"]),
            "res_base_MHz": float(point["res_base_MHz"]),
            "readout": {
                "frequency_MHz": float(readout["frequency_MHz"]),
                "length_us": float(readout["length_us"]),
                "gain": float(readout["gain"]),
                "offset_mhz": float(readout["offset_mhz"]),
                "fidelity": readout.get("fidelity"),
            },
        }
    return calibrations


def predicted_pi_seed(calibration, fixed_sigma_us):
    prediction = (
        calibration["source_pi_amp"] * calibration["source_sigma_us"]
        / fixed_sigma_us
    )
    return float(np.clip(prediction, 0.02, 0.98))


def canonical_artifacts_for_log(log_path):
    run_dir = DATA_ROOT / f"flux_{Path(log_path).stem}"
    h5_files = sorted(
        run_dir.glob("*/study_data/Data_h5/rabi_ge/*.h5"),
        key=lambda path: path.stat().st_mtime,
    )
    plot_files = sorted(
        run_dir.glob(
            "*/documentation/power_rabi_ge_plots/*.png"
        ),
        key=lambda path: path.stat().st_mtime,
    )
    if not h5_files:
        raise RuntimeError(
            f"Rabi completed without canonical HDF5 under {run_dir}"
        )
    if not plot_files:
        raise RuntimeError(
            f"Rabi completed without its canonical plot under {run_dir}"
        )
    return h5_files[-1], plot_files[-1]


def load_rabi_trace(h5_path, device):
    with h5py.File(h5_path, "r") as handle:
        group = handle[f"Q{device['index'] + 1}"]
        required = ("Gains", "Population", "I", "Q")
        missing = [name for name in required if name not in group]
        if missing:
            raise RuntimeError(f"Rabi HDF5 is missing {missing}: {h5_path}")
        gains = decode_h5_numeric(group["Gains"])
        population = decode_h5_numeric(group["Population"])
        i_data = decode_h5_numeric(group["I"])
        q_data = decode_h5_numeric(group["Q"])
        ssf = {}
        for name in ("ss_I_g", "ss_Q_g", "ss_I_e", "ss_Q_e"):
            if name not in group:
                raise RuntimeError(
                    f"Rabi HDF5 is missing calibration dataset {name}"
                )
            ssf[name] = decode_h5_numeric(group[name])
    length = min(gains.size, population.size, i_data.size, q_data.size)
    gains = np.asarray(gains[:length], dtype=float)
    population = np.asarray(population[:length], dtype=float)
    i_data = np.asarray(i_data[:length], dtype=float)
    q_data = np.asarray(q_data[:length], dtype=float)
    finite = (
        np.isfinite(gains) & np.isfinite(population)
        & np.isfinite(i_data) & np.isfinite(q_data)
    )
    order = np.argsort(gains[finite])
    return (
        gains[finite][order], population[finite][order],
        i_data[finite][order], q_data[finite][order], ssf,
    )


def smooth_trace(values, points=5):
    points = max(1, int(points))
    if points % 2 == 0:
        points += 1
    if points == 1:
        return np.asarray(values, dtype=float).copy()
    pad = points // 2
    return np.convolve(
        np.pad(values, (pad, pad), mode="edge"),
        np.ones(points, dtype=float) / points,
        mode="valid",
    )


def trace_metrics(gains, population):
    smoothed = smooth_trace(population, 5)
    lower, upper = np.percentile(smoothed, [5.0, 95.0])
    centered = smoothed - np.median(smoothed)
    noise = max(
        1.4826 * float(np.median(np.abs(np.diff(smoothed)))),
        np.finfo(float).eps,
    )
    return {
        "population_min": float(np.min(smoothed)),
        "population_max": float(np.max(smoothed)),
        "population_span_5_95": float(upper - lower),
        "largest_excursion_from_median": float(np.max(np.abs(centered))),
        "difference_noise_estimate": noise,
        "excursion_to_difference_noise": float(
            np.max(np.abs(centered)) / noise
        ),
        "smoothing_points": 5,
        "gain_at_population_max": float(gains[np.argmax(smoothed)]),
        "gain_at_population_min": float(gains[np.argmin(smoothed)]),
    }


def parse_single_match(pattern, text, label):
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {label}; found {len(matches)}"
        )
    return matches[0]


def acquire_row(device, current_ma, calibration, settings,
                reference_h5=None, existing_log_path=None):
    readout = calibration["readout"]
    pi_seed = predicted_pi_seed(
        calibration, settings["fixed_sigma_us"]
    )
    if existing_log_path is None:
        notes = (
            "Fixed-sigma current-by-gain Rabi map. Physical "
            f"{device['physical']}/{device['row']}, {current_ma:+.1f} mA; "
            "qfreq is the observed continuous-qspec peak and readout is the "
            "physical tone used for that qspec row."
        )
        text, log_path = run_round_robin(
            device, current_ma, stage_name(settings), {
                "PUCQ4_RUN_FLAGS": "rabi",
                "PUCQ4_QFREQ_MHZ": calibration["qfreq_MHz"],
                "PUCQ4_READOUT_FREQ_MHZ": readout["frequency_MHz"],
                "PUCQ4_READOUT_LENGTH_US": readout["length_us"],
                "PUCQ4_READOUT_GAIN": readout["gain"],
                "PUCQ4_READOUT_OFFSET_MHZ": readout["offset_mhz"],
                "PUCQ4_SIGMA_US": settings["fixed_sigma_us"],
                "PUCQ4_PI_AMP": pi_seed,
                "PUCQ4_RABI_START": settings["gain_start"],
                "PUCQ4_RABI_STOP": settings["gain_stop"],
                "PUCQ4_RABI_STEPS": settings["gain_steps"],
                "PUCQ4_RABI_REPS": settings["reps"],
                "PUCQ4_RABI_ROUNDS": settings["rounds"],
                "PUCQ4_RABI_RELAX_DELAY_US": (
                    settings["rabi_relax_delay_us"]
                ),
                "PUCQ4_RABI_SSF_SHOTS": settings["ssf_shots_per_state"],
                "PUCQ4_RABI_SSF_RELAX_DELAY_US": (
                    settings["ssf_relax_delay_us"]
                ),
                "PUCQ4_RABI_SCALING": "true",
                "PUCQ4_SUBSTUDY_NOTES": notes,
            },
        )
    else:
        log_path = str(existing_log_path)
        text = Path(existing_log_path).read_text(
            encoding="utf-8", errors="replace"
        )

    operating = parse_single_match(
        r"Rabi operating point: qubit_frequency_MHz=([0-9.+-eE]+), "
        r"readout_frequency_MHz=([0-9.+-eE]+), sigma_us=([0-9.+-eE]+)",
        text, "Rabi operating-point report",
    )
    reported_qfreq, reported_readout, reported_sigma = (
        float(value) for value in operating
    )
    if abs(reported_qfreq - calibration["qfreq_MHz"]) > 1e-6:
        raise RuntimeError(
            f"Rabi used qfreq {reported_qfreq} MHz; expected "
            f"{calibration['qfreq_MHz']} MHz"
        )
    if abs(reported_readout - readout["frequency_MHz"]) > 1e-6:
        raise RuntimeError(
            f"Rabi used readout {reported_readout} MHz; expected "
            f"{readout['frequency_MHz']} MHz"
        )
    if abs(reported_sigma - settings["fixed_sigma_us"]) > 1e-12:
        raise RuntimeError(
            f"Rabi used sigma {reported_sigma} us; expected fixed "
            f"{settings['fixed_sigma_us']} us"
        )

    sweep = parse_single_match(
        r"Rabi sweep settings: start=([0-9.+-eE]+), "
        r"stop=([0-9.+-eE]+), steps=([0-9]+), reps=([0-9]+), "
        r"rounds=([0-9]+), relax_delay_us=([0-9.+-eE]+), "
        r"ssf_shots_per_state=([0-9]+), "
        r"ssf_relax_delay_us=([0-9.+-eE]+)",
        text, "Rabi sweep-settings report",
    )
    sweep_values = [float(value) for value in sweep]
    expected = [
        settings["gain_start"], settings["gain_stop"],
        settings["gain_steps"], settings["reps"], settings["rounds"],
        settings["rabi_relax_delay_us"], settings["ssf_shots_per_state"],
        settings["ssf_relax_delay_us"],
    ]
    if not np.allclose(sweep_values, expected, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            f"Reported Rabi settings {sweep_values} differ from {expected}"
        )

    centroids = parse_single_match(
        r"Rabi SSF centroids: g_I=([0-9.+-eE]+), "
        r"g_Q=([0-9.+-eE]+), e_I=([0-9.+-eE]+), "
        r"e_Q=([0-9.+-eE]+)",
        text, "Rabi SSF centroid report",
    )
    g_i, g_q, e_i, e_q = (float(value) for value in centroids)
    separation = float(np.hypot(e_i - g_i, e_q - g_q))
    if not np.isfinite(separation) or separation <= 0.0:
        raise RuntimeError("Rabi SSF calibration has zero/non-finite separation")

    pi_matches = re.findall(
        r"g-e Pi amplitude for qubit\s+\d+\s+is:\s+([0-9.+-eE]+)",
        text,
    )
    if not pi_matches:
        raise RuntimeError("Could not parse the fitted Rabi pi amplitude")
    fitted_pi = float(pi_matches[-1])
    h5_path, plot_path = canonical_artifacts_for_log(log_path)
    gains, population, _, _, ssf = load_rabi_trace(h5_path, device)
    if gains.size != settings["gain_steps"]:
        raise RuntimeError(
            f"Saved {gains.size} finite gain points; expected "
            f"{settings['gain_steps']}"
        )
    if np.any(np.diff(gains) <= 0):
        raise RuntimeError("Saved Rabi gain axis is not strictly increasing")
    actual_step = float(np.median(np.diff(gains)))
    requested_step = (
        (settings["gain_stop"] - settings["gain_start"])
        / (settings["gain_steps"] - 1)
    )
    endpoint_shortfall = settings["gain_stop"] - float(gains[-1])
    maximum_endpoint_shortfall = max(0.01, 2.0 * requested_step)
    if (abs(gains[0] - settings["gain_start"]) > 1e-4
            or not 0.0 <= endpoint_shortfall <= maximum_endpoint_shortfall
            or abs(actual_step - requested_step) > 1e-4):
        raise RuntimeError(
            f"Unexpected saved gain axis {gains[0]}-{gains[-1]}, "
            f"median step {actual_step}, endpoint shortfall "
            f"{endpoint_shortfall}"
        )
    if reference_h5:
        reference_gains, _, _, _, _ = load_rabi_trace(reference_h5, device)
        if (reference_gains.size != gains.size
                or not np.allclose(
                    reference_gains, gains, rtol=0.0, atol=1e-9
                )):
            raise RuntimeError(
                "Actual gain axis differs from the completed reference row"
            )
    ssf_lengths = {name: int(values.size) for name, values in ssf.items()}
    if any(
            length != settings["ssf_shots_per_state"]
            for length in ssf_lengths.values()):
        raise RuntimeError(
            f"Unexpected saved SSF shot counts: {ssf_lengths}"
        )
    metrics = trace_metrics(gains, population)
    return {
        "status": "complete",
        "completed": datetime.now().isoformat(),
        "current_mA": current_ma,
        "qfreq_MHz": reported_qfreq,
        "qfreq_source": calibration["qfreq_source"],
        "qfreq_seed_MHz": calibration["qfreq_seed_MHz"],
        "source_qspec_h5": calibration["source_qspec_h5"],
        "source_qspec_log": calibration["source_qspec_log"],
        "source_qspec_local_snr": calibration["source_qspec_local_snr"],
        "res_base_MHz": calibration["res_base_MHz"],
        "readout": {
            **readout,
            "frequency_MHz": reported_readout,
        },
        "fixed_sigma_us": reported_sigma,
        "initial_pi_amp_seed": pi_seed,
        "fitted_pi_amp": fitted_pi,
        "actual_gain_start": float(gains[0]),
        "actual_gain_stop": float(gains[-1]),
        "actual_gain_median_step": actual_step,
        "requested_gain_stop_shortfall": endpoint_shortfall,
        "gain_steps": int(gains.size),
        "reps": settings["reps"],
        "rounds": settings["rounds"],
        "rabi_relax_delay_us": settings["rabi_relax_delay_us"],
        "ssf_shots_per_state": settings["ssf_shots_per_state"],
        "ssf_relax_delay_us": settings["ssf_relax_delay_us"],
        "ssf_centroids": {
            "g_I": g_i, "g_Q": g_q, "e_I": e_i, "e_Q": e_q,
            "separation": separation,
        },
        "saved_ssf_array_lengths": ssf_lengths,
        "population_metrics": metrics,
        "population_visibility_pass": (
            metrics["population_span_5_95"]
            >= MINIMUM_REVIEW_POPULATION_SPAN
        ),
        "log": log_path,
        "h5": str(h5_path),
        "plot": str(plot_path),
    }


def prepare_map(manifest, device_key, settings):
    device = DEVICES[device_key]
    entry = manifest["devices"].setdefault(device_key, {
        "physical": device["physical"],
        "row": device["row"],
        "python_index": device["index"],
        "yoko": device["yoko"],
        "ip": device["ip"],
    })
    signature_keys = (
        "gain_start", "gain_stop", "gain_steps", "reps", "rounds",
        "rabi_relax_delay_us", "fixed_sigma_us", "ssf_shots_per_state",
        "ssf_relax_delay_us",
    )
    new_signature = tuple(settings[key] for key in signature_keys)
    map_data = entry.get("map")
    if map_data:
        old_signature = tuple(map_data.get(key) for key in signature_keys)
        if old_signature != new_signature:
            entry.setdefault("superseded_maps", []).append(map_data)
            map_data = None
    if map_data is None:
        map_data = {"points": {}}
        entry["map"] = map_data
    points = map_data.setdefault("points", {})
    map_data.update(settings)
    map_data["points"] = points
    return map_data


def print_plan(device_key, device, currents, settings, calibrations,
               longest_path):
    nominal_seconds = (
        settings["gain_steps"] * settings["reps"] * settings["rounds"]
        * settings["rabi_relax_delay_us"] * 1e-6
    )
    settling_seconds = sum(
        device.get("settle_seconds_by_current", {}).get(current, 0)
        for current in currents
    )
    print(
        f"{device['yoko']} / physical {device['physical']} / {device['row']} "
        f"(Python index {device['index']}): {len(currents)} Rabi sweep(s), "
        f"gain {settings['gain_start']:.3f}-{settings['gain_stop']:.3f}, "
        f"{settings['gain_steps']} points, {settings['reps']} reps x "
        f"{settings['rounds']} round(s), fixed sigma "
        f"{settings['fixed_sigma_us']:.6f} us",
        flush=True,
    )
    print(
        f"Nominal programmed acquisition/settling floor: "
        f"{nominal_seconds * len(currents) / 60.0:.1f} min + "
        f"{settling_seconds / 60.0:.1f} min; longest artifact path "
        f"{len(str(longest_path))} chars",
        flush=True,
    )
    for current_ma in currents:
        calibration = calibrations[current_ma]
        readout = calibration["readout"]
        print(
            f"  {current_ma:+.1f} mA: qfreq="
            f"{calibration['qfreq_MHz']:.6f} MHz; resonator base="
            f"{calibration['res_base_MHz']:.3f} MHz; readout="
            f"{readout['frequency_MHz']:.3f} MHz / "
            f"{readout['length_us']:.1f} us / gain "
            f"{readout['gain']:.4f} / offset "
            f"{readout['offset_mhz']:+.2f} MHz",
            flush=True,
        )


def recover_completed_rows(manifest, map_data, device, currents, settings,
                           calibrations):
    for current_ma in currents:
        key = current_key(current_ma)
        existing = map_data["points"].get(key, {})
        if existing.get("status") == "complete":
            continue
        log_path = expected_log_path(device, current_ma, settings)
        if not log_path.is_file():
            continue
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        if "timetaken=" not in log_text:
            continue
        reference_h5 = next((
            point.get("h5") for point in map_data["points"].values()
            if point.get("status") == "complete"
            and Path(point.get("h5", "")).is_file()
        ), None)
        try:
            result = acquire_row(
                device, current_ma, calibrations[current_ma], settings,
                reference_h5=reference_h5,
                existing_log_path=log_path,
            )
        except Exception as exc:
            print(
                f"Existing {device['physical']} {current_ma:+.1f} mA Rabi "
                f"row was not recoverable: {exc}",
                flush=True,
            )
            continue
        result["recovered_without_reacquisition"] = True
        map_data["points"][key] = result
        manifest["updated"] = datetime.now().isoformat()
        save_manifest(manifest)
        print(
            f"Recovered completed {device['physical']} {current_ma:+.1f} mA "
            "Rabi HDF5 without touching hardware",
            flush=True,
        )


def run_device(device_key, currents, reps, force=False, plan_only=False):
    device = DEVICES[device_key]
    settings = settings_for(device_key, reps)
    longest_path = verify_paths(device, currents, settings)
    calibrations = qspec_calibrations(device_key, currents)
    print_plan(
        device_key, device, currents, settings, calibrations, longest_path
    )
    if plan_only:
        print(
            "Plan only: no source, QICK, attenuator, data, or manifest state "
            "was changed",
            flush=True,
        )
        return

    manifest = load_manifest()
    map_data = prepare_map(manifest, device_key, settings)
    save_manifest(manifest)
    if not force:
        recover_completed_rows(
            manifest, map_data, device, currents, settings, calibrations
        )
    if all(
            map_data["points"].get(current_key(current), {}).get("status")
            == "complete" and Path(
                map_data["points"][current_key(current)].get("h5", "")
            ).is_file()
            for current in currents):
        print(f"{device['physical']} requested rows are already complete")
        return

    verify_inactive_source(device_key)
    with YokogawaGS200(
            device["ip"], max_current=MAX_ABS_CURRENT_A) as source:
        try:
            identity = source.get_id()
            mode = source.get_mode()
            output = source.get_output()
            current = source.get_current()
            source_range = source.get_range()
            voltage_limit = source.get_voltage_limit()
            print(
                f"Active-source preflight: id={identity}; mode={mode}; "
                f"output={output}; current_A={current:.12g}; "
                f"range_A={source_range:.12g}; voltage_limit_V="
                f"{voltage_limit:.12g}",
                flush=True,
            )
            if not mode.startswith("CURR"):
                raise RuntimeError(
                    f"{device['yoko']} is not in current-source mode"
                )
            if abs(current) > 1e-7 or output:
                raise RuntimeError(
                    f"{device['yoko']} did not start at 0 A/output off"
                )
            source.set_range(0.01)
            if abs(source.get_range() - 0.01) > 1e-9:
                raise RuntimeError(
                    f"{device['yoko']} failed the 0.01 A range check"
                )
            source.set_current(0.0)
            source.set_output(True)
            for current_ma in currents:
                key = current_key(current_ma)
                existing = map_data["points"].get(key, {})
                if (not force and existing.get("status") == "complete"
                        and Path(existing.get("h5", "")).is_file()):
                    print(
                        f"{device['physical']} {current_ma:+.1f} mA already "
                        "complete; skipping",
                        flush=True,
                    )
                    continue
                calibration = calibrations[current_ma]
                map_data["points"][key] = {
                    "status": "running",
                    "started": datetime.now().isoformat(),
                    "current_mA": current_ma,
                    "qfreq_MHz": calibration["qfreq_MHz"],
                    "readout": calibration["readout"],
                    "fixed_sigma_us": settings["fixed_sigma_us"],
                }
                save_manifest(manifest)
                source.ramp_current(
                    current_ma * 1e-3, sweeprate=RAMP_RATE_A_PER_S
                )
                measured = source.get_current()
                if abs(measured - current_ma * 1e-3) > 1e-7:
                    raise RuntimeError(
                        f"Yoko setpoint verification failed at {current_ma} mA"
                    )
                print(
                    f"{device['yoko']} verified at {current_ma:+.1f} mA; "
                    f"qfreq={calibration['qfreq_MHz']:.6f} MHz; "
                    f"readout={calibration['readout']['frequency_MHz']:.3f} "
                    f"MHz; sigma={settings['fixed_sigma_us']:.6f} us",
                    flush=True,
                )
                settle_seconds = device.get(
                    "settle_seconds_by_current", {}
                ).get(current_ma, 0)
                if settle_seconds:
                    print(
                        f"Settling for {settle_seconds} s before acquisition",
                        flush=True,
                    )
                    time.sleep(settle_seconds)
                try:
                    reference_h5 = next((
                        point.get("h5")
                        for point in map_data["points"].values()
                        if point.get("status") == "complete"
                        and Path(point.get("h5", "")).is_file()
                    ), None)
                    result = acquire_row(
                        device, current_ma, calibration, settings,
                        reference_h5=reference_h5,
                    )
                except KeyboardInterrupt:
                    map_data["points"][key].update({
                        "status": "aborted",
                        "failed": datetime.now().isoformat(),
                        "error": "Operator interrupted the fixed-sigma Rabi row",
                    })
                    save_manifest(manifest)
                    raise
                except Exception as exc:
                    map_data["points"][key].update({
                        "status": "failed",
                        "failed": datetime.now().isoformat(),
                        "error": str(exc),
                    })
                    save_manifest(manifest)
                    raise
                map_data["points"][key] = result
                manifest["updated"] = datetime.now().isoformat()
                save_manifest(manifest)
                metrics = result["population_metrics"]
                print(
                    f"{device['physical']} {current_ma:+.1f} mA complete: "
                    f"pi={result['fitted_pi_amp']:.6f}, population 5-95 span="
                    f"{metrics['population_span_5_95']:.4f}",
                    flush=True,
                )
                if not result["population_visibility_pass"]:
                    print(
                        "WARNING: calibrated population span is below the "
                        f"{MINIMUM_REVIEW_POPULATION_SPAN:.3f} review threshold",
                        flush=True,
                    )
        finally:
            print(
                f"Returning {device['yoko']} to 0 A and disabling output",
                flush=True,
            )
            if not source.get_mode().startswith("CURR"):
                source.set_mode("current")
            if abs(source.get_current()) > 1e-7:
                source.set_output(True)
                source.ramp_current(0.0, sweeprate=RAMP_RATE_A_PER_S)
            source.set_current(0.0)
            source.set_output(False)
            if abs(source.get_current()) > 1e-7 or source.get_output():
                raise RuntimeError(
                    f"Failed to leave {device['yoko']} at 0 A/output off"
                )


def verify_all_sources_safe():
    for device_key in ("q4", "q6"):
        device = DEVICES[device_key]
        with YokogawaGS200(
                device["ip"], max_current=MAX_ABS_CURRENT_A) as source:
            mode = source.get_mode()
            current = source.get_current()
            output = source.get_output()
            print(
                f"Final source readback: {device['yoko']} mode={mode}, "
                f"current_A={current:.12g}, output={output}",
                flush=True,
            )
            if abs(current) > 1e-7 or output:
                source.set_output(True)
                source.ramp_current(0.0, sweeprate=RAMP_RATE_A_PER_S)
                source.set_output(False)
            if abs(source.get_current()) > 1e-7 or source.get_output():
                raise RuntimeError(
                    f"Final safety verification failed for {device['yoko']}"
                )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", choices=("q4", "q6", "all"), default="all"
    )
    parser.add_argument(
        "--current-ma", type=float, action="append",
        help="Run only a listed deck-grid current; repeat for multiple values",
    )
    parser.add_argument(
        "--reps", type=int, default=DEFAULT_REPS,
        help=f"Rabi repetitions per gain (default: {DEFAULT_REPS})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reacquire rows already complete for the same map settings",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Print resolved settings and provenance without touching hardware",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.reps < 200:
        raise ValueError("The production Rabi map requires at least 200 reps")
    if args.current_ma:
        currents = args.current_ma
        if any(current not in CURRENT_GRID_MA for current in currents):
            raise ValueError(f"Currents must come from {CURRENT_GRID_MA}")
    else:
        currents = ACQUISITION_ORDER_MA
    if len(set(currents)) != len(currents):
        raise ValueError("Do not request the same current more than once")
    device_keys = ("q4", "q6") if args.device == "all" else (args.device,)
    if args.plan_only:
        for device_key in device_keys:
            run_device(
                device_key, currents, args.reps, force=args.force,
                plan_only=True,
            )
        return
    try:
        for device_key in device_keys:
            run_device(
                device_key, currents, args.reps, force=args.force,
                plan_only=False,
            )
    finally:
        verify_all_sources_safe()


if __name__ == "__main__":
    main()
