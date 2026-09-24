"""Acquire fresh Ramsey corrections, tuned sigmas, and coherence maps.

Only physical Q4/M5/Yoko3 and physical Q6/M6/Yoko4 are in scope.  Each
authorized current is processed in dependency order: a fresh canonical Ramsey
frequency correction, one or more population-calibrated Rabi measurements that
tune sigma toward a 0.65 pi gain, then population-calibrated T1/T2R/T2E sweeps
using that corrected frequency and selected pulse.

Hardware is touched only from :func:`main`.  The manifest is written after each
accepted stage so an interrupted run can resume without overwriting raw data.
The inactive source is checked before a device run and guarded cleanup returns
both Yokogawas to 0 A with output disabled.
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
from pucq4_10_ramsey_corrected_rabi_maps import (
    RABI_GAIN_START,
    RABI_GAIN_STEPS,
    RABI_GAIN_STOP,
    RABI_RELAX_DELAY_US,
    RABI_REPS,
    RABI_ROUNDS,
    RABI_SSF_RELAX_DELAY_US,
    RABI_SSF_SHOTS_PER_STATE,
    acquire_ramsey,
    canonical_artifacts,
    load_rabi_trace,
    parse_single_match,
    rabi_metrics,
)


SOURCE_MANIFEST = CAMPAIGN_ROOT / "ramsey_corrected_rabi_manifest.json"
MANIFEST = CAMPAIGN_ROOT / "ramsey_sigma_coherence_manifest.json"

PI_TARGET = 0.65
PI_TOLERANCE = 0.025
PI_WINDOW = (PI_TARGET - PI_TOLERANCE, PI_TARGET + PI_TOLERANCE)
SIGMA_MIN_US = 0.025
SIGMA_MAX_US = 0.160
MAX_SIGMA_ATTEMPTS = 5
MINIMUM_RABI_POPULATION_SPAN = 0.15
MINIMUM_RABI_FIT_R2 = 0.50

# Q6 has substantially weaker Ramsey contrast at the two most-negative
# currents.  Keep the same fit acceptance thresholds and improve the evidence
# instead: these two points get more averages for their fresh correction.
HIGH_STATISTICS_RAMSEY = {
    ("q6", -7.5): {"reps": 2000, "rounds": 4},
    ("q6", -10.0): {"reps": 2000, "rounds": 4},
}

# The same two Q6 endpoints have the weakest calibrated readout contrast in
# coherence.  Reacquire their final T1/T2R/T2E evidence at higher statistics
# without repeating or replacing the accepted Ramsey and sigma stages.
HIGH_STATISTICS_COHERENCE = {
    ("q6", -7.5): {"reps": 2000, "rounds": 4},
    ("q6", -10.0): {"reps": 2000, "rounds": 4},
}

COHERENCE_REPS = 1200
COHERENCE_ROUNDS = 2
COHERENCE_RELAX_DELAY_US = 250.0
COHERENCE_SSF_SHOTS_PER_STATE = 4000
COHERENCE_SSF_RELAX_DELAY_US = 250.0
COHERENCE_SETTINGS = {
    "t1": {
        "label": "t1_ge",
        "plot_folder": "T1_ge",
        "fit_dataset": "T1",
        "start_us": 0.01,
        "stop_us": 60.0,
        "steps": 241,
    },
    "t2r": {
        "label": "t2_ge",
        "plot_folder": "Ramsey_ge_1",
        "fit_dataset": "T2",
        "start_us": 0.0,
        "stop_us": 30.0,
        "steps": 301,
        "detuning_MHz": 0.5,
    },
    "t2e": {
        "label": "t2e_ge",
        "plot_folder": "SpinEcho_ge",
        "fit_dataset": "T2E",
        "start_us": 0.01,
        "stop_us": 60.0,
        "steps": 301,
        "detuning_MHz": 0.5,
    },
}


def current_key(current_ma):
    return f"{current_ma:+.1f}"


def ramsey_settings_for(device_key, current_ma):
    return dict(HIGH_STATISTICS_RAMSEY.get((device_key, current_ma), {}))


def coherence_statistics_for(device_key, current_ma):
    return dict(HIGH_STATISTICS_COHERENCE.get(
        (device_key, current_ma),
        {"reps": COHERENCE_REPS, "rounds": COHERENCE_ROUNDS},
    ))


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_manifest():
    if MANIFEST.is_file():
        return load_json(MANIFEST)
    return {
        "created": datetime.now().isoformat(),
        "purpose": (
            "fresh Ramsey correction, current-indexed sigma tuning to pi "
            "gain 0.65, and calibrated-population T1/T2R/T2E maps"
        ),
        "source_manifest": str(SOURCE_MANIFEST),
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


def campaign_settings():
    return {
        "ramsey": {
            "implementation": (
                "round_robin_benchmark.py -> "
                "T2RMeasurement.adjust_qspec"
            ),
            "start_us": 0.0,
            "stop_us": 1.0,
            "steps": 50,
            "detuning_MHz": 4.0,
            "reps": 1000,
            "rounds": 3,
            "relax_delay_us": 4000.0,
            "seed": "previous validated Ramsey-corrected frequency",
        },
        "sigma_tuning": {
            "target_pi_gain": PI_TARGET,
            "accepted_pi_gain": list(PI_WINDOW),
            "sigma_bounds_us": [SIGMA_MIN_US, SIGMA_MAX_US],
            "max_attempts": MAX_SIGMA_ATTEMPTS,
            "gain_start": RABI_GAIN_START,
            "gain_stop": RABI_GAIN_STOP,
            "gain_steps": RABI_GAIN_STEPS,
            "reps": RABI_REPS,
            "rounds": RABI_ROUNDS,
            "relax_delay_us": RABI_RELAX_DELAY_US,
            "ssf_shots_per_state": RABI_SSF_SHOTS_PER_STATE,
            "ssf_relax_delay_us": RABI_SSF_RELAX_DELAY_US,
            "population_definition": (
                "signed current-specific SSF affine projection, ground=0, "
                "excited=1; no row normalization"
            ),
        },
        "coherence": {
            "reps": COHERENCE_REPS,
            "rounds": COHERENCE_ROUNDS,
            "relax_delay_us": COHERENCE_RELAX_DELAY_US,
            "ssf_shots_per_state_per_experiment": (
                COHERENCE_SSF_SHOTS_PER_STATE
            ),
            "ssf_relax_delay_us": COHERENCE_SSF_RELAX_DELAY_US,
            "population_definition": (
                "one built-in SSF affine calibration per T1/T2R/T2E curve; "
                "signed ground=0, excited=1; no current-row normalization"
            ),
            "experiments": COHERENCE_SETTINGS,
        },
        "runner": "round_robin_benchmark.py",
        "attenuation_dB": {
            "resonator_DAC": [10, 15],
            "qubit_DAC": [5, 4],
            "ADC": 17,
        },
    }


def source_calibrations(device_key, currents):
    if not SOURCE_MANIFEST.is_file():
        raise FileNotFoundError(SOURCE_MANIFEST)
    source = load_json(SOURCE_MANIFEST)
    points = source["devices"][device_key]["map"]["points"]
    calibrations = {}
    for current_ma in currents:
        key = current_key(current_ma)
        point = points.get(key, {})
        if point.get("status") != "complete":
            raise RuntimeError(
                f"Previous Ramsey/Rabi source is incomplete at "
                f"{device_key} {key} mA"
            )
        ramsey = point.get("ramsey", {})
        rabi = point.get("rabi", {})
        if ramsey.get("status") != "complete" or rabi.get("status") != "complete":
            raise RuntimeError(
                f"Missing completed source stage at {device_key} {key} mA"
            )
        for artifact in (ramsey.get("h5"), ramsey.get("plot"),
                         rabi.get("h5"), rabi.get("plot")):
            if not artifact or not Path(artifact).is_file():
                raise FileNotFoundError(artifact)
        pi_amp = rabi.get("fitted_pi_amp")
        if pi_amp is None:
            pi_amp = rabi.get("retained_pi_seed")
        values = [
            point["corrected_qfreq_MHz"], rabi["fixed_sigma_us"], pi_amp,
            rabi["res_base_MHz"], rabi["readout"]["frequency_MHz"],
            rabi["readout"]["length_us"], rabi["readout"]["gain"],
            rabi["readout"]["offset_mhz"],
        ]
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"Non-finite previous calibration at {device_key} {key} mA"
            )
        calibrations[current_ma] = {
            "seed_qfreq_MHz": float(point["corrected_qfreq_MHz"]),
            "seed_qfreq_source": "previous validated Ramsey correction",
            "ramsey_sigma_us": float(rabi["fixed_sigma_us"]),
            "ramsey_pi_amp": float(pi_amp),
            "source_ramsey_h5": ramsey["h5"],
            "source_rabi_h5": rabi["h5"],
            "res_base_MHz": float(rabi["res_base_MHz"]),
            "readout": {
                "frequency_MHz": float(rabi["readout"]["frequency_MHz"]),
                "length_us": float(rabi["readout"]["length_us"]),
                "gain": float(rabi["readout"]["gain"]),
                "offset_mhz": float(rabi["readout"]["offset_mhz"]),
                "fidelity": rabi["readout"].get("fidelity"),
            },
        }
    return calibrations


def prospective_artifact_path(device, current_ma, kind):
    if kind == "ramsey":
        stage = "ram2_r1000x3_235959"
        label = "t2_ge_correction_1"
    elif kind == "rabi":
        stage = "sg_a5_s160_235959"
        label = "rabi_ge"
    elif kind == "t1":
        stage = "coh_r1200x2_235959"
        label = "t1_ge"
    elif kind == "t2r":
        stage = "coh_r1200x2_235959"
        label = "t2_ge"
    else:
        stage = "coh_r1200x2_235959"
        label = "t2e_ge"
    tag = (
        f"{device['yoko'].lower()}_{device['physical'].lower()}_"
        f"{current_ma:+05.1f}mA_{stage}"
    )
    filename = (
        f"2099-12-31_23-59-59_{label}_results_batch_1_"
        "Num_per_batch1.h5"
    )
    return (
        DATA_ROOT / f"flux_{tag}" / "2099-12-31_23-59-59" /
        "study_data" / "Data_h5" / label / filename
    )


def verify_paths(device, currents):
    paths = [
        prospective_artifact_path(device, current, kind)
        for current in currents
        for kind in ("ramsey", "rabi", "t1", "t2r", "t2e")
    ]
    longest = max(paths, key=lambda path: len(str(path)))
    if len(str(longest)) >= 260:
        raise RuntimeError(
            f"Prospective Windows artifact path is {len(str(longest))} "
            f"characters: {longest}"
        )
    return longest


def fresh_stage(prefix):
    return f"{prefix}_{datetime.now().strftime('%H%M%S')}"


def load_saved_fit(h5_path, device):
    with h5py.File(h5_path, "r") as handle:
        group = handle[f"Q{device['index'] + 1}"]
        if "Fit" not in group:
            return None
        try:
            return decode_h5_numeric(group["Fit"])
        except (ValueError, TypeError):
            return None


def fit_quality(values, fitted):
    values = np.asarray(values, dtype=float)
    fitted = np.asarray(fitted, dtype=float)
    length = min(values.size, fitted.size)
    if length < 5:
        return {
            "r_squared": float("nan"), "fit_snr": 0.0,
            "residual_sigma": float("nan"),
        }
    values = values[:length]
    fitted = fitted[:length]
    finite = np.isfinite(values) & np.isfinite(fitted)
    if finite.sum() < 5:
        return {
            "r_squared": float("nan"), "fit_snr": 0.0,
            "residual_sigma": float("nan"),
        }
    values = values[finite]
    fitted = fitted[finite]
    residual = values - fitted
    residual_sum = float(np.sum(residual ** 2))
    total_sum = float(np.sum((values - np.mean(values)) ** 2))
    r_squared = 1.0 - residual_sum / max(total_sum, np.finfo(float).eps)
    residual_sigma = max(
        1.4826 * float(np.median(np.abs(residual - np.median(residual)))),
        np.finfo(float).eps,
    )
    fit_span = float(np.percentile(fitted, 95) - np.percentile(fitted, 5))
    return {
        "r_squared": r_squared,
        "fit_snr": fit_span / (2.0 * residual_sigma),
        "residual_sigma": residual_sigma,
        "fit_span_5_95": fit_span,
    }


def acquire_sigma_rabi(device, current_ma, calibration, corrected_qfreq,
                       sigma_us, attempt_number, pi_seed):
    readout = calibration["readout"]
    sigma_ns = int(round(sigma_us * 1000.0))
    stage = fresh_stage(
        f"sg_a{attempt_number}_s{sigma_ns:03d}"
    )
    notes = (
        f"Sigma calibration for {device['physical']}/{device['row']} at "
        f"{current_ma:+.1f} mA. Fresh Ramsey-corrected qubit frequency; "
        f"sigma {sigma_us:.9f} us; target pi gain {PI_TARGET:.3f}; exact "
        "previously shifted physical readout tone."
    )
    started_epoch = time.time()
    text, log_path = run_round_robin(
        device, current_ma, stage, {
            "PUCQ4_RUN_FLAGS": "rabi",
            "PUCQ4_QFREQ_MHZ": corrected_qfreq,
            "PUCQ4_READOUT_FREQ_MHZ": readout["frequency_MHz"],
            "PUCQ4_READOUT_LENGTH_US": readout["length_us"],
            "PUCQ4_READOUT_GAIN": readout["gain"],
            "PUCQ4_READOUT_OFFSET_MHZ": readout["offset_mhz"],
            "PUCQ4_SIGMA_US": sigma_us,
            "PUCQ4_PI_AMP": pi_seed,
            "PUCQ4_RABI_START": RABI_GAIN_START,
            "PUCQ4_RABI_STOP": RABI_GAIN_STOP,
            "PUCQ4_RABI_STEPS": RABI_GAIN_STEPS,
            "PUCQ4_RABI_REPS": RABI_REPS,
            "PUCQ4_RABI_ROUNDS": RABI_ROUNDS,
            "PUCQ4_RABI_RELAX_DELAY_US": RABI_RELAX_DELAY_US,
            "PUCQ4_RABI_SSF_SHOTS": RABI_SSF_SHOTS_PER_STATE,
            "PUCQ4_RABI_SSF_RELAX_DELAY_US": RABI_SSF_RELAX_DELAY_US,
            "PUCQ4_RABI_SCALING": "true",
            "PUCQ4_RABI_KEEP_SEED_ON_FIT_FAILURE": "true",
            "PUCQ4_SUBSTUDY_NOTES": notes,
        },
    )
    operating = parse_single_match(
        r"Rabi operating point: qubit_frequency_MHz=([0-9.+-eE]+), "
        r"readout_frequency_MHz=([0-9.+-eE]+), "
        r"sigma_us=([0-9.+-eE]+)",
        text, "sigma-calibration Rabi operating point",
    )
    reported_qfreq, reported_readout, reported_sigma = (
        float(value) for value in operating
    )
    if not np.allclose(
        [reported_qfreq, reported_readout, reported_sigma],
        [corrected_qfreq, readout["frequency_MHz"], sigma_us],
        rtol=0.0, atol=1e-6,
    ):
        raise RuntimeError(
            "Sigma-calibration Rabi used the wrong operating point: "
            f"{reported_qfreq}, {reported_readout}, {reported_sigma}"
        )
    sweep = parse_single_match(
        r"Rabi sweep settings: start=([0-9.+-eE]+), "
        r"stop=([0-9.+-eE]+), steps=([0-9]+), reps=([0-9]+), "
        r"rounds=([0-9]+), relax_delay_us=([0-9.+-eE]+), "
        r"ssf_shots_per_state=([0-9]+), "
        r"ssf_relax_delay_us=([0-9.+-eE]+)",
        text, "sigma-calibration Rabi settings",
    )
    expected_sweep = [
        RABI_GAIN_START, RABI_GAIN_STOP, RABI_GAIN_STEPS, RABI_REPS,
        RABI_ROUNDS, RABI_RELAX_DELAY_US, RABI_SSF_SHOTS_PER_STATE,
        RABI_SSF_RELAX_DELAY_US,
    ]
    if not np.allclose(
        [float(value) for value in sweep], expected_sweep,
        rtol=0.0, atol=1e-9,
    ):
        raise RuntimeError("Sigma-calibration Rabi settings mismatch")
    pi_matches = re.findall(
        r"g-e Pi amplitude for qubit\s+\d+\s+is:\s+([0-9.+-eE]+)",
        text,
    )
    if not pi_matches:
        raise RuntimeError("No pi amplitude was reported by the Rabi run")
    fitted_pi = float(pi_matches[-1])
    fit_available = "Rabi pi amplitude status: retained seed" not in text
    h5_path, plot_path = canonical_artifacts(
        log_path, "rabi_ge", "power_rabi_ge_plots", started_epoch
    )
    gains, population, _, _, ssf = load_rabi_trace(h5_path, device)
    if gains.size != RABI_GAIN_STEPS or np.any(np.diff(gains) <= 0):
        raise RuntimeError(
            f"Unexpected Rabi gain axis with {gains.size} points"
        )
    requested_step = (
        (RABI_GAIN_STOP - RABI_GAIN_START) / (RABI_GAIN_STEPS - 1)
    )
    actual_step = float(np.median(np.diff(gains)))
    endpoint_shortfall = RABI_GAIN_STOP - float(gains[-1])
    if (abs(float(gains[0]) - RABI_GAIN_START) > 1e-4
            or endpoint_shortfall < 0
            or endpoint_shortfall > max(0.01, 2.0 * requested_step)
            or abs(actual_step - requested_step) > 1e-4):
        raise RuntimeError(
            f"Unexpected realized Rabi axis {gains[0]}-{gains[-1]}, "
            f"step {actual_step}"
        )
    ssf_lengths = {name: int(values.size) for name, values in ssf.items()}
    if any(
        length != RABI_SSF_SHOTS_PER_STATE
        for length in ssf_lengths.values()
    ):
        raise RuntimeError(f"Unexpected Rabi SSF lengths: {ssf_lengths}")
    metrics = rabi_metrics(gains, population)
    if metrics["population_span_5_95"] < MINIMUM_RABI_POPULATION_SPAN:
        raise RuntimeError(
            "Sigma-calibration Rabi is too flat: population span "
            f"{metrics['population_span_5_95']:.4f}"
        )
    saved_fit = load_saved_fit(h5_path, device)
    quality = fit_quality(population, saved_fit) if saved_fit is not None else {
        "r_squared": float("nan"), "fit_snr": 0.0,
        "residual_sigma": float("nan"),
    }
    if (not fit_available or not np.isfinite(fitted_pi)
            or fitted_pi <= 0.0 or fitted_pi > 1.2):
        raise RuntimeError(
            f"Rabi did not return a physical fitted pi gain: {fitted_pi}"
        )
    if (not np.isfinite(quality["r_squared"])
            or quality["r_squared"] < MINIMUM_RABI_FIT_R2):
        raise RuntimeError(
            "Rabi fit quality is insufficient: R2="
            f"{quality['r_squared']:.3f}"
        )
    return {
        "status": "complete",
        "completed": datetime.now().isoformat(),
        "attempt": attempt_number,
        "qfreq_MHz": reported_qfreq,
        "res_base_MHz": calibration["res_base_MHz"],
        "readout": {**readout, "frequency_MHz": reported_readout},
        "sigma_us": reported_sigma,
        "pi_gain": fitted_pi,
        "target_error": fitted_pi - PI_TARGET,
        "target_accepted": PI_WINDOW[0] <= fitted_pi <= PI_WINDOW[1],
        "fit_quality": quality,
        "population_metrics": metrics,
        "actual_gain_start": float(gains[0]),
        "actual_gain_stop": float(gains[-1]),
        "actual_gain_median_step": actual_step,
        "gain_steps": int(gains.size),
        "reps": RABI_REPS,
        "rounds": RABI_ROUNDS,
        "relax_delay_us": RABI_RELAX_DELAY_US,
        "ssf_shots_per_state": RABI_SSF_SHOTS_PER_STATE,
        "ssf_lengths": ssf_lengths,
        "log": log_path,
        "h5": str(h5_path),
        "plot": str(plot_path),
    }


def initial_sigma(calibration):
    predicted = (
        calibration["ramsey_sigma_us"]
        * calibration["ramsey_pi_amp"] / PI_TARGET
    )
    return float(np.clip(predicted, SIGMA_MIN_US, SIGMA_MAX_US))


def tune_sigma(device, current_ma, calibration, corrected_qfreq,
               existing=None):
    existing = existing if isinstance(existing, dict) else {}
    attempts = list(existing.get("attempts", []))
    for attempt in attempts:
        if attempt.get("target_accepted"):
            return {
                **existing,
                "status": "complete",
                "selected": attempt,
                "attempts": attempts,
            }
    if attempts:
        last = attempts[-1]
        sigma_us = float(last.get("next_sigma_us", last["sigma_us"]))
        pi_seed = float(last["pi_gain"])
    else:
        sigma_us = initial_sigma(calibration)
        pi_seed = PI_TARGET
    for attempt_number in range(len(attempts) + 1, MAX_SIGMA_ATTEMPTS + 1):
        result = acquire_sigma_rabi(
            device, current_ma, calibration, corrected_qfreq,
            sigma_us, attempt_number, pi_seed,
        )
        attempts.append(result)
        if result["target_accepted"]:
            return {
                "status": "complete",
                "completed": datetime.now().isoformat(),
                "target_pi_gain": PI_TARGET,
                "accepted_pi_window": list(PI_WINDOW),
                "initial_prediction_us": initial_sigma(calibration),
                "selected": result,
                "attempts": attempts,
            }
        ratio = result["pi_gain"] / PI_TARGET
        bounded_ratio = float(np.clip(ratio, 0.75, 1.35))
        next_sigma = float(np.clip(
            sigma_us * bounded_ratio, SIGMA_MIN_US, SIGMA_MAX_US
        ))
        if abs(next_sigma - sigma_us) < 1e-9:
            raise RuntimeError(
                f"Sigma tuning reached its bound at {sigma_us:.9f} us"
            )
        result["next_sigma_us"] = next_sigma
        sigma_us = next_sigma
        pi_seed = result["pi_gain"]
    best = min(attempts, key=lambda item: abs(item["target_error"]))
    raise RuntimeError(
        f"No sigma produced pi gain in {PI_WINDOW}; best was "
        f"sigma={best['sigma_us']:.9f} us, pi={best['pi_gain']:.6f}"
    )


def coherence_overrides(corrected_qfreq, sigma_result, calibration,
                        statistics=None):
    statistics = statistics or {
        "reps": COHERENCE_REPS, "rounds": COHERENCE_ROUNDS,
    }
    readout = calibration["readout"]
    selected = sigma_result["selected"]
    result = {
        "PUCQ4_RUN_FLAGS": "t1,t2r,t2e",
        "PUCQ4_QFREQ_MHZ": corrected_qfreq,
        "PUCQ4_READOUT_FREQ_MHZ": readout["frequency_MHz"],
        "PUCQ4_READOUT_LENGTH_US": readout["length_us"],
        "PUCQ4_READOUT_GAIN": readout["gain"],
        "PUCQ4_READOUT_OFFSET_MHZ": readout["offset_mhz"],
        "PUCQ4_SIGMA_US": selected["sigma_us"],
        "PUCQ4_PI_AMP": selected["pi_gain"],
        "PUCQ4_COHERENCE_SCALING": "true",
        "PUCQ4_COHERENCE_SSF_SHOTS": COHERENCE_SSF_SHOTS_PER_STATE,
        "PUCQ4_COHERENCE_SSF_RELAX_DELAY_US": (
            COHERENCE_SSF_RELAX_DELAY_US
        ),
    }
    prefixes = {"t1": "PUCQ4_T1", "t2r": "PUCQ4_T2R", "t2e": "PUCQ4_T2E"}
    for name, prefix in prefixes.items():
        settings = COHERENCE_SETTINGS[name]
        result.update({
            f"{prefix}_START_US": settings["start_us"],
            f"{prefix}_STOP_US": settings["stop_us"],
            f"{prefix}_STEPS": settings["steps"],
            f"{prefix}_REPS": statistics["reps"],
            f"{prefix}_ROUNDS": statistics["rounds"],
            f"{prefix}_RELAX_DELAY_US": COHERENCE_RELAX_DELAY_US,
        })
        if "detuning_MHz" in settings:
            result[f"{prefix}_DETUNING_MHZ"] = settings["detuning_MHz"]
    return result


def load_coherence_trace(h5_path, device, settings):
    with h5py.File(h5_path, "r") as handle:
        group = handle[f"Q{device['index'] + 1}"]
        required = [
            "Delay Times", "I", "Q", "Population", "Fit",
            settings["fit_dataset"], "Errors", "ss_I_g", "ss_Q_g",
            "ss_I_e", "ss_Q_e",
        ]
        missing = [name for name in required if name not in group]
        if missing:
            raise RuntimeError(
                f"Coherence HDF5 is missing {missing}: {h5_path}"
            )
        delay = decode_h5_numeric(group["Delay Times"])
        population = decode_h5_numeric(group["Population"])
        fitted = decode_h5_numeric(group["Fit"])
        fit_value = float(decode_h5_numeric(
            group[settings["fit_dataset"]]
        )[0])
        fit_error = float(decode_h5_numeric(group["Errors"])[0])
        ssf = {
            name: decode_h5_numeric(group[name])
            for name in ("ss_I_g", "ss_Q_g", "ss_I_e", "ss_Q_e")
        }
    length = min(delay.size, population.size, fitted.size)
    delay = np.asarray(delay[:length], dtype=float)
    population = np.asarray(population[:length], dtype=float)
    fitted = np.asarray(fitted[:length], dtype=float)
    finite = np.isfinite(delay) & np.isfinite(population) & np.isfinite(fitted)
    order = np.argsort(delay[finite])
    return (
        delay[finite][order], population[finite][order],
        fitted[finite][order], fit_value, fit_error, ssf,
    )


def validate_coherence_experiment(h5_path, plot_path, device, name,
                                  statistics=None):
    statistics = statistics or {
        "reps": COHERENCE_REPS, "rounds": COHERENCE_ROUNDS,
    }
    settings = COHERENCE_SETTINGS[name]
    delay, population, fitted, fit_value, fit_error, ssf = (
        load_coherence_trace(h5_path, device, settings)
    )
    if delay.size != settings["steps"]:
        raise RuntimeError(
            f"{name} saved {delay.size} points; expected {settings['steps']}"
        )
    if np.any(np.diff(delay) <= 0):
        raise RuntimeError(f"{name} delay axis is not strictly increasing")
    if not np.all(np.isfinite([fit_value, fit_error])) or fit_value <= 0:
        raise RuntimeError(
            f"{name} returned non-physical fit {fit_value} +/- {fit_error} us"
        )
    if fit_value > 3.0 * settings["stop_us"]:
        raise RuntimeError(
            f"{name} fit {fit_value} us is not bounded by its delay sweep"
        )
    quality = fit_quality(population, fitted)
    lower, upper = np.percentile(population, [5.0, 95.0])
    population_span = float(upper - lower)
    if population_span < 0.03:
        raise RuntimeError(
            f"{name} population trace is flat (5-95 span {population_span})"
        )
    ssf_lengths = {key: int(value.size) for key, value in ssf.items()}
    if any(
        value != COHERENCE_SSF_SHOTS_PER_STATE
        for value in ssf_lengths.values()
    ):
        raise RuntimeError(f"{name} SSF lengths are {ssf_lengths}")
    if not Path(plot_path).is_file():
        raise FileNotFoundError(plot_path)
    return {
        "status": "complete",
        "fit_value_us": fit_value,
        "fit_error_us": fit_error,
        "fit_quality": quality,
        "population_span_5_95": population_span,
        "population_min": float(np.min(population)),
        "population_max": float(np.max(population)),
        "actual_delay_start_us": float(delay[0]),
        "actual_delay_stop_us": float(delay[-1]),
        "actual_delay_median_step_us": float(np.median(np.diff(delay))),
        "steps": int(delay.size),
        "reps": statistics["reps"],
        "rounds": statistics["rounds"],
        "relax_delay_us": COHERENCE_RELAX_DELAY_US,
        "ssf_shots_per_state": COHERENCE_SSF_SHOTS_PER_STATE,
        "ssf_lengths": ssf_lengths,
        "h5": str(h5_path),
        "plot": str(plot_path),
    }


def acquire_coherence(device, current_ma, calibration, corrected_qfreq,
                      sigma_result, statistics=None):
    statistics = statistics or {
        "reps": COHERENCE_REPS, "rounds": COHERENCE_ROUNDS,
    }
    selected = sigma_result["selected"]
    readout = calibration["readout"]
    stage = fresh_stage(
        f"coh_r{statistics['reps']}x{statistics['rounds']}"
    )
    notes = (
        f"Population-calibrated T1/T2R/T2E map row for "
        f"{device['physical']}/{device['row']} at {current_ma:+.1f} mA. "
        f"Fresh corrected qfreq {corrected_qfreq:.9f} MHz, tuned sigma "
        f"{selected['sigma_us']:.9f} us, pi gain "
        f"{selected['pi_gain']:.9f}, and exact shifted readout "
        f"{readout['frequency_MHz']:.6f} MHz. Coherence averaging "
        f"{statistics['reps']} reps x {statistics['rounds']} rounds."
    )
    overrides = coherence_overrides(
        corrected_qfreq, sigma_result, calibration, statistics=statistics
    )
    overrides["PUCQ4_SUBSTUDY_NOTES"] = notes
    started_epoch = time.time()
    text, log_path = run_round_robin(
        device, current_ma, stage, overrides
    )
    for label in ("T1", "T2R", "T2E"):
        match = parse_single_match(
            rf"{label} operating point: qubit_frequency_MHz="
            r"([0-9.+-eE]+), readout_frequency_MHz="
            r"([0-9.+-eE]+), sigma_us=([0-9.+-eE]+), "
            r"pi_amp=([0-9.+-eE]+)",
            text, f"{label} operating point",
        )
        actual = [float(value) for value in match]
        expected = [
            corrected_qfreq, readout["frequency_MHz"],
            selected["sigma_us"], selected["pi_gain"],
        ]
        if not np.allclose(actual, expected, rtol=0.0, atol=1e-6):
            raise RuntimeError(
                f"{label} used {actual}; expected {expected}"
            )
    experiments = {}
    for name, settings in COHERENCE_SETTINGS.items():
        h5_path, plot_path = canonical_artifacts(
            log_path, settings["label"], settings["plot_folder"],
            started_epoch,
        )
        result = validate_coherence_experiment(
            h5_path, plot_path, device, name, statistics=statistics
        )
        result["log"] = log_path
        experiments[name] = result
    return {
        "status": "complete",
        "completed": datetime.now().isoformat(),
        "qfreq_MHz": corrected_qfreq,
        "sigma_us": selected["sigma_us"],
        "pi_gain": selected["pi_gain"],
        "res_base_MHz": calibration["res_base_MHz"],
        "readout": readout,
        "reps": statistics["reps"],
        "rounds": statistics["rounds"],
        "population_definition": (
            "individual built-in SSF calibration per coherence curve; "
            "signed affine ground=0/excited=1 projection"
        ),
        "experiments": experiments,
        "log": log_path,
    }


def select_best_coherence(previous, candidate):
    """Keep the higher-R2 saved population fit for each experiment."""
    if not stage_is_complete(previous, nested=True):
        return candidate
    selected = {**candidate, "experiments": {}}
    choices = {}
    for name in COHERENCE_SETTINGS:
        prior_result = previous["experiments"][name]
        candidate_result = candidate["experiments"][name]
        prior_r2 = prior_result["fit_quality"]["r_squared"]
        candidate_r2 = candidate_result["fit_quality"]["r_squared"]
        if candidate_r2 >= prior_r2:
            result = dict(candidate_result)
            source = "high-statistics reacquisition"
        else:
            result = dict(prior_result)
            source = "prior acquisition retained"
        result["selection_source"] = source
        result["selection_r_squared"] = result["fit_quality"]["r_squared"]
        selected["experiments"][name] = result
        choices[name] = {
            "selected": source,
            "prior_r_squared": prior_r2,
            "candidate_r_squared": candidate_r2,
        }
    selected["selection_policy"] = (
        "highest saved signed-population fit R2 per experiment; both raw "
        "acquisitions retained"
    )
    selected["selection_choices"] = choices
    selected["completed"] = datetime.now().isoformat()
    return selected


def stage_is_complete(stage, nested=False):
    if not isinstance(stage, dict) or stage.get("status") != "complete":
        return False
    if nested:
        experiments = stage.get("experiments", {})
        return all(
            isinstance(experiments.get(name), dict)
            and experiments[name].get("status") == "complete"
            and Path(experiments[name].get("h5", "")).is_file()
            and Path(experiments[name].get("plot", "")).is_file()
            for name in COHERENCE_SETTINGS
        )
    if "h5" in stage:
        return (
            Path(stage.get("h5", "")).is_file()
            and Path(stage.get("plot", "")).is_file()
        )
    selected = stage.get("selected", {})
    return (
        isinstance(selected, dict)
        and selected.get("target_accepted") is True
        and Path(selected.get("h5", "")).is_file()
        and Path(selected.get("plot", "")).is_file()
    )


def prepare_map(manifest, device_key, settings):
    device = DEVICES[device_key]
    entry = manifest["devices"].setdefault(device_key, {
        "physical": device["physical"],
        "row": device["row"],
        "python_index": device["index"],
        "yoko": device["yoko"],
        "ip": device["ip"],
    })
    signature = json.dumps(settings, sort_keys=True)
    map_data = entry.get("map")
    if map_data and map_data.get("settings_signature") != signature:
        entry.setdefault("superseded_maps", []).append(map_data)
        map_data = None
    if map_data is None:
        map_data = {"points": {}}
        entry["map"] = map_data
    points = map_data.setdefault("points", {})
    map_data.update(settings)
    map_data["settings_signature"] = signature
    map_data["points"] = points
    return map_data


def print_plan(device_key, device, currents, calibrations, longest_path,
               redo_coherence=False):
    ramsey_minutes = sum(
        50
        * ramsey_settings_for(device_key, current).get("reps", 1000)
        * ramsey_settings_for(device_key, current).get("rounds", 3)
        * 4000.0e-6
        for current in currents
    ) / 60.0
    one_rabi_minutes = (
        RABI_GAIN_STEPS * RABI_REPS * RABI_ROUNDS
        * RABI_RELAX_DELAY_US * 1e-6 * len(currents) / 60.0
    )
    coherence_points = sum(
        settings["steps"] for settings in COHERENCE_SETTINGS.values()
    )
    coherence_minutes = sum(
        coherence_points
        * coherence_statistics_for(device_key, current)["reps"]
        * coherence_statistics_for(device_key, current)["rounds"]
        * COHERENCE_RELAX_DELAY_US * 1e-6
        for current in currents
    ) / 60.0
    settling_minutes = sum(
        device.get("settle_seconds_by_current", {}).get(current, 0)
        for current in currents
    ) / 60.0
    print(
        f"{device['yoko']} / physical {device['physical']} / "
        f"{device['row']} (Python index {device['index']}): "
        f"{len(currents)} "
        + (
            "coherence-only high-statistics reacquisition rows; complete "
            "Ramsey/sigma stages will be reused"
            if redo_coherence else
            "fresh Ramsey + adaptive sigma + coherence rows"
        ),
        flush=True,
    )
    print(
        f"Pi target {PI_TARGET:.3f}, accepted {PI_WINDOW}; sigma bounds "
        f"{SIGMA_MIN_US:.3f}-{SIGMA_MAX_US:.3f} us. Coherence grids: "
        "T1 0.01-60 us/241, T2R 0-30 us/301 at 0.5 MHz, "
        "T2E 0.01-60 us/301.", flush=True,
    )
    print(
        "Nominal programmed floor: "
        + (
            f"coherence {coherence_minutes:.1f} min + settling "
            f"{settling_minutes:.1f} min; Ramsey/Rabi 0 min (reused)"
            if redo_coherence else
            f"Ramsey {ramsey_minutes:.1f} min + one Rabi attempt/current "
            f"{one_rabi_minutes:.1f} min + coherence "
            f"{coherence_minutes:.1f} min + settling "
            f"{settling_minutes:.1f} min"
        )
        + f"; longest path {len(str(longest_path))} chars",
        flush=True,
    )
    for current_ma in currents:
        calibration = calibrations[current_ma]
        readout = calibration["readout"]
        ramsey_settings = ramsey_settings_for(device_key, current_ma)
        ramsey_reps = ramsey_settings.get("reps", 1000)
        ramsey_rounds = ramsey_settings.get("rounds", 3)
        coherence_statistics = coherence_statistics_for(
            device_key, current_ma
        )
        print(
            f"  {current_ma:+.1f} mA: seed qfreq "
            f"{calibration['seed_qfreq_MHz']:.9f} MHz; Ramsey pulse "
            f"{calibration['ramsey_sigma_us']:.6f} us / "
            f"{calibration['ramsey_pi_amp']:.6f}; initial sigma "
            f"prediction {initial_sigma(calibration):.6f} us; readout "
            f"{readout['frequency_MHz']:.3f} MHz, "
            f"{readout['length_us']:.1f} us, gain "
            f"{readout['gain']:.4f}, offset "
            f"{readout['offset_mhz']:+.2f} MHz; Ramsey "
            f"{ramsey_reps}x{ramsey_rounds}; coherence "
            f"{coherence_statistics['reps']}x"
            f"{coherence_statistics['rounds']}",
            flush=True,
        )


def verify_all_sources_safe():
    for device_key in ("q4", "q6"):
        device = DEVICES[device_key]
        with YokogawaGS200(
            device["ip"], max_current=MAX_ABS_CURRENT_A
        ) as source:
            mode = source.get_mode()
            current = source.get_current()
            output = source.get_output()
            print(
                f"Final source readback: {device['yoko']} mode={mode}, "
                f"current_A={current:.12g}, output={output}", flush=True,
            )
            if abs(current) > 1e-7 or output:
                if not mode.startswith("CURR"):
                    source.set_mode("current")
                source.set_output(True)
                source.ramp_current(0.0, sweeprate=RAMP_RATE_A_PER_S)
                source.set_current(0.0)
                source.set_output(False)
            if abs(source.get_current()) > 1e-7 or source.get_output():
                raise RuntimeError(
                    f"Final safety verification failed for {device['yoko']}"
                )


def run_device(device_key, currents, force=False, redo_coherence=False,
               plan_only=False):
    device = DEVICES[device_key]
    settings = campaign_settings()
    calibrations = source_calibrations(device_key, currents)
    longest_path = verify_paths(device, currents)
    print_plan(
        device_key, device, currents, calibrations, longest_path,
        redo_coherence=redo_coherence,
    )
    if plan_only:
        print(
            "Plan only: no source, QICK, attenuator, data, or manifest state "
            "was changed", flush=True,
        )
        return

    manifest = load_manifest()
    map_data = prepare_map(manifest, device_key, settings)
    map_data["ramsey_high_statistics_overrides"] = {
        current_key(current): ramsey_settings_for(device_key, current)
        for current in currents
        if ramsey_settings_for(device_key, current)
    }
    map_data["coherence_high_statistics_overrides"] = {
        current_key(current): coherence_statistics_for(
            device_key, current
        )
        for current in currents
        if (device_key, current) in HIGH_STATISTICS_COHERENCE
    }
    save_manifest(manifest)
    verify_inactive_source(device_key)
    with YokogawaGS200(
        device["ip"], max_current=MAX_ABS_CURRENT_A
    ) as source:
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
                f"{voltage_limit:.12g}", flush=True,
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
                if (
                    not force and not redo_coherence
                    and existing.get("status") == "complete"
                    and stage_is_complete(existing.get("ramsey"))
                    and stage_is_complete(existing.get("sigma_calibration"))
                    and stage_is_complete(existing.get("coherence"), nested=True)
                ):
                    print(
                        f"{device['physical']} {current_ma:+.1f} mA already "
                        "complete; skipping", flush=True,
                    )
                    continue
                calibration = calibrations[current_ma]
                if redo_coherence and not (
                    existing.get("status") == "complete"
                    and stage_is_complete(existing.get("ramsey"))
                    and stage_is_complete(existing.get("sigma_calibration"))
                    and stage_is_complete(
                        existing.get("coherence"), nested=True
                    )
                ):
                    raise RuntimeError(
                        "--redo-coherence requires a complete existing row "
                        f"at {device_key} {current_ma:+.1f} mA"
                    )
                if redo_coherence:
                    selected = existing["sigma_calibration"]["selected"]
                    print(
                        "Reusing accepted stages at "
                        f"{device['physical']} {current_ma:+.1f} mA: "
                        f"qfreq={existing['ramsey']['corrected_qfreq_MHz']:.9f} "
                        f"MHz, sigma={selected['sigma_us']:.9f} us, "
                        f"pi={selected['pi_gain']:.6f}",
                        flush=True,
                    )
                if force and existing:
                    map_data.setdefault("superseded_attempts", {}).setdefault(
                        key, []
                    ).append(existing)
                    existing = {}
                failure_history = list(existing.get("failure_history", []))
                if existing.get("status") in ("failed", "aborted"):
                    failure_history.append({
                        "status": existing.get("status"),
                        "time": existing.get("failed"),
                        "error": existing.get("error"),
                    })
                point = {
                    **existing,
                    "status": "running",
                    "started": datetime.now().isoformat(),
                    "current_mA": current_ma,
                    "source_calibration": calibration,
                }
                point.pop("error", None)
                point.pop("failed", None)
                if failure_history:
                    point["failure_history"] = failure_history
                map_data["points"][key] = point
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
                    f"seed={calibration['seed_qfreq_MHz']:.9f} MHz; "
                    f"readout={calibration['readout']['frequency_MHz']:.3f} "
                    "MHz", flush=True,
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
                    if force or not stage_is_complete(point.get("ramsey")):
                        ramsey_settings = ramsey_settings_for(
                            device_key, current_ma
                        )
                        ramsey_reps = ramsey_settings.get("reps", 1000)
                        ramsey_rounds = ramsey_settings.get("rounds", 3)
                        point["ramsey"] = acquire_ramsey(
                            device, current_ma, calibration,
                            stage=fresh_stage(
                                f"ram2_r{ramsey_reps}x{ramsey_rounds}"
                            ),
                            seed_description=(
                                "previous validated Ramsey-corrected "
                                "frequency seed"
                            ),
                            ramsey_settings=ramsey_settings,
                        )
                        point["corrected_qfreq_MHz"] = point["ramsey"][
                            "corrected_qfreq_MHz"
                        ]
                        manifest["updated"] = datetime.now().isoformat()
                        save_manifest(manifest)
                    corrected_qfreq = float(
                        point["ramsey"]["corrected_qfreq_MHz"]
                    )
                    if force or not stage_is_complete(
                        point.get("sigma_calibration")
                    ):
                        point["sigma_calibration"] = tune_sigma(
                            device, current_ma, calibration,
                            corrected_qfreq,
                            existing=point.get("sigma_calibration"),
                        )
                        manifest["updated"] = datetime.now().isoformat()
                        save_manifest(manifest)
                    if force or redo_coherence or not stage_is_complete(
                        point.get("coherence"), nested=True
                    ):
                        previous_coherence = point.get("coherence")
                        candidate = acquire_coherence(
                            device, current_ma, calibration,
                            corrected_qfreq, point["sigma_calibration"],
                            statistics=coherence_statistics_for(
                                device_key, current_ma
                            ),
                        )
                        if redo_coherence:
                            history = map_data.setdefault(
                                "coherence_reacquisition_history", {}
                            ).setdefault(key, [])
                            history.extend([previous_coherence, candidate])
                            point["coherence"] = select_best_coherence(
                                previous_coherence, candidate
                            )
                        else:
                            point["coherence"] = candidate
                    point["status"] = "complete"
                    point["completed"] = datetime.now().isoformat()
                    manifest["updated"] = datetime.now().isoformat()
                    save_manifest(manifest)
                except KeyboardInterrupt:
                    point.update({
                        "status": "aborted",
                        "failed": datetime.now().isoformat(),
                        "error": "Operator interrupted the calibration row",
                    })
                    save_manifest(manifest)
                    raise
                except Exception as exc:
                    point.update({
                        "status": "failed",
                        "failed": datetime.now().isoformat(),
                        "error": str(exc),
                    })
                    save_manifest(manifest)
                    raise
                selected = point["sigma_calibration"]["selected"]
                coherence = point["coherence"]["experiments"]
                print(
                    f"{device['physical']} {current_ma:+.1f} mA complete: "
                    f"qfreq={corrected_qfreq:.9f} MHz, sigma="
                    f"{selected['sigma_us']:.9f} us, pi="
                    f"{selected['pi_gain']:.6f}, T1="
                    f"{coherence['t1']['fit_value_us']:.3f} us, T2R="
                    f"{coherence['t2r']['fit_value_us']:.3f} us, T2E="
                    f"{coherence['t2e']['fit_value_us']:.3f} us",
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
        "--force", action="store_true",
        help="Reacquire rows already complete for this exact campaign",
    )
    parser.add_argument(
        "--redo-coherence", action="store_true",
        help=(
            "Reuse complete Ramsey/sigma stages, reacquire only T1/T2R/T2E, "
            "and retain the higher-R2 result per experiment"
        ),
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Print resolved settings/provenance without touching hardware",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.force and args.redo_coherence:
        raise ValueError("Use either --force or --redo-coherence, not both")
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
                device_key, currents,
                redo_coherence=args.redo_coherence, plan_only=True,
            )
        return
    try:
        for device_key in device_keys:
            run_device(
                device_key, currents, force=args.force,
                redo_coherence=args.redo_coherence, plan_only=False,
            )
    finally:
        verify_all_sources_safe()


if __name__ == "__main__":
    main()
