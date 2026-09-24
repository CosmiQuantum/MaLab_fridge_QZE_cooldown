"""Acquire Ramsey-corrected, fixed-60-ns Rabi maps for PUCQ4.

For each authorized current this controller reuses the peak from the completed
wide qspec sweep, the matching shifted resonator readout, and the pi pulse from
the preceding fixed-sigma Rabi map.  It first runs the repository's canonical
``T2RMeasurement.adjust_qspec`` path through ``round_robin_benchmark.py`` and
then uses that corrected qubit frequency in a separate canonical Rabi run.

Hardware is touched only from :func:`main`.  The inactive Yokogawa is checked
before each device and both sources are returned to 0 A/output off in guarded
cleanup.
"""

import argparse
import json
import re
import time
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import least_squares

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


SOURCE_QSPEC_MANIFEST = CAMPAIGN_ROOT / "single_sweep_qspec_manifest.json"
SOURCE_RABI_MANIFEST = CAMPAIGN_ROOT / "single_sweep_rabi_manifest.json"
MANIFEST = CAMPAIGN_ROOT / "ramsey_corrected_rabi_manifest.json"

# Preserve the user's established Ramsey-correction geometry.  Only averaging
# is raised from the historical 20 reps x 2 rounds.
RAMSEY_START_US = 0.0
RAMSEY_STOP_US = 1.0
RAMSEY_STEPS = 50
RAMSEY_DETUNING_MHZ = 4.0
RAMSEY_RELAX_DELAY_US = 4000.0
RAMSEY_REPS = 1000
RAMSEY_ROUNDS = 3

FIXED_RABI_SIGMA_US = 0.060
RABI_GAIN_START = 0.0
RABI_GAIN_STOP = 1.0
RABI_GAIN_STEPS = 201
RABI_REPS = 1200
RABI_ROUNDS = 2
RABI_RELAX_DELAY_US = 250.0
RABI_SSF_SHOTS_PER_STATE = 4000
RABI_SSF_RELAX_DELAY_US = 250.0

MINIMUM_RAMSEY_FIT_R2 = 0.50
MINIMUM_RAMSEY_FIT_SNR = 4.0
MAXIMUM_RAMSEY_CORRECTION_MHZ = 1.5
MAXIMUM_RAMSEY_FREQUENCY_STDERR_MHZ = 0.05
MAXIMUM_BUILTIN_INDEPENDENT_DELTA_MHZ = 0.08
MINIMUM_RABI_POPULATION_SPAN = 0.03


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
            "current-indexed Ramsey frequency correction followed by a common "
            "0.060 us Rabi population map"
        ),
        "source_qspec_manifest": str(SOURCE_QSPEC_MANIFEST),
        "source_rabi_manifest": str(SOURCE_RABI_MANIFEST),
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
            "start_us": RAMSEY_START_US,
            "stop_us": RAMSEY_STOP_US,
            "steps": RAMSEY_STEPS,
            "detuning_MHz": RAMSEY_DETUNING_MHZ,
            "reps": RAMSEY_REPS,
            "rounds": RAMSEY_ROUNDS,
            "relax_delay_us": RAMSEY_RELAX_DELAY_US,
            "implementation": (
                "round_robin_benchmark.py -> "
                "T2RMeasurement.adjust_qspec"
            ),
            "correction_formula": (
                "corrected_frequency = seed_frequency - fitted_fringe + "
                "intentional_detuning"
            ),
        },
        "rabi": {
            "gain_start": RABI_GAIN_START,
            "gain_stop": RABI_GAIN_STOP,
            "gain_steps": RABI_GAIN_STEPS,
            "reps": RABI_REPS,
            "rounds": RABI_ROUNDS,
            "relax_delay_us": RABI_RELAX_DELAY_US,
            "fixed_sigma_us": FIXED_RABI_SIGMA_US,
            "ssf_shots_per_state": RABI_SSF_SHOTS_PER_STATE,
            "ssf_relax_delay_us": RABI_SSF_RELAX_DELAY_US,
            "population_definition": (
                "one built-in current-specific SSF affine calibration per "
                "gain row; signed ground=0/excited=1 projection; no row "
                "normalization"
            ),
        },
        "runner": "round_robin_benchmark.py",
        "attenuation_dB": {
            "resonator_DAC": [10, 15],
            "qubit_DAC": [5, 4],
            "ADC": 17,
        },
    }


def source_calibrations(device_key, currents):
    if not SOURCE_QSPEC_MANIFEST.is_file():
        raise FileNotFoundError(SOURCE_QSPEC_MANIFEST)
    if not SOURCE_RABI_MANIFEST.is_file():
        raise FileNotFoundError(SOURCE_RABI_MANIFEST)
    qspec_manifest = load_json(SOURCE_QSPEC_MANIFEST)
    rabi_manifest = load_json(SOURCE_RABI_MANIFEST)
    qspec_points = qspec_manifest["devices"][device_key]["map"]["points"]
    rabi_points = rabi_manifest["devices"][device_key]["map"]["points"]
    calibrations = {}
    for current_ma in currents:
        key = current_key(current_ma)
        qspec = qspec_points.get(key, {})
        rabi = rabi_points.get(key, {})
        if qspec.get("status") != "complete":
            raise RuntimeError(
                f"Source qspec row is not complete for {device_key} {key} mA"
            )
        if rabi.get("status") != "complete":
            raise RuntimeError(
                f"Source Rabi row is not complete for {device_key} {key} mA"
            )
        qfreq_mhz = float(qspec["peak_metrics"]["frequency_MHz"])
        rabi_qfreq_mhz = float(rabi["qfreq_MHz"])
        if abs(qfreq_mhz - rabi_qfreq_mhz) > 1e-6:
            raise RuntimeError(
                f"Qspec/Rabi source-frequency mismatch at {device_key} "
                f"{key} mA: {qfreq_mhz} vs {rabi_qfreq_mhz} MHz"
            )
        qspec_readout = qspec["readout"]
        rabi_readout = rabi["readout"]
        for field in ("frequency_MHz", "length_us", "gain", "offset_mhz"):
            if abs(float(qspec_readout[field]) - float(rabi_readout[field])) > 1e-6:
                raise RuntimeError(
                    f"Qspec/Rabi readout mismatch for {field} at "
                    f"{device_key} {key} mA"
                )
        source_qspec_h5 = Path(qspec["h5"])
        source_rabi_h5 = Path(rabi["h5"])
        if not source_qspec_h5.is_file():
            raise FileNotFoundError(source_qspec_h5)
        if not source_rabi_h5.is_file():
            raise FileNotFoundError(source_rabi_h5)
        values = [
            qfreq_mhz,
            float(qspec["res_base_MHz"]),
            float(qspec_readout["frequency_MHz"]),
            float(qspec_readout["length_us"]),
            float(qspec_readout["gain"]),
            float(qspec_readout["offset_mhz"]),
            float(rabi["fixed_sigma_us"]),
            float(rabi["fitted_pi_amp"]),
        ]
        if not np.all(np.isfinite(values)):
            raise RuntimeError(
                f"Non-finite source calibration at {device_key} {key} mA"
            )
        calibrations[current_ma] = {
            "seed_qfreq_MHz": qfreq_mhz,
            "seed_qfreq_source": "continuous wide-qspec observed peak",
            "source_qspec_h5": str(source_qspec_h5),
            "source_qspec_log": qspec["log"],
            "ramsey_pi_amp": float(rabi["fitted_pi_amp"]),
            "ramsey_sigma_us": float(rabi["fixed_sigma_us"]),
            "source_rabi_h5": str(source_rabi_h5),
            "source_rabi_log": rabi["log"],
            "res_base_MHz": float(qspec["res_base_MHz"]),
            "readout": {
                "frequency_MHz": float(qspec_readout["frequency_MHz"]),
                "length_us": float(qspec_readout["length_us"]),
                "gain": float(qspec_readout["gain"]),
                "offset_mhz": float(qspec_readout["offset_mhz"]),
                "fidelity": qspec_readout.get("fidelity"),
            },
        }
    return calibrations


def stage_name(kind):
    if kind == "ramsey":
        return f"ramsey_corr_r{RAMSEY_REPS}x{RAMSEY_ROUNDS}"
    if kind == "rabi":
        return (
            f"ramsey_rabi_s{FIXED_RABI_SIGMA_US * 1000:.0f}ns_"
            f"n{RABI_GAIN_STEPS}_r{RABI_REPS}x{RABI_ROUNDS}"
        )
    raise ValueError(kind)


def run_tag(device, current_ma, kind):
    return (
        f"{device['yoko'].lower()}_{device['physical'].lower()}_"
        f"{current_ma:+05.1f}mA_{stage_name(kind)}"
    )


def prospective_artifact_path(device, current_ma, kind):
    tag = run_tag(device, current_ma, kind)
    if kind == "ramsey":
        label = "t2_ge_correction_1"
        filename = (
            "2099-12-31_23-59-59_t2_ge_correction_1_results_batch_1_"
            "Num_per_batch1.h5"
        )
    else:
        label = "rabi_ge"
        filename = (
            "2099-12-31_23-59-59_rabi_ge_results_batch_1_"
            "Num_per_batch1.h5"
        )
    return (
        DATA_ROOT / f"flux_{tag}" / "2099-12-31_23-59-59" /
        "study_data" / "Data_h5" / label / filename
    )


def verify_paths(device, currents):
    candidates = [
        prospective_artifact_path(device, current_ma, kind)
        for current_ma in currents for kind in ("ramsey", "rabi")
    ]
    longest = max(candidates, key=lambda path: len(str(path)))
    if len(str(longest)) >= 260:
        raise RuntimeError(
            f"Prospective Windows artifact path is {len(str(longest))} "
            f"characters: {longest}"
        )
    return longest


def parse_single_match(pattern, text, label):
    matches = re.findall(pattern, text)
    if len(matches) != 1:
        raise RuntimeError(
            f"Expected exactly one {label}; found {len(matches)}"
        )
    return matches[0]


def canonical_artifacts(log_path, label, plot_subfolder, started_epoch):
    run_dir = DATA_ROOT / f"flux_{Path(log_path).stem}"
    h5_files = sorted(
        run_dir.glob(f"*/study_data/Data_h5/{label}/*.h5"),
        key=lambda path: path.stat().st_mtime,
    )
    plot_files = sorted(
        run_dir.glob(f"*/documentation/{plot_subfolder}/*.png"),
        key=lambda path: path.stat().st_mtime,
    )
    cutoff = started_epoch - 10.0
    h5_files = [path for path in h5_files if path.stat().st_mtime >= cutoff]
    plot_files = [
        path for path in plot_files if path.stat().st_mtime >= cutoff
    ]
    if not h5_files:
        raise RuntimeError(
            f"Run completed without a new canonical {label} HDF5 under {run_dir}"
        )
    if not plot_files:
        raise RuntimeError(
            f"Run completed without a new canonical plot under {run_dir}"
        )
    return h5_files[-1], plot_files[-1]


def load_ramsey_trace(h5_path, device):
    with h5py.File(h5_path, "r") as handle:
        group = handle[f"Q{device['index'] + 1}"]
        required = (
            "Delay Times", "I", "Q", "Fit", "Correction Freq", "T2",
            "Errors",
        )
        missing = [name for name in required if name not in group]
        if missing:
            raise RuntimeError(f"Ramsey HDF5 is missing {missing}: {h5_path}")
        delay = decode_h5_numeric(group["Delay Times"])
        i_data = decode_h5_numeric(group["I"])
        q_data = decode_h5_numeric(group["Q"])
        fit = decode_h5_numeric(group["Fit"])
        fitted_frequency = float(decode_h5_numeric(
            group["Correction Freq"]
        )[0])
        t2_value = float(decode_h5_numeric(group["T2"])[0])
        t2_error = float(decode_h5_numeric(group["Errors"])[0])
    length = min(delay.size, i_data.size, q_data.size, fit.size)
    delay = np.asarray(delay[:length], dtype=float)
    i_data = np.asarray(i_data[:length], dtype=float)
    q_data = np.asarray(q_data[:length], dtype=float)
    fit = np.asarray(fit[:length], dtype=float)
    finite = (
        np.isfinite(delay) & np.isfinite(i_data) & np.isfinite(q_data)
        & np.isfinite(fit)
    )
    order = np.argsort(delay[finite])
    return (
        delay[finite][order], i_data[finite][order],
        q_data[finite][order], fit[finite][order], fitted_frequency,
        t2_value, t2_error,
    )


def damped_cosine(params, delay):
    offset, slope, amplitude, frequency, phase, decay = params
    centered = delay - np.mean(delay)
    return (
        offset + slope * centered
        + amplitude * np.exp(-delay / decay)
        * np.cos(2.0 * np.pi * frequency * delay + phase)
    )


def independent_ramsey_fit(delay, signal, frequency_seed):
    span = max(float(np.ptp(signal)), np.finfo(float).eps)
    step = float(np.median(np.diff(delay)))
    difference_noise = max(
        1.4826 * float(np.median(np.abs(np.diff(signal)))) / np.sqrt(2.0),
        span * 1e-5,
        np.finfo(float).eps,
    )
    lower_frequency = max(0.5, frequency_seed - 1.5)
    upper_frequency = min(10.0, frequency_seed + 1.5)
    lower = np.array([
        np.min(signal) - span, -5.0 * span / max(np.ptp(delay), step),
        -2.0 * span, lower_frequency, -4.0 * np.pi,
        max(2.0 * step, 0.02),
    ])
    upper = np.array([
        np.max(signal) + span, 5.0 * span / max(np.ptp(delay), step),
        2.0 * span, upper_frequency, 4.0 * np.pi, 100.0,
    ])
    best = None
    for phase_seed in (0.0, 0.5 * np.pi, np.pi, -0.5 * np.pi):
        for amplitude_sign in (1.0, -1.0):
            initial = np.array([
                np.mean(signal), 0.0, amplitude_sign * span / 2.0,
                frequency_seed, phase_seed, 2.0,
            ])
            result = least_squares(
                lambda params: (
                    damped_cosine(params, delay) - signal
                ) / difference_noise,
                initial, bounds=(lower, upper), loss="soft_l1",
                f_scale=1.0, max_nfev=50000,
            )
            residual_sum = float(np.sum(
                (damped_cosine(result.x, delay) - signal) ** 2
            ))
            if best is None or residual_sum < best[0]:
                best = (residual_sum, result)
    residual_sum, result = best
    model = damped_cosine(result.x, delay)
    total_sum = float(np.sum((signal - np.mean(signal)) ** 2))
    r_squared = 1.0 - residual_sum / max(total_sum, np.finfo(float).eps)
    dof = max(1, delay.size - result.x.size)
    try:
        covariance = np.linalg.pinv(result.jac.T @ result.jac)
        covariance *= (2.0 * result.cost / dof)
        frequency_stderr = float(np.sqrt(max(covariance[3, 3], 0.0)))
    except np.linalg.LinAlgError:
        frequency_stderr = float("inf")
    residual = signal - model
    residual_noise = max(
        1.4826 * float(np.median(np.abs(residual - np.median(residual)))),
        np.finfo(float).eps,
    )
    return {
        "frequency_MHz": float(result.x[3]),
        "frequency_stderr_MHz": frequency_stderr,
        "r_squared": r_squared,
        "amplitude": float(result.x[2]),
        "decay_us": float(result.x[5]),
        "residual_sigma": residual_noise,
        "fit_snr": abs(float(result.x[2])) / residual_noise,
        "parameters": [float(value) for value in result.x],
    }


def ramsey_metrics(delay, i_data, q_data, saved_fit, builtin_frequency):
    fit_scores = []
    for label, signal in (("I", i_data), ("Q", q_data)):
        total = float(np.sum((signal - np.mean(signal)) ** 2))
        residual = float(np.sum((signal - saved_fit) ** 2))
        fit_scores.append((
            1.0 - residual / max(total, np.finfo(float).eps),
            label, signal,
        ))
    saved_r2, saved_signal_label, saved_signal = max(
        fit_scores, key=lambda item: item[0]
    )
    independent_candidates = [
        (label, independent_ramsey_fit(delay, signal, builtin_frequency))
        for label, signal in (("I", i_data), ("Q", q_data))
    ]
    independent_label, independent = max(
        independent_candidates, key=lambda item: item[1]["r_squared"]
    )
    saved_residual = saved_signal - saved_fit
    saved_noise = max(
        1.4826 * float(np.median(np.abs(
            saved_residual - np.median(saved_residual)
        ))),
        np.finfo(float).eps,
    )
    saved_contrast = float(np.percentile(saved_fit, 95.0)
                           - np.percentile(saved_fit, 5.0))
    return {
        "saved_fit_signal": saved_signal_label,
        "saved_fit_r_squared": float(saved_r2),
        "saved_fit_peak_to_peak": saved_contrast,
        "saved_fit_residual_sigma": saved_noise,
        "saved_fit_snr": saved_contrast / (2.0 * saved_noise),
        "independent_fit_signal": independent_label,
        "independent_fit": independent,
        "builtin_independent_frequency_delta_MHz": abs(
            builtin_frequency - independent["frequency_MHz"]
        ),
    }


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


def rabi_metrics(gains, population):
    smoothed = smooth_trace(population, 5)
    lower, upper = np.percentile(smoothed, [5.0, 95.0])
    difference_noise = max(
        1.4826 * float(np.median(np.abs(np.diff(smoothed)))),
        np.finfo(float).eps,
    )
    return {
        "population_min": float(np.min(smoothed)),
        "population_max": float(np.max(smoothed)),
        "population_span_5_95": float(upper - lower),
        "difference_noise_estimate": difference_noise,
        "span_to_difference_noise": float(
            (upper - lower) / difference_noise
        ),
        "gain_at_population_max": float(gains[np.argmax(smoothed)]),
        "gain_at_population_min": float(gains[np.argmin(smoothed)]),
        "smoothing_points": 5,
    }


def acquire_ramsey(device, current_ma, calibration, stage=None,
                   seed_description="completed wide-qspec seed",
                   ramsey_settings=None):
    settings = {
        "start_us": RAMSEY_START_US,
        "stop_us": RAMSEY_STOP_US,
        "steps": RAMSEY_STEPS,
        "detuning_MHz": RAMSEY_DETUNING_MHZ,
        "reps": RAMSEY_REPS,
        "rounds": RAMSEY_ROUNDS,
        "relax_delay_us": RAMSEY_RELAX_DELAY_US,
    }
    if ramsey_settings:
        unknown = set(ramsey_settings) - set(settings)
        if unknown:
            raise ValueError(
                f"Unknown Ramsey setting overrides: {sorted(unknown)}"
            )
        settings.update(ramsey_settings)
    settings["steps"] = int(settings["steps"])
    settings["reps"] = int(settings["reps"])
    settings["rounds"] = int(settings["rounds"])
    if settings["steps"] < 3 or settings["reps"] < 1 or settings["rounds"] < 1:
        raise ValueError(f"Invalid Ramsey acquisition settings: {settings}")
    readout = calibration["readout"]
    notes = (
        f"Ramsey qubit-frequency correction using the {seed_description} "
        f"and its matching Rabi pi pulse. {device['physical']}/"
        f"{device['row']}, {current_ma:+.1f} mA. The physical shifted readout "
        "tone is supplied directly."
    )
    if stage is None:
        if settings["reps"] == RAMSEY_REPS and settings["rounds"] == RAMSEY_ROUNDS:
            stage = stage_name("ramsey")
        else:
            stage = (
                f"ramsey_corr_r{settings['reps']}x{settings['rounds']}"
            )
    started_epoch = time.time()
    text, log_path = run_round_robin(
        device, current_ma, stage, {
            "PUCQ4_RUN_FLAGS": "t2r_correction",
            "PUCQ4_QFREQ_MHZ": calibration["seed_qfreq_MHz"],
            "PUCQ4_READOUT_FREQ_MHZ": readout["frequency_MHz"],
            "PUCQ4_READOUT_LENGTH_US": readout["length_us"],
            "PUCQ4_READOUT_GAIN": readout["gain"],
            "PUCQ4_READOUT_OFFSET_MHZ": readout["offset_mhz"],
            "PUCQ4_SIGMA_US": calibration["ramsey_sigma_us"],
            "PUCQ4_PI_AMP": calibration["ramsey_pi_amp"],
            "PUCQ4_RAMSEY_CORRECTION_START_US": settings["start_us"],
            "PUCQ4_RAMSEY_CORRECTION_STOP_US": settings["stop_us"],
            "PUCQ4_RAMSEY_CORRECTION_STEPS": settings["steps"],
            "PUCQ4_RAMSEY_CORRECTION_DETUNING_MHZ": settings[
                "detuning_MHz"
            ],
            "PUCQ4_RAMSEY_CORRECTION_REPS": settings["reps"],
            "PUCQ4_RAMSEY_CORRECTION_ROUNDS": settings["rounds"],
            "PUCQ4_RAMSEY_CORRECTION_RELAX_DELAY_US": (
                settings["relax_delay_us"]
            ),
            "PUCQ4_SUBSTUDY_NOTES": notes,
        },
    )
    operating = parse_single_match(
        r"Ramsey correction operating point: seed_frequency_MHz="
        r"([0-9.+-eE]+), readout_frequency_MHz=([0-9.+-eE]+), "
        r"sigma_us=([0-9.+-eE]+), pi_amp=([0-9.+-eE]+)",
        text, "Ramsey operating-point report",
    )
    reported_seed, reported_readout, reported_sigma, reported_pi = (
        float(value) for value in operating
    )
    expected_operating = [
        calibration["seed_qfreq_MHz"], readout["frequency_MHz"],
        calibration["ramsey_sigma_us"], calibration["ramsey_pi_amp"],
    ]
    if not np.allclose(
            [reported_seed, reported_readout, reported_sigma, reported_pi],
            expected_operating, rtol=0.0, atol=1e-6):
        raise RuntimeError(
            "Reported Ramsey operating point differs from the source "
            f"calibration: {[reported_seed, reported_readout, reported_sigma, reported_pi]} "
            f"vs {expected_operating}"
        )
    sweep = parse_single_match(
        r"Ramsey correction sweep settings: start_us=([0-9.+-eE]+), "
        r"stop_us=([0-9.+-eE]+), steps=([0-9]+), "
        r"detuning_MHz=([0-9.+-eE]+), reps=([0-9]+), "
        r"rounds=([0-9]+), relax_delay_us=([0-9.+-eE]+)",
        text, "Ramsey sweep-settings report",
    )
    reported_settings = [float(value) for value in sweep]
    expected_settings = [
        settings["start_us"], settings["stop_us"], settings["steps"],
        settings["detuning_MHz"], settings["reps"], settings["rounds"],
        settings["relax_delay_us"],
    ]
    if not np.allclose(
            reported_settings, expected_settings, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            f"Reported Ramsey settings {reported_settings} differ from "
            f"{expected_settings}"
        )
    correction = parse_single_match(
        r"Ramsey correction result: seed_frequency_MHz=([0-9.+-eE]+), "
        r"fitted_fringe_MHz=([0-9.+-eE]+), "
        r"intentional_detuning_MHz=([0-9.+-eE]+), "
        r"corrected_frequency_MHz=([0-9.+-eE]+)",
        text, "Ramsey correction result",
    )
    result_seed, fitted_fringe, result_detuning, corrected_frequency = (
        float(value) for value in correction
    )
    expected_corrected = result_seed - fitted_fringe + result_detuning
    if abs(corrected_frequency - expected_corrected) > 1e-8:
        raise RuntimeError("Logged Ramsey correction formula is inconsistent")
    h5_path, plot_path = canonical_artifacts(
        log_path, "t2_ge_correction_1", "Ramsey_ge_correction_1",
        started_epoch,
    )
    (delay, i_data, q_data, saved_fit, saved_frequency, t2_value,
     t2_error) = load_ramsey_trace(h5_path, device)
    if delay.size != settings["steps"]:
        raise RuntimeError(
            f"Saved {delay.size} Ramsey points; expected {settings['steps']}"
        )
    if np.any(np.diff(delay) <= 0):
        raise RuntimeError("Saved Ramsey delay axis is not strictly increasing")
    if abs(saved_frequency - fitted_fringe) > 1e-8:
        raise RuntimeError(
            f"Saved Ramsey frequency {saved_frequency} differs from logged "
            f"{fitted_fringe} MHz"
        )
    if not np.all(np.isfinite([t2_value, t2_error])) or t2_value <= 0:
        raise RuntimeError("Ramsey fit saved a non-physical T2 result")
    metrics = ramsey_metrics(
        delay, i_data, q_data, saved_fit, fitted_fringe
    )
    correction_size = corrected_frequency - reported_seed
    failures = []
    if not 2.0 <= fitted_fringe <= 6.0:
        failures.append(
            f"fitted fringe {fitted_fringe:.6f} MHz is not near 4 MHz"
        )
    if abs(correction_size) > MAXIMUM_RAMSEY_CORRECTION_MHZ:
        failures.append(
            f"correction {correction_size:+.6f} MHz exceeds "
            f"{MAXIMUM_RAMSEY_CORRECTION_MHZ:.3f} MHz"
        )
    independent = metrics["independent_fit"]
    saved_fit_accepted = (
        metrics["saved_fit_r_squared"] >= MINIMUM_RAMSEY_FIT_R2
        and metrics["saved_fit_snr"] >= MINIMUM_RAMSEY_FIT_SNR
    )
    independent_fit_accepted = (
        independent["r_squared"] >= MINIMUM_RAMSEY_FIT_R2
        and independent["fit_snr"] >= MINIMUM_RAMSEY_FIT_SNR
    )
    metrics["saved_fit_accepted"] = saved_fit_accepted
    metrics["independent_fit_accepted"] = independent_fit_accepted
    if saved_fit_accepted:
        metrics["acceptance_basis"] = "saved fit plus independent refit"
    else:
        metrics["acceptance_basis"] = (
            f"independent {metrics['independent_fit_signal']} refit "
            "cross-quadrature confirmation of canonical frequency"
        )
    if not independent_fit_accepted:
        failures.append(
            "independent fit quality is insufficient: "
            f"R2={independent['r_squared']:.3f} "
            f"(minimum {MINIMUM_RAMSEY_FIT_R2:.2f}), "
            f"SNR={independent['fit_snr']:.2f} "
            f"(minimum {MINIMUM_RAMSEY_FIT_SNR:.1f})"
        )
    if (independent["frequency_stderr_MHz"]
            > MAXIMUM_RAMSEY_FREQUENCY_STDERR_MHZ):
        failures.append(
            "independent frequency uncertainty "
            f"{independent['frequency_stderr_MHz']:.6f} MHz exceeds "
            f"{MAXIMUM_RAMSEY_FREQUENCY_STDERR_MHZ:.3f} MHz"
        )
    if (metrics["builtin_independent_frequency_delta_MHz"]
            > MAXIMUM_BUILTIN_INDEPENDENT_DELTA_MHZ):
        failures.append(
            "built-in/independent frequency difference "
            f"{metrics['builtin_independent_frequency_delta_MHz']:.6f} MHz "
            f"exceeds {MAXIMUM_BUILTIN_INDEPENDENT_DELTA_MHZ:.3f} MHz"
        )
    if failures:
        raise RuntimeError("Ramsey correction validation failed: " + "; ".join(failures))
    if not saved_fit_accepted:
        print(
            "Ramsey validation note: the canonical fitter's endpoint-based "
            f"{metrics['saved_fit_signal']} selection was weak "
            f"(R2={metrics['saved_fit_r_squared']:.3f}, "
            f"SNR={metrics['saved_fit_snr']:.2f}); retaining its correction "
            "only because the independent "
            f"{metrics['independent_fit_signal']} fit is strong "
            f"(R2={independent['r_squared']:.3f}, "
            f"SNR={independent['fit_snr']:.2f}) and agrees within "
            f"{metrics['builtin_independent_frequency_delta_MHz']:.6f} MHz.",
            flush=True,
        )
    return {
        "status": "complete",
        "completed": datetime.now().isoformat(),
        "seed_qfreq_MHz": reported_seed,
        "fitted_fringe_MHz": fitted_fringe,
        "intentional_detuning_MHz": result_detuning,
        "frequency_correction_MHz": correction_size,
        "corrected_qfreq_MHz": corrected_frequency,
        "ramsey_sigma_us": reported_sigma,
        "ramsey_pi_amp": reported_pi,
        "readout": {**readout, "frequency_MHz": reported_readout},
        "res_base_MHz": calibration["res_base_MHz"],
        "actual_delay_start_us": float(delay[0]),
        "actual_delay_stop_us": float(delay[-1]),
        "actual_delay_median_step_us": float(np.median(np.diff(delay))),
        "steps": int(delay.size),
        "reps": settings["reps"],
        "rounds": settings["rounds"],
        "relax_delay_us": settings["relax_delay_us"],
        "t2_star_us": t2_value,
        "t2_star_error_us": t2_error,
        "fit_metrics": metrics,
        "log": log_path,
        "h5": str(h5_path),
        "plot": str(plot_path),
    }


def predicted_rabi_pi_seed(calibration):
    prediction = (
        calibration["ramsey_pi_amp"] * calibration["ramsey_sigma_us"]
        / FIXED_RABI_SIGMA_US
    )
    return float(np.clip(prediction, 0.02, 0.98))


def acquire_rabi(device, current_ma, calibration, corrected_qfreq,
                 reference_h5=None):
    readout = calibration["readout"]
    pi_seed = predicted_rabi_pi_seed(calibration)
    notes = (
        "Ramsey-corrected current-by-gain Rabi map. Physical "
        f"{device['physical']}/{device['row']}, {current_ma:+.1f} mA; fixed "
        "sigma 0.060 us for both Yokogawas. Qubit frequency comes from the "
        "validated Ramsey correction and readout is the unchanged shifted "
        "physical tone."
    )
    started_epoch = time.time()
    text, log_path = run_round_robin(
        device, current_ma, stage_name("rabi"), {
            "PUCQ4_RUN_FLAGS": "rabi",
            "PUCQ4_QFREQ_MHZ": corrected_qfreq,
            "PUCQ4_READOUT_FREQ_MHZ": readout["frequency_MHz"],
            "PUCQ4_READOUT_LENGTH_US": readout["length_us"],
            "PUCQ4_READOUT_GAIN": readout["gain"],
            "PUCQ4_READOUT_OFFSET_MHZ": readout["offset_mhz"],
            "PUCQ4_SIGMA_US": FIXED_RABI_SIGMA_US,
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
        r"readout_frequency_MHz=([0-9.+-eE]+), sigma_us=([0-9.+-eE]+)",
        text, "Rabi operating-point report",
    )
    reported_qfreq, reported_readout, reported_sigma = (
        float(value) for value in operating
    )
    if abs(reported_qfreq - corrected_qfreq) > 1e-6:
        raise RuntimeError(
            f"Rabi used qfreq {reported_qfreq} MHz; expected "
            f"{corrected_qfreq} MHz"
        )
    if abs(reported_readout - readout["frequency_MHz"]) > 1e-6:
        raise RuntimeError(
            f"Rabi used readout {reported_readout} MHz; expected "
            f"{readout['frequency_MHz']} MHz"
        )
    if abs(reported_sigma - FIXED_RABI_SIGMA_US) > 1e-12:
        raise RuntimeError(
            f"Rabi used sigma {reported_sigma} us; expected "
            f"{FIXED_RABI_SIGMA_US} us"
        )
    sweep = parse_single_match(
        r"Rabi sweep settings: start=([0-9.+-eE]+), "
        r"stop=([0-9.+-eE]+), steps=([0-9]+), reps=([0-9]+), "
        r"rounds=([0-9]+), relax_delay_us=([0-9.+-eE]+), "
        r"ssf_shots_per_state=([0-9]+), "
        r"ssf_relax_delay_us=([0-9.+-eE]+)",
        text, "Rabi sweep-settings report",
    )
    reported_settings = [float(value) for value in sweep]
    expected_settings = [
        RABI_GAIN_START, RABI_GAIN_STOP, RABI_GAIN_STEPS, RABI_REPS,
        RABI_ROUNDS, RABI_RELAX_DELAY_US, RABI_SSF_SHOTS_PER_STATE,
        RABI_SSF_RELAX_DELAY_US,
    ]
    if not np.allclose(
            reported_settings, expected_settings, rtol=0.0, atol=1e-9):
        raise RuntimeError(
            f"Reported Rabi settings {reported_settings} differ from "
            f"{expected_settings}"
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
        raise RuntimeError("Could not parse the Rabi pi-amplitude report")
    reported_pi = float(pi_matches[-1])
    fit_available = "Rabi pi amplitude status: retained seed" not in text
    h5_path, plot_path = canonical_artifacts(
        log_path, "rabi_ge", "power_rabi_ge_plots", started_epoch
    )
    gains, population, _, _, ssf = load_rabi_trace(h5_path, device)
    if gains.size != RABI_GAIN_STEPS:
        raise RuntimeError(
            f"Saved {gains.size} Rabi points; expected {RABI_GAIN_STEPS}"
        )
    if np.any(np.diff(gains) <= 0):
        raise RuntimeError("Saved Rabi gain axis is not strictly increasing")
    actual_step = float(np.median(np.diff(gains)))
    requested_step = (
        (RABI_GAIN_STOP - RABI_GAIN_START) / (RABI_GAIN_STEPS - 1)
    )
    endpoint_shortfall = RABI_GAIN_STOP - float(gains[-1])
    if (abs(gains[0] - RABI_GAIN_START) > 1e-4
            or not 0.0 <= endpoint_shortfall <= max(0.01, 2 * requested_step)
            or abs(actual_step - requested_step) > 1e-4):
        raise RuntimeError(
            f"Unexpected saved gain axis {gains[0]}-{gains[-1]}, "
            f"step {actual_step}"
        )
    if reference_h5:
        reference_gains, _, _, _, _ = load_rabi_trace(
            reference_h5, device
        )
        if (reference_gains.size != gains.size
                or not np.allclose(
                    reference_gains, gains, rtol=0.0, atol=1e-9
                )):
            raise RuntimeError(
                "Actual Rabi gain axis differs from the completed reference row"
            )
    ssf_lengths = {name: int(values.size) for name, values in ssf.items()}
    if any(
            length != RABI_SSF_SHOTS_PER_STATE
            for length in ssf_lengths.values()):
        raise RuntimeError(f"Unexpected saved SSF lengths: {ssf_lengths}")
    metrics = rabi_metrics(gains, population)
    if metrics["population_span_5_95"] < MINIMUM_RABI_POPULATION_SPAN:
        raise RuntimeError(
            "Rabi row is flat: 5-95% calibrated-population span "
            f"{metrics['population_span_5_95']:.6f} is below "
            f"{MINIMUM_RABI_POPULATION_SPAN:.3f}"
        )
    return {
        "status": "complete",
        "completed": datetime.now().isoformat(),
        "qfreq_MHz": reported_qfreq,
        "qfreq_source": "validated Ramsey correction",
        "res_base_MHz": calibration["res_base_MHz"],
        "readout": {**readout, "frequency_MHz": reported_readout},
        "fixed_sigma_us": reported_sigma,
        "initial_pi_amp_seed": pi_seed,
        "fit_available": fit_available,
        "fitted_pi_amp": reported_pi if fit_available else None,
        "retained_pi_seed": None if fit_available else reported_pi,
        "actual_gain_start": float(gains[0]),
        "actual_gain_stop": float(gains[-1]),
        "actual_gain_median_step": actual_step,
        "requested_gain_stop_shortfall": endpoint_shortfall,
        "gain_steps": int(gains.size),
        "reps": RABI_REPS,
        "rounds": RABI_ROUNDS,
        "relax_delay_us": RABI_RELAX_DELAY_US,
        "ssf_shots_per_state": RABI_SSF_SHOTS_PER_STATE,
        "ssf_relax_delay_us": RABI_SSF_RELAX_DELAY_US,
        "ssf_centroids": {
            "g_I": g_i, "g_Q": g_q, "e_I": e_i, "e_Q": e_q,
            "separation": separation,
        },
        "saved_ssf_array_lengths": ssf_lengths,
        "population_metrics": metrics,
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


def stage_is_complete(stage):
    return (
        isinstance(stage, dict) and stage.get("status") == "complete"
        and Path(stage.get("h5", "")).is_file()
        and Path(stage.get("plot", "")).is_file()
    )


def print_plan(device, currents, calibrations, longest_path):
    ramsey_seconds = (
        RAMSEY_STEPS * RAMSEY_REPS * RAMSEY_ROUNDS
        * RAMSEY_RELAX_DELAY_US * 1e-6
    )
    rabi_seconds = (
        RABI_GAIN_STEPS * RABI_REPS * RABI_ROUNDS
        * RABI_RELAX_DELAY_US * 1e-6
    )
    settling_seconds = sum(
        device.get("settle_seconds_by_current", {}).get(current, 0)
        for current in currents
    )
    print(
        f"{device['yoko']} / physical {device['physical']} / "
        f"{device['row']} (Python index {device['index']}): "
        f"{len(currents)} Ramsey correction(s) plus {len(currents)} Rabi "
        "row(s)", flush=True,
    )
    print(
        "Ramsey: established 0-1 us / 50 points / 4 MHz detuning / "
        f"4000 us relax, raised to {RAMSEY_REPS} reps x "
        f"{RAMSEY_ROUNDS} rounds. Rabi: 0-1 / {RABI_GAIN_STEPS} points / "
        f"{RABI_REPS} reps x {RABI_ROUNDS} rounds / fixed sigma "
        f"{FIXED_RABI_SIGMA_US:.3f} us.", flush=True,
    )
    print(
        "Nominal programmed acquisition/settling floor: "
        f"{(ramsey_seconds + rabi_seconds) * len(currents) / 60.0:.1f} "
        f"min + {settling_seconds / 60.0:.1f} min; longest artifact path "
        f"{len(str(longest_path))} chars", flush=True,
    )
    for current_ma in currents:
        calibration = calibrations[current_ma]
        readout = calibration["readout"]
        print(
            f"  {current_ma:+.1f} mA: qspec seed="
            f"{calibration['seed_qfreq_MHz']:.6f} MHz; Ramsey pulse "
            f"sigma/pi={calibration['ramsey_sigma_us']:.6f} us/"
            f"{calibration['ramsey_pi_amp']:.6f}; resonator base="
            f"{calibration['res_base_MHz']:.3f} MHz; readout="
            f"{readout['frequency_MHz']:.3f} MHz / "
            f"{readout['length_us']:.1f} us / gain {readout['gain']:.4f} / "
            f"offset {readout['offset_mhz']:+.2f} MHz", flush=True,
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


def run_device(device_key, currents, force=False, plan_only=False):
    device = DEVICES[device_key]
    settings = campaign_settings()
    calibrations = source_calibrations(device_key, currents)
    longest_path = verify_paths(device, currents)
    print_plan(device, currents, calibrations, longest_path)
    if plan_only:
        print(
            "Plan only: no source, QICK, attenuator, data, or manifest state "
            "was changed", flush=True,
        )
        return

    manifest = load_manifest()
    map_data = prepare_map(manifest, device_key, settings)
    save_manifest(manifest)
    if all(
            map_data["points"].get(current_key(current), {}).get("status")
            == "complete"
            and stage_is_complete(
                map_data["points"][current_key(current)].get("ramsey")
            )
            and stage_is_complete(
                map_data["points"][current_key(current)].get("rabi")
            )
            for current in currents) and not force:
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
                if (not force and existing.get("status") == "complete"
                        and stage_is_complete(existing.get("ramsey"))
                        and stage_is_complete(existing.get("rabi"))):
                    print(
                        f"{device['physical']} {current_ma:+.1f} mA already "
                        "complete; skipping", flush=True,
                    )
                    continue
                calibration = calibrations[current_ma]
                if force and existing:
                    map_data.setdefault("superseded_attempts", {}).setdefault(
                        key, []
                    ).append(existing)
                    existing = {}
                point = {
                    **existing,
                    "status": "running",
                    "started": datetime.now().isoformat(),
                    "current_mA": current_ma,
                    "source_calibration": calibration,
                }
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
                    f"seed={calibration['seed_qfreq_MHz']:.6f} MHz; "
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
                        point["ramsey"] = acquire_ramsey(
                            device, current_ma, calibration
                        )
                        point["corrected_qfreq_MHz"] = point["ramsey"][
                            "corrected_qfreq_MHz"
                        ]
                        manifest["updated"] = datetime.now().isoformat()
                        save_manifest(manifest)
                    corrected_qfreq = float(point["ramsey"][
                        "corrected_qfreq_MHz"
                    ])
                    if force or not stage_is_complete(point.get("rabi")):
                        reference_h5 = next((
                            candidate.get("rabi", {}).get("h5")
                            for candidate in map_data["points"].values()
                            if stage_is_complete(candidate.get("rabi"))
                        ), None)
                        point["rabi"] = acquire_rabi(
                            device, current_ma, calibration,
                            corrected_qfreq, reference_h5=reference_h5,
                        )
                    point["status"] = "complete"
                    point["completed"] = datetime.now().isoformat()
                    manifest["updated"] = datetime.now().isoformat()
                    save_manifest(manifest)
                except KeyboardInterrupt:
                    point.update({
                        "status": "aborted",
                        "failed": datetime.now().isoformat(),
                        "error": "Operator interrupted the Ramsey/Rabi row",
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
                print(
                    f"{device['physical']} {current_ma:+.1f} mA complete: "
                    f"corrected qfreq={corrected_qfreq:.9f} MHz, "
                    "Rabi population 5-95 span="
                    f"{point['rabi']['population_metrics']['population_span_5_95']:.4f}",
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
        "--plan-only", action="store_true",
        help="Print resolved settings/provenance without touching hardware",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if RAMSEY_REPS < 100 or RAMSEY_ROUNDS < 2:
        raise ValueError("Production Ramsey correction requires high averaging")
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
            run_device(device_key, currents, plan_only=True)
        return
    try:
        for device_key in device_keys:
            run_device(
                device_key, currents, force=args.force, plan_only=False
            )
    finally:
        verify_all_sources_safe()


if __name__ == "__main__":
    main()
