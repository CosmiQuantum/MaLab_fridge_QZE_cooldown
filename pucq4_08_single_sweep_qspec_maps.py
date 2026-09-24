"""Acquire one continuous, high-average qspec sweep per PUCQ4 flux bias.

This controller deliberately runs only ``q_spec`` through the canonical
``round_robin_benchmark.py`` path.  Physical Q4/M5 is driven by Yoko 3 and
physical Q6/M6 by Yoko 4.  Each current gets its own established resonator
readout tone and one built-in SSF g/e calibration for the complete frequency
row; there is no frequency-band stitching.

Hardware is touched only from :func:`main`.  Every source is returned to 0 A
with output disabled in a ``finally`` block.
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
    load_manifest as load_calibration_manifest,
    run_round_robin,
    verify_inactive_source,
    wide_uniform_qspec_calibration,
)


MANIFEST = CAMPAIGN_ROOT / "single_sweep_qspec_manifest.json"
DEFAULT_REPS = 500
SSF_SHOTS_PER_STATE = 2000
SSF_RELAX_DELAY_US = 250.0
MINIMUM_REPORTED_PEAK_SNR = 4.0
SETTINGS = {
    "q4": {
        "frequency_start_MHz": 5000.0,
        "frequency_stop_MHz": 7300.0,
        "frequency_step_MHz": 0.04,
    },
    "q6": {
        "frequency_start_MHz": 4000.0,
        "frequency_stop_MHz": 7300.0,
        "frequency_step_MHz": 0.04,
    },
}


def load_manifest():
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text(encoding="utf-8"))
    return {
        "created": datetime.now().isoformat(),
        "purpose": "one continuous qspec sweep per current; no band stitching",
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
    settings = dict(SETTINGS[device_key])
    settings["steps"] = int(round(
        (settings["frequency_stop_MHz"] - settings["frequency_start_MHz"])
        / settings["frequency_step_MHz"]
    )) + 1
    settings.update({
        "reps": int(reps),
        "ssf_shots_per_state": SSF_SHOTS_PER_STATE,
        "ssf_relax_delay_us": SSF_RELAX_DELAY_US,
        "runner": "round_robin_benchmark.py",
        "run_flags": ["q_spec"],
        "rounds": 1,
        "sweeps_per_current": 1,
        "population_definition": (
            "one current-specific built-in SSF affine calibration for the full "
            "continuous row; signed ground=0/excited=1 projection; no row "
            "normalization"
        ),
        "attenuation_dB": {
            "resonator_DAC": [10, 15],
            "qubit_DAC": [5, 4],
            "ADC": 17,
        },
    })
    return settings


def current_key(current_ma):
    return f"{current_ma:+.1f}"


def prospective_artifact_path(device, current_ma, settings):
    stage = (
        f"single_qspec_{settings['frequency_start_MHz']:.0f}_"
        f"{settings['frequency_stop_MHz']:.0f}_r{settings['reps']}"
    )
    tag = (
        f"{device['yoko'].lower()}_{device['physical'].lower()}_"
        f"{current_ma:+05.1f}mA_{stage}"
    )
    filename = (
        "2099-12-31_23-59-59_qspec_ge_results_batch_0_"
        "Num_per_batch1.h5"
    )
    return (
        DATA_ROOT / f"flux_{tag}" / "2099-12-31_23-59-59" /
        "study_data" / "Data_h5" / "qspec_ge" / filename
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


def canonical_h5_for_log(log_path):
    run_dir = DATA_ROOT / f"flux_{Path(log_path).stem}"
    h5_files = sorted(
        run_dir.glob("*/study_data/Data_h5/qspec_ge/*.h5"),
        key=lambda path: path.stat().st_mtime,
    )
    if not h5_files:
        raise RuntimeError(
            f"Single-sweep qspec completed without canonical HDF5 under {run_dir}"
        )
    return h5_files[-1]


def load_trace(h5_path, device):
    with h5py.File(h5_path, "r") as handle:
        group = handle[f"Q{device['index'] + 1}"]
        frequency = decode_h5_numeric(group["Frequencies"])
        population = decode_h5_numeric(group["Population"])
        i_data = decode_h5_numeric(group["I"])
        q_data = decode_h5_numeric(group["Q"])
    common_length = min(
        frequency.size, population.size, i_data.size, q_data.size
    )
    frequency = np.asarray(frequency[:common_length], dtype=float)
    population = np.asarray(population[:common_length], dtype=float)
    i_data = np.asarray(i_data[:common_length], dtype=float)
    q_data = np.asarray(q_data[:common_length], dtype=float)
    finite = (
        np.isfinite(frequency) & np.isfinite(population)
        & np.isfinite(i_data) & np.isfinite(q_data)
    )
    order = np.argsort(frequency[finite])
    return (
        frequency[finite][order], population[finite][order],
        i_data[finite][order], q_data[finite][order],
    )


def peak_metrics(frequency, population, seed_mhz):
    if frequency.size < 20:
        raise RuntimeError("Single-sweep qspec trace has fewer than 20 samples")
    step_mhz = abs(float(np.median(np.diff(frequency))))
    smoothing_points = max(3, int(round(0.12 / max(step_mhz, 1e-12))))
    if smoothing_points % 2 == 0:
        smoothing_points += 1
    pad = smoothing_points // 2
    smoothed = np.convolve(
        np.pad(population, (pad, pad), mode="edge"),
        np.ones(smoothing_points, dtype=float) / smoothing_points,
        mode="valid",
    )
    distance = np.abs(frequency - seed_mhz)
    local = distance <= 3.0
    background = (distance >= 5.0) & (distance <= 20.0)
    if local.sum() < 5 or background.sum() < 20:
        raise RuntimeError("Qspec peak window is not fully contained in the sweep")
    baseline = float(np.median(smoothed[background]))
    noise = max(
        1.4826 * float(np.median(np.abs(smoothed[background] - baseline))),
        np.finfo(float).eps,
    )
    local_indices = np.flatnonzero(local)
    peak_index = int(local_indices[np.argmax(smoothed[local])])
    dip_index = int(local_indices[np.argmin(smoothed[local])])
    positive_excursion = float(smoothed[peak_index] - baseline)
    negative_excursion = float(baseline - smoothed[dip_index])
    return {
        "frequency_MHz": float(frequency[peak_index]),
        "population": float(smoothed[peak_index]),
        "baseline_population": baseline,
        "positive_excursion": positive_excursion,
        "positive_snr": positive_excursion / noise,
        "largest_absolute_direction": (
            "positive" if positive_excursion >= negative_excursion else "negative"
        ),
        "largest_absolute_snr": max(positive_excursion, negative_excursion) / noise,
        "background_sigma": noise,
        "smoothing_points": smoothing_points,
    }


def expected_log_path(device, current_ma, settings):
    stage = (
        f"single_qspec_{settings['frequency_start_MHz']:.0f}_"
        f"{settings['frequency_stop_MHz']:.0f}_r{settings['reps']}"
    )
    tag = (
        f"{device['yoko'].lower()}_{device['physical'].lower()}_"
        f"{current_ma:+05.1f}mA_{stage}"
    )
    return CAMPAIGN_ROOT / f"{tag}.log"


def acquire_row(device, current_ma, calibration, settings, reference_h5=None,
                existing_log_path=None):
    start_mhz = settings["frequency_start_MHz"]
    stop_mhz = settings["frequency_stop_MHz"]
    readout = calibration["readout"]
    stage = (
        f"single_qspec_{start_mhz:.0f}_{stop_mhz:.0f}_"
        f"r{settings['reps']}"
    )
    if existing_log_path is None:
        text, log_path = run_round_robin(device, current_ma, stage, {
            "PUCQ4_RUN_FLAGS": "q_spec",
            # This seed drives the excited-state SSF preparation pulse.  Explicit
            # start/stop variables independently set the continuous qspec axis.
            "PUCQ4_QFREQ_MHZ": calibration["qfreq_MHz"],
            "PUCQ4_QSPEC_START_MHZ": start_mhz,
            "PUCQ4_QSPEC_STOP_MHZ": stop_mhz,
            "PUCQ4_QSPEC_STEPS": settings["steps"],
            "PUCQ4_QSPEC_REPS": settings["reps"],
            "PUCQ4_QSPEC_SSF_SHOTS": settings["ssf_shots_per_state"],
            "PUCQ4_QSPEC_SSF_RELAX_DELAY_US": settings["ssf_relax_delay_us"],
            "PUCQ4_QSPEC_GAIN": calibration["qspec_gain"],
            "PUCQ4_QSPEC_LENGTH_US": calibration["qspec_length_us"],
            "PUCQ4_QSPEC_SCALING": "true",
            "PUCQ4_QSPEC_COMMON_GRID_MODE": "true",
            "PUCQ4_QSPEC_KEEP_SEED_ON_FIT_FAILURE": "true",
            "PUCQ4_QSPEC_LOG_CALIBRATION": "true",
            # This is the already offset physical resonator tone.  Supplying the
            # base resonator frequency here would silently lose the optimized
            # current-specific readout offset.
            "PUCQ4_READOUT_FREQ_MHZ": readout["frequency_MHz"],
            "PUCQ4_READOUT_LENGTH_US": readout["length_us"],
            "PUCQ4_READOUT_GAIN": readout["gain"],
            "PUCQ4_READOUT_OFFSET_MHZ": readout["offset_mhz"],
            "PUCQ4_SIGMA_US": calibration["sigma_us"],
            "PUCQ4_PI_AMP": calibration["pi_amp"],
        })
    else:
        log_path = str(existing_log_path)
        text = Path(existing_log_path).read_text(
            encoding="utf-8", errors="replace"
        )
    readout_matches = re.findall(
        r"Qspec readout frequency: ([0-9.+-eE]+) MHz", text
    )
    if len(readout_matches) != 1:
        raise RuntimeError(
            f"Expected one qspec acquisition/readout report, found "
            f"{len(readout_matches)} in {log_path}"
        )
    reported_readout_mhz = float(readout_matches[0])
    if abs(reported_readout_mhz - readout["frequency_MHz"]) > 1e-6:
        raise RuntimeError(
            f"Qspec used {reported_readout_mhz} MHz readout, expected "
            f"{readout['frequency_MHz']} MHz"
        )
    centroid_matches = re.findall(
        r"Qspec SSF centroids: g_I=([0-9.+-eE]+), g_Q=([0-9.+-eE]+), "
        r"e_I=([0-9.+-eE]+), e_Q=([0-9.+-eE]+)",
        text,
    )
    if len(centroid_matches) != 1:
        raise RuntimeError(
            f"Expected exactly one built-in SSF calibration for the complete "
            f"row, found {len(centroid_matches)} in {log_path}"
        )
    h5_path = canonical_h5_for_log(log_path)
    frequency, population, _, _ = load_trace(h5_path, device)
    if frequency.size != settings["steps"]:
        raise RuntimeError(
            f"Saved {frequency.size} finite qspec points; expected "
            f"{settings['steps']}"
        )
    actual_step_mhz = float(np.median(np.diff(frequency)))
    requested_span_mhz = stop_mhz - start_mhz
    endpoint_shortfall_mhz = stop_mhz - float(frequency[-1])
    maximum_shortfall_mhz = max(2.0, requested_span_mhz * 0.001)
    edge_margin_mhz = min(
        calibration["qfreq_MHz"] - float(frequency[0]),
        float(frequency[-1]) - calibration["qfreq_MHz"],
    )
    if (abs(frequency[0] - start_mhz) > 0.01
            or abs(actual_step_mhz - settings["frequency_step_MHz"]) > 0.001
            or not 0.0 <= endpoint_shortfall_mhz <= maximum_shortfall_mhz
            or edge_margin_mhz < 25.0):
        raise RuntimeError(
            "Unexpected saved frequency axis: "
            f"{frequency[0]}-{frequency[-1]} MHz, step {actual_step_mhz} MHz, "
            f"endpoint shortfall {endpoint_shortfall_mhz} MHz, minimum qfreq "
            f"edge margin {edge_margin_mhz} MHz"
        )
    if reference_h5:
        reference_frequency, _, _, _ = load_trace(reference_h5, device)
        if (reference_frequency.size != frequency.size
                or not np.allclose(
                    reference_frequency, frequency, rtol=0.0, atol=1e-6
                )):
            raise RuntimeError(
                "Actual frequency axis differs from the completed reference row"
            )
    g_i, g_q, e_i, e_q = (
        float(value) for value in centroid_matches[0]
    )
    metrics = peak_metrics(
        frequency, population, calibration["qfreq_MHz"]
    )
    return {
        "status": "complete",
        "completed": datetime.now().isoformat(),
        "current_mA": current_ma,
        "qfreq_seed_MHz": calibration["qfreq_MHz"],
        "qspec_gain": calibration["qspec_gain"],
        "qspec_length_us": calibration["qspec_length_us"],
        "sigma_us": calibration["sigma_us"],
        "pi_amp": calibration["pi_amp"],
        "res_base_MHz": calibration["res_base_MHz"],
        "readout": {
            **readout,
            "frequency_MHz": reported_readout_mhz,
        },
        "calibration_source": calibration["source"],
        "actual_frequency_start_MHz": float(frequency[0]),
        "actual_frequency_stop_MHz": float(frequency[-1]),
        "actual_frequency_median_step_MHz": actual_step_mhz,
        "requested_stop_shortfall_MHz": endpoint_shortfall_mhz,
        "minimum_qfreq_edge_margin_MHz": edge_margin_mhz,
        "steps": int(frequency.size),
        "reps": settings["reps"],
        "ssf_shots_per_state": settings["ssf_shots_per_state"],
        "ssf_relax_delay_us": settings["ssf_relax_delay_us"],
        "ssf_centroids": {
            "g_I": g_i, "g_Q": g_q, "e_I": e_i, "e_Q": e_q,
        },
        "peak_metrics": metrics,
        "peak_visibility_pass": (
            metrics["positive_snr"] >= MINIMUM_REPORTED_PEAK_SNR
        ),
        "log": log_path,
        "h5": str(h5_path),
    }


def prepare_map(manifest, device_key, settings):
    device = DEVICES[device_key]
    entry = manifest["devices"].setdefault(device_key, {
        "physical": device["physical"],
        "row": device["row"],
        "yoko": device["yoko"],
        "ip": device["ip"],
    })
    new_signature = (
        settings["frequency_start_MHz"], settings["frequency_stop_MHz"],
        settings["frequency_step_MHz"], settings["steps"], settings["reps"],
        settings["ssf_shots_per_state"],
    )
    map_data = entry.get("map")
    if map_data:
        old_signature = (
            map_data.get("frequency_start_MHz"),
            map_data.get("frequency_stop_MHz"),
            map_data.get("frequency_step_MHz"), map_data.get("steps"),
            map_data.get("reps"), map_data.get("ssf_shots_per_state"),
        )
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


def run_device(device_key, currents, reps, force=False, plan_only=False):
    device = DEVICES[device_key]
    settings = settings_for(device_key, reps)
    longest_path = verify_paths(device, currents, settings)
    calibration_manifest = load_calibration_manifest()
    calibration_device = calibration_manifest["devices"].get(device_key, {})
    calibrations = {
        current_ma: wide_uniform_qspec_calibration(
            device, calibration_device, current_ma
        )
        for current_ma in currents
    }
    estimate_hours = (
        settings["steps"] * settings["reps"] * 0.000133
        * len(currents) / 3600.0
    )
    print(
        f"{device['yoko']} / {device['physical']} (index {device['index']}): "
        f"{len(currents)} continuous sweep(s), {settings['steps']:,} points "
        f"from {settings['frequency_start_MHz']:.0f} to "
        f"{settings['frequency_stop_MHz']:.0f} MHz at "
        f"{settings['frequency_step_MHz']:.3f} MHz, "
        f"{settings['reps']:,} reps/point; estimated {estimate_hours:.1f} h",
        flush=True,
    )
    print(
        f"Longest prospective artifact path: {len(str(longest_path))} chars",
        flush=True,
    )
    for current_ma in currents:
        calibration = calibrations[current_ma]
        readout = calibration["readout"]
        print(
            f"  {current_ma:+.1f} mA: f01 seed "
            f"{calibration['qfreq_MHz']:.6f} MHz; resonator base "
            f"{calibration['res_base_MHz']:.3f} MHz; physical readout "
            f"{readout['frequency_MHz']:.3f} MHz, length "
            f"{readout['length_us']:.1f} us, gain {readout['gain']:.4f}; "
            f"qspec gain {calibration['qspec_gain']:.6f}, length "
            f"{calibration['qspec_length_us']:.1f} us",
            flush=True,
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
    # A hardware run can finish successfully and then fail a stricter offline
    # validator.  Recover such a complete canonical row from its deterministic
    # log/HDF5 before opening either Yokogawa; never reacquire it by default.
    if not force:
        for current_ma in currents:
            key = current_key(current_ma)
            existing = map_data["points"].get(key, {})
            if existing.get("status") == "complete":
                continue
            log_path = expected_log_path(device, current_ma, settings)
            if not log_path.is_file() or "timetaken=" not in log_path.read_text(
                    encoding="utf-8", errors="replace"):
                continue
            reference_h5 = next((
                point.get("h5")
                for point in map_data["points"].values()
                if point.get("status") == "complete"
                and Path(point.get("h5", "")).is_file()
            ), None)
            try:
                recovered = acquire_row(
                    device, current_ma, calibrations[current_ma], settings,
                    reference_h5=reference_h5,
                    existing_log_path=log_path,
                )
            except Exception as exc:
                print(
                    f"Existing {device['physical']} {current_ma:+.1f} mA row "
                    f"was not recoverable: {exc}",
                    flush=True,
                )
            else:
                recovered["recovered_without_reacquisition"] = True
                map_data["points"][key] = recovered
                manifest["updated"] = datetime.now().isoformat()
                save_manifest(manifest)
                print(
                    f"Recovered completed {device['physical']} "
                    f"{current_ma:+.1f} mA HDF5 without touching hardware",
                    flush=True,
                )
    if all(
            map_data["points"].get(current_key(current), {}).get("status")
            == "complete" for current in currents):
        return
    verify_inactive_source(device_key)

    with YokogawaGS200(
            device["ip"], max_current=MAX_ABS_CURRENT_A) as source:
        try:
            source.set_mode("current")
            source.set_range(0.01)
            if abs(source.get_current()) > 1e-7:
                source.set_output(True)
                source.ramp_current(0.0, sweeprate=RAMP_RATE_A_PER_S)
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
                    **calibration,
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
                    f"readout={calibration['readout']['frequency_MHz']:.3f} "
                    f"MHz, gain={calibration['readout']['gain']:.4f}, "
                    f"length={calibration['readout']['length_us']:.1f} us",
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
                        "error": "Operator interrupted the continuous qspec sweep",
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
                metrics = result["peak_metrics"]
                print(
                    f"{device['physical']} {current_ma:+.1f} mA complete: "
                    f"peak {metrics['frequency_MHz']:.6f} MHz, "
                    f"positive population SNR {metrics['positive_snr']:.2f}",
                    flush=True,
                )
                if not result["peak_visibility_pass"]:
                    print(
                        "WARNING: local positive peak did not reach the "
                        f"{MINIMUM_REPORTED_PEAK_SNR:.1f} SNR review threshold",
                        flush=True,
                    )
        finally:
            print(
                f"Returning {device['yoko']} to 0 A and disabling output",
                flush=True,
            )
            source.ramp_current(0.0, sweeprate=RAMP_RATE_A_PER_S)
            source.set_output(False)
            if abs(source.get_current()) > 1e-7 or source.get_output():
                raise RuntimeError(
                    f"Failed to leave {device['yoko']} at 0 A with output off"
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
        help=f"Qspec repetitions per frequency (default: {DEFAULT_REPS})",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Reacquire rows already complete for the same settings",
    )
    parser.add_argument(
        "--plan-only", action="store_true",
        help="Print the bounded acquisition and resolved calibrations without hardware",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if args.reps < 100:
        raise ValueError("This production map requires at least 100 reps")
    if args.current_ma:
        currents = args.current_ma
        if any(current not in CURRENT_GRID_MA for current in currents):
            raise ValueError(f"Currents must come from {CURRENT_GRID_MA}")
    else:
        currents = ACQUISITION_ORDER_MA
    device_keys = ("q4", "q6") if args.device == "all" else (args.device,)
    for device_key in device_keys:
        run_device(
            device_key, currents, args.reps, force=args.force,
            plan_only=args.plan_only,
        )


if __name__ == "__main__":
    main()
