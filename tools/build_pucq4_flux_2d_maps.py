"""Build provenance-backed PUCQ4 two-Yokogawa aggregate maps.

This script is analysis-only. It reads the flux campaign manifest and numeric
round-robin HDF5 datasets, writes PNG maps plus a JSON provenance report, and
never imports experiment modules or contacts hardware.
"""

import ast
import json
import re
import warnings
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm
from scipy.optimize import curve_fit


ROOT = Path(r"M:\_Data\20250822 - Olivia\pucq4_run_started_Aug_3\PUCQ4\pucq4_first_light")
CAMPAIGN = ROOT / "yoko_flux_campaign"
MANIFEST_PATH = CAMPAIGN / "flux_campaign_manifest.json"
OUTPUT = CAMPAIGN / "aggregate_2d"
CURRENT_GRID_MA = np.array([-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0])

DEVICES = {
    "q4": {"label": "Yoko 3 / physical Q4 / M5", "group": "Q5", "run_token": "yoko3_q4"},
    "q6": {"label": "Yoko 4 / physical Q6 / M6", "group": "Q6", "run_token": "yoko4_q6"},
}

EXPERIMENTS = {
    "qspec": {"folder": "qspec_ge", "x": "Frequencies", "channels": ("I", "Q"), "axis": "Qubit frequency (MHz)", "mode": "magnitude"},
    "qspec_population": {"folder": "qspec_ge", "x": "Frequencies", "channels": ("Population",), "axis": "Qubit frequency (MHz)", "mode": "population"},
    "res_spec": {"folder": "res_ge", "x": "freq_pts", "channels": ("Amps",), "axis": "Resonator frequency (MHz)", "mode": "amplitude"},
    "rabi": {"folder": "rabi_ge", "x": "Gains", "channels": ("Population",), "axis": "DAC gain", "mode": "population"},
    "t1": {"folder": "t1_ge", "x": "Delay Times", "channels": ("I", "Q"), "axis": "Delay (us)"},
    "t2r": {"folder": "t2_ge", "x": "Delay Times", "channels": ("I", "Q"), "axis": "Delay (us)"},
    "t2e": {"folder": "t2e_ge", "x": "Delay Times", "channels": ("I", "Q"), "axis": "Delay (us)"},
}


def current_key(current_ma):
    return f"{current_ma:+.1f}"


def current_tag(current_ma):
    sign = "+" if current_ma >= 0 else "-"
    return f"{sign}{abs(current_ma):04.1f}"


def run_from_log(log_path):
    if not log_path:
        return None
    return ROOT / f"flux_{Path(log_path).stem}"


def rabi_qfreq_from_log(point, h5_path):
    """Read the scalar drive frequency from the log that produced a Rabi HDF5."""
    logs = point.get("logs", {}) if point else {}
    candidates = []
    for name in ("post_readout_calibration", "calibration"):
        if logs.get(name):
            candidates.append(logs[name])
    candidates.extend(reversed(logs.get("rabi", []) or []))
    for log_path in candidates:
        run = run_from_log(log_path)
        if run is None:
            continue
        try:
            Path(h5_path).relative_to(run)
        except ValueError:
            continue
        try:
            lines = Path(log_path).read_text(
                encoding="utf-8", errors="replace"
            ).splitlines()
        except OSError:
            continue
        matches = []
        for line in lines:
            if "Rabi configuration:" not in line:
                continue
            match = re.search(r"'qubit_freq_ge':\s*([-+0-9.eE]+)", line)
            if match:
                matches.append(float(match.group(1)))
        if matches:
            return matches[-1], str(log_path)
    return None, None


def decode_numeric(dataset, min_size=2):
    value = dataset[0] if dataset.shape else dataset[()]
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).ravel()
    if np.isscalar(value) and not isinstance(value, (bytes, str, np.bytes_)):
        return np.asarray([value], dtype=float)
    if isinstance(value, (bytes, np.bytes_)):
        text = value.decode("utf-8")
    else:
        text = str(value)
    text = text.strip()
    if not text or text in {"None", "[None]"}:
        raise ValueError("placeholder dataset")
    cleaned = text.replace("np.float64(", "").replace("np.int64(", "").replace(")", "")
    # Round-robin HDF5 stores large sweep arrays as one serialized text value.
    # ``ast.literal_eval`` can overflow CPython's parser stack on the 62,501-
    # point common qspec grid, even though the data are otherwise valid.  Parse
    # ordinary numeric list syntax with NumPy first; this is both bounded-memory
    # and substantially faster for the full current map.
    numeric_text = cleaned.translate(str.maketrans({
        "[": " ", "]": " ", "(": " ", "{": " ", "}": " ", ",": " ",
    }))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        array = np.fromstring(numeric_text, sep=" ", dtype=float)
    if array.size >= min_size:
        if not np.isfinite(array).any():
            raise ValueError("dataset has no finite values")
        return array
    try:
        parsed = ast.literal_eval(cleaned)
        array = np.asarray(parsed, dtype=float).ravel()
    except (SyntaxError, ValueError, TypeError):
        fields = [field for field in re.split(r"[,\s\[\]]+", cleaned) if field]
        array = np.asarray([float(field) for field in fields], dtype=float)
    if array.size < min_size or not np.isfinite(array).any():
        raise ValueError(f"dataset has fewer than {min_size} finite values")
    return array


def fit_local_population_peak(frequency, population, seed_mhz):
    """Fit a local positive Lorentzian without letting the multi-GHz trace steer it."""
    seed_mhz = float(seed_mhz)
    distance = np.abs(frequency - seed_mhz)
    local = (distance <= 2.0) & np.isfinite(population)
    if local.sum() < 40:
        return None
    x_data = np.asarray(frequency[local], dtype=float)
    y_data = np.asarray(population[local], dtype=float)
    y_smooth = np.convolve(
        np.pad(y_data, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid"
    )
    edge = np.abs(x_data - seed_mhz) >= 1.2
    baseline = float(np.nanmedian(y_smooth[edge]))
    center_zone = np.abs(x_data - seed_mhz) <= 0.8
    center_indices = np.flatnonzero(center_zone)
    peak_index = int(center_indices[np.nanargmax(y_smooth[center_zone])])
    center_guess = float(x_data[peak_index])
    amplitude_guess = max(float(y_smooth[peak_index] - baseline), 1e-6)

    def lorentzian(axis, offset, slope, amplitude, center, hwhm):
        return (
            offset
            + slope * (axis - seed_mhz)
            + amplitude / (1.0 + ((axis - center) / hwhm) ** 2)
        )

    try:
        params, _ = curve_fit(
            lorentzian,
            x_data,
            y_smooth,
            p0=(baseline, 0.0, amplitude_guess, center_guess, 0.15),
            bounds=(
                (-np.inf, -np.inf, 0.0, seed_mhz - 1.0, 0.02),
                (np.inf, np.inf, np.inf, seed_mhz + 1.0, 1.0),
            ),
            maxfev=30000,
        )
    except (RuntimeError, ValueError, FloatingPointError):
        return None
    offset, slope, amplitude, center, hwhm = (float(value) for value in params)
    fitted = lorentzian(x_data, *params)
    residual = y_smooth - fitted
    residual_mad = float(np.nanmedian(np.abs(residual - np.nanmedian(residual))))
    residual_noise = max(1.4826 * residual_mad, np.finfo(float).eps)
    return {
        "fit_center_MHz": center,
        "fit_fwhm_MHz": 2.0 * hwhm,
        "fit_peak_population": offset + amplitude,
        "fit_amplitude": amplitude,
        "fit_residual_noise": residual_noise,
        "fit_snr": amplitude / residual_noise,
    }


def normalized_trace(channels):
    arrays = [np.asarray(channel, dtype=float).ravel() for channel in channels]
    length = min(array.size for array in arrays)
    matrix = np.column_stack([array[:length] for array in arrays])
    finite = np.all(np.isfinite(matrix), axis=1)
    if finite.sum() < 2:
        raise ValueError("insufficient finite channel data")
    filled = matrix.copy()
    filled[~finite] = np.nan
    if filled.shape[1] == 1:
        trace = filled[:, 0]
    else:
        centered = filled - np.nanmedian(filled, axis=0)
        covariance = np.cov(centered[finite], rowvar=False)
        _, vectors = np.linalg.eigh(covariance)
        principal = vectors[:, -1]
        # Eigenvectors have an arbitrary +/- sign. Fix it deterministically so
        # neighboring bias columns cannot appear flipped solely by the PCA
        # implementation.
        anchor = int(np.argmax(np.abs(principal)))
        if principal[anchor] < 0:
            principal = -principal
        trace = centered @ principal
    lo, hi = np.nanpercentile(trace, [2, 98])
    scale = hi - lo
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("flat trace")
    return np.clip((trace - lo) / scale, 0.0, 1.0)


def preferred_runs(device_key, current_ma, experiment, point):
    logs = point.get("logs", {}) if point else {}
    runs = []
    if experiment in {"qspec", "qspec_population", "res_spec", "rabi"}:
        if experiment == "res_spec":
            run = run_from_log(logs.get("wide_res_survey"))
            if run:
                runs.append(run)
        for name in ("post_readout_calibration", "calibration"):
            run = run_from_log(logs.get(name))
            if run:
                runs.append(run)
        if experiment == "rabi":
            for log_path in reversed(logs.get("rabi", []) or []):
                run = run_from_log(log_path)
                if run:
                    runs.append(run)
    else:
        run = run_from_log(logs.get("coherence"))
        if run:
            runs.append(run)

    token = DEVICES[device_key]["run_token"]
    tag = current_tag(current_ma)
    candidates = [path for path in ROOT.glob(f"flux_{token}_{tag}mA_*") if path.is_dir()]
    stage_priority = {
        "qspec": ("post_readout_calibration", "calibration", "wide_locator_diagnostic", "locator", "readout_bootstrap_qspec"),
        "qspec_population": ("post_readout_calibration", "readout_bootstrap_qspec", "calibration"),
        "res_spec": ("wide_res_survey", "post_readout_calibration", "coherence", "calibration", "raw_iq_rabi", "locator"),
        "rabi": ("post_readout_calibration", "rabi_retry", "raw_iq_rabi_retry", "raw_iq_rabi", "calibration"),
        "t1": ("coherence",), "t2r": ("coherence",), "t2e": ("coherence",),
    }[experiment]
    if point.get("status") != "complete" and experiment == "qspec":
        # For an unresolved point, show the bounded locator evidence instead of
        # an older failed post-readout trace. The status annotation remains
        # "failed", so this column cannot be mistaken for an accepted center.
        stage_priority = ("wide_locator_diagnostic", "calibration", "locator", "post_readout_calibration")
    for stage in stage_priority:
        matching = [path for path in candidates if stage in path.name]
        runs.extend(sorted(matching, key=lambda path: path.stat().st_mtime, reverse=True))

    unique = []
    seen = set()
    for run in runs:
        if run not in seen:
            unique.append(run)
            seen.add(run)
    return unique


def load_trace(device_key, current_ma, experiment, point):
    spec = EXPERIMENTS[experiment]
    group_name = DEVICES[device_key]["group"]
    errors = []
    for run in preferred_runs(device_key, current_ma, experiment, point):
        files = sorted(run.glob(f"**/Data_h5/{spec['folder']}/*.h5"), key=lambda path: path.stat().st_mtime, reverse=True)
        for path in files:
            try:
                with h5py.File(path, "r") as handle:
                    group = handle[group_name]
                    x = decode_numeric(group[spec["x"]])
                    channels = [decode_numeric(group[name]) for name in spec["channels"]]
                    if experiment == "res_spec":
                        # Resonator HDF5 stores the swept offsets in freq_pts and
                        # the absolute run center separately in freq_center.
                        center = decode_numeric(group["freq_center"])
                        x = x + float(center[0])
                length = min([x.size] + [channel.size for channel in channels])
                if length < 2:
                    raise ValueError("trace too short")
                x = x[:length]
                mode = spec.get("mode", "projected")
                if mode == "magnitude":
                    signal = channels[0][:length] + 1j * channels[1][:length]
                    response = np.abs(signal)
                elif mode == "population":
                    response = np.asarray(channels[0][:length], dtype=float)
                elif mode == "amplitude":
                    response = np.asarray(channels[0][:length], dtype=float)
                else:
                    response = normalized_trace([channel[:length] for channel in channels])
                order = np.argsort(x)
                return x[order], response[order], path
            except (KeyError, OSError, ValueError) as exc:
                errors.append(f"{path}: {exc}")
    if errors:
        raise ValueError(errors[-1])
    raise FileNotFoundError(f"no {experiment} HDF5 candidate")


def collect_device(device_key, manifest):
    points = manifest["devices"][device_key].get("points", {})
    collected = {experiment: {} for experiment in EXPERIMENTS}
    provenance = {"device": DEVICES[device_key], "currents": {}}
    for current in CURRENT_GRID_MA:
        key = current_key(current)
        point = dict(points.get(key, {}))
        survey = manifest["devices"][device_key].get("wide_res_survey", {}).get(key)
        if survey:
            point["logs"] = dict(point.get("logs", {}))
            point["logs"]["wide_res_survey"] = survey.get("log")
        status = point.get("status", "missing")
        provenance["currents"][key] = {"status": status, "experiments": {}}
        for experiment in EXPERIMENTS:
            if experiment == "qspec_population" and status != "complete":
                provenance["currents"][key]["experiments"][experiment] = {
                    "available": False,
                    "reason": "calibrated population omitted for a non-final point",
                }
                continue
            try:
                x, response, path = load_trace(device_key, current, experiment, point)
            except (FileNotFoundError, ValueError) as exc:
                provenance["currents"][key]["experiments"][experiment] = {"available": False, "reason": str(exc)}
                continue
            collected[experiment][float(current)] = (x, response, status)
            experiment_provenance = {
                "available": True, "h5": str(path), "points": int(x.size),
                "axis_min": float(np.nanmin(x)), "axis_max": float(np.nanmax(x)),
            }
            if experiment == "rabi":
                rabi_qfreq_mhz, rabi_log = rabi_qfreq_from_log(point, path)
                if rabi_qfreq_mhz is not None:
                    final_qfreq_mhz = point.get("qfreq_MHz")
                    experiment_provenance.update({
                        "qfreq_used_MHz": rabi_qfreq_mhz,
                        "qfreq_log": rabi_log,
                        "qfreq_matches_final": bool(
                            final_qfreq_mhz is not None
                            and abs(rabi_qfreq_mhz - float(final_qfreq_mhz)) <= 1e-6
                        ),
                    })
            fit_dataset = {"t1": "T1", "t2r": "T2", "t2e": "T2E"}.get(experiment)
            if fit_dataset:
                try:
                    with h5py.File(path, "r") as handle:
                        fit_value = decode_numeric(
                            handle[DEVICES[device_key]["group"]][fit_dataset], min_size=1
                        )
                    experiment_provenance["fit_value_us"] = float(fit_value[0])
                except (KeyError, OSError, ValueError, TypeError):
                    pass
            provenance["currents"][key]["experiments"][experiment] = experiment_provenance
    return collected, provenance


def status_note(data):
    return " | ".join(f"{current:+g}: {status}" for current, (_, _, status) in sorted(data.items()))


def plot_frequency_columns(device_key, experiment, data):
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
    if data:
        axis_min = min(np.nanmin(values[0]) for values in data.values())
        axis_max = max(np.nanmax(values[0]) for values in data.values())
        grid = np.linspace(axis_min, axis_max, 2000)
        image = np.full((grid.size, CURRENT_GRID_MA.size), np.nan)
        for column, current in enumerate(CURRENT_GRID_MA):
            values = data.get(float(current))
            if not values:
                continue
            axis, response, _ = values
            unique_axis, indices = np.unique(axis, return_index=True)
            inside = (grid >= unique_axis.min()) & (grid <= unique_axis.max())
            image[inside, column] = np.interp(grid[inside], unique_axis, response[indices])
        mode = EXPERIMENTS[experiment].get("mode")
        if mode == "population":
            vmin, vmax = 0.0, 1.0
            label = "SSF-calibrated qubit population (g=0, e=1)"
        elif mode in {"magnitude", "amplitude"}:
            finite_values = np.concatenate([
                values[1][np.isfinite(values[1])] for values in data.values()
            ])
            vmin, vmax = np.nanpercentile(finite_values, [0.5, 99.5])
            label = "Measured signal magnitude (arb.)"
        else:
            vmin, vmax = 0.0, 1.0
            label = "Row-normalized measured response"
        mesh = ax.pcolormesh(CURRENT_GRID_MA, grid, np.ma.masked_invalid(image), shading="nearest",
                             cmap="viridis", vmin=vmin, vmax=vmax)
        fig.colorbar(mesh, ax=ax, label=label)
    ax.set_xticks(CURRENT_GRID_MA)
    ax.set_xlabel("Yokogawa current (mA)")
    ax.set_ylabel(EXPERIMENTS[experiment]["axis"])
    ax.set_title(f"{DEVICES[device_key]['label']} — {experiment.replace('_', ' ')} 2D sweep")
    ax.text(0.01, -0.13, "Each colored column is one acquired frequency sweep; blank regions were not measured. No interpolation or per-row normalization across bias current.",
            transform=ax.transAxes, fontsize=9)
    path = OUTPUT / f"{device_key}_{experiment}_2d.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_qspec_curve_gallery(device_key, experiment, data, manifest):
    """Show every qspec sample without compressing narrow sweeps onto one global axis."""
    if experiment not in {"qspec", "qspec_population"}:
        raise ValueError(f"Unsupported qspec gallery experiment: {experiment}")

    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    mode = EXPERIMENTS[experiment].get("mode")
    line_label = "Calibrated population" if mode == "population" else "Measured |I+iQ| (arb.)"
    for ax, current in zip(axes.ravel(), CURRENT_GRID_MA):
        values = data.get(float(current))
        if values:
            frequency, response, status = values
            ax.plot(response, frequency, color="#9ecae1", linewidth=0.45, alpha=0.55)
            if response.size >= 5 and np.isfinite(response).all():
                padded = np.pad(response, (2, 2), mode="edge")
                smooth = np.convolve(padded, np.ones(5) / 5, mode="valid")
                ax.plot(smooth, frequency, color="tab:blue", linewidth=1.1)
            ax.set_title(f"{current:+g} mA | {status} | n={frequency.size}", fontsize=10)
            ax.set_ylim(float(np.nanmin(frequency)), float(np.nanmax(frequency)))
            point = manifest["devices"][device_key].get("points", {}).get(current_key(current), {})
            center = point.get("qfreq_MHz") if point.get("status") == "complete" else None
            if center is None:
                diagnostic = manifest["devices"][device_key].get("diagnostics", {}).get(
                    current_key(current), {}
                )
                center = diagnostic.get("fit_qfreq_MHz")
            if center is not None and frequency.min() <= center <= frequency.max():
                ax.axhline(center, color="tab:orange", linestyle="--", linewidth=1.0)
                ax.text(0.98, 0.03, f"fit {center:.3f} MHz", ha="right", va="bottom",
                        transform=ax.transAxes, fontsize=7, color="darkorange")
            ax.grid(alpha=0.2)
            ax.set_facecolor("#f4fbf4" if status == "complete" else "#fff5f5")
        else:
            ax.text(0.5, 0.5, "No final calibrated trace" if mode == "population" else "No numeric trace",
                    ha="center", va="center", transform=ax.transAxes, fontsize=10)
            ax.set_title(f"{current:+g} mA | unavailable", fontsize=10)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_facecolor("#f1f1f1")
        ax.tick_params(labelsize=8)

    fig.suptitle(
        f"{DEVICES[device_key]['label']} - full {experiment.replace('_', ' ')} curves",
        fontsize=15,
    )
    fig.supxlabel(line_label, fontsize=11)
    fig.supylabel("Actual swept qubit frequency (MHz)", fontsize=11)
    suffix = "population_" if mode == "population" else ""
    path = OUTPUT / f"{device_key}_qspec_{suffix}full_curves.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def collect_uniform_qspec(device_key, manifest):
    """Load the dedicated common-frequency-grid qspec campaign without renormalizing rows."""
    device_manifest = manifest["devices"][device_key]
    uniform = device_manifest.get("uniform_qspec_map", {})
    result = {"signal": {}, "population": {}}
    provenance = {
        "frequency_start_MHz": uniform.get("frequency_start_MHz"),
        "frequency_stop_MHz": uniform.get("frequency_stop_MHz"),
        "frequency_step_MHz": uniform.get("frequency_step_MHz"),
        "actual_frequency_start_MHz": uniform.get("actual_frequency_start_MHz"),
        "actual_frequency_stop_MHz": uniform.get("actual_frequency_stop_MHz"),
        "actual_frequency_median_step_MHz": uniform.get(
            "actual_frequency_median_step_MHz"
        ),
        "steps": uniform.get("steps"),
        "reps": uniform.get("reps"),
        "ssf_shots_per_state": uniform.get("ssf_shots_per_state"),
        "ssf_relax_delay_us": uniform.get("ssf_relax_delay_us"),
        "attenuation_dB": uniform.get("attenuation_dB"),
        "population_definition": uniform.get("population_definition"),
        "currents": {},
    }
    reference_frequency = None

    def local_metrics(frequency, values, center):
        if center is None:
            return None
        distance = np.abs(frequency - float(center))
        local = (distance <= 0.5) & np.isfinite(values)
        background = (distance >= 2.0) & (distance <= 5.0) & np.isfinite(values)
        if local.sum() < 3 or background.sum() < 10:
            return None
        finite = np.isfinite(values)
        indices = np.arange(values.size)
        filled = np.interp(indices, indices[finite], values[finite])
        smooth_points = 5
        padded = np.pad(filled, (smooth_points // 2, smooth_points // 2), mode="edge")
        smoothed = np.convolve(padded, np.ones(smooth_points) / smooth_points, mode="valid")
        baseline = float(np.nanmedian(smoothed[background]))
        mad = float(np.nanmedian(np.abs(smoothed[background] - baseline)))
        noise = max(1.4826 * mad, np.finfo(float).eps)
        local_indices = np.flatnonzero(local)
        peak_index = int(local_indices[np.nanargmax(
            np.abs(smoothed[local] - baseline)
        )])
        excursion = float(abs(smoothed[peak_index] - baseline))
        return {
            "smoothing_points": smooth_points,
            "background_median": baseline,
            "local_peak_excursion": excursion,
            "robust_background_noise": noise,
            "local_peak_snr": excursion / noise,
            "local_extremum_frequency_MHz": float(frequency[peak_index]),
            "offset_from_final_center_MHz": float(frequency[peak_index] - float(center)),
        }

    for current in CURRENT_GRID_MA:
        key = current_key(current)
        point = uniform.get("points", {}).get(key, {})
        row_provenance = {"status": point.get("status", "missing")}
        provenance["currents"][key] = row_provenance
        if point.get("status") != "complete":
            continue
        path = Path(point["h5"])
        try:
            with h5py.File(path, "r") as handle:
                group = handle[DEVICES[device_key]["group"]]
                frequency = decode_numeric(group["Frequencies"])
                i_data = decode_numeric(group["I"])
                q_data = decode_numeric(group["Q"])
                population = decode_numeric(group["Population"])
        except (KeyError, OSError, ValueError) as exc:
            row_provenance.update({"available": False, "reason": str(exc), "h5": str(path)})
            continue
        length = min(frequency.size, i_data.size, q_data.size, population.size)
        frequency = frequency[:length]
        i_data = i_data[:length]
        q_data = q_data[:length]
        population = population[:length]
        order = np.argsort(frequency)
        frequency = frequency[order]
        # Keep the signal map literal: this is the acquired complex-readout
        # magnitude, not a row-centered or row-normalized contrast.  The
        # separate Population dataset carries the SSF calibration.
        signal = np.abs(i_data[order] + 1j * q_data[order])
        population = population[order]
        if reference_frequency is None:
            reference_frequency = frequency
        elif (frequency.size != reference_frequency.size
              or not np.allclose(frequency, reference_frequency, rtol=0, atol=1e-6)):
            row_provenance.update({
                "available": False,
                "reason": "frequency array does not match the dedicated common grid",
                "h5": str(path),
            })
            continue
        result["signal"][float(current)] = (frequency, signal, "complete")
        result["population"][float(current)] = (frequency, population, "complete")
        qfreq_seed = point.get("qfreq_seed_MHz")
        row_provenance.update({
            "available": True,
            "h5": str(path),
            "points": int(frequency.size),
            "axis_min": float(frequency.min()),
            "axis_max": float(frequency.max()),
            "qfreq_seed_MHz": qfreq_seed,
            "lower_frequency_margin_MHz": (
                None if qfreq_seed is None
                else float(qfreq_seed) - float(frequency.min())
            ),
            "upper_frequency_margin_MHz": (
                None if qfreq_seed is None
                else float(frequency.max()) - float(qfreq_seed)
            ),
            "qspec_gain": point.get("qspec_gain"),
            "readout": point.get("readout"),
            "signal_local_metrics": local_metrics(
                frequency, signal, point.get("qfreq_seed_MHz")
            ),
            "population_local_metrics": local_metrics(
                frequency, population, point.get("qfreq_seed_MHz")
            ),
        })
    if reference_frequency is not None:
        provenance["actual_frequency_start_MHz"] = float(reference_frequency[0])
        provenance["actual_frequency_stop_MHz"] = float(reference_frequency[-1])
        provenance["actual_frequency_median_step_MHz"] = float(
            np.median(np.diff(reference_frequency))
        )
    provenance["available_current_count"] = len(result["population"])
    provenance["all_current_rows_available"] = (
        len(result["population"]) == CURRENT_GRID_MA.size
    )
    provenance["frequency_arrays_identical"] = bool(
        reference_frequency is not None
        and all(
            row.get("available") is True
            for row in provenance["currents"].values()
        )
    )
    return result, provenance


def collect_wide_uniform_qspec(device_key, manifest):
    """Load and stitch the lower-frequency bands using saved Population data."""
    device_manifest = manifest["devices"][device_key]
    wide = device_manifest.get("wide_uniform_qspec_map", {})
    result = {}
    provenance = {
        "frequency_start_MHz": wide.get("frequency_start_MHz"),
        "frequency_stop_MHz": wide.get("frequency_stop_MHz"),
        "frequency_step_MHz": wide.get("frequency_step_MHz"),
        "combined_steps": wide.get("combined_steps"),
        "reps": wide.get("reps"),
        "segments_MHz": wide.get("segments_MHz"),
        "ssf_shots_per_state": wide.get("ssf_shots_per_state"),
        "ssf_relax_delay_us": wide.get("ssf_relax_delay_us"),
        "attenuation_dB": wide.get("attenuation_dB"),
        "population_definition": wide.get("population_definition"),
        "currents": {},
    }
    reference_frequency = None

    def local_metrics(frequency, population, center):
        if center is None:
            return None
        center = float(center)
        distance = np.abs(frequency - center)
        local = (distance <= 3.0) & np.isfinite(population)
        background = (distance >= 8.0) & (distance <= 25.0) & np.isfinite(population)
        if local.sum() < 10 or background.sum() < 20:
            return None
        finite = np.isfinite(population)
        filled = np.interp(np.arange(population.size), np.flatnonzero(finite), population[finite])
        smooth_points = 5
        smoothed = np.convolve(
            np.pad(filled, (2, 2), mode="edge"),
            np.ones(smooth_points) / smooth_points,
            mode="valid",
        )
        baseline = float(np.nanmedian(smoothed[background]))
        mad = float(np.nanmedian(np.abs(smoothed[background] - baseline)))
        noise = max(1.4826 * mad, np.finfo(float).eps)
        local_indices = np.flatnonzero(local)
        peak_index = int(local_indices[np.nanargmax(smoothed[local] - baseline)])
        excursion = float(smoothed[peak_index] - baseline)
        metrics = {
            "smoothing_points": smooth_points,
            "background_median": baseline,
            "local_peak_population": float(smoothed[peak_index]),
            "local_peak_excursion": excursion,
            "robust_background_noise": noise,
            "local_peak_snr": excursion / noise,
            "local_peak_frequency_MHz": float(frequency[peak_index]),
            "offset_from_seed_MHz": float(frequency[peak_index] - center),
        }
        fitted = fit_local_population_peak(frequency, population, center)
        if fitted is not None:
            metrics.update(fitted)
        return metrics

    for current in CURRENT_GRID_MA:
        key = current_key(current)
        point = wide.get("points", {}).get(key, {})
        row_provenance = {
            "status": point.get("status", "missing"),
            "qfreq_seed_MHz": point.get("qfreq_MHz"),
            "calibration_source": point.get("source"),
            "qspec_gain": point.get("qspec_gain"),
            "qspec_length_us": point.get("qspec_length_us"),
            "res_base_MHz": point.get("res_base_MHz"),
            "readout": point.get("readout"),
            "segments": [],
        }
        provenance["currents"][key] = row_provenance
        segments = point.get("segments", [])
        if point.get("status") != "complete" or not segments:
            continue
        frequency_parts = []
        population_parts = []
        failed = None
        for index, segment in enumerate(segments):
            path = Path(segment.get("h5", ""))
            segment_provenance = {
                "status": segment.get("status"),
                "h5": str(path),
                "requested_start_MHz": segment.get("requested_start_MHz"),
                "requested_stop_MHz": segment.get("requested_stop_MHz"),
                "ssf_centroids": segment.get("ssf_centroids"),
            }
            row_provenance["segments"].append(segment_provenance)
            if segment.get("status") != "complete" or not path.is_file():
                failed = f"segment {index + 1} is unavailable"
                break
            try:
                with h5py.File(path, "r") as handle:
                    group = handle[DEVICES[device_key]["group"]]
                    frequency = decode_numeric(group["Frequencies"])
                    population = decode_numeric(group["Population"])
            except (KeyError, OSError, ValueError) as exc:
                failed = f"segment {index + 1}: {exc}"
                break
            length = min(frequency.size, population.size)
            order = np.argsort(frequency[:length])
            frequency = frequency[:length][order]
            population = population[:length][order]
            finite = np.isfinite(frequency) & np.isfinite(population)
            frequency = frequency[finite]
            population = population[finite]
            if index + 1 < len(segments):
                next_start = float(segments[index + 1]["requested_start_MHz"])
                keep = frequency < next_start
                frequency = frequency[keep]
                population = population[keep]
            frequency_parts.append(frequency)
            population_parts.append(population)
            segment_provenance.update({
                "available": True,
                "retained_points": int(frequency.size),
                "retained_start_MHz": float(frequency[0]),
                "retained_stop_MHz": float(frequency[-1]),
            })
        if failed:
            row_provenance.update({"available": False, "reason": failed})
            continue
        frequency = np.concatenate(frequency_parts)
        population = np.concatenate(population_parts)
        if np.any(np.diff(frequency) <= 0):
            row_provenance.update({
                "available": False,
                "reason": "stitched segment frequency axis is not strictly increasing",
            })
            continue
        if reference_frequency is None:
            reference_frequency = frequency
        elif (frequency.size != reference_frequency.size
              or not np.allclose(frequency, reference_frequency, rtol=0, atol=1e-6)):
            row_provenance.update({
                "available": False,
                "reason": "stitched frequency axis differs from the first current",
            })
            continue
        result[float(current)] = (frequency, population, "complete")
        metrics = local_metrics(frequency, population, point.get("qfreq_MHz"))
        row_provenance.update({
            "available": True,
            "points": int(frequency.size),
            "actual_frequency_start_MHz": float(frequency[0]),
            "actual_frequency_stop_MHz": float(frequency[-1]),
            "actual_frequency_median_step_MHz": float(np.median(np.diff(frequency))),
            "population_local_metrics": metrics,
        })
    if reference_frequency is not None:
        provenance.update({
            "actual_frequency_start_MHz": float(reference_frequency[0]),
            "actual_frequency_stop_MHz": float(reference_frequency[-1]),
            "actual_frequency_median_step_MHz": float(np.median(np.diff(reference_frequency))),
            "actual_points": int(reference_frequency.size),
        })
    provenance["available_current_count"] = len(result)
    provenance["all_current_rows_available"] = len(result) == CURRENT_GRID_MA.size
    provenance["frequency_arrays_identical"] = bool(
        reference_frequency is not None
        and all(row.get("available") is True for row in provenance["currents"].values())
    )
    return result, provenance


def plot_wide_qspec_population_heatmap(device_key, data, manifest):
    """Plot the lower-frequency map with honest, peak-preserving rasterization."""
    fig, ax = plt.subplots(figsize=(11.5, 9))
    rows = [data.get(float(current)) for current in CURRENT_GRID_MA]
    available = [row for row in rows if row]
    if available:
        frequency = available[0][0]
        image = np.full((frequency.size, CURRENT_GRID_MA.size), np.nan)
        for column, row in enumerate(rows):
            if row:
                image[:, column] = row[1]

        # The physical lines are only about 0.1-1 MHz wide inside a multi-GHz
        # axis.  A slide raster would otherwise average them away.  Smooth five
        # acquired samples and average ordinary display bins.  Only the bins
        # within +/-1 MHz of the independently validated qspec frequency retain
        # their local maximum, so the measured line survives rasterization
        # without row normalization or alteration of the canonical HDF5 data.
        smoothed = np.full_like(image, np.nan)
        kernel = np.ones(5) / 5
        for column in range(image.shape[1]):
            values = image[:, column]
            finite = np.isfinite(values)
            if not finite.any():
                continue
            filled = np.interp(np.arange(values.size), np.flatnonzero(finite), values[finite])
            smoothed[:, column] = np.convolve(
                np.pad(filled, (2, 2), mode="edge"), kernel, mode="valid"
            )
        display_rows = min(6000, frequency.size)
        edges = np.linspace(0, frequency.size, display_rows + 1, dtype=int)
        display_frequency = np.empty(display_rows)
        display_image = np.full((display_rows, image.shape[1]), np.nan)
        for index, (start, stop) in enumerate(zip(edges[:-1], edges[1:])):
            display_frequency[index] = float(np.mean(frequency[start:stop]))
            block = smoothed[start:stop]
            if np.isfinite(block).any():
                display_image[index] = np.nanmean(block, axis=0)
        wide_points = manifest["devices"][device_key].get(
            "wide_uniform_qspec_map", {}
        ).get("points", {})
        centers = []
        center_currents = []
        for column, current in enumerate(CURRENT_GRID_MA):
            center = wide_points.get(current_key(current), {}).get("qfreq_MHz")
            if center is None:
                continue
            center = float(center)
            center_currents.append(float(current))
            centers.append(center)
            nearby = np.abs(display_frequency - center) <= 1.0
            for index in np.flatnonzero(nearby):
                start, stop = edges[index], edges[index + 1]
                display_image[index, column] = np.nanmax(smoothed[start:stop, column])
        cmap = plt.get_cmap("coolwarm").copy()
        norm = TwoSlopeNorm(vmin=-0.15, vcenter=0.0, vmax=0.35)
        mesh = ax.pcolormesh(
            CURRENT_GRID_MA, display_frequency,
            np.ma.masked_invalid(display_image), shading="nearest",
            cmap=cmap, norm=norm, rasterized=True,
        )
        fig.colorbar(
            mesh, ax=ax,
            label="Individually SSF-calibrated qubit population Pe (g=0, e=1)",
            extend="both",
        )
        if centers:
            ax.plot(center_currents, centers, color="white", linewidth=2.4, alpha=0.9)
            ax.plot(
                center_currents, centers, "o--", color="black", markersize=3.5,
                linewidth=0.9, label="Validated f01 used for each row",
            )
            ax.legend(loc="lower right")
        bin_width = float(np.median(np.diff(display_frequency)))
        ax.text(
            0.0, -0.115,
            f"All columns use the complete acquired {frequency[0]:.3f}-{frequency[-1]:.3f} MHz axis; "
            f"median acquired spacing {np.median(np.diff(frequency)):.6f} MHz.\n"
            "Each current/band uses its own built-in SSF calibration; no row normalization.\n"
            f"The raster averages {bin_width:.3f} MHz bins and preserves the measured local maximum "
            "within +/-1 MHz of f01; raw samples remain in canonical HDF5.",
            transform=ax.transAxes, fontsize=8.0,
        )
    ax.set_xticks(CURRENT_GRID_MA)
    ax.set_xlabel("Yokogawa current (mA)")
    ax.set_ylabel("Swept qubit frequency (MHz)")
    ax.set_title(
        f"{DEVICES[device_key]['label']} - lower-frequency calibrated-population qspec"
    )
    fig.subplots_adjust(left=0.10, right=0.86, bottom=0.20, top=0.93)
    path = OUTPUT / f"{device_key}_wide_lower_qspec_population_2d.png"
    fig.savefig(path, dpi=360)
    plt.close(fig)
    return path


def plot_wide_qspec_local_gallery(device_key, data, manifest):
    """Show the measured population line near the current-specific qspec seed."""
    fig, axes = plt.subplots(3, 3, figsize=(13, 10), constrained_layout=True)
    wide_points = manifest["devices"][device_key].get(
        "wide_uniform_qspec_map", {}
    ).get("points", {})
    for ax, current in zip(axes.ravel(), CURRENT_GRID_MA):
        row = data.get(float(current))
        point = wide_points.get(current_key(current), {})
        seed = point.get("qfreq_MHz")
        if row is None or seed is None:
            ax.text(0.5, 0.5, "No completed row", ha="center", va="center")
            ax.set_title(f"{current:+g} mA")
            continue
        frequency, population, _ = row
        local = np.abs(frequency - float(seed)) <= 5.0
        local_frequency = frequency[local]
        local_population = population[local]
        smooth_points = 5
        smoothed = np.convolve(
            np.pad(local_population, (2, 2), mode="edge"),
            np.ones(smooth_points) / smooth_points,
            mode="valid",
        )
        peak_index = int(np.nanargmax(smoothed))
        peak_frequency = float(local_frequency[peak_index])
        fitted = fit_local_population_peak(frequency, population, seed)
        fitted_frequency = (
            peak_frequency if fitted is None else fitted["fit_center_MHz"]
        )
        ax.plot(
            local_frequency, local_population, color="#9ecae1",
            linewidth=0.45, alpha=0.55, label="Acquired population",
        )
        ax.plot(
            local_frequency, smoothed, color="tab:blue", linewidth=1.1,
            label="5-point mean",
        )
        ax.axvline(float(seed), color="0.35", linestyle="--", linewidth=0.9)
        ax.axvline(fitted_frequency, color="tab:red", linewidth=1.1)
        ax.set_title(
            f"{current:+g} mA | local fit {fitted_frequency:.3f} MHz", fontsize=9.5
        )
        ax.grid(alpha=0.2)
        ax.tick_params(labelsize=8)
    fig.suptitle(
        f"{DEVICES[device_key]['label']} - local verification of the wide qspec peaks",
        fontsize=15,
    )
    fig.supxlabel("Actual swept qubit frequency (MHz)", fontsize=11)
    fig.supylabel("Individually SSF-calibrated qubit population", fontsize=11)
    path = OUTPUT / f"{device_key}_wide_lower_qspec_population_local_curves.png"
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path


def plot_uniform_qspec_heatmap(device_key, mode, data, manifest):
    """Plot the true common qspec grid with one global, cross-current color scale."""
    fig, ax = plt.subplots(figsize=(11.5, 9))
    rows = [data.get(float(current)) for current in CURRENT_GRID_MA]
    available = [row for row in rows if row]
    if available:
        frequency = available[0][0]
        image = np.full((frequency.size, CURRENT_GRID_MA.size), np.nan)
        for column, row in enumerate(rows):
            if row:
                image[:, column] = row[1]
        if mode == "population":
            vmin, vmax = 0.0, 1.0
            color_label = "SSF-calibrated qubit population (g=0, e=1)"
            cmap = plt.get_cmap("viridis").copy()
            cmap.set_over("#fde725")
        else:
            finite = image[np.isfinite(image)]
            vmin, vmax = np.nanpercentile(finite, [0.5, 99.5])
            color_label = "Measured IQ signal magnitude |I+iQ| (arb.)"
            cmap = "viridis"
        mesh = ax.pcolormesh(
            CURRENT_GRID_MA, frequency, np.ma.masked_invalid(image),
            shading="nearest", cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True,
        )
        fig.colorbar(
            mesh, ax=ax, label=color_label,
            extend="both" if mode == "population" else "neither",
        )
        centers = []
        center_currents = []
        points = manifest["devices"][device_key].get("points", {})
        for current in CURRENT_GRID_MA:
            center = points.get(current_key(current), {}).get("qfreq_MHz")
            if center is not None:
                center_currents.append(current)
                centers.append(center)
        if centers:
            ax.plot(center_currents, centers, color="white", linewidth=2.3, alpha=0.9)
            ax.plot(center_currents, centers, "o--", color="black", markersize=3.5,
                    linewidth=0.9, label="Final local qspec center")
            ax.legend(loc="lower right")
    ax.set_xticks(CURRENT_GRID_MA)
    ax.set_xlabel("Yokogawa current (mA)")
    ax.set_ylabel("Swept qubit frequency (MHz)")
    title_mode = "calibrated population" if mode == "population" else "measured signal magnitude"
    ax.set_title(f"{DEVICES[device_key]['label']} - common-grid qspec {title_mode}")
    if available:
        actual_axis_note = (
            f"Every column uses the identical actual {frequency[0]:.6f}-"
            f"{frequency[-1]:.6f} MHz axis (median spacing "
            f"{np.median(np.diff(frequency)):.6f} MHz). "
        )
    else:
        actual_axis_note = "No common-grid row is available. "
    calibration_note = (
        "Population uses one SSF calibration for all frequencies at that bias and no row normalization."
    )
    if device_key == "q4" and mode == "population":
        calibration_note += " The -10 mA column is low-confidence (SSF 0.124; provisional Rabi)."
    ax.text(
        0.0, -0.095,
        actual_axis_note + "\n" + calibration_note,
        transform=ax.transAxes, fontsize=9,
    )
    fig.subplots_adjust(left=0.10, right=0.86, bottom=0.15, top=0.93)
    path = OUTPUT / f"{device_key}_uniform_qspec_{mode}_2d.png"
    fig.savefig(path, dpi=360)
    plt.close(fig)
    return path


def plot_uniform_qspec_gallery(device_key, mode, data, manifest):
    """Show every common-grid curve on the full axis, plus a same-data local inset."""
    fig, axes = plt.subplots(3, 3, figsize=(14, 11), constrained_layout=True)
    all_values = np.concatenate([row[1][np.isfinite(row[1])] for row in data.values()]) if data else np.array([])
    if mode == "population":
        x_min, x_max = -0.03, max(1.05, float(np.nanpercentile(all_values, 99.8))) if all_values.size else 1.05
        x_max = min(x_max, 1.5)
        x_label = "SSF-calibrated population (g=0, e=1)"
    else:
        x_min, x_max = (np.nanpercentile(all_values, [0.2, 99.8]) if all_values.size else (0.0, 1.0))
        x_label = "Measured IQ signal magnitude |I+iQ| (arb.)"
    points = manifest["devices"][device_key].get("points", {})
    for ax, current in zip(axes.ravel(), CURRENT_GRID_MA):
        row = data.get(float(current))
        if not row:
            ax.text(0.5, 0.5, "Unavailable", ha="center", va="center", transform=ax.transAxes)
            ax.set_title(f"{current:+g} mA")
            ax.set_ylim(5950, 7200)
            continue
        frequency, response, _ = row
        ax.plot(response, frequency, color="tab:blue", linewidth=0.35)
        ax.set_xlim(float(x_min), float(x_max))
        ax.set_ylim(float(frequency.min()), float(frequency.max()))
        center = points.get(current_key(current), {}).get("qfreq_MHz")
        if center is not None:
            ax.axhline(center, color="tab:orange", linestyle="--", linewidth=0.9)
            local = (frequency >= center - 2.0) & (frequency <= center + 2.0)
            inset = ax.inset_axes([0.49, 0.55, 0.48, 0.40])
            inset.plot(response[local], frequency[local], color="tab:blue", linewidth=0.7)
            inset.axhline(center, color="tab:orange", linestyle="--", linewidth=0.7)
            if np.any(local):
                local_values = response[local]
                lo, hi = np.nanpercentile(local_values, [0.5, 99.5])
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    inset.set_xlim(float(lo), float(hi))
            inset.set_ylim(center - 2.0, center + 2.0)
            inset.tick_params(labelsize=5, pad=1)
            inset.set_title("same data, +/-2 MHz", fontsize=5, pad=1)
        ax.set_title(f"{current:+g} mA | n={frequency.size}", fontsize=10)
        ax.grid(alpha=0.18)
        ax.tick_params(labelsize=7)
    fig.suptitle(
        f"{DEVICES[device_key]['label']} - complete common-grid qspec curves ({mode})",
        fontsize=15,
    )
    fig.supxlabel(x_label, fontsize=11)
    if data:
        reference_axis = next(iter(data.values()))[0]
        axis_label = (
            "Swept qubit frequency (MHz); full actual "
            f"{reference_axis[0]:.3f}-{reference_axis[-1]:.3f} MHz axis"
        )
    else:
        axis_label = "Swept qubit frequency (MHz)"
    fig.supylabel(axis_label, fontsize=11)
    path = OUTPUT / f"{device_key}_uniform_qspec_{mode}_full_curves.png"
    fig.savefig(path, dpi=300)
    plt.close(fig)
    return path


def plot_uniform_qspec_frequency_check(
        device_key, provenance, local_provenance, manifest):
    """Compare full-grid population extrema with recorded Rabi frequencies."""
    rabi_currents = []
    rabi_frequencies = []
    mismatched_currents = []
    mismatched_final_frequencies = []
    map_currents = []
    map_frequencies = []
    offsets = []
    snrs = []
    points = manifest["devices"][device_key].get("points", {})
    rows = provenance.get("currents", {})
    for current in CURRENT_GRID_MA:
        key = current_key(current)
        final_frequency = points.get(key, {}).get("qfreq_MHz")
        rabi = (local_provenance.get("currents", {}).get(key, {})
                .get("experiments", {}).get("rabi", {}))
        rabi_frequency = rabi.get("qfreq_used_MHz")
        metrics = rows.get(key, {}).get("population_local_metrics")
        if rabi_frequency is not None:
            rabi_currents.append(float(current))
            rabi_frequencies.append(float(rabi_frequency))
            if not rabi.get("qfreq_matches_final") and final_frequency is not None:
                mismatched_currents.append(float(current))
                mismatched_final_frequencies.append(float(final_frequency))
        if metrics:
            map_currents.append(float(current))
            map_frequencies.append(float(metrics["local_extremum_frequency_MHz"]))
            reference = rabi_frequency if rabi_frequency is not None else final_frequency
            offsets.append(
                float(metrics["local_extremum_frequency_MHz"]) - float(reference)
            )
            snrs.append(float(metrics["local_peak_snr"]))

    fig, (ax_frequency, ax_offset) = plt.subplots(
        2, 1, figsize=(11.5, 8.5), sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.0]}, constrained_layout=True,
    )
    if rabi_currents:
        ax_frequency.plot(
            rabi_currents, rabi_frequencies, "o-", color="black",
            label="Actual f01 recorded in selected Rabi run",
        )
    if mismatched_currents:
        ax_frequency.plot(
            mismatched_currents, mismatched_final_frequencies, "s", color="red",
            label="Final local f01 mismatch",
        )
    if map_currents:
        ax_frequency.plot(
            map_currents, map_frequencies, "x--", color="tab:orange",
            markersize=8, label="Common-grid population extremum",
        )
        bars = ax_offset.bar(map_currents, offsets, width=1.25, color="tab:blue", alpha=0.8)
        for bar, snr in zip(bars, snrs):
            ax_offset.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"SNR {snr:.1f}", ha="center",
                va="bottom" if bar.get_height() >= 0 else "top", fontsize=7,
            )
    ax_frequency.set_ylabel("Qubit frequency (MHz)")
    ax_frequency.set_title(
        f"{DEVICES[device_key]['label']} - full-grid frequency consistency"
    )
    ax_frequency.legend(loc="best")
    ax_offset.axhline(0.0, color="black", linewidth=0.8)
    ax_offset.set_ylabel("Map extremum - Rabi f01 (MHz)")
    ax_offset.set_xlabel("Yokogawa current (mA)")
    ax_offset.set_xticks(CURRENT_GRID_MA)
    for axis in (ax_frequency, ax_offset):
        axis.grid(alpha=0.25)
    path = OUTPUT / f"{device_key}_uniform_qspec_frequency_check.png"
    fig.savefig(path, dpi=240)
    plt.close(fig)
    return path


def qspec_fwhm_from_log(point):
    if point.get("status") == "complete" and point.get("qspec_fwhm_MHz") is not None:
        return abs(float(point["qspec_fwhm_MHz"]))
    logs = point.get("logs", {})
    for name in ("post_readout_calibration", "calibration"):
        path = logs.get(name)
        if not path or not Path(path).exists():
            continue
        text = Path(path).read_text(errors="replace")
        matches = re.findall(r"g-e Qubit \d+ FWHM: ([0-9.+-eE]+) MHz", text)
        if matches:
            return abs(float(matches[-1]))
    return None


def plot_found_frequencies(device_key, manifest):
    points = manifest["devices"][device_key].get("points", {})
    q_rows, res_rows, width_rows = [], [], []
    for current in CURRENT_GRID_MA:
        point = points.get(current_key(current), {})
        if point.get("status") != "complete":
            continue
        if point.get("qfreq_MHz") is not None:
            q_rows.append((current, point["qfreq_MHz"]))
        if point.get("res_base_MHz") is not None:
            res_rows.append((current, point["res_base_MHz"]))
        width = qspec_fwhm_from_log(point)
        if width is not None:
            width_rows.append((current, width))

    fig, (ax_q, ax_r) = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    if q_rows:
        ax_q.plot([row[0] for row in q_rows], [row[1] for row in q_rows], "o-")
    if res_rows:
        ax_r.plot([row[0] for row in res_rows], [row[1] for row in res_rows], "o-", color="tab:orange")
    ax_q.set(xlabel="Yokogawa current (mA)", ylabel="Found qubit frequency (MHz)", title="Qubit transition center")
    ax_r.set(xlabel="Yokogawa current (mA)", ylabel="Found resonator frequency (MHz)", title="Resonator center")
    for axis in (ax_q, ax_r):
        axis.set_xticks(CURRENT_GRID_MA)
        axis.grid(alpha=0.25)
    fig.suptitle(DEVICES[device_key]["label"] + " — accepted fitted centers")
    center_path = OUTPUT / f"{device_key}_found_frequencies_vs_current.png"
    fig.savefig(center_path, dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6), constrained_layout=True)
    if width_rows:
        currents = np.asarray([row[0] for row in width_rows], dtype=float)
        widths = np.asarray([row[1] for row in width_rows], dtype=float)
        ax.plot(currents, widths, "o-", color="tab:green", label="Final measured FWHM")
        outside = (widths < 0.1) | (widths > 0.2)
        if np.any(outside):
            ax.scatter(currents[outside], widths[outside], color="tab:red", zorder=4,
                       label="Outside nominal linewidth target")
        if device_key == "q4" and np.any(currents == 5.0):
            index = int(np.flatnonzero(currents == 5.0)[0])
            ax.annotate("population-optimal,\npower-broadened",
                        (currents[index], widths[index]), xytext=(-92, -10),
                        textcoords="offset points", fontsize=8,
                        arrowprops={"arrowstyle": "->", "color": "tab:red"})
    ax.axhspan(0.1, 0.2, color="tab:green", alpha=0.15, label="Target 0.1–0.2 MHz")
    ax.set(xlabel="Yokogawa current (mA)", ylabel="Qspec FWHM (MHz)",
           title=DEVICES[device_key]["label"] + " — final measured qspec linewidth")
    ax.set_xticks(CURRENT_GRID_MA)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fwhm_path = OUTPUT / f"{device_key}_qspec_fwhm_vs_current.png"
    fig.savefig(fwhm_path, dpi=180)
    plt.close(fig)
    return center_path, fwhm_path


def plot_interpolated_map(device_key, experiment, data):
    fig, ax = plt.subplots(figsize=(12, 7))
    if data:
        axis_min = min(np.nanmin(values[0]) for values in data.values())
        axis_max = max(np.nanmax(values[0]) for values in data.values())
        grid = np.linspace(axis_min, axis_max, 500)
        image = np.full((grid.size, CURRENT_GRID_MA.size), np.nan)
        for column, current in enumerate(CURRENT_GRID_MA):
            values = data.get(float(current))
            if not values:
                continue
            axis, response, _ = values
            unique_axis, indices = np.unique(axis, return_index=True)
            inside = (grid >= unique_axis.min()) & (grid <= unique_axis.max())
            image[inside, column] = np.interp(grid[inside], unique_axis, response[indices])
        mesh = ax.pcolormesh(CURRENT_GRID_MA, grid, np.ma.masked_invalid(image), shading="nearest",
                             cmap="viridis", vmin=0, vmax=1)
        color_label = (
            "SSF-calibrated qubit population (g=0, e=1)"
            if experiment == "rabi"
            else "Row-normalized measured response"
        )
        fig.colorbar(mesh, ax=ax, label=color_label)
    ax.set_xticks(CURRENT_GRID_MA)
    ax.set_xlabel("Yokogawa current (mA)")
    ax.set_ylabel(EXPERIMENTS[experiment]["axis"])
    ax.set_title(f"{DEVICES[device_key]['label']} — {experiment.upper()} 2D sweep")
    note = (
        "Blank columns have no numeric acquisition; values are interpolated only "
        "within each measured trace."
    )
    if experiment == "rabi":
        note += (
            "\nEvery acquired DAC-gain axis is sorted ascending with its population "
            "samples; no column is reversed."
        )
    ax.text(0.01, -0.13, note, transform=ax.transAxes, fontsize=9)
    fig.subplots_adjust(left=0.10, right=0.86, bottom=0.18, top=0.92)
    path = OUTPUT / f"{device_key}_{experiment}_2d.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_sigma_pi(device_key, manifest):
    points = manifest["devices"][device_key].get("points", {})
    currents, sigmas, pi_amps = [], [], []
    deferred_currents, deferred_sigmas, deferred_pi_amps = [], [], []
    for current in CURRENT_GRID_MA:
        point = points.get(current_key(current), {})
        if point.get("status") != "complete":
            continue
        sigma = point.get("sigma_us")
        pi_amp = point.get("pi_amp")
        if sigma is None or pi_amp is None:
            continue
        if point.get("sigma_recalibration_required"):
            deferred_currents.append(current)
            deferred_sigmas.append(sigma)
            deferred_pi_amps.append(pi_amp)
        else:
            currents.append(current)
            sigmas.append(sigma)
            pi_amps.append(pi_amp)
    fig, (ax_map, ax_pi) = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    if currents:
        scatter = ax_map.scatter(currents, sigmas, c=pi_amps, cmap="coolwarm", vmin=0.6, vmax=0.7,
                                 s=160, edgecolor="black", label="Accepted full point")
        fig.colorbar(scatter, ax=ax_map, label="Measured pi amplitude (DAC gain)")
        ax_pi.plot(currents, pi_amps, "o-", label="Accepted pi amplitude")
    if deferred_currents:
        deferred = ax_map.scatter(
            deferred_currents, deferred_sigmas, c=deferred_pi_amps,
            cmap="coolwarm", vmin=0.6, vmax=0.7, s=190, marker="D",
            edgecolor="black", linewidth=1.5,
            label="Completed row; sigma retuning deferred",
        )
        if not currents:
            fig.colorbar(deferred, ax=ax_map, label="Measured pi amplitude (DAC gain)")
        ax_pi.scatter(
            deferred_currents, deferred_pi_amps, marker="D", s=100,
            color="tab:orange", edgecolor="black",
            label="Completed row; sigma retuning deferred",
        )
    ax_map.set(xlabel="Yokogawa current (mA)", ylabel="Gaussian sigma (us)",
               title="Accepted current-indexed sigma map")
    ax_pi.axhspan(0.6, 0.7, color="tab:green", alpha=0.18, label="Required window")
    ax_pi.set(xlabel="Yokogawa current (mA)", ylabel="Pi amplitude (DAC gain)",
              title="Pi-amplitude acceptance")
    ax_pi.legend(loc="best")
    ax_map.legend(loc="best")
    for axis in (ax_map, ax_pi):
        axis.set_xticks(CURRENT_GRID_MA)
        axis.grid(alpha=0.25)
    fig.suptitle(DEVICES[device_key]["label"])
    path = OUTPUT / f"{device_key}_sigma_pi_2d.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(MANIFEST_PATH.read_text())
    report = {"manifest": str(MANIFEST_PATH), "output": str(OUTPUT), "devices": {}}
    written = []
    for device_key in DEVICES:
        collected, provenance = collect_device(device_key, manifest)
        report["devices"][device_key] = provenance
        written.append(plot_frequency_columns(device_key, "qspec", collected["qspec"]))
        written.append(plot_frequency_columns(device_key, "qspec_population", collected["qspec_population"]))
        written.append(plot_qspec_curve_gallery(device_key, "qspec", collected["qspec"], manifest))
        written.append(plot_qspec_curve_gallery(
            device_key, "qspec_population", collected["qspec_population"], manifest
        ))
        written.append(plot_frequency_columns(device_key, "res_spec", collected["res_spec"]))
        written.extend(plot_found_frequencies(device_key, manifest))
        for experiment in ("rabi", "t1", "t2r", "t2e"):
            written.append(plot_interpolated_map(device_key, experiment, collected[experiment]))
        written.append(plot_sigma_pi(device_key, manifest))
        uniform_data, uniform_provenance = collect_uniform_qspec(device_key, manifest)
        report["devices"][device_key]["uniform_qspec_map"] = uniform_provenance
        if uniform_data["signal"] or uniform_data["population"]:
            for mode in ("signal", "population"):
                written.append(plot_uniform_qspec_heatmap(
                    device_key, mode, uniform_data[mode], manifest
                ))
                written.append(plot_uniform_qspec_gallery(
                    device_key, mode, uniform_data[mode], manifest
                ))
            if uniform_data["population"]:
                written.append(plot_uniform_qspec_frequency_check(
                    device_key, uniform_provenance, provenance, manifest
                ))
        wide_data, wide_provenance = collect_wide_uniform_qspec(device_key, manifest)
        report["devices"][device_key]["wide_uniform_qspec_map"] = wide_provenance
        if wide_data:
            written.append(plot_wide_qspec_population_heatmap(
                device_key, wide_data, manifest
            ))
            written.append(plot_wide_qspec_local_gallery(
                device_key, wide_data, manifest
            ))
    report_path = OUTPUT / "flux_2d_provenance.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
