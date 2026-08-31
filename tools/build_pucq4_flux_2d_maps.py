"""Build provenance-backed PUCQ4 two-Yokogawa aggregate maps.

This script is analysis-only. It reads the flux campaign manifest and numeric
round-robin HDF5 datasets, writes PNG maps plus a JSON provenance report, and
never imports experiment modules or contacts hardware.
"""

import ast
import json
import re
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np


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
    "qspec": {"folder": "qspec_ge", "x": "Frequencies", "channels": ("I", "Q"), "axis": "Qubit frequency (MHz)"},
    "res_spec": {"folder": "res_ge", "x": "freq_pts", "channels": ("Amps",), "axis": "Resonator frequency (MHz)"},
    "rabi": {"folder": "rabi_ge", "x": "Gains", "channels": ("I", "Q"), "axis": "DAC gain"},
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


def decode_numeric(dataset):
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
    try:
        parsed = ast.literal_eval(cleaned)
        array = np.asarray(parsed, dtype=float).ravel()
    except (SyntaxError, ValueError, TypeError):
        fields = [field for field in re.split(r"[,\s\[\]]+", cleaned) if field]
        array = np.asarray([float(field) for field in fields], dtype=float)
    if array.size < 2 or not np.isfinite(array).any():
        raise ValueError("dataset has fewer than two finite values")
    return array


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
        trace = centered @ vectors[:, -1]
    lo, hi = np.nanpercentile(trace, [2, 98])
    scale = hi - lo
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("flat trace")
    return np.clip((trace - lo) / scale, 0.0, 1.0)


def preferred_runs(device_key, current_ma, experiment, point):
    logs = point.get("logs", {}) if point else {}
    runs = []
    if experiment in {"qspec", "res_spec", "rabi"}:
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
        "res_spec": ("post_readout_calibration", "coherence", "calibration", "raw_iq_rabi", "locator"),
        "rabi": ("post_readout_calibration", "rabi_retry", "raw_iq_rabi_retry", "raw_iq_rabi", "calibration"),
        "t1": ("coherence",), "t2r": ("coherence",), "t2e": ("coherence",),
    }[experiment]
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
                length = min([x.size] + [channel.size for channel in channels])
                if length < 2:
                    raise ValueError("trace too short")
                x = x[:length]
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
        point = points.get(key, {})
        status = point.get("status", "missing")
        provenance["currents"][key] = {"status": status, "experiments": {}}
        for experiment in EXPERIMENTS:
            try:
                x, response, path = load_trace(device_key, current, experiment, point)
            except (FileNotFoundError, ValueError) as exc:
                provenance["currents"][key]["experiments"][experiment] = {"available": False, "reason": str(exc)}
                continue
            collected[experiment][float(current)] = (x, response, status)
            provenance["currents"][key]["experiments"][experiment] = {
                "available": True, "h5": str(path), "points": int(x.size),
                "axis_min": float(np.nanmin(x)), "axis_max": float(np.nanmax(x)),
            }
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
        mesh = ax.pcolormesh(CURRENT_GRID_MA, grid, np.ma.masked_invalid(image), shading="nearest",
                             cmap="viridis", vmin=0, vmax=1)
        fig.colorbar(mesh, ax=ax, label="Row-normalized measured signal magnitude")
    ax.set_xticks(CURRENT_GRID_MA)
    ax.set_xlabel("Yokogawa current (mA)")
    ax.set_ylabel(EXPERIMENTS[experiment]["axis"])
    ax.set_title(f"{DEVICES[device_key]['label']} — {experiment.replace('_', ' ')} 2D sweep")
    ax.text(0.01, -0.13, "Each colored column is one acquired frequency sweep; blank regions were not measured. No interpolation across bias current.",
            transform=ax.transAxes, fontsize=9)
    path = OUTPUT / f"{device_key}_{experiment}_2d.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def qspec_fwhm_from_log(point):
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
        ax.plot([row[0] for row in width_rows], [row[1] for row in width_rows], "o-", color="tab:green")
    ax.axhspan(0.1, 0.2, color="tab:green", alpha=0.15, label="Target 0.1–0.2 MHz")
    ax.set(xlabel="Yokogawa current (mA)", ylabel="Qspec FWHM (MHz)",
           title=DEVICES[device_key]["label"] + " — accepted qspec linewidth")
    ax.set_xticks(CURRENT_GRID_MA)
    ax.grid(alpha=0.25)
    ax.legend(loc="best")
    fwhm_path = OUTPUT / f"{device_key}_qspec_fwhm_vs_current.png"
    fig.savefig(fwhm_path, dpi=180)
    plt.close(fig)
    return center_path, fwhm_path


def plot_interpolated_map(device_key, experiment, data):
    fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
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
        fig.colorbar(mesh, ax=ax, label="Row-normalized measured response")
    ax.set_xticks(CURRENT_GRID_MA)
    ax.set_xlabel("Yokogawa current (mA)")
    ax.set_ylabel(EXPERIMENTS[experiment]["axis"])
    ax.set_title(f"{DEVICES[device_key]['label']} — {experiment.upper()} 2D sweep")
    ax.text(0.01, -0.13, "Blank columns have no numeric acquisition; values are interpolated only within each measured trace.",
            transform=ax.transAxes, fontsize=9)
    path = OUTPUT / f"{device_key}_{experiment}_2d.png"
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return path


def plot_sigma_pi(device_key, manifest):
    points = manifest["devices"][device_key].get("points", {})
    currents, sigmas, pi_amps = [], [], []
    provisional_currents, provisional_sigmas, provisional_pi_amps = [], [], []
    for current in CURRENT_GRID_MA:
        point = points.get(current_key(current), {})
        if point.get("status") != "complete":
            continue
        sigma = point.get("sigma_us")
        pi_amp = point.get("pi_amp")
        if sigma is None or pi_amp is None:
            continue
        currents.append(current)
        sigmas.append(sigma)
        pi_amps.append(pi_amp)
    for current in CURRENT_GRID_MA:
        point = points.get(current_key(current), {})
        pulse = point.get("pulse_calibration", {})
        if pulse.get("status") != "revalidation_required":
            continue
        provisional_currents.append(current)
        provisional_sigmas.append(pulse["sigma_us"])
        provisional_pi_amps.append(pulse["raw_iq_pi_amp"])
    fig, (ax_map, ax_pi) = plt.subplots(1, 2, figsize=(13, 6), constrained_layout=True)
    if currents:
        scatter = ax_map.scatter(currents, sigmas, c=pi_amps, cmap="coolwarm", vmin=0.6, vmax=0.7,
                                 s=160, edgecolor="black", label="Accepted full point")
        fig.colorbar(scatter, ax=ax_map, label="Measured pi amplitude (DAC gain)")
        ax_pi.plot(currents, pi_amps, "o-", label="Accepted pi amplitude")
    if provisional_currents:
        provisional = ax_map.scatter(
            provisional_currents, provisional_sigmas, c=provisional_pi_amps,
            cmap="coolwarm", vmin=0.6, vmax=0.7, s=190, marker="D",
            edgecolor="black", linewidth=1.5, label="Raw-I/Q in window; revalidation required",
        )
        if not currents:
            fig.colorbar(provisional, ax=ax_map, label="Measured pi amplitude (DAC gain)")
        ax_pi.scatter(
            provisional_currents, provisional_pi_amps, marker="D", s=100,
            color="tab:orange", edgecolor="black", label="Revalidation required",
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
        written.append(plot_frequency_columns(device_key, "res_spec", collected["res_spec"]))
        written.extend(plot_found_frequencies(device_key, manifest))
        for experiment in ("rabi", "t1", "t2r", "t2e"):
            written.append(plot_interpolated_map(device_key, experiment, collected[experiment]))
        written.append(plot_sigma_pi(device_key, manifest))
    report_path = OUTPUT / "flux_2d_provenance.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(report_path)
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
