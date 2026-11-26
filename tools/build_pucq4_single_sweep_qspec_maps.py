"""Build PUCQ4 qspec heatmaps from continuous one-sweep-per-current HDF5.

This is analysis-only.  It never imports an experiment controller or touches
hardware.  The canonical HDF5 arrays remain unchanged.  The heatmaps use every
point from the same identical-frequency-grid traces as the local curve panels.
For a readable full-band raster, they subtract only a slowly varying per-row
baseline, then use peak-preserving display bins and a disclosed three-pixel
ridge.  No row amplitude scaling or peak normalization is applied.
"""

import argparse
import json
import re
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import PowerNorm


ROOT = Path(
    r"M:\_Data\20250822 - Olivia\pucq4_run_started_Oct_7\PUCQ4"
    r"\pucq4_first_light\yoko_flux_campaign"
)
MANIFEST = ROOT / "single_sweep_qspec_manifest.json"
OUTPUT = ROOT / "single_sweep_qspec_2d"
PROVENANCE = OUTPUT / "single_sweep_qspec_provenance.json"
CURRENT_GRID_MA = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]
DEVICES = {
    "q4": {"group": "Q5", "label": "Yoko 3 / physical Q4 / M5"},
    "q6": {"group": "Q6", "label": "Yoko 4 / physical Q6 / M6"},
}
DISPLAY_FREQUENCY_BINS = 1500
BASELINE_BLOCK_WIDTH_MHZ = 20.0
PEAK_PRESERVE_HALF_WIDTH_MHZ = 1.0
PEAK_RIDGE_DISPLAY_ROWS = 3
POPULATION_CONTRAST_LIMITS = (0.0, 0.35)
CONTRAST_GAMMA = 0.55


def decode_h5_numeric(dataset):
    value = dataset[0] if dataset.shape else dataset[()]
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).ravel()
    if isinstance(value, (bytes, np.bytes_)):
        text = value.decode("utf-8")
    else:
        text = str(value)
    fields = [
        field for field in re.split(r"[,\s\[\]]+", text.strip()) if field
    ]
    return np.asarray([float(field) for field in fields], dtype=float)


def load_trace(path, group_name):
    with h5py.File(path, "r") as handle:
        group = handle[group_name]
        frequency = decode_h5_numeric(group["Frequencies"])
        population = decode_h5_numeric(group["Population"])
        i_data = decode_h5_numeric(group["I"])
        q_data = decode_h5_numeric(group["Q"])
    length = min(
        frequency.size, population.size, i_data.size, q_data.size
    )
    frequency = np.asarray(frequency[:length], dtype=float)
    population = np.asarray(population[:length], dtype=float)
    i_data = np.asarray(i_data[:length], dtype=float)
    q_data = np.asarray(q_data[:length], dtype=float)
    finite = (
        np.isfinite(frequency) & np.isfinite(population)
        & np.isfinite(i_data) & np.isfinite(q_data)
    )
    order = np.argsort(frequency[finite])
    return (
        frequency[finite][order], population[finite][order],
        i_data[finite][order], q_data[finite][order],
    )


def smooth_trace(values, points=5):
    points = max(1, int(points))
    if points % 2 == 0:
        points += 1
    if points == 1:
        return values.copy()
    pad = points // 2
    return np.convolve(
        np.pad(values, (pad, pad), mode="edge"),
        np.ones(points, dtype=float) / points,
        mode="valid",
    )


def blockwise_median_baseline(frequency, values, width_mhz):
    """Interpolate robust block medians into a slow baseline."""
    spacing = float(np.median(np.diff(frequency)))
    block_samples = max(3, int(round(float(width_mhz) / spacing)))
    starts = np.arange(0, values.size, block_samples, dtype=int)
    centers = []
    medians = []
    for start in starts:
        stop = min(values.size, start + block_samples)
        centers.append(float(np.mean(frequency[start:stop])))
        medians.append(float(np.median(values[start:stop])))
    return np.interp(
        frequency, np.asarray(centers), np.asarray(medians),
        left=medians[0], right=medians[-1],
    )


def baseline_subtracted_trace(frequency, values, smoothing_points):
    smoothed = smooth_trace(values, points=smoothing_points)
    baseline = blockwise_median_baseline(
        frequency, smoothed, BASELINE_BLOCK_WIDTH_MHZ
    )
    return smoothed - baseline


def load_device(manifest, device_key):
    device = DEVICES[device_key]
    map_data = manifest.get("devices", {}).get(device_key, {}).get("map", {})
    points = map_data.get("points", {})
    missing = [
        f"{current:+.1f}" for current in CURRENT_GRID_MA
        if points.get(f"{current:+.1f}", {}).get("status") != "complete"
    ]
    if missing:
        raise RuntimeError(
            f"{device_key} single-sweep map is incomplete at {missing}"
        )
    rows = []
    reference_frequency = None
    for current in CURRENT_GRID_MA:
        record = points[f"{current:+.1f}"]
        path = Path(record["h5"])
        if not path.is_file():
            raise FileNotFoundError(path)
        frequency, population, i_data, q_data = load_trace(
            path, device["group"]
        )
        if reference_frequency is None:
            reference_frequency = frequency
        elif (frequency.size != reference_frequency.size
              or not np.allclose(
                  frequency, reference_frequency, rtol=0.0, atol=1e-6
              )):
            raise RuntimeError(
                f"{device_key} {current:+.1f} mA does not share one actual axis"
            )
        centroids = record["ssf_centroids"]
        ssf_separation = float(np.hypot(
            centroids["e_I"] - centroids["g_I"],
            centroids["e_Q"] - centroids["g_Q"],
        ))
        rows.append({
            "current_mA": current,
            "record": record,
            "frequency_MHz": frequency,
            "population": population,
            "complex_signal": i_data + 1j * q_data,
            "magnitude": np.abs(i_data + 1j * q_data),
            "ssf_axis_signal": population * ssf_separation,
        })
    return map_data, rows


def peak_preserving_bins(
        frequency, traces, target_bins, peak_frequencies,
        preserve_half_width_mhz=PEAK_PRESERVE_HALF_WIDTH_MHZ):
    samples = frequency.size
    bin_count = min(int(target_bins), samples)
    edges = np.linspace(0, samples, bin_count + 1, dtype=int)
    centers = np.empty(bin_count, dtype=float)
    display = np.empty((bin_count, len(traces)), dtype=float)
    for index in range(bin_count):
        start = edges[index]
        stop = max(start + 1, edges[index + 1])
        centers[index] = float(np.mean(frequency[start:stop]))
        for column, trace in enumerate(traces):
            intersects_peak = (
                frequency[start] <= (
                    peak_frequencies[column] + preserve_half_width_mhz
                )
                and frequency[stop - 1] >= (
                    peak_frequencies[column] - preserve_half_width_mhz
                )
            )
            reducer = np.max if intersects_peak else np.mean
            display[index, column] = float(reducer(trace[start:stop]))
    return centers, display


def thicken_peak_ridge(display_frequency, display, peak_frequencies):
    """Keep a narrow measured line visible after slide-size rasterization."""
    half_rows = PEAK_RIDGE_DISPLAY_ROWS // 2
    result = display.copy()
    for column, peak_frequency in enumerate(peak_frequencies):
        center = int(np.argmin(np.abs(display_frequency - peak_frequency)))
        source_start = max(0, center - 1)
        source_stop = min(display.shape[0], center + 2)
        peak_value = float(np.max(display[source_start:source_stop, column]))
        start = max(0, center - half_rows)
        stop = min(display.shape[0], center + half_rows + 1)
        result[start:stop, column] = np.maximum(
            result[start:stop, column], peak_value
        )
    return result


def coordinate_edges(values):
    values = np.asarray(values, dtype=float)
    midpoints = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def plot_population_heatmap(device_key, map_data, rows):
    frequency = rows[0]["frequency_MHz"]
    contrast = [
        baseline_subtracted_trace(frequency, row["population"], 5)
        for row in rows
    ]
    measured = [
        row["record"]["peak_metrics"]["frequency_MHz"] for row in rows
    ]
    display_frequency, display = peak_preserving_bins(
        frequency, contrast, DISPLAY_FREQUENCY_BINS, measured
    )
    display = thicken_peak_ridge(display_frequency, display, measured)
    current_edges = coordinate_edges(CURRENT_GRID_MA)
    frequency_edges = coordinate_edges(display_frequency)
    fig, ax = plt.subplots(figsize=(8.2, 6.4), constrained_layout=True)
    mesh = ax.pcolormesh(
        current_edges, frequency_edges, display,
        cmap="magma", shading="flat",
        norm=PowerNorm(
            gamma=CONTRAST_GAMMA,
            vmin=POPULATION_CONTRAST_LIMITS[0],
            vmax=POPULATION_CONTRAST_LIMITS[1],
        ),
        rasterized=True,
    )
    seeds = [row["record"]["qfreq_seed_MHz"] for row in rows]
    ax.plot(
        CURRENT_GRID_MA, seeds, color="white", linewidth=0.8,
        linestyle="--", alpha=0.75, label="validated qspec seed",
    )
    ax.scatter(
        CURRENT_GRID_MA, measured, s=18, facecolors="none",
        edgecolors="white", linewidths=0.8, label="peak in this sweep",
    )
    ax.set(
        xlabel="Yokogawa current (mA)",
        ylabel="Actual swept qubit frequency (MHz)",
        title=(
            f"{DEVICES[device_key]['label']} — continuous wide qspec contrast\n"
            "same full frequency grid and qspec parameters as the local panels; "
            f"{map_data['reps']:,} reps/frequency"
        ),
    )
    ax.set_xticks(CURRENT_GRID_MA)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.85)
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label(
        f"SSF-calibrated $\Delta P_e$ from {BASELINE_BLOCK_WIDTH_MHZ:.0f}-MHz "
        "local baseline"
    )
    output = (
        OUTPUT / f"{device_key}_single_sweep_qspec_population_contrast_2d.png"
    )
    fig.savefig(output, dpi=400)
    plt.close(fig)
    return output, float(np.median(np.diff(display_frequency)))


def plot_magnitude_heatmap(device_key, map_data, rows):
    frequency = rows[0]["frequency_MHz"]
    contrast = []
    for row in rows:
        signal = baseline_subtracted_trace(
            frequency, row["ssf_axis_signal"], 11
        )
        contrast.append(np.abs(signal))
    measured = [
        row["record"]["peak_metrics"]["frequency_MHz"] for row in rows
    ]
    display_frequency, display = peak_preserving_bins(
        frequency, contrast, DISPLAY_FREQUENCY_BINS,
        measured,
    )
    display = thicken_peak_ridge(display_frequency, display, measured)
    local_peak_values = []
    for trace, peak_frequency in zip(contrast, measured):
        local = np.abs(frequency - peak_frequency) <= 1.0
        local_peak_values.append(float(np.max(trace[local])))
    vmax = max(
        np.finfo(float).eps,
        float(np.percentile(local_peak_values, 75.0)),
    )
    fig, ax = plt.subplots(figsize=(8.2, 6.4), constrained_layout=True)
    mesh = ax.pcolormesh(
        coordinate_edges(CURRENT_GRID_MA),
        coordinate_edges(display_frequency), display,
        cmap="magma", shading="flat",
        norm=PowerNorm(gamma=CONTRAST_GAMMA, vmin=0.0, vmax=vmax),
        rasterized=True,
    )
    ax.set(
        xlabel="Yokogawa current (mA)",
        ylabel="Actual swept qubit frequency (MHz)",
        title=(
            f"{DEVICES[device_key]['label']} — continuous wide qspec "
            "SSF-axis signal-amplitude contrast\n"
            "same full frequency grid and qspec parameters as the local panels; "
            f"{map_data['reps']:,} reps/frequency"
        ),
    )
    ax.set_xticks(CURRENT_GRID_MA)
    ax.plot(
        CURRENT_GRID_MA, measured, color="white", linewidth=0.8,
        linestyle="--", alpha=0.8,
    )
    ax.scatter(
        CURRENT_GRID_MA, measured, s=18, facecolors="none",
        edgecolors="white", linewidths=0.8,
    )
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label(
        f"$|\\Delta S_{{\\mathrm{{SSF}}}}|$ from "
        f"{BASELINE_BLOCK_WIDTH_MHZ:.0f}-MHz local baseline "
        "(raw I/Q units)"
    )
    output = OUTPUT / f"{device_key}_single_sweep_qspec_signal_contrast_2d.png"
    fig.savefig(output, dpi=400)
    plt.close(fig)
    return output


def plot_local_curves(device_key, rows):
    fig, axes = plt.subplots(
        3, 3, figsize=(12.4, 7.0), sharey=False, constrained_layout=True
    )
    for ax, row in zip(axes.ravel(), rows):
        record = row["record"]
        seed = float(record["qfreq_seed_MHz"])
        local = np.abs(row["frequency_MHz"] - seed) <= 3.0
        frequency = row["frequency_MHz"][local]
        population = row["population"][local]
        smoothed = smooth_trace(population, points=5)
        measured = record["peak_metrics"]["frequency_MHz"]
        ax.plot(frequency, population, color="#9ecae1", linewidth=0.65)
        ax.plot(frequency, smoothed, color="#08519c", linewidth=1.3)
        ax.axvline(seed, color="black", linestyle="--", linewidth=0.8)
        ax.axvline(measured, color="#d62728", linewidth=0.8)
        ax.set_title(
            f"{row['current_mA']:+.1f} mA | SNR "
            f"{record['peak_metrics']['positive_snr']:.1f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)
        lower, upper = np.percentile(smoothed, [1.0, 99.0])
        lower = min(float(lower), float(np.min(smoothed)))
        upper = max(float(upper), float(np.max(smoothed)))
        padding = max(0.02, 0.07 * (upper - lower))
        ax.set_ylim(lower - padding, upper + padding)
        ax.grid(alpha=0.18)
    fig.supxlabel("Actual swept qubit frequency (MHz)")
    fig.supylabel("SSF-calibrated $P_e$ (labeled physical scale per panel)")
    fig.suptitle(
        f"{DEVICES[device_key]['label']} — peaks from the continuous sweeps\n"
        "independent panel limits; no population normalization",
        fontsize=13,
    )
    output = OUTPUT / f"{device_key}_single_sweep_qspec_local_curves.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def build(manifest_path=MANIFEST, device_keys=None):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if device_keys is None:
        device_keys = list(DEVICES)
    provenance = {
        "created": None,
        "source_manifest": str(manifest_path),
        "display": {
            "frequency_bins": DISPLAY_FREQUENCY_BINS,
            "population_smoothing_points": 5,
            "signal_smoothing_points": 11,
            "frequency_reduction": (
                "five-sample population or eleven-sample SSF-axis smoothing; "
                "subtract an interpolated 20-MHz block-median baseline; "
                "ordinary display bins use their mean, bins intersecting "
                "+/-1 MHz of the independently measured local peak retain "
                "their maximum, and the peak is repeated over three display "
                "rows so it survives PowerPoint rasterization"
            ),
            "row_offset_removed": True,
            "row_amplitude_normalization": False,
            "baseline_block_width_MHz": BASELINE_BLOCK_WIDTH_MHZ,
            "peak_preserve_half_width_MHz": PEAK_PRESERVE_HALF_WIDTH_MHZ,
            "peak_ridge_display_rows": PEAK_RIDGE_DISPLAY_ROWS,
            "contrast_power_gamma": CONTRAST_GAMMA,
            "population_contrast_color_limits": list(
                POPULATION_CONTRAST_LIMITS
            ),
            "raw_data_modified": False,
        },
        "devices": {},
    }
    for device_key in device_keys:
        map_data, rows = load_device(manifest, device_key)
        population_path, display_step = plot_population_heatmap(
            device_key, map_data, rows
        )
        magnitude_path = plot_magnitude_heatmap(device_key, map_data, rows)
        local_path = plot_local_curves(device_key, rows)
        provenance["devices"][device_key] = {
            "label": DEVICES[device_key]["label"],
            "population_heatmap": str(population_path),
            "magnitude_heatmap": str(magnitude_path),
            "local_curves": str(local_path),
            "actual_frequency_start_MHz": rows[0]["frequency_MHz"][0],
            "actual_frequency_stop_MHz": rows[0]["frequency_MHz"][-1],
            "actual_frequency_step_MHz": float(np.median(
                np.diff(rows[0]["frequency_MHz"])
            )),
            "display_frequency_step_MHz": display_step,
            "frequency_arrays_identical": True,
            "all_current_rows_available": True,
            "minimum_positive_peak_snr": min(
                row["record"]["peak_metrics"]["positive_snr"] for row in rows
            ),
        }
    provenance["created"] = __import__("datetime").datetime.now().isoformat()
    PROVENANCE.write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(PROVENANCE)
    return provenance


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=MANIFEST)
    parser.add_argument(
        "--device", choices=("all", "q4", "q6"), default="all",
        help="Build both device maps or one completed device map",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device_keys = list(DEVICES) if args.device == "all" else [args.device]
    build(args.manifest, device_keys=device_keys)


if __name__ == "__main__":
    main()
