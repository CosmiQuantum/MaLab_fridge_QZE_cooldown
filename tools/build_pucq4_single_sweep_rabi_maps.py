"""Build fixed-sigma PUCQ4 current-by-gain Rabi maps from canonical HDF5.

This is analysis-only.  It never imports an experiment controller or touches
hardware.  The heatmaps use the saved calibrated population on one identical
realized gain grid at every current; no row offset, amplitude, peak, range, or
noise normalization is applied.  A fixed three-sample mean is display-only and
the canonical HDF5 arrays remain unchanged.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(
    r"M:\_Data\20250822 - Olivia\pucq4_run_started_Oct_7\PUCQ4"
    r"\pucq4_first_light\yoko_flux_campaign"
)
MANIFEST = ROOT / "single_sweep_rabi_manifest.json"
OUTPUT = ROOT / "single_sweep_rabi_2d"
PROVENANCE = OUTPUT / "single_sweep_rabi_provenance.json"
CURRENT_GRID_MA = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]
DEVICES = {
    "q4": {"group": "Q5", "label": "Yoko 3 / physical Q4 / M5"},
    "q6": {"group": "Q6", "label": "Yoko 4 / physical Q6 / M6"},
}
DISPLAY_SMOOTHING_POINTS = 3
# One shared physical population scale for both devices.  The wider bounds keep
# the Q6 -7.5 mA row's measured affine-SSF overshoot visible without per-row
# normalization or clipping it back to the nominal g=0/e=1 anchors.
POPULATION_LIMITS = (-0.30, 1.50)


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


def smooth_trace(values, points=DISPLAY_SMOOTHING_POINTS):
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


def load_trace(path, group_name):
    with h5py.File(path, "r") as handle:
        group = handle[group_name]
        gains = decode_h5_numeric(group["Gains"])
        population = decode_h5_numeric(group["Population"])
        fitted = (
            decode_h5_numeric(group["Fit"])
            if "Fit" in group else np.asarray([], dtype=float)
        )
    length = min(gains.size, population.size)
    gains = np.asarray(gains[:length], dtype=float)
    population = np.asarray(population[:length], dtype=float)
    finite = np.isfinite(gains) & np.isfinite(population)
    order = np.argsort(gains[finite])
    gains = gains[finite][order]
    population = population[finite][order]
    if fitted.size == length:
        fitted = np.asarray(fitted, dtype=float)[finite][order]
    else:
        fitted = np.asarray([], dtype=float)
    return gains, population, fitted


def load_device(manifest, device_key):
    map_data = (
        manifest.get("devices", {}).get(device_key, {}).get("map", {})
    )
    points = map_data.get("points", {})
    missing = [
        f"{current:+.1f}" for current in CURRENT_GRID_MA
        if points.get(f"{current:+.1f}", {}).get("status") != "complete"
    ]
    if missing:
        raise RuntimeError(
            f"{device_key} fixed-sigma Rabi map is incomplete at {missing}"
        )
    rows = []
    reference_gain = None
    for current in CURRENT_GRID_MA:
        record = points[f"{current:+.1f}"]
        path = Path(record["h5"])
        if not path.is_file():
            raise FileNotFoundError(path)
        gains, population, fitted = load_trace(
            path, DEVICES[device_key]["group"]
        )
        if reference_gain is None:
            reference_gain = gains
        elif (gains.size != reference_gain.size
              or not np.allclose(
                  gains, reference_gain, rtol=0.0, atol=1e-9
              )):
            raise RuntimeError(
                f"{device_key} {current:+.1f} mA does not share one gain axis"
            )
        rows.append({
            "current_mA": current,
            "record": record,
            "gains": gains,
            "population": population,
            "fit": fitted,
        })
    return map_data, rows


def coordinate_edges(values):
    values = np.asarray(values, dtype=float)
    midpoints = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def plot_population_heatmap(device_key, map_data, rows):
    gains = rows[0]["gains"]
    display = np.column_stack([
        smooth_trace(row["population"]) for row in rows
    ])
    fig, ax = plt.subplots(figsize=(8.2, 6.4), constrained_layout=True)
    mesh = ax.pcolormesh(
        coordinate_edges(CURRENT_GRID_MA), coordinate_edges(gains), display,
        cmap="viridis", shading="flat", vmin=POPULATION_LIMITS[0],
        vmax=POPULATION_LIMITS[1], rasterized=True,
    )
    fitted_pi = [row["record"]["fitted_pi_amp"] for row in rows]
    ax.plot(
        CURRENT_GRID_MA, fitted_pi, color="white", linewidth=0.9,
        linestyle="--", alpha=0.85, label="fitted first pi extremum",
    )
    ax.scatter(
        CURRENT_GRID_MA, fitted_pi, s=20, facecolors="none",
        edgecolors="white", linewidths=0.8,
    )
    ax.set(
        xlabel="Yokogawa current (mA)",
        ylabel="Actual qubit drive gain (DAC units)",
        title=(
            f"{DEVICES[device_key]['label']} — fixed-sigma Rabi population\n"
            f"sigma={map_data['fixed_sigma_us']:.3f} us; common gain grid; "
            f"{map_data['reps']} reps x {map_data['rounds']} round"
        ),
    )
    ax.set_xticks(CURRENT_GRID_MA)
    ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("SSF-calibrated qubit population $P_e$ (g=0, e=1)")
    output = OUTPUT / f"{device_key}_fixed_sigma_rabi_population_2d.png"
    fig.savefig(output, dpi=400)
    plt.close(fig)
    return output


def plot_local_curves(device_key, map_data, rows):
    fig, axes = plt.subplots(
        3, 3, figsize=(12.4, 7.0), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for ax, row in zip(axes.ravel(), rows):
        gains = row["gains"]
        population = row["population"]
        smoothed = smooth_trace(population)
        ax.plot(gains, population, color="#9ecae1", linewidth=0.65)
        ax.plot(gains, smoothed, color="#08519c", linewidth=1.25)
        if row["fit"].size == gains.size:
            ax.plot(gains, row["fit"], color="#d62728", linewidth=1.0)
        ax.axvline(
            row["record"]["fitted_pi_amp"], color="black",
            linestyle="--", linewidth=0.8,
        )
        ax.set_title(
            f"{row['current_mA']:+.1f} mA | pi="
            f"{row['record']['fitted_pi_amp']:.3f}",
            fontsize=9,
        )
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.18)
    axes[0, 0].set_xlim(rows[0]["gains"][0], rows[0]["gains"][-1])
    axes[0, 0].set_ylim(*POPULATION_LIMITS)
    fig.supxlabel("Actual qubit drive gain (DAC units)")
    fig.supylabel("SSF-calibrated qubit population $P_e$")
    fig.suptitle(
        f"{DEVICES[device_key]['label']} — fixed-sigma Rabi curves\n"
        f"sigma={map_data['fixed_sigma_us']:.3f} us; "
        "one physical population scale; no row normalization",
        fontsize=13,
    )
    output = OUTPUT / f"{device_key}_fixed_sigma_rabi_population_curves.png"
    fig.savefig(output, dpi=300)
    plt.close(fig)
    return output


def build(manifest_path=MANIFEST, device_keys=None):
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if device_keys is None:
        device_keys = list(DEVICES)
    provenance = {
        "created": datetime.now().isoformat(),
        "source_manifest": str(manifest_path),
        "display": {
            "population_smoothing_points": DISPLAY_SMOOTHING_POINTS,
            "population_color_limits": list(POPULATION_LIMITS),
            "row_offset_removed": False,
            "row_amplitude_normalization": False,
            "peak_normalization": False,
            "raw_data_modified": False,
        },
        "devices": {},
    }
    for device_key in device_keys:
        map_data, rows = load_device(manifest, device_key)
        heatmap = plot_population_heatmap(device_key, map_data, rows)
        curves = plot_local_curves(device_key, map_data, rows)
        provenance["devices"][device_key] = {
            "label": DEVICES[device_key]["label"],
            "population_heatmap": str(heatmap),
            "population_curves": str(curves),
            "fixed_sigma_us": map_data["fixed_sigma_us"],
            "actual_gain_start": rows[0]["gains"][0],
            "actual_gain_stop": rows[0]["gains"][-1],
            "actual_gain_median_step": float(np.median(
                np.diff(rows[0]["gains"])
            )),
            "gain_arrays_identical": True,
            "all_current_rows_available": True,
            "minimum_population_span_5_95": min(
                row["record"]["population_metrics"][
                    "population_span_5_95"
                ] for row in rows
            ),
        }
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
        help="Build both completed device maps or one device map",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device_keys = list(DEVICES) if args.device == "all" else [args.device]
    build(args.manifest, device_keys=device_keys)


if __name__ == "__main__":
    main()
