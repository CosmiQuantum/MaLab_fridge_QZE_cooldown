"""Build Ramsey-correction galleries and corrected-frequency Rabi maps.

This analysis-only utility reads the campaign manifest and canonical HDF5
artifacts.  Heatmap columns are ordered by increasing current, rows by
increasing realized gain, and color is the saved SSF-calibrated population.
No row normalization or offset is applied.
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
MANIFEST = ROOT / "ramsey_corrected_rabi_manifest.json"
OUTPUT = ROOT / "ramsey_corrected_rabi_2d"
PROVENANCE = OUTPUT / "ramsey_corrected_rabi_provenance.json"
CURRENT_GRID_MA = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]
DEVICES = {
    "q4": {"group": "Q5", "label": "Yoko 3 / physical Q4 / M5"},
    "q6": {"group": "Q6", "label": "Yoko 4 / physical Q6 / M6"},
}
DISPLAY_SMOOTHING_POINTS = 3
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


def load_ramsey_trace(path, group_name):
    with h5py.File(path, "r") as handle:
        group = handle[group_name]
        delay = decode_h5_numeric(group["Delay Times"])
        i_data = decode_h5_numeric(group["I"])
        q_data = decode_h5_numeric(group["Q"])
        fitted = decode_h5_numeric(group["Fit"])
    length = min(delay.size, i_data.size, q_data.size, fitted.size)
    delay = np.asarray(delay[:length], dtype=float)
    i_data = np.asarray(i_data[:length], dtype=float)
    q_data = np.asarray(q_data[:length], dtype=float)
    fitted = np.asarray(fitted[:length], dtype=float)
    finite = (
        np.isfinite(delay) & np.isfinite(i_data) & np.isfinite(q_data)
        & np.isfinite(fitted)
    )
    order = np.argsort(delay[finite])
    return (
        delay[finite][order], i_data[finite][order],
        q_data[finite][order], fitted[finite][order],
    )


def load_rabi_trace(path, group_name):
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
    map_data = manifest.get("devices", {}).get(device_key, {}).get("map", {})
    points = map_data.get("points", {})
    missing = [
        f"{current:+.1f}" for current in CURRENT_GRID_MA
        if points.get(f"{current:+.1f}", {}).get("status") != "complete"
    ]
    if missing:
        raise RuntimeError(
            f"{device_key} Ramsey/Rabi campaign is incomplete at {missing}"
        )
    rows = []
    reference_gain = None
    for current in CURRENT_GRID_MA:
        record = points[f"{current:+.1f}"]
        ramsey_path = Path(record["ramsey"]["h5"])
        rabi_path = Path(record["rabi"]["h5"])
        if not ramsey_path.is_file():
            raise FileNotFoundError(ramsey_path)
        if not rabi_path.is_file():
            raise FileNotFoundError(rabi_path)
        delay, i_data, q_data, ramsey_fit = load_ramsey_trace(
            ramsey_path, DEVICES[device_key]["group"]
        )
        gains, population, rabi_fit = load_rabi_trace(
            rabi_path, DEVICES[device_key]["group"]
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
            "delay": delay,
            "ramsey_I": i_data,
            "ramsey_Q": q_data,
            "ramsey_fit": ramsey_fit,
            "gains": gains,
            "population": population,
            "rabi_fit": rabi_fit,
        })
    return map_data, rows


def coordinate_edges(values):
    values = np.asarray(values, dtype=float)
    midpoints = (values[:-1] + values[1:]) / 2.0
    first = values[0] - (midpoints[0] - values[0])
    last = values[-1] + (values[-1] - midpoints[-1])
    return np.concatenate(([first], midpoints, [last]))


def plot_ramsey_gallery(device_key, map_data, rows):
    fig, axes = plt.subplots(
        3, 3, figsize=(12.6, 7.0), constrained_layout=True,
    )
    for ax, row in zip(axes.ravel(), rows):
        metrics = row["record"]["ramsey"]["fit_metrics"]
        signal_label = metrics["saved_fit_signal"]
        signal = row["ramsey_I"] if signal_label == "I" else row["ramsey_Q"]
        ax.plot(
            row["delay"], signal, color="#6baed6", linewidth=0.9,
            marker=".", markersize=2.8, label=f"saved {signal_label}",
        )
        ax.plot(
            row["delay"], row["ramsey_fit"], color="#d62728",
            linewidth=1.25, label="built-in fit",
        )
        ramsey = row["record"]["ramsey"]
        ax.set_title(
            f"{row['current_mA']:+.1f} mA | fringe "
            f"{ramsey['fitted_fringe_MHz']:.4f} MHz\n"
            f"f01 {ramsey['seed_qfreq_MHz']:.3f} -> "
            f"{ramsey['corrected_qfreq_MHz']:.3f} MHz",
            fontsize=8.2,
        )
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.18)
    axes[0, 0].legend(fontsize=6.5, loc="best")
    fig.supxlabel("Actual Ramsey delay (us)")
    fig.supylabel("Selected averaged I/Q quadrature (a.u.)")
    ramsey_settings = map_data["ramsey"]
    fig.suptitle(
        f"{DEVICES[device_key]['label']} - Ramsey qubit-frequency correction\n"
        f"{ramsey_settings['reps']} reps x {ramsey_settings['rounds']} rounds; "
        f"{ramsey_settings['detuning_MHz']:.1f} MHz intentional detuning; "
        "canonical built-in fits",
        fontsize=12.5,
    )
    output = OUTPUT / f"{device_key}_ramsey_frequency_correction_curves.png"
    fig.savefig(output, dpi=320)
    plt.close(fig)
    return output


def plot_rabi_heatmap(device_key, map_data, rows):
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
    fit_rows = [
        row for row in rows if row["record"]["rabi"].get("fit_available")
        and row["record"]["rabi"].get("fitted_pi_amp") is not None
    ]
    if fit_rows:
        currents = [row["current_mA"] for row in fit_rows]
        fitted_pi = [
            row["record"]["rabi"]["fitted_pi_amp"] for row in fit_rows
        ]
        ax.plot(
            currents, fitted_pi, color="white", linewidth=0.9,
            linestyle="--", alpha=0.85, label="fitted first pi extremum",
        )
        ax.scatter(
            currents, fitted_pi, s=20, facecolors="none",
            edgecolors="white", linewidths=0.8,
        )
        ax.legend(loc="upper right", fontsize=8, framealpha=0.85)
    rabi_settings = map_data["rabi"]
    ax.set(
        xlabel="Yokogawa current (mA)",
        ylabel="Actual qubit drive gain (DAC units)",
        title=(
            f"{DEVICES[device_key]['label']} - Ramsey-corrected Rabi population\n"
            f"sigma={rabi_settings['fixed_sigma_us']:.3f} us; "
            f"{rabi_settings['reps']} reps x {rabi_settings['rounds']} rounds"
        ),
    )
    ax.set_xticks(CURRENT_GRID_MA)
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("SSF-calibrated qubit population $P_e$ (g=0, e=1)")
    output = OUTPUT / f"{device_key}_ramsey_corrected_rabi_population_2d.png"
    fig.savefig(output, dpi=400)
    plt.close(fig)
    return output


def plot_rabi_curves(device_key, map_data, rows):
    fig, axes = plt.subplots(
        3, 3, figsize=(12.4, 7.0), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for ax, row in zip(axes.ravel(), rows):
        gains = row["gains"]
        population = row["population"]
        ax.plot(gains, population, color="#9ecae1", linewidth=0.65)
        ax.plot(
            gains, smooth_trace(population), color="#08519c",
            linewidth=1.25,
        )
        rabi = row["record"]["rabi"]
        if row["rabi_fit"].size == gains.size:
            ax.plot(gains, row["rabi_fit"], color="#d62728", linewidth=1.0)
        if rabi.get("fit_available") and rabi.get("fitted_pi_amp") is not None:
            ax.axvline(
                rabi["fitted_pi_amp"], color="black", linestyle="--",
                linewidth=0.8,
            )
            pi_label = f"pi={rabi['fitted_pi_amp']:.3f}"
        else:
            pi_label = "pi fit unavailable"
        ax.set_title(
            f"{row['current_mA']:+.1f} mA | {pi_label}\n"
            f"f01={rabi['qfreq_MHz']:.3f} MHz",
            fontsize=8.2,
        )
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.18)
    axes[0, 0].set_xlim(rows[0]["gains"][0], rows[0]["gains"][-1])
    axes[0, 0].set_ylim(*POPULATION_LIMITS)
    fig.supxlabel("Actual qubit drive gain (DAC units)")
    fig.supylabel("SSF-calibrated qubit population $P_e$")
    rabi_settings = map_data["rabi"]
    fig.suptitle(
        f"{DEVICES[device_key]['label']} - Ramsey-corrected Rabi curves\n"
        f"sigma={rabi_settings['fixed_sigma_us']:.3f} us; one physical "
        "population scale; no row normalization",
        fontsize=12.5,
    )
    output = OUTPUT / f"{device_key}_ramsey_corrected_rabi_population_curves.png"
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
            "rabi_population_smoothing_points": DISPLAY_SMOOTHING_POINTS,
            "rabi_population_color_limits": list(POPULATION_LIMITS),
            "current_axis_order": "ascending",
            "gain_axis_order": "ascending",
            "row_offset_removed": False,
            "row_amplitude_normalization": False,
            "peak_normalization": False,
            "raw_data_modified": False,
        },
        "devices": {},
    }
    for device_key in device_keys:
        map_data, rows = load_device(manifest, device_key)
        ramsey_gallery = plot_ramsey_gallery(device_key, map_data, rows)
        rabi_heatmap = plot_rabi_heatmap(device_key, map_data, rows)
        rabi_curves = plot_rabi_curves(device_key, map_data, rows)
        provenance["devices"][device_key] = {
            "label": DEVICES[device_key]["label"],
            "ramsey_gallery": str(ramsey_gallery),
            "rabi_population_heatmap": str(rabi_heatmap),
            "rabi_population_curves": str(rabi_curves),
            "fixed_rabi_sigma_us": map_data["rabi"]["fixed_sigma_us"],
            "actual_gain_start": rows[0]["gains"][0],
            "actual_gain_stop": rows[0]["gains"][-1],
            "actual_gain_median_step": float(np.median(
                np.diff(rows[0]["gains"])
            )),
            "gain_arrays_identical": True,
            "all_current_rows_available": True,
            "minimum_population_span_5_95": min(
                row["record"]["rabi"]["population_metrics"][
                    "population_span_5_95"
                ] for row in rows
            ),
            "maximum_ramsey_frequency_stderr_MHz": max(
                row["record"]["ramsey"]["fit_metrics"][
                    "independent_fit"
                ]["frequency_stderr_MHz"] for row in rows
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
        "--device", choices=("all", "q4", "q6"), default="all"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device_keys = list(DEVICES) if args.device == "all" else [args.device]
    build(args.manifest, device_keys=device_keys)


if __name__ == "__main__":
    main()
