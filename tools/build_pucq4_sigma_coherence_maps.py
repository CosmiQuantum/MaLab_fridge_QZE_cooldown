"""Build sigma/current and calibrated-population coherence maps for PUCQ4.

This analysis-only utility reads the resumable campaign manifest and canonical
HDF5 artifacts.  Current is always the x axis, actual realized delay in us is
the y axis for T1/T2R/T2E, and color is the saved signed SSF-calibrated qubit
population.  No current-row normalization or offset is applied.
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
    r"M:\_Data\20250822 - Olivia\pucq4_run_started_Aug_3\PUCQ4"
    r"\pucq4_first_light\yoko_flux_campaign"
)
MANIFEST = ROOT / "ramsey_sigma_coherence_manifest.json"
OUTPUT = ROOT / "ramsey_sigma_coherence_2d"
PROVENANCE = OUTPUT / "ramsey_sigma_coherence_provenance.json"
FREQUENCY_SUMMARY_PROVENANCE = (
    OUTPUT / "frequency_axis_summary_provenance.json"
)
FREQUENCY_ONLY_SUMMARY_PROVENANCE = (
    OUTPUT / "frequency_only_summary_provenance.json"
)
CURRENT_GRID_MA = np.asarray(
    [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]
)
DEVICES = {
    "q4": {"group": "Q5", "label": "Yoko 3 / physical Q4 / M5"},
    "q6": {"group": "Q6", "label": "Yoko 4 / physical Q6 / M6"},
}
EXPERIMENTS = {
    "t1": {"title": "T1", "fit_label": "T1"},
    "t2r": {"title": "T2R Ramsey", "fit_label": "T2R"},
    "t2e": {"title": "T2E spin echo", "fit_label": "T2E"},
}
DISPLAY_SMOOTHING_POINTS = 3
POPULATION_LIMITS = (-0.30, 1.50)
SUMMARY_METRICS = {
    "sigma": {
        "title": "Selected Gaussian sigma",
        "ylabel": "Selected Gaussian sigma (ns)",
        "color": "#08519c",
        "decimals": 1,
    },
    "t1": {
        "title": "Fitted T1",
        "ylabel": "Fitted T1 (us)",
        "color": "#238b45",
        "decimals": 3,
    },
    "t2r": {
        "title": "Fitted T2R Ramsey",
        "ylabel": "Fitted T2R (us)",
        "color": "#6a51a3",
        "decimals": 3,
    },
    "t2e": {
        "title": "Fitted T2E spin echo",
        "ylabel": "Fitted T2E (us)",
        "color": "#cb181d",
        "decimals": 3,
    },
}


def format_averaging(averaging_groups):
    default_statistics, default_currents = max(
        averaging_groups.items(), key=lambda item: len(item[1])
    )
    default_reps, default_rounds = default_statistics
    if len(default_currents) == len(CURRENT_GRID_MA):
        return (
            f"{default_reps} reps x {default_rounds} rounds at all currents"
        )
    exceptions = []
    for (reps, rounds), currents in averaging_groups.items():
        if (reps, rounds) == default_statistics:
            continue
        exception_currents = ", ".join(
            f"{current:+.1f}" for current in currents
        )
        verb = "uses" if len(currents) == 1 else "use"
        exceptions.append(
            f"{exception_currents} mA {verb} {reps} reps x {rounds} rounds"
        )
    return (
        f"{default_reps} reps x {default_rounds} rounds at other currents; "
        + "; ".join(exceptions)
    )


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
    values = np.asarray(values, dtype=float)
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


def coordinate_edges(values):
    values = np.asarray(values, dtype=float)
    midpoints = (values[:-1] + values[1:]) / 2.0
    return np.concatenate((
        [values[0] - (midpoints[0] - values[0])],
        midpoints,
        [values[-1] + (values[-1] - midpoints[-1])],
    ))


def load_ramsey(path, group_name):
    with h5py.File(path, "r") as handle:
        group = handle[group_name]
        delay = decode_h5_numeric(group["Delay Times"])
        i_data = decode_h5_numeric(group["I"])
        q_data = decode_h5_numeric(group["Q"])
        fitted = decode_h5_numeric(group["Fit"])
    length = min(delay.size, i_data.size, q_data.size, fitted.size)
    finite = (
        np.isfinite(delay[:length]) & np.isfinite(i_data[:length])
        & np.isfinite(q_data[:length]) & np.isfinite(fitted[:length])
    )
    order = np.argsort(delay[:length][finite])
    return (
        delay[:length][finite][order], i_data[:length][finite][order],
        q_data[:length][finite][order], fitted[:length][finite][order],
    )


def load_rabi(path, group_name):
    with h5py.File(path, "r") as handle:
        group = handle[group_name]
        gain = decode_h5_numeric(group["Gains"])
        population = decode_h5_numeric(group["Population"])
        fitted = decode_h5_numeric(group["Fit"])
    length = min(gain.size, population.size, fitted.size)
    finite = (
        np.isfinite(gain[:length]) & np.isfinite(population[:length])
        & np.isfinite(fitted[:length])
    )
    order = np.argsort(gain[:length][finite])
    return (
        gain[:length][finite][order],
        population[:length][finite][order],
        fitted[:length][finite][order],
    )


def load_coherence(path, group_name):
    with h5py.File(path, "r") as handle:
        group = handle[group_name]
        delay = decode_h5_numeric(group["Delay Times"])
        population = decode_h5_numeric(group["Population"])
        fitted = decode_h5_numeric(group["Fit"])
    length = min(delay.size, population.size, fitted.size)
    finite = (
        np.isfinite(delay[:length]) & np.isfinite(population[:length])
        & np.isfinite(fitted[:length])
    )
    order = np.argsort(delay[:length][finite])
    return (
        delay[:length][finite][order],
        population[:length][finite][order],
        fitted[:length][finite][order],
    )


def load_device(manifest, device_key):
    map_data = manifest["devices"][device_key]["map"]
    points = map_data["points"]
    rows = []
    reference_delays = {}
    for current in CURRENT_GRID_MA:
        key = f"{current:+.1f}"
        record = points.get(key, {})
        if record.get("status") != "complete":
            raise RuntimeError(
                f"{device_key} campaign is incomplete at {key} mA"
            )
        selected = record["sigma_calibration"]["selected"]
        paths = [
            Path(record["ramsey"]["h5"]), Path(selected["h5"]),
        ] + [
            Path(record["coherence"]["experiments"][name]["h5"])
            for name in EXPERIMENTS
        ]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(path)
        group_name = DEVICES[device_key]["group"]
        delay, ramsey_i, ramsey_q, ramsey_fit = load_ramsey(
            paths[0], group_name
        )
        gain, rabi_population, rabi_fit = load_rabi(paths[1], group_name)
        coherence = {}
        for name in EXPERIMENTS:
            experiment = record["coherence"]["experiments"][name]
            expt_delay, population, fitted = load_coherence(
                experiment["h5"], group_name
            )
            if np.any(np.diff(expt_delay) <= 0.0):
                raise RuntimeError(
                    f"{device_key} {name} delay axis is not increasing"
                )
            if name not in reference_delays:
                reference_delays[name] = expt_delay
            elif expt_delay.size != reference_delays[name].size:
                raise RuntimeError(
                    f"{device_key} {name} delay-axis sizes differ: "
                    f"{reference_delays[name].size} versus {expt_delay.size}"
                )
            coherence[name] = {
                "delay": expt_delay,
                "population": population,
                "fit": fitted,
                "metadata": experiment,
            }
        rows.append({
            "current_mA": float(current),
            "record": record,
            "ramsey_delay": delay,
            "ramsey_I": ramsey_i,
            "ramsey_Q": ramsey_q,
            "ramsey_fit": ramsey_fit,
            "gain": gain,
            "rabi_population": rabi_population,
            "rabi_fit": rabi_fit,
            "coherence": coherence,
        })
    return map_data, rows


def load_frequency_summary_rows(manifest, device_key):
    """Pair each final metric with the corrected f01 in the same bias row."""
    points = manifest["devices"][device_key]["map"]["points"]
    rows = []
    for current in CURRENT_GRID_MA:
        key = f"{current:+.1f}"
        record = points.get(key, {})
        if record.get("status") != "complete":
            raise RuntimeError(
                f"{device_key} campaign is incomplete at {key} mA"
            )
        corrected_qfreq = float(
            record["ramsey"]["corrected_qfreq_MHz"]
        )
        selected_rabi = record["sigma_calibration"]["selected"]
        coherence = record["coherence"]
        frequency_sources = {
            "selected Rabi": float(selected_rabi["qfreq_MHz"]),
            "coherence": float(coherence["qfreq_MHz"]),
        }
        for source_name, source_frequency in frequency_sources.items():
            if not np.isclose(
                source_frequency, corrected_qfreq, rtol=0.0, atol=1e-9
            ):
                raise RuntimeError(
                    f"{device_key} {key} mA {source_name} frequency "
                    f"{source_frequency:.12f} MHz does not match corrected "
                    f"f01 {corrected_qfreq:.12f} MHz"
                )
        row = {
            "current_mA": float(current),
            "corrected_qfreq_MHz": corrected_qfreq,
            "sigma": float(selected_rabi["sigma_us"]) * 1000.0,
        }
        for experiment in EXPERIMENTS:
            row[experiment] = float(
                coherence["experiments"][experiment]["fit_value_us"]
            )
        if not all(np.isfinite(value) for value in row.values()):
            raise RuntimeError(
                f"{device_key} {key} mA contains a non-finite summary value"
            )
        rows.append(row)
    return rows


def add_frequency_lookup_axis(ax, currents, corrected_qfreqs):
    """Add exact per-current f01 labels without implying a linear transform."""
    frequency_axis = ax.twiny()
    frequency_axis.set_xlim(ax.get_xlim())
    frequency_axis.set_xticks(currents)
    frequency_axis.set_xticklabels(
        [f"{frequency:.3f}" for frequency in corrected_qfreqs],
        rotation=45,
        ha="center",
        rotation_mode="anchor",
        fontsize=7.4,
    )
    frequency_axis.tick_params(axis="x", length=3, pad=1.5)
    frequency_axis.set_xlabel(
        "Ramsey-corrected f01 paired at each bias (MHz; nonlinear lookup)",
        fontsize=9,
        labelpad=5,
    )
    return frequency_axis


def plot_frequency_summary_metric(device_key, rows, metric):
    settings = SUMMARY_METRICS[metric]
    currents = np.asarray([row["current_mA"] for row in rows])
    corrected_qfreqs = np.asarray([
        row["corrected_qfreq_MHz"] for row in rows
    ])
    values = np.asarray([row[metric] for row in rows])
    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    ax.plot(
        currents, values, color=settings["color"], marker="o",
        markersize=5.5, linewidth=1.8,
    )
    ax.set_xlim(-11.0, 11.0)
    ax.set_xticks(currents)
    ax.set_xlabel("Yokogawa current (mA)")
    ax.set_ylabel(settings["ylabel"])
    ax.grid(alpha=0.25)
    ax.margins(y=0.16)
    value_format = f"{{:.{settings['decimals']}f}}"
    for current, value in zip(currents, values):
        ax.annotate(
            value_format.format(value), (current, value),
            xytext=(0, 7), textcoords="offset points", ha="center",
            fontsize=7.0, color=settings["color"],
        )
    add_frequency_lookup_axis(ax, currents, corrected_qfreqs)
    ax.set_title(
        f"{DEVICES[device_key]['label']}\n{settings['title']} by bias row",
        fontsize=11.5,
        pad=10,
    )
    path = OUTPUT / f"{device_key}_{metric}_current_qfreq_axes.png"
    fig.savefig(path, dpi=360)
    plt.close(fig)
    return path


def plot_frequency_only_metric(device_key, rows, metric):
    """Plot a metric against corrected f01 in strictly ascending order."""
    settings = SUMMARY_METRICS[metric]
    frequency_sorted_rows = sorted(
        rows, key=lambda row: row["corrected_qfreq_MHz"]
    )
    corrected_qfreqs = np.asarray([
        row["corrected_qfreq_MHz"] for row in frequency_sorted_rows
    ])
    values = np.asarray([row[metric] for row in frequency_sorted_rows])
    if np.any(np.diff(corrected_qfreqs) <= 0.0):
        raise RuntimeError(
            f"{device_key} corrected f01 values are not unique"
        )
    fig, ax = plt.subplots(figsize=(7.0, 6.0), constrained_layout=True)
    ax.plot(
        corrected_qfreqs, values, color=settings["color"], marker="o",
        markersize=5.5, linewidth=1.8,
    )
    frequency_span = float(np.ptp(corrected_qfreqs))
    ax.set_xlim(
        corrected_qfreqs[0] - 0.04 * frequency_span,
        corrected_qfreqs[-1] + 0.04 * frequency_span,
    )
    ax.set_xlabel("Ramsey-corrected qubit frequency f01 (MHz)")
    ax.set_ylabel(settings["ylabel"])
    ax.ticklabel_format(axis="x", style="plain", useOffset=False)
    ax.locator_params(axis="x", nbins=7)
    ax.grid(alpha=0.25)
    ax.margins(y=0.16)
    value_format = f"{{:.{settings['decimals']}f}}"
    for index, (frequency, value) in enumerate(zip(
            corrected_qfreqs, values)):
        vertical_offset = 7 if index % 2 == 0 else -13
        ax.annotate(
            value_format.format(value), (frequency, value),
            xytext=(0, vertical_offset), textcoords="offset points",
            ha="center", fontsize=7.0, color=settings["color"],
        )
    ax.set_title(
        f"{DEVICES[device_key]['label']}\n"
        f"{settings['title']} versus corrected f01 (frequency-sorted)",
        fontsize=11.5,
    )
    path = OUTPUT / f"{device_key}_{metric}_vs_qfreq.png"
    fig.savefig(path, dpi=360)
    plt.close(fig)
    return path


def build_frequency_summaries(manifest_path=MANIFEST, device_keys=None):
    """Build manifest-only dual-axis summaries; no raw data are modified."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if device_keys is None:
        device_keys = list(DEVICES)
    provenance = {
        "created": datetime.now().isoformat(),
        "source_manifest": str(manifest_path),
        "source_manifest_updated": manifest.get("updated"),
        "axis_semantics": {
            "primary_bottom_axis": "Yokogawa current in mA",
            "secondary_top_axis": (
                "exact Ramsey-corrected f01 from the same current row in "
                "MHz; categorical per-tick lookup, not a linear transform"
            ),
            "pairing": (
                "sigma, T1, T2R, and T2E are paired only with the corrected "
                "f01 stored in the same device/current manifest row"
            ),
        },
        "raw_data_modified": False,
        "devices": {},
    }
    for device_key in device_keys:
        rows = load_frequency_summary_rows(manifest, device_key)
        plots = {
            metric: str(plot_frequency_summary_metric(
                device_key, rows, metric
            )) for metric in SUMMARY_METRICS
        }
        provenance["devices"][device_key] = {
            "label": DEVICES[device_key]["label"],
            "plots": plots,
            "rows": rows,
        }
    FREQUENCY_SUMMARY_PROVENANCE.write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(FREQUENCY_SUMMARY_PROVENANCE)
    return provenance


def build_frequency_only_summaries(manifest_path=MANIFEST, device_keys=None):
    """Build frequency-only summaries from the validated manifest rows."""
    manifest_path = Path(manifest_path)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    if device_keys is None:
        device_keys = list(DEVICES)
    provenance = {
        "created": datetime.now().isoformat(),
        "source_manifest": str(manifest_path),
        "source_manifest_updated": manifest.get("updated"),
        "axis_semantics": {
            "x_axis": (
                "actual Ramsey-corrected f01 in MHz, sorted strictly "
                "ascending before plotting"
            ),
            "current_axis_present": False,
            "pairing": (
                "sigma, T1, T2R, and T2E remain paired with corrected f01 "
                "from the same original device/current manifest row"
            ),
        },
        "raw_data_modified": False,
        "devices": {},
    }
    for device_key in device_keys:
        rows = load_frequency_summary_rows(manifest, device_key)
        frequency_sorted_rows = sorted(
            rows, key=lambda row: row["corrected_qfreq_MHz"]
        )
        plots = {
            metric: str(plot_frequency_only_metric(
                device_key, rows, metric
            )) for metric in SUMMARY_METRICS
        }
        provenance["devices"][device_key] = {
            "label": DEVICES[device_key]["label"],
            "plots": plots,
            "frequency_sorted_rows": frequency_sorted_rows,
        }
    FREQUENCY_ONLY_SUMMARY_PROVENANCE.write_text(
        json.dumps(provenance, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(FREQUENCY_ONLY_SUMMARY_PROVENANCE)
    return provenance


def plot_ramsey_gallery(device_key, map_data, rows):
    fig, axes = plt.subplots(
        3, 3, figsize=(12.6, 7.0), constrained_layout=True
    )
    for ax, row in zip(axes.ravel(), rows):
        ramsey = row["record"]["ramsey"]
        selected = ramsey["fit_metrics"]["saved_fit_signal"]
        signal = row["ramsey_I"] if selected == "I" else row["ramsey_Q"]
        ax.plot(
            row["ramsey_delay"], signal, color="#6baed6", linewidth=0.9,
            marker=".", markersize=2.6,
        )
        ax.plot(
            row["ramsey_delay"], row["ramsey_fit"], color="#d62728",
            linewidth=1.2,
        )
        ax.set_title(
            f"{row['current_mA']:+.1f} mA | {selected} | "
            f"df={ramsey['frequency_correction_MHz']:+.4f} MHz\n"
            f"f01={ramsey['corrected_qfreq_MHz']:.4f} MHz",
            fontsize=8.1,
        )
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.18)
    fig.supxlabel("Actual Ramsey delay (us)")
    fig.supylabel("Selected averaged I/Q quadrature (a.u.)")
    settings = map_data["ramsey"]
    averaging_groups = {}
    for row in rows:
        ramsey = row["record"]["ramsey"]
        statistics = (int(ramsey["reps"]), int(ramsey["rounds"]))
        averaging_groups.setdefault(statistics, []).append(
            row["current_mA"]
        )
    averaging_text = format_averaging(averaging_groups)
    fig.suptitle(
        f"{DEVICES[device_key]['label']} - fresh Ramsey corrections\n"
        f"{averaging_text}; {settings['detuning_MHz']:.1f} MHz "
        "intentional detuning",
        fontsize=12.5,
    )
    path = OUTPUT / f"{device_key}_fresh_ramsey_corrections.png"
    fig.savefig(path, dpi=320)
    plt.close(fig)
    return path


def plot_sigma_vs_current(device_key, map_data, rows):
    sigma = np.asarray([
        row["record"]["sigma_calibration"]["selected"]["sigma_us"]
        for row in rows
    ])
    pi_gain = np.asarray([
        row["record"]["sigma_calibration"]["selected"]["pi_gain"]
        for row in rows
    ])
    attempts = np.asarray([
        len(row["record"]["sigma_calibration"]["attempts"])
        for row in rows
    ])
    fig, (ax_sigma, ax_pi) = plt.subplots(
        2, 1, figsize=(8.5, 7.0), sharex=True, constrained_layout=True
    )
    ax_sigma.plot(
        CURRENT_GRID_MA, sigma * 1000.0, color="#08519c", marker="o",
        linewidth=1.8,
    )
    for current, value, count in zip(CURRENT_GRID_MA, sigma, attempts):
        ax_sigma.annotate(
            f"{value * 1000.0:.1f} ns\n{count} run{'s' if count != 1 else ''}",
            (current, value * 1000.0), xytext=(0, 7),
            textcoords="offset points", ha="center", fontsize=7,
        )
    ax_sigma.set_ylabel("Selected Gaussian sigma (ns)")
    ax_sigma.grid(alpha=0.25)
    ax_pi.axhspan(
        map_data["sigma_tuning"]["accepted_pi_gain"][0],
        map_data["sigma_tuning"]["accepted_pi_gain"][1],
        color="#c7e9c0", alpha=0.65, label="accepted window",
    )
    ax_pi.axhline(
        map_data["sigma_tuning"]["target_pi_gain"], color="#238b45",
        linestyle="--", linewidth=1.2, label="target 0.65",
    )
    ax_pi.plot(
        CURRENT_GRID_MA, pi_gain, color="#cb181d", marker="o",
        linewidth=1.6, label="measured pi gain",
    )
    ax_pi.set_xlabel("Yokogawa current (mA)")
    ax_pi.set_ylabel("Fitted pi gain (DAC units)")
    ax_pi.set_xticks(CURRENT_GRID_MA)
    ax_pi.grid(alpha=0.25)
    ax_pi.legend(fontsize=8, loc="best")
    fig.suptitle(
        f"{DEVICES[device_key]['label']} - optimized sigma versus current\n"
        f"201 gain points, {map_data['sigma_tuning']['reps']} reps x "
        f"{map_data['sigma_tuning']['rounds']} rounds per attempt",
        fontsize=12.5,
    )
    path = OUTPUT / f"{device_key}_sigma_vs_current.png"
    fig.savefig(path, dpi=360)
    plt.close(fig)
    return path


def plot_rabi_gallery(device_key, rows):
    fig, axes = plt.subplots(
        3, 3, figsize=(12.5, 7.0), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for ax, row in zip(axes.ravel(), rows):
        selected = row["record"]["sigma_calibration"]["selected"]
        ax.plot(
            row["gain"], row["rabi_population"], color="#9ecae1",
            linewidth=0.6,
        )
        ax.plot(
            row["gain"], row["rabi_fit"], color="#d62728",
            linewidth=1.15,
        )
        ax.axvline(
            selected["pi_gain"], color="black", linestyle="--",
            linewidth=0.8,
        )
        ax.set_title(
            f"{row['current_mA']:+.1f} mA | sigma "
            f"{selected['sigma_us'] * 1000:.1f} ns\n"
            f"pi={selected['pi_gain']:.4f}, "
            f"R2={selected['fit_quality']['r_squared']:.3f}",
            fontsize=8.0,
        )
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.18)
    axes[0, 0].set_ylim(*POPULATION_LIMITS)
    fig.supxlabel("Actual qubit drive gain (DAC units)")
    fig.supylabel("SSF-calibrated qubit population $P_e$")
    fig.suptitle(
        f"{DEVICES[device_key]['label']} - final sigma-calibration Rabi curves",
        fontsize=12.5,
    )
    path = OUTPUT / f"{device_key}_selected_sigma_rabi_curves.png"
    fig.savefig(path, dpi=320)
    plt.close(fig)
    return path


def plot_coherence_heatmap(device_key, map_data, rows, experiment):
    traces = [row["coherence"][experiment] for row in rows]
    averaging_groups = {}
    for row, trace in zip(rows, traces):
        statistics = (
            int(trace["metadata"]["reps"]),
            int(trace["metadata"]["rounds"]),
        )
        averaging_groups.setdefault(statistics, []).append(
            row["current_mA"]
        )
    averaging_text = format_averaging(averaging_groups)
    fit_values = np.asarray([
        trace["metadata"]["fit_value_us"] for trace in traces
    ])
    fig, ax = plt.subplots(figsize=(8.4, 6.4), constrained_layout=True)
    current_edges = coordinate_edges(CURRENT_GRID_MA)
    mesh = None
    for column, trace in enumerate(traces):
        mesh = ax.pcolormesh(
            current_edges[column:column + 2],
            coordinate_edges(trace["delay"]),
            smooth_trace(trace["population"])[:, np.newaxis],
            shading="flat", cmap="viridis", vmin=POPULATION_LIMITS[0],
            vmax=POPULATION_LIMITS[1], rasterized=True,
        )
    ax.plot(
        CURRENT_GRID_MA, fit_values, color="white", linestyle="--",
        linewidth=1.0, label=f"fitted {EXPERIMENTS[experiment]['fit_label']}",
    )
    ax.scatter(
        CURRENT_GRID_MA, fit_values, s=22, facecolors="none",
        edgecolors="white", linewidths=0.8,
    )
    settings = map_data["coherence"]["experiments"][experiment]
    subtitle = (
        f"{settings['start_us']:.2f}-{settings['stop_us']:.0f} us, "
        f"{settings['steps']} points"
    )
    if "detuning_MHz" in settings:
        subtitle += f", {settings['detuning_MHz']:.1f} MHz detuning"
    ax.set(
        xlabel="Yokogawa current (mA)",
        ylabel="Actual delay (us)",
        title=(
            f"{DEVICES[device_key]['label']} - "
            f"{EXPERIMENTS[experiment]['title']} population map\n"
            f"requested {subtitle}\n{averaging_text}"
        ),
    )
    ax.set_xticks(CURRENT_GRID_MA)
    ax.legend(fontsize=8, loc="upper right")
    colorbar = fig.colorbar(mesh, ax=ax, pad=0.02)
    colorbar.set_label("SSF-calibrated qubit population $P_e$ (g=0, e=1)")
    path = OUTPUT / f"{device_key}_{experiment}_population_2d.png"
    fig.savefig(path, dpi=400)
    plt.close(fig)
    return path


def plot_coherence_gallery(device_key, rows, experiment):
    fig, axes = plt.subplots(
        3, 3, figsize=(12.5, 7.0), sharex=True, sharey=True,
        constrained_layout=True,
    )
    for ax, row in zip(axes.ravel(), rows):
        trace = row["coherence"][experiment]
        metadata = trace["metadata"]
        ax.plot(
            trace["delay"], trace["population"], color="#9ecae1",
            linewidth=0.65,
        )
        ax.plot(
            trace["delay"], trace["fit"], color="#d62728",
            linewidth=1.1,
        )
        ax.set_title(
            f"{row['current_mA']:+.1f} mA | "
            f"{metadata['fit_value_us']:.3f} +/- "
            f"{metadata['fit_error_us']:.3f} us\n"
            f"R2={metadata['fit_quality']['r_squared']:.3f}",
            fontsize=8.0,
        )
        ax.tick_params(labelsize=7)
        ax.grid(alpha=0.18)
    axes[0, 0].set_ylim(*POPULATION_LIMITS)
    fig.supxlabel("Actual delay (us)")
    fig.supylabel("SSF-calibrated qubit population $P_e$")
    fig.suptitle(
        f"{DEVICES[device_key]['label']} - "
        f"{EXPERIMENTS[experiment]['title']} curves",
        fontsize=12.5,
    )
    path = OUTPUT / f"{device_key}_{experiment}_population_curves.png"
    fig.savefig(path, dpi=320)
    plt.close(fig)
    return path


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
            "current_axis_order": "ascending",
            "delay_axis_order": "ascending",
            "row_offset_removed": False,
            "row_amplitude_normalization": False,
            "peak_normalization": False,
            "raw_data_modified": False,
        },
        "devices": {},
    }
    for device_key in device_keys:
        map_data, rows = load_device(manifest, device_key)
        ramsey = plot_ramsey_gallery(device_key, map_data, rows)
        sigma = plot_sigma_vs_current(device_key, map_data, rows)
        rabi = plot_rabi_gallery(device_key, rows)
        experiments = {}
        for name in EXPERIMENTS:
            heatmap = plot_coherence_heatmap(
                device_key, map_data, rows, name
            )
            curves = plot_coherence_gallery(device_key, rows, name)
            experiments[name] = {
                "heatmap": str(heatmap),
                "curves": str(curves),
                "selection_source_by_current": [
                    row["coherence"][name]["metadata"].get(
                        "selection_source", "original acquisition"
                    ) for row in rows
                ],
                "reacquisition_comparison_by_current": {
                    f"{row['current_mA']:+.1f}": row["record"][
                        "coherence"
                    ]["selection_choices"][name]
                    for row in rows
                    if name in row["record"]["coherence"].get(
                        "selection_choices", {}
                    )
                },
                "delay_axes": {
                    "all_same_length": len({
                        row["coherence"][name]["delay"].size
                        for row in rows
                    }) == 1,
                    "arrays_identical": all(
                        np.array_equal(
                            row["coherence"][name]["delay"],
                            rows[0]["coherence"][name]["delay"],
                        ) for row in rows[1:]
                    ),
                    "maximum_pointwise_spread_us": float(np.max(np.ptp(
                        np.column_stack([
                            row["coherence"][name]["delay"]
                            for row in rows
                        ]), axis=1,
                    ))),
                    "start_us_by_current": [
                        float(row["coherence"][name]["delay"][0])
                        for row in rows
                    ],
                    "stop_us_by_current": [
                        float(row["coherence"][name]["delay"][-1])
                        for row in rows
                    ],
                    "heatmap_coordinates": (
                        "one explicit quad strip per current using native "
                        "delay centers and midpoint edges; no delay-axis "
                        "interpolation"
                    ),
                },
                "fit_values_us": [
                    row["coherence"][name]["metadata"]["fit_value_us"]
                    for row in rows
                ],
            }
        provenance["devices"][device_key] = {
            "label": DEVICES[device_key]["label"],
            "ramsey_gallery": str(ramsey),
            "sigma_vs_current": str(sigma),
            "selected_rabi_curves": str(rabi),
            "experiments": experiments,
            "all_current_rows_available": True,
            "selected_sigma_us": [
                row["record"]["sigma_calibration"]["selected"]["sigma_us"]
                for row in rows
            ],
            "selected_pi_gain": [
                row["record"]["sigma_calibration"]["selected"]["pi_gain"]
                for row in rows
            ],
            "corrected_qfreq_MHz": [
                row["record"]["ramsey"]["corrected_qfreq_MHz"]
                for row in rows
            ],
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
    parser.add_argument(
        "--frequency-summary-only", action="store_true",
        help=(
            "Build only the manifest-backed sigma/T1/T2R/T2E plots with "
            "current and corrected-f01 x axes"
        ),
    )
    parser.add_argument(
        "--frequency-only-summary", action="store_true",
        help=(
            "Build only sigma/T1/T2R/T2E versus corrected f01, sorted by "
            "frequency with no current axis"
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device_keys = list(DEVICES) if args.device == "all" else [args.device]
    if args.frequency_only_summary:
        build_frequency_only_summaries(
            args.manifest, device_keys=device_keys
        )
        return
    if not args.frequency_summary_only:
        build(args.manifest, device_keys=device_keys)
    build_frequency_summaries(args.manifest, device_keys=device_keys)


if __name__ == "__main__":
    main()
