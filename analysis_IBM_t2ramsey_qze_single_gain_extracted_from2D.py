from analysis_006_T1_vs_time_plots import T1VsTime
from analysis_007_T2R_vs_time_plots import T2rVsTime
from analysis_008_T2E_vs_time_plots import T2eVsTime
from analysis_003_q_freqs_vs_time_plots import QubitFreqsVsTime

import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict

# Configuration
save_figs = True
save_individual_qspec = False
figure_quality = 100
final_figure_quality = 200
FRIDGE = "QUIET"
qubits = [4]
path = '2d_higher_n_bar'
CHI_MHZ = -0.137

for qubit in qubits:
    run_name = f'bob_run_started_Aug_23/squill/{path}/all_qubits/'

    p = Path('M:/_Data/20250822 - Olivia/' + run_name)
    top_folder_dates = [item.name for item in p.iterdir() if item.is_dir() and "qubit" in item.name]

    # ============================================================
    # 1) Run QSpec analysis to get nbar for each round
    # ============================================================
    q_vs_time = QubitFreqsVsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs,
                                 False, 'None', run_name, fridge=FRIDGE, exp_name='ge', qubit=qubit)
    _, _, amps_qspec, gains_qspec, rounds_qspec, freqs_qspec = q_vs_time.run_q_sweep_new(exp_extension='_ge', scaling=True)

    # calculate_nbar returns:
    #   {round_id_str: {"gains": [g1, g2, ...], "lorentzian": [nbar1, nbar2, ...], "gaussian": None}}
    n_bars = q_vs_time.calculate_nbar(amps_qspec, gains_qspec, rounds_qspec, freqs_qspec, chi_MHz=CHI_MHZ)

    # Build gain -> list of nbar values across all rounds
    # Then average to get gain -> mean_nbar
    gain_to_nbars = defaultdict(list)
    for r_id, entry in n_bars.items():
        gains_list_r = entry.get("gains", [])
        nbar_list_r = entry.get("lorentzian", None)
        if nbar_list_r is None:
            continue
        for g_val, nb_val in zip(gains_list_r, nbar_list_r):
            if np.isfinite(g_val) and np.isfinite(nb_val):
                gain_to_nbars[round(g_val, 6)].append(nb_val)

    # Mean nbar per gain
    gain_to_mean_nbar = {}
    for g_val, nb_list in gain_to_nbars.items():
        gain_to_mean_nbar[g_val] = float(np.mean(nb_list))

    print("\n=== Gain -> Mean nbar mapping ===")
    for g in sorted(gain_to_mean_nbar.keys()):
        print(f"  gain={g:.6f}  ->  mean nbar={gain_to_mean_nbar[g]:.4f}  (from {len(gain_to_nbars[g])} rounds)")

    # ============================================================
    # 2) Run T1 analysis (same as original script 1)
    # ============================================================
    t1_vs_time = T1VsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs, False,
                         'None', run_name, FRIDGE, exp_name='ge', qubit=qubit, t1_slice='10us')
    _, _, amps_t1, gains_t1, rounds_t1, delay_times_t1 = t1_vs_time.run_t1_sweep_new(exp_extension='_ge', scaling=True, weighted_mean=True)

    # ============================================================
    # Helper: map gain values to nbar using the calibration
    # ============================================================
    def gain_to_nbar_lookup(gain_val, mapping, atol=1e-3):
        """Look up nbar for a gain value using the gain_to_mean_nbar mapping."""
        gain_val = float(gain_val)
        for g_key, nb in mapping.items():
            if abs(gain_val - g_key) < atol:
                return nb
        return np.nan

    def make_nbar_labels(gain_labels, mapping, atol=1e-3):
        """Convert a list of gain labels to nbar labels."""
        return [gain_to_nbar_lookup(g, mapping, atol=atol) for g in gain_labels]

    # ============================================================
    # 3) All the helper functions from original script 1
    # ============================================================

    def unique_gains_raw(gains_dict, q_key, round_to=6):
        """Get unique gain values present for a qubit (rounded)."""
        vals = []
        for entry in gains_dict.get(q_key, []):
            arr = np.asarray(entry, float).ravel()
            arr = arr[np.isfinite(arr)]
            if arr.size:
                vals.append(arr)
        if not vals:
            return np.array([])
        v = np.concatenate(vals)
        return np.unique(np.round(v, round_to))


    def filter_by_gains_same_format_tol(
            amps, gains, rounds, delay_times,
            gains_keep,
            *,
            atol=1e-3,
            rtol=1e-6,
    ):
        def _as_1d_float_array(x):
            return np.asarray(x, dtype=float).ravel()

        gains_keep = _as_1d_float_array(gains_keep)
        if gains_keep.size == 0:
            return {}, {}, {}, {}

        def _in_keep(g_arr):
            g_arr = np.asarray(g_arr, dtype=float).ravel()
            return np.any(np.isclose(g_arr[:, None], gains_keep[None, :], atol=atol, rtol=rtol), axis=1)

        amps_f, gains_f, rounds_f, delay_f = {}, {}, {}, {}

        for q in amps.keys():
            amps_q = amps.get(q, [])
            gains_q = gains.get(q, [])
            rounds_q = rounds.get(q, [])
            delay_q = delay_times.get(q, [])

            n = min(len(amps_q), len(gains_q), len(rounds_q), len(delay_q))
            if n == 0 or not (len(amps_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
                continue

            amps_out, gains_out, rounds_out, delay_out = [], [], [], []

            for i in range(n):
                a_samples = _as_1d_float_array(amps_q[i])
                if a_samples.size == 0:
                    continue

                g_i = gains_q[i]
                d_i = delay_q[i]

                g_arr = _as_1d_float_array(g_i) if isinstance(g_i, (list, tuple, np.ndarray)) else None
                if g_arr is None or g_arr.size == 1:
                    try:
                        g_scalar = float(g_arr[0] if (g_arr is not None and g_arr.size == 1) else g_i)
                    except Exception:
                        continue
                    g_arr = np.full(a_samples.shape, g_scalar, dtype=float)
                    scalar_gain_entry = True
                else:
                    scalar_gain_entry = False
                    if g_arr.size != a_samples.size:
                        continue

                d_arr = _as_1d_float_array(d_i) if isinstance(d_i, (list, tuple, np.ndarray)) else None
                if d_arr is None or d_arr.size == 1:
                    try:
                        d_scalar = float(d_arr[0] if (d_arr is not None and d_arr.size == 1) else d_i)
                    except Exception:
                        continue
                    d_arr = np.full(a_samples.shape, d_scalar, dtype=float)
                else:
                    if d_arr.size != a_samples.size:
                        continue

                finite = np.isfinite(a_samples) & np.isfinite(g_arr) & np.isfinite(d_arr)
                if not np.any(finite):
                    continue

                if scalar_gain_entry:
                    keep_entire = bool(np.any(np.isclose(g_arr[finite][0], gains_keep, atol=atol, rtol=rtol)))
                    if not keep_entire:
                        continue
                    a_keep = a_samples[finite]
                    g_keep = g_arr[finite]
                    d_keep = d_arr[finite]
                else:
                    m = finite & _in_keep(g_arr)
                    if not np.any(m):
                        continue
                    a_keep = a_samples[m]
                    g_keep = g_arr[m]
                    d_keep = d_arr[m]

                amps_out.append(a_keep)
                gains_out.append(g_keep)
                delay_out.append(d_keep)
                rounds_out.append(rounds_q[i])

            if amps_out:
                amps_f[q] = amps_out
                gains_f[q] = gains_out
                delay_f[q] = delay_out
                rounds_f[q] = rounds_out

        return amps_f, gains_f, rounds_f, delay_f


    def split_same_format_by_gain(
            amps_f, gains_f, rounds_f, delay_f,
            gains_list,
            *,
            atol=1e-3,
            rtol=1e-6,
    ):
        gains_list = np.asarray(gains_list, float).ravel()

        amps_by_gain, delay_by_gain, rounds_by_gain, labels = [], [], [], []

        for g_target in gains_list:
            amps_g, delay_g, rounds_g = {}, {}, {}

            for q in amps_f.keys():
                aL = amps_f.get(q, [])
                gL = gains_f.get(q, [])
                dL = delay_f.get(q, [])
                rL = rounds_f.get(q, [])

                n = min(len(aL), len(gL), len(dL), len(rL))
                if n == 0:
                    continue

                a_out, d_out, r_out = [], [], []
                for i in range(n):
                    a = np.asarray(aL[i], float).ravel()
                    gg = np.asarray(gL[i], float).ravel()
                    dd = np.asarray(dL[i], float).ravel()
                    if not (a.size == gg.size == dd.size):
                        continue

                    m = np.isfinite(a) & np.isfinite(gg) & np.isfinite(dd) & np.isclose(gg, g_target, atol=atol,
                                                                                        rtol=rtol)
                    if not np.any(m):
                        continue

                    a_out.append(a[m])
                    d_out.append(dd[m])
                    r_out.append(rL[i])

                if a_out:
                    amps_g[q] = a_out
                    delay_g[q] = d_out
                    rounds_g[q] = r_out

            amps_by_gain.append(amps_g)
            delay_by_gain.append(delay_g)
            rounds_by_gain.append(rounds_g)
            labels.append(float(g_target))

        return amps_by_gain, delay_by_gain, rounds_by_gain, labels


    def _flatten_finite_amp_samples_for_qubit(amps_g_dict, q_key):
        vals = []
        for entry in amps_g_dict.get(q_key, []):
            a = np.asarray(entry, float).ravel()
            a = a[np.isfinite(a)]
            if a.size:
                vals.append(a)
        if not vals:
            return np.array([], dtype=float)
        return np.concatenate(vals)


    def _collect_vals_for_gain(amps_g: dict, q_key):
        if q_key is None:
            if isinstance(amps_g, dict) and len(amps_g) == 1:
                q_use = next(iter(amps_g.keys()))
            else:
                return None, None
        else:
            q_use = q_key

        arr_list = amps_g.get(q_use, [])
        if not arr_list:
            return q_use, None

        vals = []
        for a in arr_list:
            a = np.asarray(a, float).ravel()
            a = a[np.isfinite(a)]
            if a.size:
                vals.append(a)

        if not vals:
            return q_use, None

        return q_use, np.concatenate(vals)


    # ============================================================
    # 4) MODIFIED plotting functions: x-axis = nbar instead of gain
    # ============================================================

    def plot_t1_boxplot_by_nbar(
            amps_list,
            gain_labels,
            nbar_labels,
            *,
            q_key,
            save_path,
            filename="t1_boxplot_by_nbar.png",
            title=None,
            ylabel=r"$T_1$ ($\mu$s)",
    ):
        """
        Recreates the T1 distribution boxplot but with average nbar on the x-axis
        instead of gain. Includes overlaid scatter points colored per gain/nbar.
        """
        os.makedirs(save_path, exist_ok=True)

        gains = np.asarray(gain_labels, dtype=float).ravel()
        nbars = np.asarray(nbar_labels, dtype=float).ravel()
        order = np.argsort(nbars)

        # Default color cycle matching the original plot
        default_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                          '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

        data_for_box = []
        nbar_tick_labels = []
        colors_used = []

        for idx, j in enumerate(order):
            nb = nbars[j]
            amps_g = amps_list[j]
            _, v = _collect_vals_for_gain(amps_g, q_key=q_key)
            if v is None or v.size == 0:
                continue
            data_for_box.append(v)
            nbar_tick_labels.append(f"{nb:.4g}")
            colors_used.append(default_colors[idx % len(default_colors)])

        if not data_for_box:
            print("No data for nbar boxplot.")
            return None

        fig, ax = plt.subplots(figsize=(8, 6))

        # Draw boxplots
        positions = list(range(1, len(data_for_box) + 1))
        bp = ax.boxplot(data_for_box, positions=positions, widths=0.5,
                        patch_artist=False, showfliers=False,
                        medianprops=dict(color='orange', linewidth=2))

        # Overlay scatter points colored per nbar group
        for i, (pos, vals, color) in enumerate(zip(positions, data_for_box, colors_used)):
            ax.scatter(np.full(vals.size, pos), vals, alpha=0.5, s=20, color=color,
                       edgecolors='none', zorder=3, label=f"$\\bar{{n}}$ = {nbar_tick_labels[i]}")
        ax.set_xticks(positions)
        ax.set_xticklabels(nbar_tick_labels)
        ax.set_xlabel(r"$\bar{n}$")
        ax.set_ylabel(ylabel)

        if title is None:
            title = f"$T_1$ distribution by $\\bar{{n}}$"
        ax.set_title(title)

        ax.legend(loc="upper right", title=r"$\bar{n}$", fontsize=9)
        #ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()

        out = os.path.join(save_path, filename)
        fig.savefig(out, dpi=500)
        plt.close(fig)
        print(f"Saved nbar boxplot: {out}")
        return out


    def plot_histograms_by_gain(
            amps_list,
            gain_labels,
            nbar_labels,
            *,
            title_prefix,
            xlabel,
            save_path,
            filename_prefix,
            q_key=None,
            bins="auto",
            xlim=None,
    ):
        """
        Saves one histogram per gain, with mean/median/±1σ marked.
        Also saves a relative error (CV = σ/mean) vs nbar summary plot.
        """
        os.makedirs(save_path, exist_ok=True)
        gains = np.asarray(gain_labels, dtype=float).ravel()
        nbars = np.asarray(nbar_labels, dtype=float).ravel()
        order = np.argsort(nbars)
        nbars_s = nbars[order]
        gains_s = gains[order]
        outs = []
        q_use_final = None

        nbars_for_summary = []
        means_for_summary = []
        stds_for_summary = []
        rel_errors_for_summary = []

        for j, (nb, g) in enumerate(zip(nbars_s, gains_s)):
            amps_g = amps_list[order[j]]
            q_use, v = _collect_vals_for_gain(amps_g, q_key=q_key)
            if q_use is None or v is None or v.size == 0:
                continue
            q_use_final = q_use
            mean = float(np.mean(v))
            median = float(np.median(v))
            std = float(np.std(v, ddof=1)) if v.size > 1 else 0.0
            rel_err = (std / abs(mean)) if abs(mean) > 0 else np.nan

            nbars_for_summary.append(nb)
            means_for_summary.append(mean)
            stds_for_summary.append(std)
            rel_errors_for_summary.append(rel_err)

            fig, ax = plt.subplots(figsize=(12, 6))
            ax.hist(v, bins=bins, alpha=0.6, edgecolor="black", label=f"N = {v.size}")
            ax.axvspan(mean - std, mean + std, color="red", alpha=0.10)
            ax.axvline(mean, color="red", linewidth=3, label=f"Mean = {mean:.4g}")
            ax.axvline(median, color="orange", linestyle="--", linewidth=3, label=f"Median = {median:.4g}")
            ax.axvline(mean - std, color="red", linestyle=":", linewidth=3, label=f"Mean − 1σ = {(mean - std):.4g}")
            ax.axvline(mean + std, color="red", linestyle=":", linewidth=3, label=f"Mean + 1σ = {(mean + std):.4g}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Count")
            ax.set_title(
                f"{title_prefix} — nbar = {nb:.4g}\n"
                f"Qubit: {q_use}    |    "
                f"σ = {std:.4g}    |    "
                f"Relative Error (σ/mean) = {rel_err:.2%}"
            )
            ax.grid(True, alpha=0.3)
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.legend(loc="upper right")
            fig.tight_layout()
            out = os.path.join(save_path, f"{filename_prefix}_nbar_{nb:.4f}_q{q_use}.png")
            fig.savefig(out, dpi=300)
            plt.close(fig)
            outs.append(out)

        # Relative error vs nbar summary plot
        if len(nbars_for_summary) > 0:
            nbars_arr = np.asarray(nbars_for_summary, float)
            rel_arr = np.asarray(rel_errors_for_summary, float)

            fig_re, ax_re = plt.subplots(figsize=(8, 5))
            ax_re.plot(nbars_arr, rel_arr * 100, 'o-', color='steelblue', markersize=8, linewidth=2)

            for nv, rv in zip(nbars_arr, rel_arr):
                ax_re.annotate(f"{rv:.1%}", (nv, rv * 100),
                               textcoords="offset points", xytext=(0, 10),
                               ha='center', fontsize=9)

            ax_re.set_xlabel(r"$\bar{n}$")
            ax_re.set_ylabel("Relative Error σ/mean (%)")
            ax_re.set_title(
                f"{title_prefix}: Relative Error vs nbar\n"
                f"Qubit: {q_use_final}"
            )
            ax_re.grid(True, alpha=0.3)
            fig_re.tight_layout()

            out_re = os.path.join(save_path, f"{filename_prefix}_relative_error_vs_nbar_q{q_use_final}.png")
            fig_re.savefig(out_re, dpi=300)
            plt.close(fig_re)
            outs.append(out_re)

        return outs


    def plot_histograms_by_gain_logscale(
            amps_list,
            gain_labels,
            nbar_labels,
            *,
            title_prefix,
            xlabel,
            save_path,
            filename_prefix,
            q_key=None,
            bins="auto",
            xlim=None,
    ):
        """
        Same as plot_histograms_by_gain but takes ln() of values first.
        X-axis labels use nbar.
        """
        os.makedirs(save_path, exist_ok=True)
        gains = np.asarray(gain_labels, dtype=float).ravel()
        nbars = np.asarray(nbar_labels, dtype=float).ravel()
        order = np.argsort(nbars)
        nbars_s = nbars[order]
        gains_s = gains[order]
        outs = []
        q_use_final = None

        nbars_for_summary = []
        stds_log_for_summary = []

        for j, (nb, g) in enumerate(zip(nbars_s, gains_s)):
            amps_g = amps_list[order[j]]
            q_use, v = _collect_vals_for_gain(amps_g, q_key=q_key)
            if q_use is None or v is None or v.size == 0:
                continue

            v = v[v > 0]
            if v.size == 0:
                continue

            q_use_final = q_use
            lnv = np.log(v)

            mean_log = float(np.mean(lnv))
            median_log = float(np.median(lnv))
            std_log = float(np.std(lnv, ddof=1)) if lnv.size > 1 else 0.0

            geom_mean = np.exp(mean_log)
            exp_median = np.exp(median_log)
            exp_upper = np.exp(mean_log + std_log)
            exp_lower = np.exp(mean_log - std_log)

            nbars_for_summary.append(nb)
            stds_log_for_summary.append(std_log)

            fig, ax = plt.subplots(figsize=(12, 6))

            ax.hist(lnv, bins=bins, alpha=0.6, edgecolor="black", label=f"N = {v.size}")
            ax.axvspan(mean_log - std_log, mean_log + std_log, color="red", alpha=0.10)
            ax.axvline(mean_log, color="red", linewidth=3,
                       label=f"Mean(ln) = {mean_log:.4g}  →  e^mean = {geom_mean:.4g}")
            ax.axvline(median_log, color="orange", linestyle="--", linewidth=3,
                       label=f"Median(ln) = {median_log:.4g}  →  e^med = {exp_median:.4g}")
            ax.axvline(mean_log - std_log, color="red", linestyle=":", linewidth=3,
                       label=f"Mean−1σ = {mean_log - std_log:.4g}  →  e^ = {exp_lower:.4g}")
            ax.axvline(mean_log + std_log, color="red", linestyle=":", linewidth=3,
                       label=f"Mean+1σ = {mean_log + std_log:.4g}  →  e^ = {exp_upper:.4g}")

            ax.set_xlabel(f"ln({xlabel})")
            ax.set_ylabel("Count")
            ax.set_title(
                f"ln({title_prefix}) — nbar = {nb:.4g}\n"
                f"Qubit: {q_use}    |    "
                f"σ_log = {std_log:.4g}"
            )
            ax.grid(True, alpha=0.3)
            if xlim is not None:
                ax.set_xlim(xlim)
            ax.legend(loc="upper right", fontsize=8)
            fig.tight_layout()

            out = os.path.join(save_path, f"{filename_prefix}_logscale_nbar_{nb:.4f}_q{q_use}.png")
            fig.savefig(out, dpi=300)
            plt.close(fig)
            outs.append(out)

        # σ_log vs nbar summary plot
        if len(nbars_for_summary) > 0:
            nbars_arr = np.asarray(nbars_for_summary, float)
            std_arr = np.asarray(stds_log_for_summary, float)

            fig_s, ax_s = plt.subplots(figsize=(8, 5))
            ax_s.plot(nbars_arr, std_arr, 'o-', color='steelblue', markersize=8, linewidth=2)

            for nv, sv in zip(nbars_arr, std_arr):
                ax_s.annotate(f"{sv:.4f}", (nv, sv),
                              textcoords="offset points", xytext=(0, 10),
                              ha='center', fontsize=9)

            ax_s.set_xlabel(r"$\bar{n}$")
            ax_s.set_ylabel("σ in log-space")
            ax_s.set_title(
                f"ln({title_prefix}): σ_log vs nbar\n"
                f"Qubit: {q_use_final}\n"
                f"(This should be identical for T1 and Γ)"
            )
            ax_s.grid(True, alpha=0.3)
            fig_s.tight_layout()

            out_s = os.path.join(save_path, f"{filename_prefix}_logscale_sigma_vs_nbar_q{q_use_final}.png")
            fig_s.savefig(out_s, dpi=300)
            plt.close(fig_s)
            outs.append(out_s)

        return outs


    def plot_mean_median_std_by_gain(
            amps_list,
            gain_labels,
            nbar_labels,
            *,
            title,
            save_path,
            filename,
            q_key=None,
            ylabel="Amplitude (a.u.)",
    ):
        gains = np.asarray(gain_labels, dtype=float).ravel()
        nbars = np.asarray(nbar_labels, dtype=float).ravel()

        means = np.full(nbars.shape, np.nan, dtype=float)
        medians = np.full(nbars.shape, np.nan, dtype=float)
        stds = np.full(nbars.shape, np.nan, dtype=float)

        for j, amps_g in enumerate(amps_list):
            q_use, v = _collect_vals_for_gain(amps_g, q_key=q_key)
            if q_use is None or v is None or v.size == 0:
                continue

            means[j] = np.mean(v)
            medians[j] = np.median(v)
            stds[j] = np.std(v, ddof=1) if v.size > 1 else 0.0

        order = np.argsort(nbars)
        nbars_s = nbars[order]
        means_s = means[order]
        medians_s = medians[order]
        stds_s = stds[order]

        os.makedirs(save_path, exist_ok=True)

        plt.figure()
        plt.errorbar(nbars_s, means_s, yerr=stds_s, fmt="o", capsize=4, label="Mean ± 1σ")
        plt.plot(nbars_s, medians_s, marker="s", linestyle="-", label="Median")
        plt.xlabel(r"$\bar{n}$")
        plt.ylabel(ylabel)
        plt.title(title if q_key is None else f"{title}\nQubit: {q_key}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out = os.path.join(save_path, filename)
        plt.savefig(out, dpi=300)
        plt.close()
        return out


    def plot_mean_median_std_multi_gain_by_dataset_index(
            amps_list,
            *,
            gains,
            nbar_labels,
            save_path,
            q_key,
            colors,
            filename_prefix,
            title_prefix="",
    ):
        """
        Per-dataset-index summary stats, with legend labels showing nbar instead of gain.
        """
        os.makedirs(save_path, exist_ok=True)

        if colors is None or len(colors) < len(amps_list):
            raise ValueError(
                f"`colors` must be a list with >= number of gains ({len(amps_list)}). "
            )

        fig, ax = plt.subplots(figsize=(10, 5))

        for k, amps_g in enumerate(amps_list):
            aL = amps_g.get(q_key, [])
            if not aL:
                continue

            xs, means, meds, stds = [], [], [], []

            for i, arr in enumerate(aL):
                a = np.asarray(arr, float).ravel()
                a = a[np.isfinite(a)]
                if a.size == 0:
                    continue

                xs.append(i)
                means.append(np.mean(a))
                meds.append(np.median(a))
                stds.append(np.std(a, ddof=1) if a.size > 1 else 0.0)

            if not xs:
                continue

            xs = np.asarray(xs, int)
            means = np.asarray(means, float)
            meds = np.asarray(meds, float)
            stds = np.asarray(stds, float)

            c = colors[k]
            label = f"nbar={nbar_labels[k]:.4g}"

            ax.errorbar(
                xs, means, yerr=stds,
                fmt="o-",
                capsize=3,
                color=c,
                label=label + " (mean±1σ)"
            )

            ax.plot(
                xs, meds,
                linestyle="--",
                marker="s",
                color=c,
                alpha=0.9,
                label=label + " (median)"
            )

        ax.set_xlabel("Dataset index")
        ax.set_ylabel("Value")
        if title_prefix:
            ax.set_title(f"{title_prefix} summary stats: mean, median, ±1σ ({q_key})")
        else:
            ax.set_title(f"Summary stats: mean, median, ±1σ ({q_key})")

        ax.grid(True, alpha=0.3)
        ax.legend(ncol=2, fontsize=8)
        fig.tight_layout()

        out = os.path.join(save_path, f"{filename_prefix}_{q_key}.png")
        fig.savefig(out, dpi=200)
        plt.close(fig)
        return out


    # ============================================================
    # 5) Filter, split, and plot (same pipeline as original)
    # ============================================================

    q = qubits[0]

    print("\nUnique gains present in RAW data for this qubit:")
    print(unique_gains_raw(gains_t1, q, round_to=6))

    gains_keep = [0, 0.034243, 0.054386, 0.080571, 0.0987]

    # Re-filter with tolerance
    amps_f, gains_f, rounds_f, delay_f = filter_by_gains_same_format_tol(
        amps_t1, gains_t1, rounds_t1, delay_times_t1,
        gains_keep=gains_keep,
        atol=1e-3, rtol=1e-6
    )

    print("Unique gains present AFTER filtering:")
    print(unique_gains_raw(gains_f, q, round_to=6))

    # Split per gain for plotting
    amps_list, delay_list, rounds_list, gain_labels = split_same_format_by_gain(
        amps_f, gains_f, rounds_f, delay_f,
        gains_list=gains_keep,
        atol=1e-3, rtol=1e-6
    )

    # Convert gain_labels to nbar_labels
    nbar_labels = make_nbar_labels(gain_labels, gain_to_mean_nbar)
    print("\n=== Gain labels -> nbar labels ===")
    for g, nb in zip(gain_labels, nbar_labels):
        print(f"  gain={g:.6f}  ->  nbar={nb:.4f}" if np.isfinite(nb) else f"  gain={g:.6f}  ->  nbar=NaN (no calibration!)")

    # Plot (multi-gain) — note: the scatter plot still uses gain internally,
    # but the histograms/summary plots below will use nbar on x-axis
    fig, ax, amps_list_g, amps_list_t1_fit, gain_labels = t1_vs_time.plot_t1_scatter_multi_gain_by_dataset_index(
        amps_list,
        delay_list,
        gain_labels,
        f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis_gain_nbar/',
        q_key=q,
        return_distributions=True
    )

    # Recompute nbar_labels in case gain_labels was modified by the plotting function
    nbar_labels = make_nbar_labels(gain_labels, gain_to_mean_nbar)

    save_dir = f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis_gain_nbar/'

    # ============================================================
    # 6) NEW: T1 boxplot with nbar on x-axis (replaces gain x-axis)
    # ============================================================
    plot_t1_boxplot_by_nbar(
        amps_list_t1_fit,
        gain_labels,
        nbar_labels,
        q_key=q,
        save_path=save_dir,
        filename=f"t1_boxplot_by_nbar_Q{q}.pdf",
        title=f"$T_1$ distribution by $\\bar{{n}}$",
        ylabel=r"$T_1$ ($\mu$s)",
    )

    # ============================================================
    # 7) Histogram and summary plots with nbar x-axis
    # ============================================================

    # T1 histograms (one per nbar)
    plot_histograms_by_gain(
        amps_list_t1_fit,
        gain_labels,
        nbar_labels,
        title_prefix="T1 Distribution",
        xlabel=r"$T_1$ ($\mu$s)",
        save_path=save_dir,
        filename_prefix=f"t1_hist_{q}",
        q_key=q,
        bins="auto",
    )
    plot_histograms_by_gain_logscale(
        amps_list_t1_fit,
        gain_labels,
        nbar_labels,
        title_prefix="T1 Distribution",
        xlabel=r"$T_1$ ($\mu$s)",
        save_path=save_dir,
        filename_prefix=f"t1_hist_{q}",
        q_key=q,
        bins="auto",
    )

    # Gamma histograms (one per nbar)
    plot_histograms_by_gain(
        amps_list_g,
        gain_labels,
        nbar_labels,
        title_prefix="Gamma (1/T1) Distribution",
        xlabel=r"$\Gamma_1$ (1/$\mu$s)",
        save_path=save_dir,
        filename_prefix=f"gamma_hist_{q}",
        q_key=q,
        bins="auto",
    )
    plot_histograms_by_gain_logscale(
        amps_list_g,
        gain_labels,
        nbar_labels,
        title_prefix="Gamma (1/T1) Distribution",
        xlabel=r"$\Gamma_1$ (1/$\mu$s)",
        save_path=save_dir,
        filename_prefix=f"gamma_hist_{q}",
        q_key=q,
        bins="auto",
    )

    # Mean/median/std vs nbar
    plot_mean_median_std_by_gain(
        amps_list_t1_fit, gain_labels, nbar_labels,
        title="T1: Mean / Median with ±1σ",
        save_path=save_dir,
        filename=f"t1_mean_median_std_vs_nbar_{q}.png",
        q_key=q,
        ylabel=r"$T_1$ ($\mu$s)",
    )

    plot_mean_median_std_by_gain(
        amps_list_g, gain_labels, nbar_labels,
        title="Gamma: Mean / Median with ±1σ",
        save_path=save_dir,
        filename=f"gamma_mean_median_std_vs_nbar_{q}.png",
        q_key=q,
        ylabel=r"$\Gamma_1$ (1/$\mu$s)",
    )
    print("\n=== Total datapoints per nbar ===")
    for k, (g, nb) in enumerate(zip(gain_labels, nbar_labels)):
        _, v = _collect_vals_for_gain(amps_list_t1_fit[k], q_key=q)
        count = v.size if v is not None else 0
        print(f"  nbar={nb:.4f}  (gain={g:.6f})  ->  {count} datapoints")