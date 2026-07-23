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

# How often to place an x-axis tick label (every Nth nbar) so the axis with
# 50+ points does not get crowded.
LABEL_EVERY = 5

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
    # 2) Run T1 analysis (same as original script)
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
        best_key = None
        best_diff = atol
        for g_key, nb in mapping.items():
            diff = abs(gain_val - g_key)
            if diff < best_diff:
                best_diff = diff
                best_key = g_key
        return mapping[best_key] if best_key is not None else np.nan

    def make_nbar_labels(gain_labels, mapping, atol=1e-3):
        """Convert a list of gain labels to nbar labels."""
        return [gain_to_nbar_lookup(g, mapping, atol=atol) for g in gain_labels]

    # ============================================================
    # 3) Helper functions from original script (filtering / splitting)
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
    # 4a) Fit T1 curves -> T1 and Gamma distributions per gain.
    #     Replaces plot_t1_scatter_multi_gain_by_dataset_index so that NO
    #     side-effect files are written (no per_dataset_fits_by_gain folder,
    #     no boxplots, no summary csv/npz). Same return format as that method's
    #     return_distributions=True output.
    # ============================================================
    def fit_t1_gamma_distributions(
            amps_list, delay_list, gain_labels,
            *, q_key,
            min_points=6, maxfev=20000,
            reject_nonpositive=True, max_T1_us=500.0,
    ):
        from scipy.optimize import curve_fit

        def _initial_guess(x, y):
            x = np.asarray(x, float)
            y = np.asarray(y, float)
            a_guess = float(np.nanmax(y) - np.nanmin(y)) if y.size else 1.0
            c_guess = float((x[-1] - x[0]) / 5.0) if len(x) > 1 else 1.0
            d_guess = float(np.nanmin(y)) if y.size else 0.0
            return [a_guess, 0.0, max(c_guess, 1e-6), d_guess]

        lower = [-np.inf, -np.inf, 0.0, -np.inf]
        upper = [np.inf, np.inf, np.inf, np.inf]

        amps_list_t1_out, amps_list_g_out = [], []
        for gi in range(len(gain_labels)):
            amps_dict = amps_list[gi]
            delay_dict = delay_list[gi]
            t1_vals, gamma_vals = [], []

            if q_key in amps_dict and q_key in delay_dict:
                aL = amps_dict[q_key]
                dL = delay_dict[q_key]
                for di in range(min(len(aL), len(dL))):
                    x = np.asarray(dL[di], float).ravel()
                    y = np.asarray(aL[di], float).ravel()
                    if x.size == 0 or x.size != y.size:
                        continue
                    good = np.isfinite(x) & np.isfinite(y)
                    x = x[good]
                    y = y[good]
                    if x.size < min_points:
                        continue
                    order = np.argsort(x)
                    x = x[order]
                    y = y[order]
                    try:
                        popt, _ = curve_fit(
                            t1_vs_time.exponential, x, y,
                            p0=_initial_guess(x, y),
                            bounds=(lower, upper), method="trf", maxfev=maxfev,
                        )
                    except Exception:
                        continue
                    T1_est = float(popt[2])
                    if not np.isfinite(T1_est):
                        continue
                    if reject_nonpositive and T1_est <= 0:
                        continue
                    if max_T1_us is not None and T1_est > float(max_T1_us):
                        continue
                    t1_vals.append(T1_est)
                    gamma_vals.append(1.0 / T1_est)

            amps_list_t1_out.append({q_key: [np.asarray(t1_vals, float)]})
            amps_list_g_out.append({q_key: [np.asarray(gamma_vals, float)]})

        return amps_list_g_out, amps_list_t1_out


    # ============================================================
    # 4) NEW: scatter plot of a distribution vs nbar
    #    (one scatter point per datapoint, at the true nbar x-position;
    #     only every LABEL_EVERY-th nbar gets an x tick label)
    # ============================================================
    def plot_scatter_by_nbar(
            amps_list,
            nbar_labels,
            *,
            q_key,
            save_path,
            filename,
            ylabel,
            title,
            label_every=LABEL_EVERY,
            point_color='steelblue',
            median_color='crimson',
            show_median_line=True,
    ):
        """
        Scatter every datapoint in each nbar group against its nbar value.
        The x-axis is a genuine linear scale in nbar (points sit at their true
        nbar value); only every `label_every`-th nbar receives an x-axis tick
        label so the axis stays readable with 50+ points.
        A per-nbar median trend line is overlaid unless show_median_line=False.
        """
        os.makedirs(save_path, exist_ok=True)

        nbars = np.asarray(nbar_labels, dtype=float).ravel()
        order = np.argsort(nbars)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.set_xscale('linear')

        used_nbars = []
        median_x, median_y = [], []
        n_total = 0

        for j in order:
            nb = nbars[j]
            if not np.isfinite(nb):
                continue
            amps_g = amps_list[j]
            _, v = _collect_vals_for_gain(amps_g, q_key=q_key)
            if v is None or v.size == 0:
                continue

            ax.scatter(np.full(v.size, nb), v, alpha=0.4, s=18,
                       color=point_color, edgecolors='none', zorder=3)
            median_x.append(nb)
            median_y.append(float(np.median(v)))
            used_nbars.append(nb)
            n_total += v.size

        if not used_nbars:
            print(f"No data for scatter plot: {filename}")
            plt.close(fig)
            return None

        # Overlay per-nbar median trend (the red line), unless suppressed
        if show_median_line:
            ax.plot(median_x, median_y, '-', color=median_color, linewidth=1.5,
                    marker='D', markersize=5, zorder=4, label='per-$\\bar{n}$ median')

        # Label only every Nth nbar to avoid crowding
        used_nbars = np.asarray(sorted(set(used_nbars)))
        tick_positions = used_nbars[::label_every]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels([f"{nb:.3g}" for nb in tick_positions], rotation=45, ha='right')

        ax.set_xlabel(r"$\bar{n}$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title}\nQubit {q_key}  |  {len(used_nbars)} $\\bar{{n}}$ values  |  {n_total} points")
        ax.grid(True, alpha=0.3)
        if show_median_line:
            ax.legend(loc="best")
        fig.tight_layout()

        out = os.path.join(save_path, filename)
        fig.savefig(out, dpi=300)
        plt.close(fig)
        print(f"Saved scatter vs nbar: {out}")
        return out


    # ============================================================
    # 5) Use ALL nbar/gain datapoints (the full 2D sweep), not a
    #    hand-picked subset. Derive the gain step size / list from data.
    # ============================================================

    q = qubits[0]

    all_gains = unique_gains_raw(gains_t1, q, round_to=6)
    print(f"\nUnique gains present in RAW data for qubit {q} ({all_gains.size} values):")
    print(all_gains)
    if all_gains.size > 1:
        steps = np.diff(all_gains)
        print(f"Gain step size (median of diffs): {np.median(steps):.6f}  "
              f"(min={steps.min():.6f}, max={steps.max():.6f})")

    gains_keep = all_gains

    # Re-filter with tolerance (keeps everything, but normalizes structure)
    amps_f, gains_f, rounds_f, delay_f = filter_by_gains_same_format_tol(
        amps_t1, gains_t1, rounds_t1, delay_times_t1,
        gains_keep=gains_keep,
        atol=1e-3, rtol=1e-6
    )

    print("Unique gains present AFTER filtering:")
    print(unique_gains_raw(gains_f, q, round_to=6))

    # Split per gain for fitting
    amps_list, delay_list, rounds_list, gain_labels = split_same_format_by_gain(
        amps_f, gains_f, rounds_f, delay_f,
        gains_list=gains_keep,
        atol=1e-3, rtol=1e-6
    )

    # Convert gain_labels to nbar_labels
    nbar_labels = make_nbar_labels(gain_labels, gain_to_mean_nbar)
    print("\n=== Gain labels -> nbar labels ===")
    for g, nb in zip(gain_labels, nbar_labels):
        print(f"  gain={g:.6f}  ->  nbar={nb:.4f}" if np.isfinite(nb)
              else f"  gain={g:.6f}  ->  nbar=NaN (no calibration!)")

    # Output folder: separate from the original analysis_gain_nbar
    save_dir = f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis_more_nbars/'
    os.makedirs(save_dir, exist_ok=True)

    # Run the T1 fits to obtain gamma (1/T1) and T1 distributions per gain.
    # This local fit produces amps_list_g (gamma) and amps_list_t1_fit (T1)
    # WITHOUT writing any per_dataset_fits_by_gain / boxplot / summary files.
    # gain_labels order is preserved, so nbar_labels stays aligned.
    amps_list_g, amps_list_t1_fit = fit_t1_gamma_distributions(
        amps_list, delay_list, gain_labels, q_key=q,
    )

    # ============================================================
    # 6) The only deliverable plots: T1 vs nbar and Gamma vs nbar
    # ============================================================
    # T1 vs nbar (linear nbar x-axis) — with the red per-nbar median line.
    plot_scatter_by_nbar(
        amps_list_t1_fit,
        nbar_labels,
        q_key=q,
        save_path=save_dir,
        filename=f"t1_vs_nbar_scatter_Q{q}.png",
        ylabel=r"$T_1$ ($\mu$s)",
        title=r"$T_1$ vs $\bar{n}$",
        show_median_line=True,
    )

    # T1 vs nbar (linear nbar x-axis) — scatter only, no red median line.
    plot_scatter_by_nbar(
        amps_list_t1_fit,
        nbar_labels,
        q_key=q,
        save_path=save_dir,
        filename=f"t1_vs_nbar_scatter_Q{q}_no_median.png",
        ylabel=r"$T_1$ ($\mu$s)",
        title=r"$T_1$ vs $\bar{n}$",
        show_median_line=False,
    )

    plot_scatter_by_nbar(
        amps_list_g,
        nbar_labels,
        q_key=q,
        save_path=save_dir,
        filename=f"gamma_vs_nbar_scatter_Q{q}.png",
        ylabel=r"$\Gamma_1$ (1/$\mu$s)",
        title=r"$\Gamma_1$ (1/$T_1$) vs $\bar{n}$",
    )

    print("\n=== Total datapoints per nbar ===")
    for k, (g, nb) in enumerate(zip(gain_labels, nbar_labels)):
        _, v = _collect_vals_for_gain(amps_list_t1_fit[k], q_key=q)
        count = v.size if v is not None else 0
        nb_str = f"{nb:.4f}" if np.isfinite(nb) else "NaN"
        print(f"  nbar={nb_str}  (gain={g:.6f})  ->  {count} datapoints")