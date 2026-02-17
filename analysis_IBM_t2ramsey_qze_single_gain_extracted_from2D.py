from analysis_006_T1_vs_time_plots import T1VsTime
from analysis_007_T2R_vs_time_plots import T2rVsTime
from analysis_008_T2E_vs_time_plots import T2eVsTime
from analysis_003_q_freqs_vs_time_plots import QubitFreqsVsTime

# Configuration
save_figs = True
save_individual_qspec = False
figure_quality = 100
final_figure_quality = 200
FRIDGE = "QUIET"
qubits = [4]
path = '2d_higher_n_bar'

for qubit in qubits:
    # run_name = f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/'
    # top_folder_dates = [f'qubit_{qubit}round{round}' for round in range(113)]
    run_name = f'bob_run_started_Aug_23/squill/{path}/all_qubits/'
    from pathlib import Path

    p = Path('M:/_Data/20250822 - Olivia/' + run_name)
    top_folder_dates = [item.name for item in p.iterdir() if item.is_dir() and "qubit" in item.name]


    # # T1 Analysis
    t1_vs_time = T1VsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs, False,
                         'None', run_name, FRIDGE, exp_name='ge', qubit=qubit, t1_slice='10us')
    _, _, amps_t1, gains_t1, rounds_t1, delay_times_t1 = t1_vs_time.run_t1_sweep_new(exp_extension='_ge', scaling=True, weighted_mean=True)

    import numpy as np


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
            atol=1e-3,  # <-- IMPORTANT: allow matching like 0.0800000002
            rtol=1e-6,
    ):
        """
        Same as your filter_by_gains_same_format, but with a default tolerance
        that will actually match float-ish gain values.
        """

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
        """
        Split filtered dicts into list-of-dicts, one per gain, so the plotting function
        can truly plot multiple gains.
        """
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


    import os
    import numpy as np
    import matplotlib.pyplot as plt


    def _flatten_finite_amp_samples_for_qubit(amps_g_dict, q_key):
        """
        amps_g_dict: dict like amps_g[q] = [array1, array2, ...] (per dataset)
        Returns 1D array of all finite samples for the qubit across datasets.
        """
        vals = []
        for entry in amps_g_dict.get(q_key, []):
            a = np.asarray(entry, float).ravel()
            a = a[np.isfinite(a)]
            if a.size:
                vals.append(a)
        if not vals:
            return np.array([], dtype=float)
        return np.concatenate(vals)


    def save_gain_summary_stats_plot(
            amps_by_gain_list,
            gain_labels,
            q_key,
            save_dir,
            filename=None,
    ):
        """
        Creates and saves a plot with:
          - mean with ±1 std error bars
          - median markers
        for each gain, using amplitude samples aggregated across datasets.
        """
        os.makedirs(save_dir, exist_ok=True)

        means, stds, medians, gains_used = [], [], [], []

        for amps_g, g in zip(amps_by_gain_list, gain_labels):
            a = _flatten_finite_amp_samples_for_qubit(amps_g, q_key)
            if a.size == 0:
                continue
            gains_used.append(float(g))
            means.append(float(np.mean(a)))
            stds.append(float(np.std(a, ddof=1)) if a.size > 1 else 0.0)
            medians.append(float(np.median(a)))

        if not gains_used:
            print(f"[summary plot] No finite amplitude data found for q_key={q_key}. Nothing saved.")
            return

        x = np.asarray(gains_used, float)
        means = np.asarray(means, float)
        stds = np.asarray(stds, float)
        medians = np.asarray(medians, float)

        # Sort by gain so the plot reads nicely
        order = np.argsort(x)
        x, means, stds, medians = x[order], means[order], stds[order], medians[order]

        plt.figure()
        plt.errorbar(x, means, yerr=stds, fmt='o', capsize=4, label='Mean ± 1σ')
        plt.plot(x, medians, 'x', label='Median')

        plt.xlabel("Gain")
        plt.ylabel("Amplitude")
        plt.title(f"{q_key}: Mean/Median amplitude vs Gain (±1σ)")
        plt.legend()
        plt.tight_layout()

        if filename is None:
            filename = f"{q_key}_gain_summary_mean_median_std.png"
        out_path = os.path.join(save_dir, filename)
        plt.savefig(out_path, dpi=200)
        plt.close()
        print(f"[summary plot] Saved: {out_path}")


    # -------------------- USE IT --------------------

    q = qubits[0]

    print("Unique gains present in RAW data for this qubit:")
    print(unique_gains_raw(gains_t1, q, round_to=6))

    # Re-filter with tolerance
    amps_f, gains_f, rounds_f, delay_f = filter_by_gains_same_format_tol(
        amps_t1, gains_t1, rounds_t1, delay_times_t1,
        gains_keep=[0.0,0.034243,0.054386, 0.080571, 0.0987],
        atol=1e-3, rtol=1e-6
    )

    print("Unique gains present AFTER filtering:")
    print(unique_gains_raw(gains_f, q, round_to=6))

    # Split per gain for plotting
    amps_list, delay_list, rounds_list, gain_labels = split_same_format_by_gain(
        amps_f, gains_f, rounds_f, delay_f,
        gains_list=[0.0,0.034243,0.054386, 0.080571, 0.0987],
        atol=1e-3, rtol=1e-6
    )

    # Plot (multi-gain) — note: plotting function doesn't use gains_f dict; it uses labels
    t1_vs_time.plot_t1_scatter_multi_gain_by_dataset_index(
        amps_list,
        delay_list,
        gains=gain_labels,
        save_path=f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis_gain/',
        q_key=q
    )

    import os
    import numpy as np
    import matplotlib.pyplot as plt


    def plot_mean_median_std_multi_gain_by_dataset_index(
            amps_list,
            *,
            gains,
            save_path,
            q_key,
            colors,
            filename_prefix,
            title_prefix="",
    ):
        """
        For each gain (amps_list[k]), plot per-dataset-index summary stats of amps:
          - mean with ±1 std error bars
          - median as a line/marker
        Saves one figure to: save_path/{filename_prefix}_{q_key}.png

        amps_list format (as produced by split_same_format_by_gain):
          amps_list[gain_idx][q_key] -> list of 1D arrays, one per dataset index
        """
        os.makedirs(save_path, exist_ok=True)

        # Basic validation
        if colors is None or len(colors) < len(amps_list):
            raise ValueError(
                f"`colors` must be a list with >= number of gains ({len(amps_list)}). "
                "Use the same colors list you used for the boxplot."
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
            label = f"gain={gains[k]:.6g}" if gains is not None else f"gain_idx={k}"

            # Mean ± 1σ
            ax.errorbar(
                xs, means, yerr=stds,
                fmt="o-",  # markers + line
                capsize=3,
                color=c,
                label=label + " (mean±1σ)"
            )

            # Median (same color, different marker/linestyle)
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

    def plot_mean_median_std_by_gain(
        amps_list,
        gain_labels,
        *,
        title,
        save_path,
        filename,
        q_key=None,
    ):
        """
        amps_list: list of dicts, length = #gains.
                  amps_list[j] is dict {q_key: [arrays_per_dataset_index]}
                  like produced by split_same_format_by_gain.
        gain_labels: list of float gains, same length as amps_list.
        Saves a figure with mean ± 1σ error bars and median vs gain.
        """
        gains = np.asarray(gain_labels, dtype=float).ravel()

        means = np.full(gains.shape, np.nan, dtype=float)
        medians = np.full(gains.shape, np.nan, dtype=float)
        stds = np.full(gains.shape, np.nan, dtype=float)

        for j, amps_g in enumerate(amps_list):
            if q_key is None:
                # If not provided, try to infer the only key
                if isinstance(amps_g, dict) and len(amps_g) == 1:
                    q_use = next(iter(amps_g.keys()))
                else:
                    continue
            else:
                q_use = q_key

            arr_list = amps_g.get(q_use, [])
            if not arr_list:
                continue

            # Flatten all samples for this (gain, qubit) across dataset indices
            vals = []
            for a in arr_list:
                a = np.asarray(a, float).ravel()
                a = a[np.isfinite(a)]
                if a.size:
                    vals.append(a)

            if not vals:
                continue

            v = np.concatenate(vals)
            means[j] = np.mean(v)
            medians[j] = np.median(v)
            stds[j] = np.std(v, ddof=1) if v.size > 1 else 0.0

        # Sort by gain for nicer plotting
        order = np.argsort(gains)
        gains_s = gains[order]
        means_s = means[order]
        medians_s = medians[order]
        stds_s = stds[order]

        os.makedirs(save_path, exist_ok=True)

        plt.figure()
        plt.errorbar(gains_s, means_s, yerr=stds_s, fmt='o', capsize=4, label='Mean ± 1σ')
        plt.plot(gains_s, medians_s, marker='s', linestyle='-', label='Median')
        plt.xlabel("Gain")
        plt.ylabel("Amplitude (a.u.)")
        plt.title(title if q_key is None else f"{title}\nQubit: {q_key}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()

        out = os.path.join(save_path, filename)
        plt.savefig(out, dpi=300)
        plt.close()
        return out


        q = qubits[0]

        print("Unique gains present in RAW data for this qubit:")
        print(unique_gains_raw(gains_t1, q, round_to=6))

        amps_f, gains_f, rounds_f, delay_f = filter_by_gains_same_format_tol(
            amps_t1, gains_t1, rounds_t1, delay_times_t1,
            gains_keep=[0.0, 0.034243, 0.054386, 0.080571, 0.0987],
            atol=1e-3, rtol=1e-6
        )

        print("Unique gains present AFTER filtering:")
        print(unique_gains_raw(gains_f, q, round_to=6))

        amps_list, delay_list, rounds_list, gain_labels = split_same_format_by_gain(
            amps_f, gains_f, rounds_f, delay_f,
            gains_list=[0.0, 0.034243, 0.054386, 0.080571, 0.0987],
            atol=1e-3, rtol=1e-6
        )

        # Your existing plot (box/scatter/whatever this function saves)
        save_dir = f'M:/_Data/20250822 - Olivia/bob_run_started_Feb_11/squill/{path}/all_qubits/analysis_gain/'
        t1_vs_time.plot_t1_scatter_multi_gain_by_dataset_index(
            amps_list,
            delay_list,
            gains=gain_labels,
            save_path=save_dir,
            q_key=q
        )

        # NEW: mean/median/std plot for T1
        plot_mean_median_std_by_gain(
            amps_list,
            gain_labels,
            title="T1: Mean / Median with ±1σ",
            save_path=save_dir,
            filename=f"t1_mean_median_std_{q}.png",
            q_key=q
        )

        # -------------------- GAMMA PIPELINE (mirror T1 + new summary plot) --------------------
        # Assumes you have analogous dicts: amps_gamma, gains_gamma, rounds_gamma, delay_times_gamma
        # If your gamma plot function is different, keep it; we just add the summary plot.

        print("Unique gains present in RAW GAMMA data for this qubit:")
        print(unique_gains_raw(gains_gamma, q, round_to=6))

        amps_f_g, gains_f_g, rounds_f_g, delay_f_g = filter_by_gains_same_format_tol(
            amps_gamma, gains_gamma, rounds_gamma, delay_times_gamma,
            gains_keep=[0.0, 0.034243, 0.054386, 0.080571, 0.0987],
            atol=1e-3, rtol=1e-6
        )

        print("Unique gains present AFTER filtering (GAMMA):")
        print(unique_gains_raw(gains_f_g, q, round_to=6))

        amps_list_g, delay_list_g, rounds_list_g, gain_labels_g = split_same_format_by_gain(
            amps_f_g, gains_f_g, rounds_f_g, delay_f_g,
            gains_list=[0.0, 0.034243, 0.054386, 0.080571, 0.0987],
            atol=1e-3, rtol=1e-6
        )

        # Whatever your existing gamma box/scatter plot call is, keep it here.
        # Example (rename to your real plotting function):
        # gamma_vs_time.plot_gamma_scatter_multi_gain_by_dataset_index(
        #     amps_list_g,
        #     delay_list_g,
        #     gains=gain_labels_g,
        #     save_path=save_dir,
        #     q_key=q
        # )

        # NEW: mean/median/std plot for GAMMA
        plot_mean_median_std_by_gain(
            amps_list_g,
            gain_labels_g,
            title="Gamma: Mean / Median with ±1σ",
            save_path=save_dir,
            filename=f"gamma_mean_median_std_{q}.png",
            q_key=q
        )