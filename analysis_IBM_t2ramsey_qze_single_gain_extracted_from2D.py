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
    run_name = f'bob_run_started_Feb_11/squill/{path}/all_qubits/'
    top_folder_dates = [f'qubit_{qubit}round{round}' for round in range(113)]


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
        save_path=f'M:/_Data/20250822 - Olivia/bob_run_started_Feb_11/squill/{path}/all_qubits/analysis_gain/',
        q_key=q
    )

