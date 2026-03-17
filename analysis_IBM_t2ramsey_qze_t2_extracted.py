from analysis_006_T1_vs_time_plots import T1VsTime
from analysis_007_T2R_vs_time_plots import T2rVsTime
from analysis_008_T2E_vs_time_plots import T2eVsTime
from analysis_003_q_freqs_vs_time_plots import QubitFreqsVsTime

# Configuration
save_figs = True
save_individual_qspec = False  # Set to False to skip saving individual qspec plots
nbar_from_ramsey = False
figure_quality = 100
final_figure_quality = 200
FRIDGE = "QUIET"
qubits = [5]
path = '2d_test_high_res'#
ramsey_nbar_path='/bob_run_started_Feb_11/squill/ramsey_n_bar_calibration_more_fringes/run/'

for qubit in qubits:
    run_name = f'bob_run_started_Feb_11/squill/{path}/all_qubits/'
    top_folder_dates = [f'qubit_{qubit}round{round}' for round in [0]]
    
    # QSpec analysis
    q_vs_time = QubitFreqsVsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs,
                                 False, 'None', run_name, fridge=FRIDGE, exp_name='ge', qubit=qubit)
    _, _, amps_qspec, gains_qspec, rounds_qspec, freqs_qspec = q_vs_time.run_q_sweep_new(exp_extension='_ge', scaling=True)

    # Calculate nbar (no plotting)
    if nbar_from_ramsey:
        top_folder_dates_ramsey_nbar=['2025-12-10_13-20-47']#'2025-12-25_22-10-45'
        t2_vs_time = T2rVsTime(figure_quality, final_figure_quality, 6, top_folder_dates_ramsey_nbar, save_figs, False,
                               'None', ramsey_nbar_path, fridge=FRIDGE, exp_name='ge', qubit=qubit)
        n_bars=t2_vs_time.run_ramsey_nbar(exp_name='StarkRamsey', save_path=f'M:/_Data/20250822 - Olivia/bob_run_started_Feb_11/squill/{path}/all_qubits/analysis_extracted_t2/',
                                              )



    else:
        n_bars = q_vs_time.calculate_nbar(amps_qspec, gains_qspec, rounds_qspec, freqs_qspec, chi_MHz=-0.137)
        q_vs_time.plot_all_q_heatmaps_nbar(amps_qspec, gains_qspec, rounds_qspec, freqs_qspec,
                                                    f'M:/_Data/20250822 - Olivia/bob_run_started_Feb_11/squill/{path}/all_qubits/analysis_extracted_t2/',
                                                    individual_subfolder="individual_specs")

    t2_out=q_vs_time.plot_all_q_heatmaps_new_format_return_t2(amps_qspec, gains_qspec, rounds_qspec, freqs_qspec,
                                              f'M:/_Data/20250822 - Olivia/bob_run_started_Feb_11/squill/{path}/all_qubits/analysis_extracted_t2/',
                                              n_bar=n_bars, save_individual_plots=save_individual_qspec)
    g_t2 = t2_out["0"]["gains"]  # gain per column
    t2_us = t2_out["0"]["T2_us"]  # T2 per gain (microseconds)
    fwhm = t2_out["0"]["fwhm_MHz"]  # FWHM per gain (MHz) used to compute it

    # T1 analysis
    t1_vs_time = T1VsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs, False,
                         'None', run_name, FRIDGE, exp_name='ge', qubit=qubit, t1_slice='10us')
    _, _, amps_t1, gains_t1, rounds_t1, delay_times_t1 = t1_vs_time.run_t1_sweep_new(exp_extension='_ge', scaling=True, weighted_mean=True)

    t1_vs_time.fit_and_save_t1_slices_new_format(amps_t1, gains_t1, rounds_t1, delay_times_t1,
                                                  f'M:/_Data/20250822 - Olivia/bob_run_started_Feb_11/squill/{path}/all_qubits/analysis_extracted_t2/',
                                                  n_bar=n_bars)
    t1_out=t1_vs_time.plot_all_t1_heatmaps_new_format(amps_t1, gains_t1, rounds_t1, delay_times_t1,
                                                f'M:/_Data/20250822 - Olivia/bob_run_started_Feb_11/squill/{path}/all_qubits/analysis_extracted_t2/',
                                                n_bar=n_bars, use_linear_x=True)
    g_t1 = t1_out["0"]["gains"]
    t1_us = t1_out["0"]["T1_us"]

    import numpy as np

    save_dir = f"M:/_Data/20250822 - Olivia/bob_run_started_Feb_11/squill/{path}/all_qubits/analysis_extracted_t2/"
    round_id=0
    # t_phi now
    g_t2 = np.asarray(t2_out[str(round_id)]["gains"], float)
    t2_us = np.asarray(t2_out[str(round_id)]["T2_us"], float)

    g_t1 = np.asarray(t1_out[str(round_id)]["gains"], float)
    t1_us = np.asarray(t1_out[str(round_id)]["T1_us"], float)

    # keep finite
    m2 = np.isfinite(g_t2) & np.isfinite(t2_us) & (t2_us > 0)
    m1 = np.isfinite(g_t1) & np.isfinite(t1_us) & (t1_us > 0)
    g_t2, t2_us = g_t2[m2], t2_us[m2]
    g_t1, t1_us = g_t1[m1], t1_us[m1]

    # ---- match gains ----
    # Try exact match first (after rounding to kill float noise)
    rdigits = 12
    t1_map = {round(float(g), rdigits): float(t) for g, t in zip(g_t1, t1_us)}

    gains_common = []
    t1_common = []
    t2_common = []

    for g, t2v in zip(g_t2, t2_us):
        key = round(float(g), rdigits)
        if key in t1_map:
            gains_common.append(float(g))
            t2_common.append(float(t2v))
            t1_common.append(float(t1_map[key]))

    # If that failed (float mismatch), fall back to nearest-neighbor matching
    if len(gains_common) < max(3, int(0.5 * min(len(g_t2), len(g_t1)))):
        gains_common = []
        t1_common = []
        t2_common = []

        # tolerance: small fraction of gain span, with an absolute floor
        span = float(np.nanmax(np.r_[g_t1, g_t2]) - np.nanmin(np.r_[g_t1, g_t2]))
        tol = max(1e-0, 1e-6 * span)

        for g, t2v in zip(g_t2, t2_us):
            j = int(np.argmin(np.abs(g_t1 - g)))
            if abs(g_t1[j] - g) <= tol:
                gains_common.append(float(g))
                t2_common.append(float(t2v))
                t1_common.append(float(t1_us[j]))

    gains_common = np.asarray(gains_common, float)
    t1_common = np.asarray(t1_common, float)
    t2_common = np.asarray(t2_common, float)

    # ---- compute Tphi ----
    # Work in seconds for safety
    T1 = t1_common * 1e-6
    T2 = t2_common * 1e-6

    inv_Tphi = (1.0 / T2) - (1.0 / (2.0 * T1))

    # invalid if inv_Tphi <= 0 (would imply T2 limited purely by T1 or noise/bad fit)
    tphi_s = np.where(inv_Tphi > 0, 1.0 / inv_Tphi, np.nan)
    tphi_us = tphi_s * 1e6
    import os
    # ---- plot + save ----
    os.makedirs(save_dir, exist_ok=True)

    ordg = np.argsort(gains_common)
    g_plot = gains_common[ordg]
    tphi_plot = tphi_us[ordg]
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    ax.plot(g_plot, tphi_plot, marker="o", linewidth=1.5)
    ax.set_title(rf"$T_\phi$ vs Gain — Round {round_id}")
    ax.set_xlabel("Gain (a.u.)")
    ax.set_ylabel(r"$T_\phi$ ($\mu$s)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_png = os.path.join(save_dir, f"tphi_vs_gain_round{round_id}.png")
    fig.savefig(out_png, dpi=300, transparent=False)
    plt.close(fig)
    print(f"Saved Tphi vs gain to: {out_png}")