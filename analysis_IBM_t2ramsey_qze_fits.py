from analysis_006_T1_vs_time_plots import T1VsTime
from analysis_007_T2R_vs_time_plots import T2rVsTime
from analysis_008_T2E_vs_time_plots import T2eVsTime
from analysis_003_q_freqs_vs_time_plots import QubitFreqsVsTime

# Configuration
save_figs = True
save_individual_qspec = False  # Set to False to skip saving individual qspec plots
nbar_from_ramsey = True
figure_quality = 100
final_figure_quality = 200
FRIDGE = "QUIET"
qubits = [4]
path = '2d_tiny_n_bar'#'2d_less_reps_more_qspec_steps'#
ramsey_nbar_path='/bob_run_started_Aug_23/squill/ramsey_n_bar_calibration_more_fringes/run/'

for qubit in qubits:
    run_name = f'bob_run_started_Aug_23/squill/{path}/all_qubits/'
    top_folder_dates = [f'qubit_{qubit}round{round}' for round in range(90,92)]
    
    # QSpec analysis
    q_vs_time = QubitFreqsVsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs,
                                 False, 'None', run_name, fridge=FRIDGE, exp_name='ge', qubit=qubit)
    _, _, amps_qspec, gains_qspec, rounds_qspec, freqs_qspec = q_vs_time.run_q_sweep_new(exp_extension='_ge', scaling=True)

    # Calculate nbar (no plotting)
    if nbar_from_ramsey:
        top_folder_dates_ramsey_nbar=['2025-12-19_13-29-47']#'2025-12-25_22-10-45'
        t2_vs_time = T2rVsTime(figure_quality, final_figure_quality, 6, top_folder_dates_ramsey_nbar, save_figs, False,
                               'None', ramsey_nbar_path, fridge=FRIDGE, exp_name='ge', qubit=qubit)

        n_bars=t2_vs_time.run_ramsey_nbar(exp_name='StarkRamsey', save_path=f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis_new_fwhm_qspec_nbar/Ramsey_stark/',
                                              )



    else:
        n_bars = q_vs_time.calculate_nbar(amps_qspec, gains_qspec, rounds_qspec, freqs_qspec, chi_MHz=-0.137)
        q_vs_time.plot_all_q_heatmaps_nbar(amps_qspec, gains_qspec, rounds_qspec, freqs_qspec,
                                                    f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis_new_fwhm_qspec_nbar/',
                                                    individual_subfolder="individual_specs")

    q_vs_time.plot_all_q_heatmaps_new_format(amps_qspec, gains_qspec, rounds_qspec, freqs_qspec,
                                              f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis_new_fwhm_qspec_nbar/',
                                              n_bar=n_bars, save_individual_plots=save_individual_qspec)

    # # T1 analysis
    # t1_vs_time = T1VsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs, False,
    #                      'None', run_name, FRIDGE, exp_name='ge', qubit=qubit, t1_slice='10us')
    # _, _, amps_t1, gains_t1, rounds_t1, delay_times_t1 = t1_vs_time.run_t1_sweep_new(exp_extension='_ge', scaling=True, weighted_mean=True)
    #
    # t1_vs_time.fit_and_save_t1_slices_new_format(amps_t1, gains_t1, rounds_t1, delay_times_t1,
    #                                               f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis_new_fwhm_qspec_nbar/',
    #                                               n_bar=n_bars)
    # t1_vs_time.plot_all_t1_heatmaps_new_format(amps_t1, gains_t1, rounds_t1, delay_times_t1,
    #                                             f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis_new_fwhm_qspec_nbar/',
    #                                             n_bar=n_bars, use_linear_x=True)
    #
    # # T2R analysis
    # t2_vs_time = T2rVsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs, False,
    #                        'None', run_name, fridge=FRIDGE, exp_name='ge', qubit=qubit)
    # _, _, amps_t2, gains_t2, rounds_t2, delay_times_t2 = t2_vs_time.run_t2_sweep_new(exp_extension='_ge', scaling=True)
    #
    # t2_vs_time.plot_all_t2_heatmaps_new_format(amps_t2, gains_t2, rounds_t2, delay_times_t2,
    #                                             f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis_new_fwhm_qspec_nbar/',
    #                                             n_bar=n_bars, use_linear_x=False)
    #
    # t2_vs_time.plot_all_t2_curves(amps_t2, gains_t2, rounds_t2, delay_times_t2,
    #                                f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis_new_fwhm_qspec_nbar/',
    #                                n_bar=n_bars)
    #
    # # T2E analysis
    # t2e_vs_time = T2eVsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs,
    #                        False, 'None', run_name, fridge=FRIDGE, exp_name='ge', qubit=qubit)
    # _, _, amps_t2e, gains_t2e, rounds_t2e, delay_times_t2e = t2e_vs_time.run_t2_sweep_new(exp_extension='_ge', scaling=True)
    #
    # t2e_vs_time.plot_all_t2_heatmaps_new_format(amps_t2e, gains_t2e, rounds_t2e, delay_times_t2e,
    #                                            f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis_new_fwhm_qspec_nbar/',
    #                                            n_bar=n_bars, save_individual_plots=save_individual_qspec)
