from analysis_006_T1_vs_time_plots import T1VsTime
from analysis_007_T2R_vs_time_plots import T2rVsTime
from analysis_008_T2E_vs_time_plots import T2eVsTime
from analysis_003_q_freqs_vs_time_plots import QubitFreqsVsTime
from pathlib import Path

# Configuration
save_figs = True
save_individual_qspec = False
figure_quality = 100
final_figure_quality = 200
FRIDGE = "QUIET"
qubits = [4]
path = 'single_gain_debug'

for qubit in qubits:
    run_name = f'bob_run_started_Aug_23/squill/{path}/all_qubits/'
    p = Path('M:/_Data/20250822 - Olivia/'+run_name)
    top_folder_dates = [item.name for item in p.iterdir() if item.is_dir() and "qubit" in item.name]
    # QSpec Analysis
    q_vs_time = QubitFreqsVsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs,
                                         False, 'None', run_name, fridge=FRIDGE, exp_name='ge', qubit=qubit)
    _, _, amps_qspec_hg, dates_qspec_hg, rounds_qspec_hg, freqs_qspec_hg = q_vs_time.run_q_sweep_single_gain(
        exp_extension='_ge', scaling=True)

    q_vs_time.plot_all_q_heatmaps_single_gain(amps_qspec_hg, dates_qspec_hg, rounds_qspec_hg, freqs_qspec_hg,
                                                      f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/',
                                                    save_individual_plots=save_individual_qspec)


    # T1 Analysis
    t1_vs_time = T1VsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs, False,
                         'None', run_name, FRIDGE, exp_name='ge', qubit=qubit, t1_slice='10us')
    _, _, amps_t1, dates_t1, rounds_t1, delay_times_t1 = t1_vs_time.run_t1_sweep_single_gain(exp_extension='_ge', scaling=True)

    t1_vs_time.plot_all_t1_heatmaps_single_gain(amps_t1, dates_t1, rounds_t1, delay_times_t1,
                                              f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/',
                                              save_individual_plots=save_individual_qspec)

    # T2R Analysis
    t2_vs_time = T2rVsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs, False,
                           'None', run_name, fridge=FRIDGE, exp_name='ge', qubit=qubit)
    _, _, amps_t2, dates_t2, rounds_t2, delay_times_t2 = t2_vs_time.run_t2_sweep_single_gain(exp_extension='_ge', scaling=True)

    t2_vs_time.plot_all_t2_heatmaps_single_gain(amps_t2, dates_t2, rounds_t2, delay_times_t2,
                                              f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/',
                                              save_individual_plots=save_individual_qspec)

    # T2E Analysis
    t2e_vs_time = T2eVsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs, False,
                           'None', run_name, fridge=FRIDGE, exp_name='ge', qubit=qubit)
    _, _, amps_t2e, dates_t2e, rounds_t2e, delay_times_t2e = t2e_vs_time.run_t2_sweep_single_gain(exp_extension='_ge', scaling=True)

    t2e_vs_time.plot_all_t2_heatmaps_single_gain(amps_t2e, dates_t2e, rounds_t2e, delay_times_t2e,
                                              f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/',
                                              save_individual_plots=save_individual_qspec)

   