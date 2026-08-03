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
path = 'repeat_t1_fluctuations_interweaved_gains_overnight'
gains=['gain0p08','gain0']
amps_t1_gains=[]
dates_t1_gains=[]
delay_times_t1_gains=[]
for qubit in qubits:
    for gain in gains:
        run_name = f'bob_run_started_Aug_3_2026/squill/{path}/{gain}/'
        p = Path('M:/_Data/20250822 - Olivia/'+run_name)
        top_folder_dates = [item.name for item in p.iterdir() if item.is_dir() and "qubit" in item.name]
        # # QSpec Analysis
        # q_vs_time = QubitFreqsVsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs,
        #                                      False, 'None', run_name, fridge=FRIDGE, exp_name='ge', qubit=qubit)
        # _, _, amps_qspec_hg, dates_qspec_hg, rounds_qspec_hg, freqs_qspec_hg = q_vs_time.run_q_sweep_single_gain(
        #     exp_extension='_ge', scaling=True)
        #
        # q_vs_time.plot_all_q_heatmaps_single_gain(amps_qspec_hg, dates_qspec_hg, rounds_qspec_hg, freqs_qspec_hg,
        #                                                   f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_3_2026/squill/{path}/all_qubits/analysis/',
        #                                                 save_individual_plots=save_individual_qspec)


        # T1 Analysis
        t1_vs_time = T1VsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs, False,
                             'None', run_name, FRIDGE, exp_name='ge', qubit=qubit, t1_slice='10us')
        _, _, amps_t1, dates_t1, rounds_t1, delay_times_t1 = t1_vs_time.run_t1_sweep_single_gain(exp_extension='_ge', scaling=True)
        amps_t1_gains.append(amps_t1)
        dates_t1_gains.append(dates_t1)
        delay_times_t1_gains.append(delay_times_t1)
        # t1_vs_time.plot_t1_heatmap_single_gain(amps_t1, dates_t1, delay_times_t1,
        #                                           f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_3_2026/squill/{path}/{gain}/analysis/', q_key=qubits[0],
        #                                        gain=gain)
        # t1_vs_time.plot_t1_scatter_single_gain(amps_t1, dates_t1, delay_times_t1,
        #                                        f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_3_2026/squill/{path}/{gain}/analysis/',
        #                                        q_key=qubits[0],
        #                                        gain=gain)
    t1_vs_time.plot_t1_scatter_multi_gain_by_round(amps_t1_gains, dates_t1_gains, delay_times_t1_gains,gains,
                                               f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_3_2026/squill/{path}/{gain}/analysis/',
                                               q_key=qubits[0])