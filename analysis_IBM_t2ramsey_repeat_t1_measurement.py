
from analysis_006_T1_vs_time_plots import T1VsTime
from analysis_007_T2R_vs_time_plots import T2rVsTime
from analysis_003_q_freqs_vs_time_plots import QubitFreqsVsTime
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE

###################################################### Set These #######################################################
save_figs = True
fit_saved = False
show_legends = False
signal = 'None'
run_number = 3 #starting from first run with qubits. Run 1 = run4a at quiet, run 2 = run5a at quiet, etc
figure_quality = 100 #ramp this up to like 500 for presentation plots
final_figure_quality = 200

path = 'QZE_IBM_t1_consistency_gain0p01_q5_round0_morelottareps'
FRIDGE = "QUIET"
run_notes = ('Added IR shielding, better cryo terminators, thermalizing with 0dB attenuator ') #please make it brief for the plot
qubits=[4]

Ie_calibration = {i: [] for i in range(6)}
Ig_calibration = {i: [] for i in range(6)}
Qe_calibration = {i: [] for i in range(6)}
Qg_calibration = {i: [] for i in range(6)}

Is = {i: [] for i in range(6)}
Qs = {i: [] for i in range(6)}
amps = {i: [] for i in range(6)}
gains = {i: [] for i in range(6)}
rounds = {i: [] for i in range(6)}
delay_times = {i: [] for i in range(6)}

Is_t2 = {i: [] for i in range(6)}
Qs_t2 = {i: [] for i in range(6)}
amps_t2 = {i: [] for i in range(6)}
gains_t2 = {i: [] for i in range(6)}
rounds_t2 = {i: [] for i in range(6)}
delay_times_t2 = {i: [] for i in range(6)}

Is_qspec = {i: [] for i in range(6)}
Qs_qspec = {i: [] for i in range(6)}
amps_qspec = {i: [] for i in range(6)}
gains_qspec = {i: [] for i in range(6)}
rounds_qspec = {i: [] for i in range(6)}
freqs_qspec = {i: [] for i in range(6)}
for qubit in qubits:
    run_name = f'bob_run_started_Aug_23/squill/{path}/all_qubits/'
    top_folder_dates = []
    for round in range(50):
        top_folder_dates.append(f'qubit_{qubit}round{round}')

    t1_vs_time = T1VsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs,
                          fit_saved,
                          signal, run_name, FRIDGE, exp_name='ge', qubit=qubit, t1_slice=f'{slice}us')
    Is1, Qs1, amps1, gains1, rounds1, delay_times1 = t1_vs_time.run_t1_sweep(exp_extension='_ge', scaling=True)

    t2_vs_time = T2rVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs,
                          fit_saved,
                          signal, run_name,fridge=FRIDGE,  exp_name='ge', qubit=qubit)
    Is_t2[qubit] = Is1[qubit]
    Qs_t2[qubit] = Qs1[qubit]
    amps_t2[qubit] = amps1[qubit]
    gains_t2[qubit] = gains1[qubit]
    rounds_t2[qubit] = rounds1[qubit]
    delay_times_t2[qubit] = delay_times1[qubit]
    t2_vs_time.plot_all_t2_rounds(amps_t2, gains_t2, rounds_t2, delay_times_t2,
                                               f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/')
    t2_vs_time.plot_all_t2_rounds_IQ(Is_t2, Qs_t2, gains_t2, rounds_t2, delay_times_t2,
                                               f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/')

    t2_vs_time.plot_all_t2_rounds_IQAmp(Is_t2, Qs_t2, gains_t2, rounds_t2, delay_times_t2,
                                                   f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/')
