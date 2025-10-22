
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
path= 'QZE_IBM_q5_rotated_phase_tuned_rabi_debug'
for qubit in qubits:
    run_name = f'bob_run_started_Aug_23/squill/{path}/all_qubits/'
    top_folder_dates = []
    for round in range(4):
        top_folder_dates.append(f'qubit_{qubit}round{round}')
    from analysis_004_pi_amp_vs_time_plots import PiAmpsVsTime
    pi_amps_vs_time = PiAmpsVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs,
                                   fit_saved, signal, run_name)
    date_times_pi_amps, pi_amps, Ie_calibration1, Ig_calibration1,Qe_calibration1,Qg_calibration1 = pi_amps_vs_time.run_rabi_w_calibration(plot_depths=False, exp_extension='_ge')

    t1_vs_time = T1VsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                     signal, run_name, FRIDGE,exp_name = 'ge', qubit=qubit, t1_slice=f'{slice}us')
    Is1, Qs1, amps1, gains1, rounds1, delay_times1, Ig_calibration_t1, \
        Ie_calibration_t1, Qe_calibration_t1, Qg_calibration_t1, steps = t1_vs_time.run_t1_sweep(exp_extension='_ge',
                                                                                           scaling=True,
                                                                                           return_calibration_data=True)

    Is[qubit]=Is1[qubit]
    Qs[qubit]=Qs1[qubit]
    amps[qubit]=amps1[qubit]
    gains[qubit]=gains1[qubit]
    rounds[qubit]=rounds1[qubit]
    delay_times[qubit]=delay_times1[qubit]
    Ie_calibration[qubit] = Ie_calibration1[qubit]
    Ig_calibration[qubit] = Ig_calibration1[qubit]
    Qe_calibration[qubit] = Qe_calibration1[qubit]
    Qg_calibration[qubit] = Qg_calibration1[qubit]

    t1_vs_time.plot_all_t1_heatmaps_single_calibration(Is, Qs, Ig_calibration[qubit][0], \
        Ie_calibration[qubit][0], Qe_calibration[qubit][0], Qg_calibration[qubit][0], gains, rounds, delay_times,
                                    f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/single_calibration_from_rabi/')


    t2_vs_time = T2rVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs,
                          fit_saved,
                          signal, run_name,fridge=FRIDGE,  exp_name='ge', qubit=qubit)
    Is1_t2, Qs1_t2, amps1_t2, gains1_t2, rounds1_t2, delay_times1_t2, Ig_calibration1_t2, \
        Ie_calibration1_t2, Qe_calibration1_t2, Qg_calibration1_t2, steps = t2_vs_time.run_t2_sweep(exp_extension='_ge',
                                                                                                    scaling=True,
                                                                                                    return_calibration_data=True)
    Is_t2[qubit] = Is1_t2[qubit]
    Qs_t2[qubit] = Qs1_t2[qubit]
    amps_t2[qubit] = amps1_t2[qubit]
    gains_t2[qubit] = gains1_t2[qubit]
    rounds_t2[qubit] = rounds1_t2[qubit]
    delay_times_t2[qubit] = delay_times1_t2[qubit]

    t2_vs_time.plot_all_t2_heatmaps_single_calibration(Is_t2, Qs_t2, Ig_calibration[qubit], \
        Ie_calibration[qubit][0], Qe_calibration[qubit][0], Qg_calibration[qubit][0], gains_t2, rounds_t2, delay_times_t2,
                                               f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/single_calibration_from_rabi/')

    q_vs_time = QubitFreqsVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs,
                           fit_saved,
                           signal, run_name, fridge=FRIDGE, exp_name='ge', qubit=qubit)
    Is1_qspec, Qs1_qspec, amps1_qspec, gains1_qspec, rounds1_qspec, delay_times1_qspec, Ig_calibration1_qspec, \
        Ie_calibration1_qspec, Qe_calibration1_qspec, Qg_calibration1_qspec, steps = q_vs_time.run_q_sweep(
        exp_extension='_ge',
        scaling=True,
        return_calibration_data=True)
    Is_qspec[qubit] = Is1_qspec[qubit]
    Qs_qspec[qubit] = Qs1_qspec[qubit]
    amps_qspec[qubit] = amps1_qspec[qubit]
    gains_qspec[qubit] = gains1_qspec[qubit]
    rounds_qspec[qubit] = rounds1_qspec[qubit]
    freqs_qspec[qubit] = delay_times1_qspec[qubit]

    q_vs_time.plot_all_q_heatmaps_single_calibration(Is_qspec, Qs_qspec, Ig_calibration[qubit][0], \
        Ie_calibration[qubit][0], Qe_calibration[qubit][0], Qg_calibration[qubit][0], gains_qspec, rounds_qspec, freqs_qspec,
                                               f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/single_calibration_from_rabi/')