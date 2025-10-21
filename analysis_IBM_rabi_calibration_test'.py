
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
path= 'QZE_IBM_q5'
for qubit in qubits:
    run_name = f'bob_run_started_Aug_23/squill/{path}/all_qubits/'
    top_folder_dates = []
    for round in range(4):
        top_folder_dates.append(f'qubit_{qubit}round{round}')
    from analysis_004_pi_amp_vs_time_plots import PiAmpsVsTime
    pi_amps_vs_time = PiAmpsVsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs,
                                   fit_saved, signal, run_name)
    date_times_pi_amps, pi_amps,I_rabi,Q_rabi,gains_rabi, Ie_calibration1_rabi, Ig_calibration1_rabi,Qe_calibration1_rabi\
        ,Qg_calibration1_rabi = pi_amps_vs_time.run_rabi_w_calibration(plot_depths=False, exp_extension='_ge', return_data=True)

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
    Ie_calibration[qubit] = Ie_calibration_t1[qubit]
    Ig_calibration[qubit] = Ig_calibration_t1[qubit]
    Qe_calibration[qubit] = Qe_calibration_t1[qubit]
    Qg_calibration[qubit] = Qg_calibration_t1[qubit]
    from section_005_single_shot_ge import SingleShot

    ss = SingleShot(qubit, 6, f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/',
                    0, True)

    worse_calibration_dict = t1_vs_time.plot_worse_ssf_only(amps, gains, rounds, delay_times,
                                                          f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/rabi_worse_calibration/',
                                                          ss_class_instance=ss,  # your object that has hist_ssf(...)
                                                          ss_cfg={"steps": steps},
                                                          Ig_calibration=Ig_calibration,
                                                          Ie_calibration=Ie_calibration,
                                                          Qg_calibration=Qg_calibration,
                                                          Qe_calibration=Qe_calibration)

    import numpy as np

    Ig_calibration1 = np.asarray(worse_calibration_dict['Ig'], dtype=float).ravel()
    Ie_calibration1 = np.asarray(worse_calibration_dict['Ie'], dtype=float).ravel()
    Qe_calibration1 = np.asarray(worse_calibration_dict['Qe'], dtype=float).ravel()
    Qg_calibration1 = np.asarray(worse_calibration_dict['Qg'], dtype=float).ravel()

    best_calibration_dict = t1_vs_time.plot_best_ssf_only(amps, gains, rounds, delay_times,
                                                            f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/rabi_worse_calibration/',
                                                            ss_class_instance=ss,  # your object that has hist_ssf(...)
                                                            ss_cfg={"steps": steps},
                                                            Ig_calibration=Ig_calibration,
                                                            Ie_calibration=Ie_calibration,
                                                            Qg_calibration=Qg_calibration,
                                                            Qe_calibration=Qe_calibration)

    import numpy as np

    Ig_calibration1_best = np.asarray(best_calibration_dict['Ig'], dtype=float).ravel()
    Ie_calibration1_best = np.asarray(best_calibration_dict['Ie'], dtype=float).ravel()
    Qe_calibration1_best = np.asarray(best_calibration_dict['Qe'], dtype=float).ravel()
    Qg_calibration1_best = np.asarray(best_calibration_dict['Qg'], dtype=float).ravel()

    from section_006_amp_rabi_ge import AmplitudeRabiExperiment
    rabi = AmplitudeRabiExperiment(qubit, 6, f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/rabi_worse_calibration/', 0,None, True)
    Ig_calibration1_rabi[qubit] = np.asarray(Ig_calibration1_rabi[qubit], dtype=float)
    Ie_calibration1_rabi[qubit] = np.asarray(Ie_calibration1_rabi[qubit], dtype=float)
    Qg_calibration1_rabi[qubit] = np.asarray(Qg_calibration1_rabi[qubit], dtype=float)
    Qe_calibration1_rabi[qubit] = np.asarray(Qe_calibration1_rabi[qubit], dtype=float)
    q1_fit_cosine, pi_amp = rabi.plot_results(I_rabi, Q_rabi, gains_rabi,
                                              scaling=True, Ie=Ie_calibration1_rabi[qubit], Ig=Ig_calibration1_rabi[qubit],
                                              Qe=Qe_calibration1_rabi[qubit], Qg=Qg_calibration1_rabi[qubit], file_ext='rabi_calibration')
    q1_fit_cosine, pi_amp = rabi.plot_results(I_rabi, Q_rabi, gains_rabi,
                                              scaling=True, Ie=Ie_calibration1,
                                              Ig=Ig_calibration1,
                                              Qe=Qe_calibration1, Qg=Qg_calibration1,
                                              file_ext='worse_calibration')

    q1_fit_cosine, pi_amp = rabi.plot_results(I_rabi, Q_rabi, gains_rabi,
                                              scaling=True, Ie=Ie_calibration1_best,
                                              Ig=Ig_calibration1_best,
                                              Qe=Qe_calibration1_best, Qg=Qg_calibration1_best,
                                              file_ext='best_calibration')