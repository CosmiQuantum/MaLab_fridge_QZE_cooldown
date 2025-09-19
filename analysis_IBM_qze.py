
from analysis_006_T1_vs_time_plots import T1VsTime
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
qubits=[3]
Is = {i: [] for i in range(6)}
Qs = {i: [] for i in range(6)}
amps = {i: [] for i in range(6)}
gains = {i: [] for i in range(6)}
rounds = {i: [] for i in range(6)}
delay_times = {i: [] for i in range(6)}
for qubit in qubits:
    slices = [5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
    for slice in slices:
        run_name = f'bob_run_started_Aug_23/squill/QZE_IBM_with_scaling/repeat_rounds_t1_slice_{slice}us/'
        top_folder_dates = []
        for round in range(4):
            top_folder_dates.append(f'qubit_{qubit}round{round}')
        t1_vs_time = T1VsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                         signal, run_name, FRIDGE,exp_name = 'ge', qubit=qubit, t1_slice=f'{slice}us')
        # Is,Qs,amps,gains = t1_vs_time.run_IBM_qze()
        # t1_vs_time.plot_IBM_qze(amps,gains, f'M:/_Data/20250822 - Olivia/{run_name}/analysis/')
        # t1_vs_time.plot_IBM_qze_normal(amps,gains, f'M:/_Data/20250822 - Olivia/{run_name}/analysis/')
        Is1,Qs1,amps1,gains1,rounds1 = t1_vs_time.run_IBM_qze_rounds(exp_extension='_ge', scaling=True)
        Is[qubit].append(Is1[qubit])
        Qs[qubit].append(Qs1[qubit])
        amps[qubit].append(amps1[qubit])
        gains[qubit].append(gains1[qubit])
        rounds[qubit].append(rounds1[qubit])
        delay_times[qubit].append(slice)
        # t1_vs_time.plot_IBM_qze_compare(amps1,gains1,rounds1, f'M:/_Data/20250822 - Olivia/{run_name}/analysis/')
        # t1_vs_time.plot_IBM_qze_normal_compare(amps1,gains1,rounds1, f'M:/_Data/20250822 - Olivia/{run_name}/analysis/')
# t1_vs_time.plot_all_t1_heatmaps(amps,gains,rounds,delay_times, f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/QZE_IBM_with_scaling/analysis/')
# t1_vs_time.plot_t1_vs_delay_per_gain(amps,gains,rounds,delay_times, f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/QZE_IBM_with_scaling/analysis/')
run_name = f'bob_run_started_Aug_23/squill/QZE_IBM_with_scaling_and_base_t1_start_lowgain/repeat_rounds_t1_slice_5us/'
top_folder_dates = [f'qubit_{qubit}round0']
t1 = T1VsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, FRIDGE)
t1_delay_times, t1_amps = t1.run(exp_extension='_ge_base', just_data=True)
t1_vs_time.plot_t1_vs_delay_per_gain_vs_base_t1(amps,gains,rounds,delay_times,t1_delay_times,t1_amps,  f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/QZE_IBM_with_scaling/analysis/')