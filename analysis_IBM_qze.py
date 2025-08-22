
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
run_name = 'run7/6transmon/QZE_IBM/final_run7_300_t1_points_1000_avgs_slice30us_try2/'

FRIDGE = "QUIET"
run_notes = ('Added IR shielding, better cryo terminators, thermalizing with 0dB attenuator ') #please make it brief for the plot
top_folder_dates = [] #,'qubit_1','qubit_2','qubit_3', 'qubit_4','qubit_5'
for round in range(2):
    top_folder_dates.append(f'qubit_0round{round}')
# ################################################ 01: Get all data ######################################################

t1_vs_time = T1VsTime(figure_quality, final_figure_quality, tot_num_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, FRIDGE)
# Is,Qs,amps,gains = t1_vs_time.run_IBM_qze()
# t1_vs_time.plot_IBM_qze(amps,gains, f'/data/QICK_data/{run_name}/analysis/')
# t1_vs_time.plot_IBM_qze_normal(amps,gains, f'/data/QICK_data/{run_name}/analysis/')
Is,Qs,amps,gains,rounds = t1_vs_time.run_IBM_qze_rounds()
t1_vs_time.plot_IBM_qze_compare(amps,gains,rounds, f'/data/QICK_data/{run_name}/analysis/')
t1_vs_time.plot_IBM_qze_normal_compare(amps,gains,rounds, f'/data/QICK_data/{run_name}/analysis/')

