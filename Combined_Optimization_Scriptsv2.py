import sys
import os
import numpy as np
import datetime
sys.path.append(os.path.abspath("/home/kanyang/Github/4x2Loud_tprocV2"))
from section_001_time_of_flight import TOFExperiment
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_005_single_shot_ge import GainFrequencySweep
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from section_003_punch_out_ge_mux import PunchOut
from system_config import QICK_experiment
from expt_config import *
import h5py
import time
import matplotlib.pyplot as plt
import copy
from tqdm import tqdm


signal = 'None'        #'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization has happened
save_figs = True   # save plots for everything as you go along the RR script?
live_plot = False    # for live plotting open http://localhost:8097/ on firefox
fit_data = False # always set to False
unmask = False
FRIDGE = "LOUD"
number_of_qubits = 6
list_of_all_qubits = [0,1,2,3,4,5]

# For Nexus
# outerFolder = os.path.join("/home/nexusadmin/qick/NEXUS_sandbox/Data/Run30", str(datetime.date.today())) #change run number in each new run

# For Quiet
#substudy = 'optimization_q3'#'readout_length_opt_round2'#"readout_gain_offset_optimization"
# outerFolder = os.path.join("M:/_Data/20250822 - Olivia/6transmon_run6/", str(datetime.date.today()))
#outerFolder = os.path.join("M:/_Data/20250822 - Olivia/run6/6transmon/StarkShift/DAC0_check/Optimization/run2/", str(datetime.date.today()))
#outerFolder = os.path.join(f"M:/_Data/20250822 - Olivia/run6/6transmon/TLS_Comprehensive_Study/readout_optimization_{datetime.date.today().strftime('%Y-%m-%d')}", datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
#outerFolder = os.path.join(f"M:/_Data/20250822 - Olivia//bob_run_started_Aug_23/squill/{substudy}/{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}/") #
def create_folder_if_not_exists(folder_path):
    """Creates a folder at the given path if it doesn't already exist."""
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)


################################################ Data Saving Setup ##################################################
#Folders

data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

run_name = 'run5'
device_name = 'rfsoc-4x2-loopback'  # 'saph-6transmon'#  'sil-6transmon'
substudy_txt_notes = ('Test-Loopback')# ('This data was taken after reverting back to only 1 channel on the qick box. T1 shots saved as well as averaged IQ data.\n') # Initial qubit checkouts quiet run 8

################################################ Data Saving Setup ##################################################
# Folders
study = 'tests-round_robin' #qubit_checkouts
sub_study ='tests'# 'source_on_25dBDAC' #pre_AB_paper_data_still_optimizing, two_photon_peak_search, AB_Paper_Data_24hrs, ABpaperdata3rdbatch_21dB_DACatten_Q1to5_t1shots_optional
#ABpaperdata3rdbatch_21dB_DACatten_Q1to6_t1shots_optional, ABpaperdata_21dB_DACatten_Q1to6_t1shots_optional_newopt, 18dB_DAC_testdata_allQs_exceptQ4, cooldown_run8b_19dB_DAC_allQs
data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")


if not os.path.exists(f"/data/QICK_data/{run_name}/"):
    os.makedirs(f"/data/QICK_data/{run_name}/")
if not os.path.exists(f"/data/QICK_data/{run_name}/{device_name}/"):
    os.makedirs(f"/data/QICK_data/{run_name}/{device_name}/")
studyFolder = os.path.join(f"/data/QICK_data/{run_name}/{device_name}/", study)
if not os.path.exists(studyFolder):
    os.makedirs(studyFolder)
subStudyFolder = os.path.join(studyFolder, sub_study)
if not os.path.exists(subStudyFolder):
    os.makedirs(subStudyFolder)

dataSetFolder = os.path.join(subStudyFolder, data_set)
optimizationFolder = os.path.join(dataSetFolder, 'optimization')
studyFolder = os.path.join(dataSetFolder, 'study_data')
studyDocumentationFolder = os.path.join(dataSetFolder, 'documentation')
subStudyDataFolder = os.path.join(dataSetFolder, 'study_data')
if not os.path.exists(studyDocumentationFolder):
    os.makedirs(studyDocumentationFolder)
if not os.path.exists(optimizationFolder):
    os.makedirs(optimizationFolder)
if not os.path.exists(subStudyDataFolder):
    os.makedirs(subStudyDataFolder)

outerFolder = dataSetFolder 

# Where to save readout length sweep data
prefix = str(datetime.date.today())
output_folder_length =outerFolder + "/study_data/Data_h5/ge_readout_length_optimization/"
create_folder_if_not_exists(output_folder_length)

#Where to save the RR plots
outerfolder_plots =  os.path.join(dataSetFolder, 'documentation')  #outerFolder + "/documentation/"

n = 1  # Number of rounds
n_loops = 3  # Number of repetitions per length to average

# List of qubits to measure
Qs = [1]

#Change for NEXUS vs QUIET
res_leng_vals = [1.5]*6
res_gain = [0.15,0.1157, 0.042, 0.066, 0.2833, 0.15]
freq_offsets = [0, -0.25, 0.1, 0.233, -0.08, -0.15]
punch_out_vals = [0.075] *6

optimal_lengths = [None] * 6 # creates list where the script will be storing the optimal readout lengths for each qubit. We currently have 6 qubits in total.
res_freq_ge = [None] * 6 # creates list where the script will be storing the freq of each resonator, to use in the 2d sweep

j=0 #round number, from RR code. Not really used here since we just run it once for each qubit

lengs = np.arange(0.5, 15, 0.5)
start=time.time()
for QubitIndex in tqdm(Qs):
    # Get the config for this qubit
    experiment = QICK_experiment(outerFolder, DAC_attenuator1 = 10, DAC_attenuator2 = 15, qubit_DAC_attenuator1 = 5,
                                     qubit_DAC_attenuator2 = 4, ADC_attenuator = 17,
                                 fridge=FRIDGE)

    experiment.readout_cfg['res_gain_ge'] = res_gain[QubitIndex]
    experiment.readout_cfg['res_gain_ef'] = experiment.readout_cfg['res_gain_ef'][QubitIndex]
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]
    experiment.readout_cfg['res_freq_ge'] = experiment.readout_cfg['res_freq_ge'][QubitIndex]

    experiment.qubit_cfg['qubit_freq_ge'] = experiment.qubit_cfg['qubit_freq_ge'][QubitIndex]
    experiment.qubit_cfg['qubit_gain_ge'] = experiment.qubit_cfg['qubit_gain_ge'][QubitIndex]

    ################################################## Res spec ####################################################

    # res_spec = ResonanceSpectroscopy(QubitIndex, number_of_qubits, outerfolder_plots, j, True,
    #                                  experiment, unmasking_resgain = unmask)
    # res_freqs, freq_pts, freq_center, amps, res_spec_config = res_spec.run()
    # print(res_freqs)
    # base_res_freq=res_freqs[0]
    # offset = freq_offsets[
    #     QubitIndex]  # use optimized offset values or whats set at top of script based on pre_optimize flag
    # offset_res_freqs = [r + offset for r in res_freqs]



    # # Used later when optimizing res gains and freqs, decide if you want to set the offsets to zero or not for the first round
    # this_res_freq = offset_res_freqs
    # res_freq_ge = float(this_res_freq[0])
    # experiment.readout_cfg['res_freq_ge'] = res_freq_ge
    # print(experiment.readout_cfg['res_freq_ge'])
    # del res_spec

    ################################################## Qubit spec ##################################################

    # q_spec = QubitSpectroscopy(QubitIndex, number_of_qubits, outerfolder_plots, j, signal,
    #                            True, experiment, live_plot, unmasking_resgain = unmask)
    # qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq, qubit_spec_config = q_spec.run()

    # # if these are None, fit didnt work
    # if (qspec_I_fit is None and qspec_Q_fit is None and qubit_freq is None):
    #     print('QSpec fit didnt work, skipping the rest of this qubit')
    #     continue  # skip the rest of this qubit

    # experiment.qubit_cfg['qubit_freq_ge'] = float(qubit_freq)
    # print('Qubit freq for qubit ', QubitIndex + 1, ' is: ', float(qubit_freq))
    # del q_spec


    ###################################################### Rabi ####################################################
    # increase_qubit_reps = False  # if you want to increase the reps for a qubit, set to True
    # qubit_to_increase_reps_for = 0  # only has impact if previous line is True
    # multiply_qubit_reps_by = 2  # only has impact if the line two above is True
    # print('ge Rabi')
    # rabi = AmplitudeRabiExperiment(QubitIndex, number_of_qubits, outerfolder_plots, j, signal,
    #                                False, experiment=experiment, live_plot=live_plot,
    #                                increase_qubit_reps=increase_qubit_reps, qubit_to_increase_reps_for=qubit_to_increase_reps_for,
    #                                multiply_qubit_reps_by=multiply_qubit_reps_by, unmasking_resgain = unmask)
    # rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp, sys_config_to_save = rabi.run()

    # # if these are None, fit didnt work
    # if (rabi_fit is None and pi_amp is None):
    #     print('Rabi fit didnt work, skipping the rest of this qubit')
    #     continue  # skip the rest of this qubit

    # experiment.qubit_cfg['pi_amp'] = float(pi_amp)
    # print('Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))

    # t2r = T2RMeasurement(QubitIndex, tot_num_of_qubits, outerfolder_plots, j, signal, save_figs,
    #                      experiment=experiment, live_plot=live_plot, fit_data=True,
    #                      increase_qubit_reps=increase_qubit_reps,
    #                      qubit_to_increase_reps_for=qubit_to_increase_reps_for,
    #                      multiply_qubit_reps_by=multiply_qubit_reps_by,
    #                      unmasking_resgain=unmask, correction=True,
    #                      correction_round=1)
    # t2r_est_1, t2r_err_1, t2r_I_1, t2r_Q_1, t2r_delay_times_1, fit_ramsey_1, sys_config_t2r_1, ramsey_found_q_freq = t2r.adjust_qspec(
    #     thresholding=False, correction=True)

    # experiment.qubit_cfg['qubit_freq_ge'] = experiment.qubit_cfg['qubit_freq_ge'] - ramsey_found_q_freq + \
    #                                         expt_cfg['Ramsey_ge_correction']['ramsey_freq']
    # del t2r

    # correct again
    # t2r = T2RMeasurement(QubitIndex, tot_num_of_qubits, outerfolder_plots, j, signal, save_figs,
    #                      experiment=experiment, live_plot=live_plot, fit_data=True,
    #                      increase_qubit_reps=increase_qubit_reps,
    #                      qubit_to_increase_reps_for=qubit_to_increase_reps_for,
    #                      multiply_qubit_reps_by=multiply_qubit_reps_by,
    #                      unmasking_resgain=unmask, correction=True,
    #                      correction_round=2)
    # t2r_est_2, t2r_err_2, t2r_I_2, t2r_Q_2, t2r_delay_times_2, fit_ramsey_2, sys_config_t2r_2, ramsey_found_q_freq = t2r.adjust_qspec(
    #     thresholding=False, correction=True)

    # experiment.qubit_cfg['qubit_freq_ge'] = experiment.qubit_cfg[
    #                                             'qubit_freq_ge'] - ramsey_found_q_freq + \
    #                                         expt_cfg['Ramsey_ge_correction']['ramsey_freq']
    # del t2r

    # # correct rabi
    # rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, outerfolder_plots, j, signal,
    #                                save_figs=save_figs, save_shots=False,
    #                                experiment=experiment, live_plot=live_plot,
    #                                increase_qubit_reps=increase_qubit_reps,
    #                                qubit_to_increase_reps_for=qubit_to_increase_reps_for,
    #                                multiply_qubit_reps_by=multiply_qubit_reps_by,
    #                                unmasking_resgain=unmask,
    #                                correction=True)
    # (rabi_I_corrected, rabi_Q_corrected, rabi_gains_corrected, rabi_fit_corrected, pi_amp_corrected,
    #  sys_config_rabi_corrected) = rabi.run(thresholding=False)

    # # if these are None, fit didnt work
    # if (rabi_fit is None and pi_amp is None):
    #     continue  # skip the rest of this qubit

    # experiment.qubit_cfg['pi_amp'] = float(pi_amp)
    # del rabi

    #MAKE DEEP COPY OF CONFIG, IMPORTANT!!!
    tuned_experiment = copy.deepcopy(experiment)

    # #-----------Sweeping Readout Length----------------------------
    QubitIndex = int(QubitIndex)  # Ensure QubitIndex is an integer

    avg_fids = []
    rms_fids = []

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    h5_filename = os.path.join(output_folder_length, f"ge_readoutlength_sweep_Q{QubitIndex + 1}_data_{timestamp}.h5")
    with h5py.File(h5_filename, 'w') as h5_file:
        # Top-level group for the qubit
        qubit_group = h5_file.create_group(f"Qubit_{QubitIndex + 1}")
        fids = []  # Store fidelity values for each loop
        ground_iq_data = []  # Store ground state IQ data for each loop
        excited_iq_data = []  # Store excited state IQ data for each loop

        # Iterate over each readout pulse length
        for leng in lengs:
            print(len)
            # Subgroup for each readout length within the round
            length_group = qubit_group.create_group(f"Length_{leng}")

            for k in range(n_loops):  # loops for each read out length
                # ------------------------Single Shot-------------------------
                # Initialize experiment for each loop iteration
                experiment = copy.deepcopy(tuned_experiment)
                # Set specific configuration values for each iteration
                experiment.readout_cfg['res_length'] = leng  # Set the current readout pulse length

                # Set gain for the current qubit
                #gain = res_gain[QubitIndex]
                #res_gains = experiment.set_gain_filter_ge(QubitIndex, gain)  # Set gain for current qubit only

                #experiment.readout_cfg['res_gain_ge'] = gain
                try:
                    ss = SingleShot(QubitIndex, number_of_qubits, outerFolder,  j, save_figs, experiment, unmasking_resgain = unmask)  # updated way
                    fid, angle, iq_list_g, iq_list_e, ss_config = ss.run()
                except:
                    continue

                #print(ss_config)
                fids.append(fid)

                # Append IQ data for each loop
                ground_iq_data.append(iq_list_g)
                excited_iq_data.append(iq_list_e)

                # Save individual fidelity and IQ data for this loop
                loop_group = length_group.create_group(f"Loop_{k + 1}")
                loop_group.create_dataset("fidelity", data=fid)
                loop_group.create_dataset("ground_iq_data", data=iq_list_g)
                loop_group.create_dataset("excited_iq_data", data=iq_list_e)

                del experiment

            # Calculate average and RMS for fidelities across loops
            avg_fid = np.mean(fids)
            print(avg_fid)
            rms_fid = np.std(fids)
            avg_fids.append(avg_fid)
            rms_fids.append(rms_fid)

            # Calculate average IQ data across all loops
            avg_ground_iq = np.mean(ground_iq_data, axis=0)
            avg_excited_iq = np.mean(excited_iq_data, axis=0)

            fids.clear()
            ground_iq_data.clear()
            excited_iq_data.clear()

        # Save the averages and RMS to the HDF5 file for this length
        length_group.create_dataset("avg_fidelity", data=avg_fids)
        length_group.create_dataset("rms_fidelity", data=rms_fids)
        length_group.create_dataset("avg_ground_iq_data", data=avg_ground_iq)
        length_group.create_dataset("avg_excited_iq_data", data=avg_excited_iq)



    # avg_max = max(avg_fids[:10])
    avg_max = max(avg_fids)
    avg_max_index = avg_fids.index(avg_max)
    max_len = lengs[avg_max_index]
    optimal_lengths[QubitIndex] = max_len

    # Plot the average fidelity vs. pulse length with error bars for each qubit
    plt.figure()
    plt.errorbar(lengs, avg_fids, yerr=rms_fids, fmt='-o', color='black')
    plt.axvline(x=max_len, linestyle="--", color="red")
    plt.text(max_len + 0.1, avg_fids[0], f'{max_len:.4f}', color='red')
    plt.xlabel('Readout and Pulse Length')
    plt.ylabel('Fidelity')
    plt.title(f'Avg Fidelity vs. Readout and Pulse Length for Qubit {QubitIndex + 1}, ({n_loops} repetitions)' , fontsize=10)
    path = os.path.join(outerfolder_plots, 'readout_length_ge')
    create_folder_if_not_exists(path)
    file_nm = os.path.join(path, f'ge_readoutlength_sweep_Q{QubitIndex + 1}_{timestamp}.png')
    plt.savefig(file_nm, dpi=300)
    print('res leng sweep plot saved to:', outerfolder_plots)
    #plt.show()
    plt.close()

    del avg_fids, rms_fids, avg_ground_iq, avg_excited_iq, loop_group, length_group

    #---------------------Res Gain and Res Freq Sweeps------------------------
    # optimal_lengths = [5]*6#[5,4.2,8.3,7.9,7.5,6.4]
    # date_str = str(datetime.date.today())
    # output_folder = outerFolder + "/study_data/Data_h5/2D_Gain_Freq_Sweeps/"
    # # Ensure the output folder exists
    # os.makedirs(output_folder, exist_ok=True)

    # # Define sweeping parameters
    # # if QubitIndex == 0 or QubitIndex == 1 or QubitIndex == 2 or QubitIndex == 5:
    # #     gain_range = [0.8, 1.0]
    # # elif QubitIndex == 3 or QubitIndex == 4:
    # #     gain_range = [0.46,0.66]  # Gain range in a.u.
    # gain_range=[0.09,0.12]#res_gain[QubitIndex]]
    # freq_steps = 7
    # gain_steps =7

    # print(f'Starting Qubit {QubitIndex + 1} res gain and res freq measurements.')
    # # Select the reference frequency for the current resonator
    # reference_frequency = experiment.readout_cfg['res_freq_ge']#base_res_freq

    # freq_range = [reference_frequency -0.25, reference_frequency + 0.25]# Frequency range in MHz
    # #freq_range = [reference_frequency -0.2, (reference_frequency + 0.2) + 1]  # Frequency range in MHz

    # experiment = copy.deepcopy(tuned_experiment)
    # sweep = GainFrequencySweep(QubitIndex, number_of_qubits, list_of_all_qubits, experiment, optimal_lengths=optimal_lengths, output_folder=output_folder, unmasking_resgain = unmask)
    # results = sweep.run_sweep(freq_range, gain_range, freq_steps, gain_steps)
    # results = np.array(results)

    # timestamp = time.strftime("%H%M%S")
    # h5_file = os.path.join(output_folder, f"Gain_Freq_Sweep_Qubit_{QubitIndex + 1}_{timestamp}.h5")

    # with h5py.File(h5_file, "w") as f:
    #     # Store the data
    #     f.create_dataset("results", data=results)
    #     # Store metadata
    #     f.attrs["gain_range"] = gain_range
    #     f.attrs["freq_range"] = freq_range
    #     f.attrs["reference_frequency"] = reference_frequency
    #     f.attrs["freq_steps"] = freq_steps
    #     f.attrs["gain_steps"] = gain_steps
    #     f.attrs["optimal_length"] = optimal_lengths[QubitIndex]

    # #print(f"Saved data for Qubit {QubitIndex + 1} to {h5_file}")

    # plt.imshow(results, aspect='auto',
    #            extent=[gain_range[0], gain_range[1], freq_range[0] - reference_frequency,
    #                    freq_range[1] - reference_frequency],
    #            origin='lower')
    # plt.colorbar(label="Fidelity")
    # plt.xlabel("Readout pulse gain (a.u.)")  # Gain on x-axis
    # plt.ylabel("Readout frequency offset (MHz)")  # Frequency on y-axis
    # plt.title(f"Gain-Frequency Sweep for Qubit {QubitIndex + 1}")
    # # plt.show()
    # path = os.path.join(outerfolder_plots, '2D_GainFreq_Sweep')
    # create_folder_if_not_exists(path)
    # file_nm = os.path.join(path, f'ge_readoutlength_sweep_Q{QubitIndex + 1}_{timestamp}.png')
    # plt.savefig(file_nm, dpi=600, bbox_inches='tight')

    # plt.close()  # Close the plot to free up memory
    # del results, sweep

end=time.time()
print('timetaken=',end-start)
