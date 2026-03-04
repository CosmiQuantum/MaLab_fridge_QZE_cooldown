import copy
import sys
import os
import numpy as np

# from long_qubit_spectroscopy import fh_config

np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import time
import logging
import visdom
import gc, copy
import time
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_001_time_of_flight import TOFExperiment
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_002_res_spec_ef import ResonanceSpectroscopyEF
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_004_qubit_spec_ef import EFQubitSpectroscopy
from section_004_qubit_spec_fh_V2 import FHQubitSpectroscopy
from section_006_amp_rabi_ef import EF_AmplitudeRabiExperiment
from section_006_amp_fh import FH_AmplitudeRabiExperiment
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_011_qubit_temperatures_efRabipt3 import Temps_EFAmpRabiExperiment
from section_007_T1_ge import T1Measurement
from section_005_single_shot_ge import SingleShot
from section_005_single_shot_ef import SingleShot_ef
# from section_005_single_shot_gef import SingleShot_ef # Old way: Fix for example Unmask
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from system_config import QICK_experiment
from section_003_punch_out_ge_mux import PunchOut
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
################################################ Run Configurations ####################################################
st = time.time()
#
n= 1000000
pre_optimize = False
freq_offset_steps = 10
ssf_avgs_per_opt_pt = 5
save_r = 1                           # how many rounds to save after
signal = 'None'                      # 'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization
save_figs = True                     # save plots for everything as you go along the RR script?
live_plot = False                    # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = True                      # fit the data here and save or plot the fits?
save_data_h5 = True                  # save all of the data to h5 files?
verbose = True                       # print everything to the console in real time, good for debugging, bad for memory
qick_verbose = True                  # qick verbose prints the progress bar for each qick experiment as it is happening (the red bar that fills out as more experiment rounds/reps are being done)
debug_mode = True                    # if True, it disables the continuing function of RR if an error pops up in a class -- errors now stop the RR script
thresholding = False                 # use internal QICK threshold for ratio of Binary values on y for rabi/t1/t2r/t2e, or analog avg when false
increase_qubit_reps = False          # if you want to increase the reps for a qubit, set to True
unmask = True                        # Do you want to use the unmasking feature to increase resonator gain?
qubit_to_increase_reps_for = 0       # only has impact if previous line is True
multiply_qubit_reps_by = 2           # only has impact if the line two above is True

Qs_to_look_at = [5]     # only list the qubits you want to do the RR for

#Data saving info
run_name = 'bob_run_started_Feb_11'
device_name = 'squill'
substudy_txt_notes = ('track res and q spec')
study ='find_ef_qfreqs'#'higher_spec_transitions'
sub_study = f'qubit_' + str(5)
data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# set which of the following you'd like to run to 'True'
run_flags = {"tof": False, "res_spec": True, "q_spec": True, "ss":  False, "rabi":  True, "ss_gef": False, "test_act": False, "fh_rabi": False,
             "t1":  False, "t2r": False, "t2r_correction":True, "t2e":  False, "ef_res_spec": True, "ef_q_spec": True, "fh_q_spec": False, "rabi_pop_meas": False, "ef_Rabi": False, "ef_ss": False}


# optimization outputs from qick board, unmasking set to true
res_leng_vals = [9]*6
res_gain = [0.25,0.25,0.25,0.25,0.24,0.25]
freq_offsets = [0,0,0,0,-0.25,-0.15]

qubit_freqs_ef = [None]*6
increase_steps_to_ef = 600
ef_res_sample_number = 1
number_of_qubits = 6
figure_quality = 200
################################################ Data Saving Setup ##################################################
#Folders

data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

if not os.path.exists(f"M:/_Data/20250822 - Olivia/{run_name}/"):
    os.makedirs(f"M:/_Data/20250822 - Olivia/{run_name}/")
if not os.path.exists(f"M:/_Data/20250822 - Olivia/{run_name}/{device_name}/"):
    os.makedirs(f"M:/_Data/20250822 - Olivia/{run_name}/{device_name}/")
studyFolder = os.path.join(f"M:/_Data/20250822 - Olivia/{run_name}/{device_name}/", study)
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

file_path = os.path.join(studyDocumentationFolder, 'sub_study_notes.txt')
with open(file_path, "w", encoding="utf-8") as file:
    file.write(substudy_txt_notes)

################################################## Configure logging ###################################################
''' We need to create a custom logger and disable propagation like this
to remove the logs from the underlying qick from saving to the log file for RR'''

log_file = os.path.join(studyDocumentationFolder, "RR_script.log")
rr_logger = logging.getLogger("custom_logger_for_rr_only")
rr_logger.setLevel(logging.DEBUG)

file_handler = logging.FileHandler(log_file, mode='a')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)

rr_logger.addHandler(file_handler)
rr_logger.propagate = False  #dont propagate logs from underlying qick package

####################################################### RR #############################################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Define what to save to h5 files
tof_keys = ['Dates', 'iq_list','t', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num','Recycled QFreq',
              'Exp Config', 'Syst Config']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
ss_ef_keys = ['Fidelity', 'Angle', 'Dates', 'I_e', 'Q_e', 'I_f', 'Q_f', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t2r_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
t2r_correction_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Correction Freq', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
t2e_keys = ['T2E', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
rabi_keys_ef_Qtemps = ['Dates', 'Qfreq_ge', 'I1', 'Q1', 'Gains1', 'Fit1', 'I2', 'Q2', 'Gains2', 'Fit2', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
ss_keys_gef = ['Fidelity', 'Angle_ef', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'I_f', 'Q_f', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
act_keys = [ 'actI', 'actQ','noactI', 'noactQ', 'Syst Config']
#initialize a simple list to store the qspec values in incase a fit fails
stored_qspec_list = [None] * tot_num_of_qubits
# True
if live_plot:
    # Check if visdom is connected right away, otherwise, throw an error
    if not (viz := visdom.Visdom()).check_connection(timeout_seconds=5):
        raise RuntimeError("Visdom server not connected!, Type \'visdom\' into the command line and go to "
                           "http://localhost:8097/ on firefox")

# initialize a dictionary to store those values
tof_data = create_data_dict(tof_keys, save_r, list_of_all_qubits)
res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
ef_ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
t2r_correction_1_data = create_data_dict(t2r_correction_keys, save_r, list_of_all_qubits)
t2r_correction_2_data = create_data_dict(t2r_correction_keys, save_r, list_of_all_qubits)
rabi_corrected_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)

ef_res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
ef_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
ef_rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
fh_rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
fh_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
rabi_data_ef_Qtemps = create_data_dict(rabi_keys_ef_Qtemps, save_r, list_of_all_qubits)
ss_data_gef = create_data_dict(ss_keys_gef, save_r, list_of_all_qubits)
act_data = create_data_dict(act_keys, save_r, list_of_all_qubits)

batch_num=0
j = 0
angles=[]
while j < n:
    j += 1
    for QubitIndex in Qs_to_look_at:
        recycled_qfreq = False

        #Get the config for this qubit
        experiment = QICK_experiment(optimizationFolder, DAC_attenuator1 = 10, DAC_attenuator2 = 15, qubit_DAC_attenuator1 = 5,
                                     qubit_DAC_attenuator2 = 4, ADC_attenuator = 30, fridge=FRIDGE) # ADC_attenuator MUST be above 16dB
        experiment.create_folder_if_not_exists(optimizationFolder)

        experiment.readout_cfg['res_gain_ge'] = res_gain[QubitIndex]
        experiment.readout_cfg['res_gain_ef'] = res_gain[QubitIndex]
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]
        experiment.readout_cfg['res_freq_ge'] = experiment.readout_cfg['res_freq_ge'][QubitIndex] + freq_offsets[QubitIndex]
        experiment.readout_cfg['res_freq_ef'] = experiment.readout_cfg['res_freq_ef'][QubitIndex]

        experiment.qubit_cfg['qubit_freq_ge'] = experiment.qubit_cfg['qubit_freq_ge'][QubitIndex]
        experiment.qubit_cfg['qubit_gain_ge'] = experiment.qubit_cfg['qubit_gain_ge'][QubitIndex]

        # ################################################# g-e Res spec ####################################################
        # if run_flags["res_spec"]:
        #     try:
        #         res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs,
        #                                          experiment=experiment, verbose=verbose, logger=rr_logger,
        #                                          unmasking_resgain=unmask)
        #         res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
        #
        #         offset = freq_offsets[
        #             QubitIndex]  # use optimized offset values or whats set at top of script based on pre_optimize flag
        #         offset_res_freqs = [r + offset for r in res_freqs]
        #         experiment.readout_cfg['res_freq_ge'] = offset_res_freqs[0]
        #         del res_spec
        #
        #     except Exception as e:
        #         if debug_mode:
        #             raise e  # In debug mode, re-raise the exception immediately
        #         else:
        #             rr_logger.exception(f'Got the following error, continuing: {e}')
        #             if verbose: print(f'Got the following error, continuing: {e}')
        #             continue  # skip the rest of this qubit

        ################################################## g-e Qubit spec ##################################################
        if run_flags["q_spec"]:
            try:
                q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j,
                                           signal, save_figs, plot_fit=True, experiment=experiment,
                                           live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                           unmasking_resgain=unmask)
                (qspec_I, qspec_Q, qspec_freqs, qspec_fit, qubit_freq, sys_config_qspec, ss_Q_e_qspec, ss_Q_g_qspec,
                 ss_I_e_qspec,
                 ss_I_g_qspec, I_shots_qspec, Q_shots_qspec) = q_spec.run(scaling=True)

                if qubit_freq is None:
                    if stored_qspec_list[QubitIndex] is not None:
                        experiment.qubit_cfg['qubit_freq_ge'] = stored_qspec_list[QubitIndex]
                        rr_logger.warning(f"Using previous stored value: {stored_qspec_list[QubitIndex]}")
                        recycled_qfreq = True
                        qubit_freq = stored_qspec_list[QubitIndex]
                        experiment.qubit_cfg['qubit_freq_ge'] = float(qubit_freq)
                        stored_qspec_list[QubitIndex] = float(qubit_freq)
                        if verbose:
                            print(f"Using previous stored value: {qubit_freq}")
                    else:
                        rr_logger.warning(f"No stored g-e qubit spec value for qubit {QubitIndex}; skipping iteration.")
                        if verbose:
                            print('No stored g-e qubit spec value for qubit {QubitIndex}; skipping iteration.')
                        del q_spec

                    continue
                else:
                    experiment.qubit_cfg['qubit_freq_ge'] = float(qubit_freq)
                    stored_qspec_list[QubitIndex] = float(qubit_freq)
                rr_logger.info(f"g-e Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
                if verbose:
                    print(f"g-e Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
                del q_spec

            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"RR g-e QSpec error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"RR g-e QSpec error on qubit {QubitIndex}: {e}")
                continue
        ###################################################### g-e Rabi ####################################################
        if run_flags["rabi"]:
            # try:
            rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                                           save_figs=save_figs, save_shots=False,
                                           experiment=experiment, live_plot=live_plot,
                                           increase_qubit_reps=increase_qubit_reps,
                                           qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                           multiply_qubit_reps_by=multiply_qubit_reps_by,
                                           verbose=verbose, logger=rr_logger, unmasking_resgain=unmask)
            (rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp,
             sys_config_rabi, ss_Q_e, ss_Q_g, ss_I_e, ss_I_g, rabi_I_shots, rabi_Q_shots) = rabi.run(
                thresholding=thresholding, scaling=True)

            # if these are None, fit didnt work
            if (rabi_fit is None and pi_amp is None):
                rr_logger.info('g-e Rabi fit didnt work, skipping the rest of this qubit')
                if verbose: print('g-e Rabi fit didnt work, skipping the rest of this qubit')
                continue  # skip the rest of this qubit

            experiment.qubit_cfg['pi_amp'] = float(pi_amp)
            rr_logger.info(f'g-e Pi amplitude for qubit {QubitIndex + 1} is: {float(pi_amp)}')
            if verbose: print('g-e Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))
            del rabi

        ################################################ Qubit Spec EF ################################################
        if run_flags["ef_q_spec"]:
            rr_logger.info("----------------- Starting Qubit Spec EF  -----------------")
            if verbose:
                print("----------------- Starting Qubit Spec EF  -----------------")


            ef_q_spec = EFQubitSpectroscopy(QubitIndex, number_of_qubits, studyDocumentationFolder, j, signal,
                           True, experiment, live_plot, unmasking_resgain = unmask)

            efqspec_I, efqspec_Q, efqspec_freqs, sys_config_qspec_ef, efqspec_I_fit, efqspec_Q_fit, efqubit_freq = ef_q_spec.run()

            qubit_freqs_ef[QubitIndex] = efqubit_freq
            experiment.qubit_cfg['qubit_freq_ef'][QubitIndex] = float(efqubit_freq)

            if verbose:
                print(f"EF Qubit {QubitIndex + 1} frequency: {efqubit_freq}")

            del ef_q_spec


        ############################################### Collect Results ################################################
        if save_data_h5:

            # ---------------------Collect g-e QSpec Results----------------
            if run_flags["q_spec"]:

                qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1]=(
                    time.mktime(datetime.datetime.now().timetuple()))
                qspec_data[QubitIndex]['I'][j - batch_num * save_r - 1] = qspec_I
                qspec_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = qspec_Q
                qspec_data[QubitIndex]['Frequencies'][j - batch_num * save_r - 1] = qspec_freqs
                qspec_data[QubitIndex]['I Fit'][j - batch_num * save_r - 1] = qspec_fit
                # qspec_data[QubitIndex]['Q Fit'][j - batch_num * save_r - 1] = qspec_Q_fit
                qspec_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                qspec_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                qspec_data[QubitIndex]['Recycled QFreq'][j - batch_num * save_r - 1] = recycled_qfreq
                qspec_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                qspec_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_qspec


            # ---------------------Collect e-f qspec Results----------------
            if run_flags["ef_q_spec"]:
                ef_qspec_data[QubitIndex]['Dates'][0] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                ef_qspec_data[QubitIndex]['I'][0] = efqspec_I
                ef_qspec_data[QubitIndex]['Q'][0] = efqspec_Q
                ef_qspec_data[QubitIndex]['Frequencies'][0] = efqspec_freqs
                ef_qspec_data[QubitIndex]['I Fit'][0] = efqspec_I_fit
                ef_qspec_data[QubitIndex]['Q Fit'][0] = efqspec_Q_fit
                ef_qspec_data[QubitIndex]['Round Num'][0] = j
                ef_qspec_data[QubitIndex]['Batch Num'][0] = batch_num
                ef_qspec_data[QubitIndex]['Recycled QFreq'][0] = False  # no rr so no recycling here
                ef_qspec_data[QubitIndex]['Exp Config'][0] = expt_cfg
                ef_qspec_data[QubitIndex]['Syst Config'][0] = sys_config_qspec_ef

        del experiment

    ################################################## Potentially Save ################################################
    if save_data_h5:
        # Check if you are at the right round number
        # If so, then save all of the data and change the round num so you replace data starting next round
        if j % save_r == 0:
            batch_num+=1

            # --------------------------save g-e Res Spec-----------------------
            if run_flags["res_spec"]:
                saver_res = Data_H5(subStudyDataFolder, res_data, batch_num, save_r)
                saver_res.save_to_h5('res_ge')
                del saver_res
                del res_data

            # --------------------------save g-e QSpec-----------------------
            if run_flags["q_spec"]:
                saver_qspec = Data_H5(subStudyDataFolder, qspec_data, batch_num, save_r)
                saver_qspec.save_to_h5('qspec_ge')
                del saver_qspec
                del qspec_data


            # --------------------------save e-f qspec-----------------------
            if run_flags["ef_q_spec"]:
                saver_ef_qspec = Data_H5(subStudyDataFolder, ef_qspec_data, batch_num, save_r)
                saver_ef_qspec.save_to_h5('qspec_ef')
                del saver_ef_qspec
                del ef_qspec_data


    # reset all dictionaries to none for safety
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    fh_rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
    ss_data_ef = create_data_dict(ss_keys, save_r, list_of_all_qubits)
    ef_ss_data = create_data_dict(ss_ef_keys, save_r, list_of_all_qubits)
    ef_res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    ef_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    fh_qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    rabi_data_ef_Qtemps = create_data_dict(rabi_keys_ef_Qtemps, save_r, list_of_all_qubits)
    t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
    t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
    t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
    ss_data_gef = create_data_dict(ss_keys_gef, save_r, list_of_all_qubits)
    act_data = create_data_dict(act_keys, save_r, list_of_all_qubits)
    t2r_correction_1_data = create_data_dict(t2r_correction_keys, save_r, list_of_all_qubits)
    t2r_correction_2_data = create_data_dict(t2r_correction_keys, save_r, list_of_all_qubits)
    rabi_corrected_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)

en=time.time()
print('timetaken=',en-st)