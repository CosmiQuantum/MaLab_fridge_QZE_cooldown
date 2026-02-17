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

Qs_to_look_at = [0,1,2,3,4,5]     # only list the qubits you want to do the RR for

#Data saving info
run_name = 'bob_run_started_Feb_11'
device_name = 'squill'
substudy_txt_notes = ('track res and q spec')
study ='track_res_spec_lower_gain0p05_len_4us'#'higher_spec_transitions'
sub_study = f'qubit_' + str(4)
data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# set which of the following you'd like to run to 'True'
run_flags = {"tof": False, "res_spec": True, "q_spec": True, "ss":  False, "rabi":  True, "ss_gef": False, "test_act": False, "fh_rabi": False,
             "t1":  False, "t2r": False, "t2r_correction":True, "t2e":  False, "ef_res_spec": True, "ef_q_spec": False, "fh_q_spec": False, "rabi_pop_meas": False, "ef_Rabi": False, "ef_ss": False}


# optimization outputs from qick board, unmasking set to true
res_leng_vals =[4]*6 #[10]*6
res_gain =[0.05,0.05, 0.05, 0.05, 0.05, 0.05] #[0.15,0.2, 0.2, 0.2, 0.2833, 0.15]
freq_offsets = [-0.15,0,-0.15,-0.15,0,0]

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
        experiment.readout_cfg['res_freq_ge'] = experiment.readout_cfg['res_freq_ge'][QubitIndex]

        experiment.qubit_cfg['qubit_freq_ge'] = experiment.qubit_cfg['qubit_freq_ge'][QubitIndex]
        experiment.qubit_cfg['qubit_gain_ge'] = experiment.qubit_cfg['qubit_gain_ge'][QubitIndex]
        ###################################################### TOF #####################################################
        if run_flags["tof"]:
            tof        = TOFExperiment(QubitIndex, studyDocumentationFolder, experiment, j, save_figs, unmasking_resgain = unmask)
            t, iq_list, tof_config = tof.run()
            del tof

        ################################################# g-e Res spec ####################################################
        if run_flags["res_spec"]:
            try:
                res_spec   = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs,
                                                   experiment = experiment, verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
                res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()

                offset = freq_offsets[QubitIndex] #use optimized offset values or whats set at top of script based on pre_optimize flag
                offset_res_freqs = [r + offset for r in res_freqs]
                experiment.readout_cfg['res_freq_ge'] = offset_res_freqs[0]
                del res_spec

            except Exception as e:
                if debug_mode:
                    raise e # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
                    continue #skip the rest of this qubit

        # ################### Roll Signal into I (need to configure for recent updates) ################################
        # #get the average theta value, then use that to rotate the signal. Plug that value into system_config res_phase
        # leng=4
        # ss = SingleShotGE(QubitIndex, outerFolder, experiment, j, save_figs)
        # fid, angle, iq_list_g, iq_list_e = ss.run()
        # angles.append(angle)
        # #rr_logger.info(angles)
        # #rr_logger.info('avg theta: ', np.average(angles))
        # del ss

        ############################################# res spec ef ####################################################
        if run_flags["ef_res_spec"]:
            rr_logger.info("----------------- Starting Res Spec EF  -----------------")
            if verbose:
                print("----------------- Starting Res Spec EF  -----------------")

            ef_res_freqs_samples = []
            for sample in range(ef_res_sample_number):
                try:
                    ef_res_spec = ResonanceSpectroscopyEF(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, sample,
                                                          save_figs, experiment=experiment, verbose=verbose,
                                                          logger=rr_logger, qick_verbose=qick_verbose, unmasking_resgain = unmask)
                    ef_res_freqs, ef_freq_pts, ef_freq_center, ef_amps, sys_config_rspec_ef = ef_res_spec.run()
                    ef_res_freqs_samples.append(ef_res_freqs)
                    rr_logger.info(f"EF ResSpec sample {sample} for qubit {QubitIndex + 1}: {ef_res_freqs}")

                    del ef_res_spec

                except Exception as e:
                    if debug_mode:
                        raise  # In debug mode, re-raise the exception immediately
                    rr_logger.exception(f"EF ResSpec error on qubit {QubitIndex + 1} sample {sample}: {e}")


            if ef_res_freqs_samples:
                # Average the resonator frequency values across samples
                avg_ef_res_freqs = np.mean(np.array(ef_res_freqs_samples), axis=0).tolist()
            else:
                rr_logger.error(f"No resonator spectroscopy data collected for qubit {QubitIndex + 1}.")

            experiment.readout_cfg['res_freq_ef'] = ef_res_freqs[0]

            rr_logger.info(f"Avg. EF resonator frequencies for qubit {QubitIndex + 1}: {avg_ef_res_freqs}")
            if verbose:
                print(f"Avg. EF resonator frequencies for qubit {QubitIndex + 1}: {avg_ef_res_freqs}")


        ############################################### Collect Results ################################################
        if save_data_h5:

            # ---------------------Collect g-e Res Spec Results----------------
            if run_flags["res_spec"]:
                res_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                res_data[QubitIndex]['freq_pts'][j - batch_num * save_r - 1] = freq_pts
                res_data[QubitIndex]['freq_center'][j - batch_num * save_r - 1] = freq_center
                res_data[QubitIndex]['Amps'][j - batch_num * save_r - 1] = amps
                res_data[QubitIndex]['Found Freqs'][j - batch_num * save_r - 1] = res_freqs
                res_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                res_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                res_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
                res_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rspec


            # ---------------------Collect e-f res spec Results----------------
            if run_flags["ef_res_spec"]:
                ef_res_data[QubitIndex]['Dates'][0] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                ef_res_data[QubitIndex]['freq_pts'][0] = ef_freq_pts
                ef_res_data[QubitIndex]['freq_center'][0] = ef_freq_center
                ef_res_data[QubitIndex]['Amps'][0] = ef_amps
                ef_res_data[QubitIndex]['Found Freqs'][0] = ef_res_freqs
                ef_res_data[QubitIndex]['Round Num'][0] = j
                ef_res_data[QubitIndex]['Batch Num'][0] = batch_num
                ef_res_data[QubitIndex]['Exp Config'][0] = expt_cfg
                ef_res_data[QubitIndex]['Syst Config'][0] = sys_config_rspec_ef


        del experiment

    ################################################## Potentially Save ################################################
    if save_data_h5:
        # Check if you are at the right round number
        # If so, then save all of the data and change the round num so you replace data starting next round
        if j % save_r == 0:
            batch_num+=1

            # -----------------------------save tof----------------------------
            if run_flags["tof"]:
                saver_res = Data_H5(subStudyDataFolder, tof_data, batch_num, save_r)
                saver_res.save_to_h5('tof')
                del saver_res
                del res_data
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

            # --------------------------save g-e Rabi-----------------------
            if run_flags["rabi"]:
                saver_rabi = Data_H5(subStudyDataFolder, rabi_data, batch_num, save_r)
                saver_rabi.save_to_h5('rabi_ge')
                del saver_rabi
                del rabi_data

            # --------------------------save g-e SS-----------------------
            if run_flags["ss"]:
                saver_ss = Data_H5(subStudyDataFolder, ss_data, batch_num, save_r)
                saver_ss.save_to_h5('ss_ge')
                del saver_ss
                del ss_data

            # --------------------------save e-f res spec-----------------------
            if run_flags["ef_res_spec"]:
                saver_ef_res = Data_H5(subStudyDataFolder, ef_res_data, batch_num, save_r)  # save
                saver_ef_res.save_to_h5('res_ef')
                del saver_ef_res
                del ef_res_data
            # --------------------------save e-f qspec-----------------------
            if run_flags["ef_q_spec"]:
                saver_ef_qspec = Data_H5(subStudyDataFolder, ef_qspec_data, batch_num, save_r)
                saver_ef_qspec.save_to_h5('qspec_ef')
                del saver_ef_qspec
                del ef_qspec_data
            # --------------------------save e-f Rabi-----------------------
            if run_flags["ef_Rabi"]:
                saver_ef_rabi = Data_H5(subStudyDataFolder, ef_rabi_data, batch_num, save_r)
                saver_ef_rabi.save_to_h5('rabi_ef')
                del saver_ef_rabi
                del ef_rabi_data

            # --------------------------save e-f Rabi-----------------------
            if run_flags["fh_rabi"]:
                saver_fh_rabi = Data_H5(subStudyDataFolder, fh_rabi_data, batch_num, save_r)
                saver_fh_rabi.save_to_h5('rabi_fh')
                del saver_fh_rabi
                del fh_rabi_data
            # --------------------------save f-h qspec-----------------------
            if run_flags["fh_q_spec"]:
                saver_ef_qspec = Data_H5(subStudyDataFolder, fh_qspec_data, batch_num, save_r)
                saver_ef_qspec.save_to_h5('qspec_fh')
                del saver_ef_qspec
                del fh_qspec_data

            # --------save rabi population measurements (qubit temperature data) -----------------------
            if run_flags["rabi_pop_meas"]:
                saver_rabi_Qtemps = Data_H5(subStudyDataFolder, rabi_data_ef_Qtemps, batch_num, save_r)
                saver_rabi_Qtemps.save_to_h5('q_temperatures')
                del saver_rabi_Qtemps
                del rabi_data_ef_Qtemps
            # --------------------------save g-e t1-----------------------
            if run_flags["t1"]:
                saver_t1 = Data_H5(subStudyDataFolder, t1_data, batch_num, save_r)
                saver_t1.save_to_h5('t1_ge')
                del saver_t1
                del t1_data

            #--------------------------save g-e t2r-----------------------
            if run_flags["t2r"]:
                saver_t2r = Data_H5(subStudyDataFolder, t2r_data, batch_num, save_r)
                saver_t2r.save_to_h5('t2_ge')
                del saver_t2r
                del t2r_data

            # --------------------------save g-e t2r correction 1-----------------------
            if run_flags["t2r_correction"]:
                saver_t2r = Data_H5(subStudyDataFolder, t2r_correction_1_data, batch_num, save_r)
                saver_t2r.save_to_h5('t2_ge_correction_1')
                del saver_t2r
                del t2r_correction_1_data

                saver_t2r = Data_H5(subStudyDataFolder, t2r_correction_2_data, batch_num, save_r)
                saver_t2r.save_to_h5('t2_ge_correction_2')
                del saver_t2r
                del t2r_correction_2_data

                saver_rabi = Data_H5(subStudyDataFolder, rabi_corrected_data, batch_num, save_r)
                saver_rabi.save_to_h5('rabi_ge_corrected')
                del saver_rabi
                del rabi_corrected_data

            #--------------------------save g-e t2e-----------------------
            if run_flags["t2e"]:
                saver_t2e = Data_H5(subStudyDataFolder, t2e_data, batch_num, save_r)
                saver_t2e.save_to_h5('t2e_ge')
                del saver_t2e
                del t2e_data

            # --------------------------save e-f SS-----------------------
            if run_flags["ef_ss"]:
                saver_ss_ef = Data_H5(subStudyDataFolder, ef_ss_data, batch_num, save_r)
                saver_ss_ef.save_to_h5('ef_ss')
                del saver_ss_ef
                del ef_ss_data

            # --------------------------save g-e-f SS-----------------------
            if run_flags["ss_gef"]:
                saver_ss = Data_H5(subStudyDataFolder, ss_data_gef, batch_num, save_r)
                saver_ss.save_to_h5('SS_gef')
                del saver_ss
                del ss_data_gef

            # --------------------------save g-e SS-----------------------
            if run_flags["test_act"]:
                saver_act = Data_H5(subStudyDataFolder, act_data, batch_num, save_r)
                saver_act.save_to_h5('test_act')
                del saver_act
                del act_data

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