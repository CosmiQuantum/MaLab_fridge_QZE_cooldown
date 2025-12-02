import sys
import os
from copy import deepcopy

import numpy as np

np.set_printoptions(threshold=int(1e15)) #need this so it saves absolutely everything returned from the classes
import datetime
import time
import logging
import visdom
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_006p5_length_rabi_ge import LengthRabiExperiment
from section_005_single_shot_ge import SingleShot
from section_007_T1_ge import T1Measurement
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
from section_008_save_data_to_h5 import Data_H5
from system_config import QICK_experiment
from expt_config import expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE
from section_007_T1_ge_IBM_zeno import T1Measurement_with_Zeno
from section_007_T1_ge_zeno_loop import T1Measurement_with_Zeno_loop
from section_009_T2R_ge_zeno import T2RMeasurementZeno
from section_010_T2E_ge_zeno import T2EMeasurementZeno
from section_004_qubit_spec_ge_zeno import QubitSpectroscopyZeno
################################################ Run Configurations ####################################################
zero_qubit_drive_gain = False
constant_zeno_pulse = True
adapt_starked_qubit_freq = False
wait_for_res_ring_up = True
n= 1
unmask = True
save_r = 1                           # how many rounds to save after
signal = 'None'                      # 'I', or 'Q' depending on where the signal is (after optimization). Put 'None' if no optimization
save_figs = True                     # save plots for everything as you go along the RR script?
live_plot = False                    # for live plotting do "visdom" in comand line and then open http://localhost:8097/ on firefox
fit_data = False                     # fit the data here and save or plot the fits?
save_data_h5 = True                  # save all of the data to h5 files?
verbose = True                       # print everything to the console in real time, good for debugging, bad for memory
debug_mode = True                  # if True, it disables the continuing function of RR if an error pops up in a class -- errors now stop the RR script
thresholding = False                 # use internal QICK threshold for ratio of Binary values on y for rabi/t1/t2r/t2e, or analog avg when false
increase_qubit_reps = False          # if you want to increase the reps for a qubit, set to True
qubit_to_increase_reps_for = 0       # only has impact if previous line is True
multiply_qubit_reps_by = 2           # only has impact if the line two above is True
Qs_to_look_at = [4]        # only list the qubits you want to do the RR for

#Data saving info
run_name = 'bob_run_started_Aug_23'
device_name = 'squill'
substudy_txt_notes = ('script longitudinal_qze_005_ge_ramsey.py')

study = '2d_less_reps_more_qspec_steps'

################################################ optimization outputs ##################################################
# Optimization parameters for resonator spectroscopy
res_leng_vals = [10]*6
res_gain = [0.15,0.2, 0.2, 0.2, 0.2833, 0.15]
freq_offsets = [0, -0.15, -0.15,-0.15, -0.08, -0.15]
####################################################### RR #############################################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}

# Define what to save to h5 files
t2r_correction_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Correction Freq', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config', 'I_shots', 'Q_shots']
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num','Recycled QFreq',
              'Exp Config', 'Syst Config','ss_Q_e', 'ss_Q_g','ss_I_e', 'ss_I_g', 'I_shots', 'Q_shots', 'Gains']
rabi_keys = ['Dates', 'I', 'Q', 'Gains', 'Fit', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config',  'ss_Q_e', 'ss_Q_g','ss_I_e', 'ss_I_g', 'I_shots', 'Q_shots']
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config', 'I_shots', 'Q_shots']
ss_ef_keys = ['Fidelity', 'Angle', 'Dates', 'I_e', 'Q_e', 'I_f', 'Q_f', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config', 'I_shots', 'Q_shots']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config','ss_Q_e', 'ss_Q_g','ss_I_e', 'ss_I_g', 'I_shots', 'Q_shots', 'Gains']
t2r_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config', 'ss_Q_e', 'ss_Q_g','ss_I_e', 'ss_I_g', 'I_shots', 'Q_shots', 'Gains']
t2e_keys = ['T2E', 'Errors', 'Dates', 'I', 'Q', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config', 'ss_Q_e', 'ss_Q_g','ss_I_e', 'ss_I_g', 'I_shots', 'Q_shots', 'Gains']
rabi_keys_ef_Qtemps = ['Dates', 'Qfreq_ge', 'I1', 'Q1', 'Gains1', 'Fit1', 'I2', 'Q2', 'Gains2', 'Fit2', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config', 'I_shots', 'Q_shots']
ss_keys_gef = ['Fidelity', 'Angle_ef', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'I_f', 'Q_f', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config', 'I_shots', 'Q_shots']

#initialize a simple list to store the qspec values in incase a fit fails
stored_qspec_list = [None] * tot_num_of_qubits

if live_plot:
    # Check if visdom is connected right away, otherwise, throw an error
    if not (viz := visdom.Visdom()).check_connection(timeout_seconds=5):
        raise RuntimeError("Visdom server not connected!, Type \'visdom\' into the command line and go to "
                           "http://localhost:8097/ on firefox")



############################## Get first the res, qubit freqs #######################################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}


res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)



#Vary the readout pulse height, (number of photons int he resonator) and look at rabi (fig 2 here
# https://iopscience.iop.org/article/10.1088/1367-2630/17/6/063035/pdf)
batch_num=0
j = 0
# slices=[5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100]
for QubitIndex in Qs_to_look_at:
    sub_study = 'all_qubits'
    data_set = f'qubit_' + str(
        QubitIndex) + f'round0'  # + '_' +datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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
    rr_logger.propagate = False  # dont propagate logs from underlying qick package

    ####################################### start everything #######################################
    # Get the config for this qubit
    experiment = QICK_experiment(optimizationFolder, DAC_attenuator1=10, DAC_attenuator2=15,
                                 qubit_DAC_attenuator1=5,
                                 qubit_DAC_attenuator2=4, ADC_attenuator=17,
                                 fridge=FRIDGE)  # ADC_attenuator MUST be above 16dB
    experiment.create_folder_if_not_exists(optimizationFolder)

    experiment.readout_cfg['res_gain_ge'] = res_gain[QubitIndex]
    experiment.readout_cfg['res_gain_ef'] = res_gain[QubitIndex]
    experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]
    experiment.readout_cfg['res_freq_ge'] = experiment.readout_cfg['res_freq_ge'][QubitIndex]

    experiment.qubit_cfg['qubit_freq_ge'] = experiment.qubit_cfg['qubit_freq_ge'][QubitIndex]
    experiment.qubit_cfg['qubit_gain_ge'] = experiment.qubit_cfg['qubit_gain_ge'][QubitIndex]
    ################################ Do Res spec once per qubit and store the value ####################################
    ################################################# g-e Res spec ####################################################

    res_spec = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs,
                                     experiment=experiment, verbose=verbose, logger=rr_logger,
                                     unmasking_resgain=unmask)
    res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
    offset = freq_offsets[
        QubitIndex]  # use optimized offset values or whats set at top of script based on pre_optimize flag
    offset_res_freqs = [r + offset for r in res_freqs]# [experiment.readout_cfg['res_freq_ge']]]#res_freqs]
    experiment.readout_cfg['res_freq_ge'] = offset_res_freqs[0]
    del res_spec

    res_data[QubitIndex]['Dates'][0] = (
        time.mktime(datetime.datetime.now().timetuple()))
    res_data[QubitIndex]['freq_pts'][0] = freq_pts
    res_data[QubitIndex]['freq_center'][0] = freq_center
    res_data[QubitIndex]['Amps'][0] = amps
    res_data[QubitIndex]['Found Freqs'][0] = res_freqs
    res_data[QubitIndex]['Batch Num'][0] = 0
    res_data[QubitIndex]['Exp Config'][0] = expt_cfg
    res_data[QubitIndex]['Syst Config'][0] = sys_config_rspec

    saver_res = Data_H5(optimizationFolder, res_data, 0, save_r)  # save
    saver_res.save_to_h5('Res')
    del saver_res
    del res_data
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)  # initialize again to a blank for saftey

    ############ Qubit Spec ##############
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)

    try:
        q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j,
                                   signal, save_figs, plot_fit=True, experiment=experiment,
                                   live_plot=live_plot, verbose=verbose, logger=rr_logger, unmasking_resgain=unmask)
        (qspec_I, qspec_Q, qspec_freqs, qspec_fit, qubit_freq, sys_config_qspec, ss_Q_e_qspec, ss_Q_g_qspec,ss_I_e_qspec,
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

        qspec_data[QubitIndex]['Dates'][0] = (
            time.mktime(datetime.datetime.now().timetuple()))
        qspec_data[QubitIndex]['I'][0] = qspec_I
        qspec_data[QubitIndex]['Q'][0] = qspec_Q
        qspec_data[QubitIndex]['Frequencies'][0] = qspec_freqs
        qspec_data[QubitIndex]['I Fit'][0] = qspec_fit
        qspec_data[QubitIndex]['Round Num'][0] = 0
        qspec_data[QubitIndex]['Batch Num'][0] = 0
        qspec_data[QubitIndex]['Recycled QFreq'][0] = False  # no rr so no recycling here
        qspec_data[QubitIndex]['Exp Config'][0] = expt_cfg
        qspec_data[QubitIndex]['Syst Config'][0] = sys_config_qspec
        qspec_data[QubitIndex]['ss_Q_e'][0] = ss_Q_e_qspec
        qspec_data[QubitIndex]['ss_Q_g'][0] = ss_Q_g_qspec
        qspec_data[QubitIndex]['ss_I_e'][0] = ss_I_e_qspec
        qspec_data[QubitIndex]['ss_I_g'][0] = ss_I_g_qspec
        qspec_data[QubitIndex]['I_shots'][0] = I_shots_qspec
        qspec_data[QubitIndex]['Q_shots'][0] = Q_shots_qspec

        saver_qspec = Data_H5(optimizationFolder, qspec_data, 0, save_r)
        saver_qspec.save_to_h5('QSpec')
        del saver_qspec
        del qspec_data

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

    # reinitialize
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)

    ################### amp rabi ################
    rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)

    try:
        rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                                       save_figs=save_figs, save_shots=False,
                                       experiment=experiment, live_plot=live_plot,
                                       increase_qubit_reps=increase_qubit_reps,
                                       qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                       multiply_qubit_reps_by=multiply_qubit_reps_by,
                                       verbose=verbose, logger=rr_logger, unmasking_resgain=unmask)
        (rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp,
         sys_config_rabi,  ss_Q_e, ss_Q_g,ss_I_e, ss_I_g, rabi_I_shots, rabi_Q_shots) = rabi.run(thresholding=thresholding, scaling=True)

        # if these are None, fit didnt work
        if (rabi_fit is None and pi_amp is None):
            rr_logger.info('g-e Rabi fit didnt work, skipping the rest of this qubit')
            if verbose: print('g-e Rabi fit didnt work, skipping the rest of this qubit')
            continue  # skip the rest of this qubit

        experiment.qubit_cfg['pi_amp'] = float(pi_amp)
        rr_logger.info(f'g-e Pi amplitude for qubit {QubitIndex + 1} is: {float(pi_amp)}')
        if verbose: print('g-e Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))
        del rabi

        rabi_data[QubitIndex]['Dates'][0] = (
            time.mktime(datetime.datetime.now().timetuple()))
        rabi_data[QubitIndex]['I'][0] = rabi_I
        rabi_data[QubitIndex]['Q'][0] = rabi_Q
        rabi_data[QubitIndex]['Gains'][0] = rabi_gains
        rabi_data[QubitIndex]['Fit'][0] = rabi_fit
        rabi_data[QubitIndex]['Round Num'][0] = 0
        rabi_data[QubitIndex]['Batch Num'][0] = 0
        rabi_data[QubitIndex]['Exp Config'][0] = expt_cfg
        rabi_data[QubitIndex]['Syst Config'][0] = sys_config_rabi
        rabi_data[QubitIndex]['ss_Q_e'][0] = ss_Q_e
        rabi_data[QubitIndex]['ss_Q_g'][0] = ss_Q_g
        rabi_data[QubitIndex]['ss_I_e'][0] = ss_I_e
        rabi_data[QubitIndex]['ss_I_g'][0] = ss_I_g
        rabi_data[QubitIndex]['I_shots'][0] = rabi_I_shots
        rabi_data[QubitIndex]['Q_shots'][0] = rabi_Q_shots

        saver_rabi = Data_H5(optimizationFolder, rabi_data, 0, save_r)
        saver_rabi.save_to_h5('Rabi')
        del saver_rabi
        del rabi_data
    except Exception as e:
        if debug_mode:
            raise e
        rr_logger.exception(f"Rabi error on qubit {QubitIndex}: {e}")
        continue

        rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    ########################################### tune up freq using ramsey ######################################
    t2r_correction_1_data = create_data_dict(t2r_correction_keys, save_r, list_of_all_qubits)
    t2r_correction_2_data = create_data_dict(t2r_correction_keys, save_r, list_of_all_qubits)
    rabi_corrected_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    try:
        t2r = T2RMeasurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
                             experiment=experiment, live_plot=live_plot, fit_data=True,
                             increase_qubit_reps=increase_qubit_reps,
                             qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                             multiply_qubit_reps_by=multiply_qubit_reps_by,
                             verbose=verbose, logger=rr_logger, unmasking_resgain=unmask, correction=True,
                             correction_round=1)
        t2r_est_1, t2r_err_1, t2r_I_1, t2r_Q_1, t2r_delay_times_1, fit_ramsey_1, sys_config_t2r_1, \
            ramsey_found_q_freq, t2r_I_shots_1, t2r_Q_shots_1 = t2r.adjust_qspec(
            thresholding=thresholding, correction=True)

        experiment.qubit_cfg['qubit_freq_ge'] = experiment.qubit_cfg['qubit_freq_ge'] - ramsey_found_q_freq + \
                                                expt_cfg['Ramsey_ge_correction']['ramsey_freq']
        del t2r

        # correct rabi
        rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                                       save_figs=save_figs, save_shots=False,
                                       experiment=experiment, live_plot=live_plot,
                                       increase_qubit_reps=increase_qubit_reps,
                                       qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                       multiply_qubit_reps_by=multiply_qubit_reps_by,
                                       verbose=verbose, logger=rr_logger, unmasking_resgain=unmask,
                                       correction=True)
        (rabi_I_corrected, rabi_Q_corrected, rabi_gains_corrected, rabi_fit_corrected, pi_amp_corrected,
         sys_config_rabi_corrected, ss_Q_e2, ss_Q_g2, ss_I_e2, ss_I_g2, I_shots_rabi_corr, Q_shots_rabi_corr) = rabi.run(thresholding=thresholding, scaling=True)

        # if these are None, fit didnt work
        if (rabi_fit_corrected is None and pi_amp_corrected is None):
            rr_logger.info('g-e Rabi fit didnt work, skipping the rest of this qubit')
            if verbose: print('g-e Rabi fit didnt work, skipping the rest of this qubit')
            continue  # skip the rest of this qubit

        experiment.qubit_cfg['pi_amp'] = float(pi_amp_corrected)
        rr_logger.info(f'corrected g-e Pi amplitude for qubit {QubitIndex + 1} is: {float(pi_amp_corrected)}')

        t2r_correction_1_data[QubitIndex]['T2'][j - batch_num * save_r - 1] = t2r_est_1
        t2r_correction_1_data[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t2r_err_1
        t2r_correction_1_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
            time.mktime(datetime.datetime.now().timetuple()))
        t2r_correction_1_data[QubitIndex]['I'][j - batch_num * save_r - 1] = t2r_I_1
        t2r_correction_1_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = t2r_Q_1
        t2r_correction_1_data[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t2r_delay_times_1
        t2r_correction_1_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = fit_ramsey_1
        t2r_correction_1_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
        t2r_correction_1_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
        t2r_correction_1_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
        t2r_correction_1_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2r_1
        t2r_correction_1_data[QubitIndex]['Correction Freq'][j - batch_num * save_r - 1] = ramsey_found_q_freq
        t2r_correction_1_data[QubitIndex]['I_shots'][0] = t2r_I_shots_1
        t2r_correction_1_data[QubitIndex]['Q_shots'][0] = t2r_Q_shots_1


        rabi_corrected_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
            time.mktime(datetime.datetime.now().timetuple()))
        rabi_corrected_data[QubitIndex]['I'][j - batch_num * save_r - 1] = rabi_I_corrected
        rabi_corrected_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = rabi_Q_corrected
        rabi_corrected_data[QubitIndex]['Gains'][j - batch_num * save_r - 1] = rabi_gains_corrected
        rabi_corrected_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = rabi_fit_corrected
        rabi_corrected_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
        rabi_corrected_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
        rabi_corrected_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
        rabi_corrected_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rabi_corrected
        rabi_corrected_data[QubitIndex]['ss_Q_e'][0] = ss_Q_e2
        rabi_corrected_data[QubitIndex]['ss_Q_g'][0] = ss_Q_g2
        rabi_corrected_data[QubitIndex]['ss_I_e'][0] = ss_I_e2
        rabi_corrected_data[QubitIndex]['ss_I_g'][0] = ss_I_g2
        rabi_corrected_data[QubitIndex]['I_shots'][0] = I_shots_rabi_corr
        rabi_corrected_data[QubitIndex]['Q_shots'][0] = Q_shots_rabi_corr

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

        # if verbose: print('corrected g-e Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))
        del rabi

    except Exception as e:
        if debug_mode:
            raise e
        rr_logger.exception(f"Rabi error on qubit {QubitIndex}: {e}")
        continue
    ########################################## g-e Single Shot Measurements ############################################
    ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
    ss = SingleShot(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs, experiment=experiment,
                    verbose=verbose, logger=rr_logger, unmasking_resgain=unmask)
    fid, angle, iq_list_g, iq_list_e, sys_config_ss = ss.run()
    I_g = iq_list_g[0][0].T[0]
    Q_g = iq_list_g[0][0].T[1]
    I_e = iq_list_e[0][0].T[0]
    Q_e = iq_list_e[0][0].T[1]

    ss_data[QubitIndex]['Fidelity'][j - batch_num * save_r - 1] = fid
    ss_data[QubitIndex]['Angle'][j - batch_num * save_r - 1] = angle
    ss_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
        time.mktime(datetime.datetime.now().timetuple()))
    ss_data[QubitIndex]['I_g'][j - batch_num * save_r - 1] = I_g
    ss_data[QubitIndex]['Q_g'][j - batch_num * save_r - 1] = Q_g
    ss_data[QubitIndex]['I_e'][j - batch_num * save_r - 1] = I_e
    ss_data[QubitIndex]['Q_e'][j - batch_num * save_r - 1] = Q_e
    ss_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
    ss_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
    ss_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
    ss_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_ss

    saver_ss = Data_H5(subStudyDataFolder, ss_data, batch_num, save_r)
    saver_ss.save_to_h5('ss_ge')
    del saver_ss
    del ss_data

    del ss

    ###################################################### g-e T1 ######################################################
    t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
    t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
    t1 = T1Measurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                       True,
                       experiment=experiment,
                       live_plot=live_plot, fit_data=fit_data,
                       increase_qubit_reps=increase_qubit_reps,
                       qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                       multiply_qubit_reps_by=multiply_qubit_reps_by,
                       verbose=verbose, logger=rr_logger, unmasking_resgain=unmask)
    t1_est, t1_err, t1_I, t1_Q, t1_delay_times, q1_fit_exponential, \
        sys_config_t1, ss_Q_e_t11, ss_Q_g_t11, ss_I_e_t11, ss_I_g_t11, I_shots_t11, Q_shots_t11 = t1.run(
        thresholding=thresholding, scaling=True)

    t1_data[QubitIndex]['T1'][j - batch_num * save_r - 1] = t1_est
    t1_data[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t1_err
    t1_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
        time.mktime(datetime.datetime.now().timetuple()))
    t1_data[QubitIndex]['I'][j - batch_num * save_r - 1] = t1_I
    t1_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = t1_Q
    t1_data[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t1_delay_times
    t1_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = q1_fit_exponential
    t1_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
    t1_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
    t1_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
    t1_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t1
    t1_data[QubitIndex]['ss_Q_e'][0] = ss_Q_e_t11
    t1_data[QubitIndex]['ss_Q_g'][0] = ss_Q_g_t11
    t1_data[QubitIndex]['ss_I_e'][0] = ss_I_e_t11
    t1_data[QubitIndex]['ss_I_g'][0] = ss_I_g_t11
    t1_data[QubitIndex]['I_shots'][0] = I_shots_t11
    t1_data[QubitIndex]['Q_shots'][0] = Q_shots_t11

    saver_t1 = Data_H5(subStudyDataFolder, t1_data, batch_num, save_r)
    saver_t1.save_to_h5('T1_ge_base')
    del saver_t1
    del t1
    ###################################################### g-e T2R #####################################################
    t2r = T2RMeasurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                         save_figs,
                         experiment=experiment, live_plot=live_plot, fit_data=fit_data,
                         increase_qubit_reps=increase_qubit_reps,
                         qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                         multiply_qubit_reps_by=multiply_qubit_reps_by,
                         verbose=verbose, logger=rr_logger, unmasking_resgain=unmask)
    t2r_est, t2r_err, t2r_I, t2r_Q, t2r_delay_times, fit_ramsey, sys_config_t2r, ss_Q_e_t2r, ss_Q_g_t2r,\
        ss_I_e_t2r, ss_I_g_t2r, I_shots_t2r, Q_shots_t2r = t2r.run(
        thresholding=thresholding, scaling=True)
    t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
    t2r_data[QubitIndex]['T2'][j - batch_num * save_r - 1] = t2r_est
    t2r_data[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t2r_err
    t2r_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
        time.mktime(datetime.datetime.now().timetuple()))
    t2r_data[QubitIndex]['I'][j - batch_num * save_r - 1] = t2r_I
    t2r_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = t2r_Q
    t2r_data[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t2r_delay_times
    t2r_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = fit_ramsey
    t2r_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
    t2r_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
    t2r_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
    t2r_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2r
    t2r_data[QubitIndex]['ss_Q_e'][0] = ss_Q_e_t2r
    t2r_data[QubitIndex]['ss_Q_g'][0] = ss_Q_g_t2r
    t2r_data[QubitIndex]['ss_I_e'][0] = ss_I_e_t2r
    t2r_data[QubitIndex]['ss_I_g'][0] = ss_I_g_t2r
    t2r_data[QubitIndex]['I_shots'][0] = I_shots_t2r
    t2r_data[QubitIndex]['Q_shots'][0] = Q_shots_t2r

    saver_t2 = Data_H5(subStudyDataFolder, t2r_data, batch_num, save_r)
    saver_t2.save_to_h5('T2_ge_base')
    del t2r
    t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
    t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
    ##################################################### g-e T2E #####################################################
    t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
    t2e = T2EMeasurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                         save_figs,
                         experiment=experiment, live_plot=live_plot, fit_data=fit_data,
                         increase_qubit_reps=increase_qubit_reps,
                         qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                         multiply_qubit_reps_by=multiply_qubit_reps_by,
                         verbose=verbose, logger=rr_logger, unmasking_resgain=unmask)
    t2e_est, t2e_err, t2e_I, t2e_Q, t2e_delay_times, fit_ramsey, sys_config_t2e, ss_Q_e_t2e, ss_Q_g_t2e, \
        ss_I_e_t2e, ss_I_g_t2e, I_shots_t2e, Q_shots_t2e = t2e.run(
        thresholding=thresholding, scaling=True)
    t2e_data[QubitIndex]['T2E'][j - batch_num * save_r - 1] = t2e_est
    t2e_data[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t2e_err
    t2e_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
        time.mktime(datetime.datetime.now().timetuple()))
    t2e_data[QubitIndex]['I'][j - batch_num * save_r - 1] = t2e_I
    t2e_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = t2e_Q
    t2e_data[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t2e_delay_times
    t2e_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = fit_ramsey
    t2e_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
    t2e_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
    t2e_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
    t2e_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2e
    t2e_data[QubitIndex]['ss_Q_e'][0] = ss_Q_e_t2e
    t2e_data[QubitIndex]['ss_Q_g'][0] = ss_Q_g_t2e
    t2e_data[QubitIndex]['ss_I_e'][0] = ss_I_e_t2e
    t2e_data[QubitIndex]['ss_I_g'][0] = ss_I_g_t2e
    t2e_data[QubitIndex]['I_shots'][0] = I_shots_t2e
    t2e_data[QubitIndex]['Q_shots'][0] = Q_shots_t2e

    saver_t2e = Data_H5(subStudyDataFolder, t2e_data, batch_num, save_r)
    saver_t2e.save_to_h5('T2E_ge_base')
    del t2e
    t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
    ########################################### go through each gain ######################################

    for repeat_round in range(20,1000):
        data_set = f'qubit_' + str(
            QubitIndex) + f'round{repeat_round}'  # + '_' +datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

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


        ############################################## Start IBM like experiment ###########################################
        exp = deepcopy(experiment) #before updating for qze
        # exp.readout_cfg['res_gain_qze'] = gain
        exp.readout_cfg['res_freq_qze'] = exp.readout_cfg['res_freq_ge']
        exp.readout_cfg['res_phase_qze'] = exp.readout_cfg['res_phase']


        ############################### High Gain Qubit Spec #####################################################
        normal_gain = exp.qubit_cfg['qubit_gain_ge']
        exp.qubit_cfg['qubit_gain_ge'] = 0.02
        qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)

        try:
            q_spec = QubitSpectroscopyZeno(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j,
                                           signal, save_figs, plot_fit=True, experiment=exp,
                                           live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                           unmasking_resgain=unmask)
            (qspec_I, qspec_Q, qspec_freqs, qspec_fit, qubit_freq, sys_config_qspec, ss_Q_e_qspec, ss_Q_g_qspec,
             ss_I_e_qspec, ss_I_g_qspec, I_shots_qspec, Q_shots_qspec, gains_qspec) = q_spec.run(scaling=True, qze_pulse='const')

            qspec_data[QubitIndex]['Dates'][0] = (
                time.mktime(datetime.datetime.now().timetuple()))
            qspec_data[QubitIndex]['I'][0] = qspec_I
            qspec_data[QubitIndex]['Q'][0] = qspec_Q
            qspec_data[QubitIndex]['Frequencies'][0] = qspec_freqs
            qspec_data[QubitIndex]['I Fit'][0] = qspec_fit
            qspec_data[QubitIndex]['Round Num'][0] = 0
            qspec_data[QubitIndex]['Batch Num'][0] = 0
            qspec_data[QubitIndex]['Recycled QFreq'][0] = False  # no rr so no recycling here
            qspec_data[QubitIndex]['Exp Config'][0] = expt_cfg
            qspec_data[QubitIndex]['Syst Config'][0] = sys_config_qspec
            qspec_data[QubitIndex]['ss_Q_e'][0] = ss_Q_e_qspec
            qspec_data[QubitIndex]['ss_Q_g'][0] = ss_Q_g_qspec
            qspec_data[QubitIndex]['ss_I_e'][0] = ss_I_e_qspec
            qspec_data[QubitIndex]['ss_I_g'][0] = ss_I_g_qspec
            qspec_data[QubitIndex]['I_shots'][0] = I_shots_qspec
            qspec_data[QubitIndex]['Q_shots'][0] = Q_shots_qspec
            qspec_data[QubitIndex]['Gains'][0] = gains_qspec

            saver_qspec = Data_H5(subStudyDataFolder, qspec_data, 0, save_r)
            saver_qspec.save_to_h5('QSpec_zeno_high_gain')
            del saver_qspec
            del qspec_data

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
            del exp
            continue

        # reinitialize
        qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
        exp.qubit_cfg['qubit_gain_ge'] = normal_gain
        ############################################### Qubit Spec ########################################################
        qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)

        try:
            q_spec = QubitSpectroscopyZeno(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j,
                                           signal, save_figs, plot_fit=True, experiment=exp,
                                           live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                           unmasking_resgain=unmask)
            (qspec_I, qspec_Q, qspec_freqs, qspec_fit, qubit_freq, sys_config_qspec, ss_Q_e_qspec, ss_Q_g_qspec,
             ss_I_e_qspec, ss_I_g_qspec, I_shots_qspec, Q_shots_qspec, gains_qspec) = q_spec.run(scaling=True,qze_pulse='const')

            experiment.qubit_cfg['corrected_qspec_freq']=updated_qfreqs

            qspec_data[QubitIndex]['Dates'][0] = (
                time.mktime(datetime.datetime.now().timetuple()))
            qspec_data[QubitIndex]['I'][0] = qspec_I
            qspec_data[QubitIndex]['Q'][0] = qspec_Q
            qspec_data[QubitIndex]['Frequencies'][0] = qspec_freqs
            qspec_data[QubitIndex]['I Fit'][0] = qspec_fit
            qspec_data[QubitIndex]['Round Num'][0] = 0
            qspec_data[QubitIndex]['Batch Num'][0] = 0
            qspec_data[QubitIndex]['Recycled QFreq'][0] = False  # no rr so no recycling here
            qspec_data[QubitIndex]['Exp Config'][0] = expt_cfg
            qspec_data[QubitIndex]['Syst Config'][0] = sys_config_qspec
            qspec_data[QubitIndex]['ss_Q_e'][0] = ss_Q_e_qspec
            qspec_data[QubitIndex]['ss_Q_g'][0] = ss_Q_g_qspec
            qspec_data[QubitIndex]['ss_I_e'][0] = ss_I_e_qspec
            qspec_data[QubitIndex]['ss_I_g'][0] = ss_I_g_qspec
            qspec_data[QubitIndex]['I_shots'][0] = I_shots_qspec
            qspec_data[QubitIndex]['Q_shots'][0] = Q_shots_qspec
            qspec_data[QubitIndex]['Gains'][0] = gains_qspec

            saver_qspec = Data_H5(subStudyDataFolder, qspec_data, 0, save_r)
            saver_qspec.save_to_h5('QSpec_zeno')
            del saver_qspec
            del qspec_data

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
            del exp
            continue

        # reinitialize
        qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
        # ############################################# Ramsey freq correction #########################################
        t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
        try:
            t2r = T2RMeasurementZeno(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                                     save_figs,
                                     experiment=exp, live_plot=live_plot, fit_data=fit_data,
                                     increase_qubit_reps=increase_qubit_reps,
                                     qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                     multiply_qubit_reps_by=multiply_qubit_reps_by,
                                     verbose=verbose, logger=rr_logger, unmasking_resgain=unmask, correction=True)
            t2r_est, t2r_err, t2r_I, t2r_Q, t2r_delay_times, fit_ramsey, sys_config_t2r, ss_Q_e_t2r, ss_Q_g_t2r, \
                ss_I_e_t2r, ss_I_g_t2r, I_shots_t2r, Q_shots_t2r, gains_t2r,ramsey_corrected_freqsv = t2r.run(
                thresholding=thresholding, scaling=True, qze_pulse='const')
            t2r_data[QubitIndex]['T2'][j - batch_num * save_r - 1] = t2r_est
            t2r_data[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t2r_err
            t2r_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                time.mktime(datetime.datetime.now().timetuple()))
            t2r_data[QubitIndex]['I'][j - batch_num * save_r - 1] = t2r_I
            t2r_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = t2r_Q
            t2r_data[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t2r_delay_times
            t2r_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = fit_ramsey
            t2r_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            t2r_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            t2r_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
            t2r_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2r
            t2r_data[QubitIndex]['ss_Q_e'][0] = ss_Q_e_t2r
            t2r_data[QubitIndex]['ss_Q_g'][0] = ss_Q_g_t2r
            t2r_data[QubitIndex]['ss_I_e'][0] = ss_I_e_t2r
            t2r_data[QubitIndex]['ss_I_g'][0] = ss_I_g_t2r
            t2r_data[QubitIndex]['I_shots'][0] = I_shots_t2r
            t2r_data[QubitIndex]['Q_shots'][0] = Q_shots_t2r
            t2r_data[QubitIndex]['Gains'][0] = gains_t2r

            saver_t2 = Data_H5(subStudyDataFolder, t2r_data, batch_num, save_r)
            saver_t2.save_to_h5('T2_ge_zeno_correction')
            del t2r
        except:
            continue
        t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)

        ###################################################### T1 ######################################################
        try:
            t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
            t1 = T1Measurement_with_Zeno_loop(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                                              save_figs=True,
                                              experiment=exp,
                                              live_plot=live_plot, fit_data=True,
                                              increase_qubit_reps=increase_qubit_reps,
                                              qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                              multiply_qubit_reps_by=multiply_qubit_reps_by,
                                              verbose=verbose, logger=rr_logger, unmasking_resgain=unmask,
                                              )
            t1_est, t1_err, t1_I, t1_Q, t1_delay_times, q1_fit_exponential, sys_config_t1, \
                ss_Q_e_t1, ss_Q_g_t1, ss_I_e_t1, ss_I_g_t1, I_shots_t1, Q_shots_t1, gains_t1 = t1.run(
                thresholding=thresholding, scaling=True, qze_pulse='const')

            t1_data[QubitIndex]['T1'][j - batch_num * save_r - 1] = t1_est
            t1_data[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t1_err
            t1_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                time.mktime(datetime.datetime.now().timetuple()))
            t1_data[QubitIndex]['I'][j - batch_num * save_r - 1] = t1_I
            t1_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = t1_Q
            t1_data[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t1_delay_times
            t1_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = q1_fit_exponential
            t1_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            t1_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            t1_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
            t1_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t1
            t1_data[QubitIndex]['ss_Q_e'][0] = ss_Q_e_t1
            t1_data[QubitIndex]['ss_Q_g'][0] = ss_Q_g_t1
            t1_data[QubitIndex]['ss_I_e'][0] = ss_I_e_t1
            t1_data[QubitIndex]['ss_I_g'][0] = ss_I_g_t1
            t1_data[QubitIndex]['I_shots'][0] = I_shots_t1
            t1_data[QubitIndex]['Q_shots'][0] = Q_shots_t1
            t1_data[QubitIndex]['Gains'][0] = gains_t1

            saver_t1 = Data_H5(subStudyDataFolder, t1_data, batch_num, save_r)
            saver_t1.save_to_h5('T1_ge_zeno')
            del saver_t1
            del t1_data
            del t1
        except:
            continue
        t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
        # ###################################################### T2R ######################################################
        t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
        try:
            t2r = T2RMeasurementZeno(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                                 save_figs,
                                 experiment=exp, live_plot=live_plot, fit_data=fit_data,
                                 increase_qubit_reps=increase_qubit_reps,
                                 qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                 multiply_qubit_reps_by=multiply_qubit_reps_by,
                                 verbose=verbose, logger=rr_logger, unmasking_resgain=unmask)
            t2r_est, t2r_err, t2r_I, t2r_Q, t2r_delay_times, fit_ramsey, sys_config_t2r, ss_Q_e_t2r, ss_Q_g_t2r, \
                ss_I_e_t2r, ss_I_g_t2r, I_shots_t2r, Q_shots_t2r, gains_t2r = t2r.run(
                thresholding=thresholding, scaling=True, qze_pulse='const')
            t2r_data[QubitIndex]['T2'][j - batch_num * save_r - 1] = t2r_est
            t2r_data[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t2r_err
            t2r_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                time.mktime(datetime.datetime.now().timetuple()))
            t2r_data[QubitIndex]['I'][j - batch_num * save_r - 1] = t2r_I
            t2r_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = t2r_Q
            t2r_data[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t2r_delay_times
            t2r_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = fit_ramsey
            t2r_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            t2r_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            t2r_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
            t2r_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2r
            t2r_data[QubitIndex]['ss_Q_e'][0] = ss_Q_e_t2r
            t2r_data[QubitIndex]['ss_Q_g'][0] = ss_Q_g_t2r
            t2r_data[QubitIndex]['ss_I_e'][0] = ss_I_e_t2r
            t2r_data[QubitIndex]['ss_I_g'][0] = ss_I_g_t2r
            t2r_data[QubitIndex]['I_shots'][0] = I_shots_t2r
            t2r_data[QubitIndex]['Q_shots'][0] = Q_shots_t2r
            t2r_data[QubitIndex]['Gains'][0] = gains_t2r

            saver_t2 = Data_H5(subStudyDataFolder, t2r_data, batch_num, save_r)
            saver_t2.save_to_h5('T2_ge_zeno')
            del t2r
        except:
            continue
        t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)


        # # ###################################################### T2E ######################################################
        t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
        try:
            t2e = T2EMeasurementZeno(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                                     save_figs,
                                     experiment=exp, live_plot=live_plot, fit_data=fit_data,
                                     increase_qubit_reps=increase_qubit_reps,
                                     qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                     multiply_qubit_reps_by=multiply_qubit_reps_by,
                                     verbose=verbose, logger=rr_logger, unmasking_resgain=unmask)
            t2e_est, t2e_err, t2e_I, t2e_Q, t2e_delay_times, fit_ramsey, sys_config_t2e, ss_Q_e_t2e, ss_Q_g_t2e, \
                ss_I_e_t2e, ss_I_g_t2e, I_shots_t2e, Q_shots_t2e, gains_t2e = t2e.run(
                thresholding=thresholding, scaling=True)
            t2e_data[QubitIndex]['T2E'][j - batch_num * save_r - 1] = t2e_est
            t2e_data[QubitIndex]['Errors'][j - batch_num * save_r - 1] = t2e_err
            t2e_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                time.mktime(datetime.datetime.now().timetuple()))
            t2e_data[QubitIndex]['I'][j - batch_num * save_r - 1] = t2e_I
            t2e_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = t2e_Q
            t2e_data[QubitIndex]['Delay Times'][j - batch_num * save_r - 1] = t2e_delay_times
            t2e_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = fit_ramsey
            t2e_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
            t2e_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
            t2e_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = expt_cfg
            t2e_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2e
            t2e_data[QubitIndex]['ss_Q_e'][0] = ss_Q_e_t2e
            t2e_data[QubitIndex]['ss_Q_g'][0] = ss_Q_g_t2e
            t2e_data[QubitIndex]['ss_I_e'][0] = ss_I_e_t2e
            t2e_data[QubitIndex]['ss_I_g'][0] = ss_I_g_t2e
            t2e_data[QubitIndex]['I_shots'][0] = I_shots_t2e
            t2e_data[QubitIndex]['Q_shots'][0] = Q_shots_t2e
            t2e_data[QubitIndex]['Gains'][0] = gains_t2e

            saver_t2e = Data_H5(subStudyDataFolder, t2e_data, batch_num, save_r)
            saver_t2e.save_to_h5('T2E_ge_zeno')
            del t2e
        except:
            continue
        t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)
        del exp
    del experiment
    # except:
    #     del experiment
    #     continue





