import copy
import sys
import os
import atexit
import json
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
from section_002_res_spec_f import ResonanceSpectroscopyFH
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
from system_config import (QICK_experiment, get_pucq4_flux_calibration,
                           resolve_pucq4_flux_config, normalize_pucq4_flux_current)
from malab_yokogawa_gs200 import YokogawaGS200
from section_003_punch_out_ge_mux import PunchOut
from expt_config import (expt_cfg, list_of_all_qubits, tot_num_of_qubits, FRIDGE,
                         apply_pucq4_flux_experiment_config,
                         pucq4_round_robin_profile, apply_pucq4_round_robin_profile)


def set_yoko_bias(yoko3_mA=0.0, yoko4_mA=0.0):
    """Set the documented biases at 0.5 mA/s; zero means output off."""
    currents = tuple(normalize_pucq4_flux_current(i) for i in (yoko3_mA, yoko4_mA))
    states, errors = [], []
    for ip, current in zip(("192.168.1.73", "192.168.1.77"), currents):
        try:
            with YokogawaGS200(ip, max_current=0.010) as source:
                if source.get_mode() != "CURR" or abs(source.get_range() - 0.01) > 1e-9:
                    raise RuntimeError("%s must already be in current mode / 10 mA range" % ip)
                if not source.get_output():
                    source.set_current(0.0)  # enable only from zero
                    if current:
                        source.set_output(True)
                source.ramp_current(current * 1e-3, sweeprate=0.0005)
                source.set_output(bool(current))
                state = dict(ip=ip, identity=source.get_id(), mode=source.get_mode(),
                             current_A=source.get_current(), output=source.get_output(),
                             range_A=source.get_range(), voltage_limit_V=source.get_voltage_limit())
                if abs(state["current_A"] - current * 1e-3) > 1e-7 or state["output"] != bool(current):
                    raise RuntimeError("Yoko readback mismatch: %s" % state)
                states.append(state)
        except Exception as exc:
            errors.append("%s: %s" % (ip, exc))
            if any(currents):
                raise
    if errors:  # zeroing still attempts BOTH sources if one connection fails
        raise RuntimeError("; ".join(errors))
    print("Verified Yoko biases: " + json.dumps(states), flush=True)
    return states

################################################ Run Configurations ####################################################
st = time.time()
#
n = 1                              # one calibration/coherence round
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

Qs_to_look_at = [0, 1, 2, 3, 4, 5]  # M1..M6; physical Q4 is index 4, Q6 is index 5

# Set these two values, then run: the script ramps/verifies the sources and
# selects the matching pulse/readout/acquisition settings. Both return to zero
# with outputs off when finished. Only the nine documented grid values work.
# Current-sweep calibrations use one nonzero Yoko at a time; joint biases are unverified.
YOKO3_CURRENT_MA = 0.0  # physical Q4, resonator row M5, Python index 4
YOKO4_CURRENT_MA = 0.0  # physical Q6, resonator row M6, Python index 5

# Optional environment overrides are used by the bounded PUCQ4 flux controller.
# Ordinary interactive runs see exactly the defaults above. Values are explicit
# and unit-labelled so the controller can reuse this orchestrator and all of its
# normal plot/HDF5 paths without changing any underlying QICK program.
if os.environ.get('PUCQ4_QS'):
    Qs_to_look_at = [int(value) for value in os.environ['PUCQ4_QS'].split(',')]
YOKO3_CURRENT_MA = float(os.environ.get('PUCQ4_YOKO3_CURRENT_MA', YOKO3_CURRENT_MA))
YOKO4_CURRENT_MA = float(os.environ.get('PUCQ4_YOKO4_CURRENT_MA', YOKO4_CURRENT_MA))
# Use a consistent endpoint settling interval after the verified ramp.
# Matched endpoint runs retain qspec to track observed frequency variation.
YOKO_SETTLE_SECONDS = 600 if YOKO3_CURRENT_MA in (-10.0, 10.0) else 0
selected_flux_calibration = get_pucq4_flux_calibration(
    yoko3_current_mA=YOKO3_CURRENT_MA,
    yoko4_current_mA=YOKO4_CURRENT_MA,
)
selected_flux_config = resolve_pucq4_flux_config(
    yoko3_current_mA=YOKO3_CURRENT_MA,
    yoko4_current_mA=YOKO4_CURRENT_MA,
)
apply_pucq4_flux_experiment_config(selected_flux_calibration)

#Data saving info
run_name = 'pucq4_run_started_Aug_3'
device_name = 'PUCQ4'
_bias_description = (
    'zero-current' if YOKO3_CURRENT_MA == 0.0 and YOKO4_CURRENT_MA == 0.0
    else 'selected-current'
)
substudy_txt_notes = ('PUCQ4 %s round-robin reproduction using persistent '
                      'pulse, readout, averaging and coherence timing profiles.'
                      % _bias_description)
substudy_txt_notes = os.environ.get(
    'PUCQ4_SUBSTUDY_NOTES', substudy_txt_notes
)
substudy_txt_notes += (
    '\nSelected Python indices: %s. Flux selectors: Yoko 3/Q4/M5 = %.1f mA; '
    'Yoko 4/Q6/M6 = %.1f mA. Parameters resolved from the final '
    'current-indexed pulse/readout calibration table.'
    % (Qs_to_look_at, YOKO3_CURRENT_MA, YOKO4_CURRENT_MA)
)
study ='find_resonators_sandbox'
if YOKO3_CURRENT_MA == 0.0 and YOKO4_CURRENT_MA == 0.0:
    sub_study = 'rr6_0mA'
else:
    sub_study = (
        'rr6_y3_%+.1f_y4_%+.1f'
        % (YOKO3_CURRENT_MA, YOKO4_CURRENT_MA)
    )
sub_study = os.environ.get('PUCQ4_SUB_STUDY', sub_study)
data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# set which of the following you'd like to run to 'True'
run_flags = {"tof": False, "res_spec": True, "q_spec": True, "ss": True, "rabi": True, "len_rabi": False, "ss_gef": False, "test_act": False, "fh_rabi": False,
             "t1": True, "t2r": True, "t2r_correction":False, "t2e": True, "ef_res_spec":  False, "ef_q_spec":  False, "fh_q_spec":False,
             "rabi_pop_meas": False, "ef_Rabi":  False, "ef_ss": False, "res_spec_fh":  False}
if os.environ.get('PUCQ4_RUN_FLAGS'):
    enabled = {name.strip() for name in os.environ['PUCQ4_RUN_FLAGS'].split(',') if name.strip()}
    run_flags = {name: name in enabled for name in run_flags}
t2r_fit_starts_override = None
if os.environ.get('PUCQ4_T2R_FIT_STARTS'):
    # Optional diagnostic starts for the existing free fit; never a T2 guess.
    t2r_fit_starts_override = json.loads(os.environ['PUCQ4_T2R_FIT_STARTS'])
    if (len(Qs_to_look_at) != 1 or not run_flags['t2r']
            or not isinstance(t2r_fit_starts_override, dict)
            or 'f' not in t2r_fit_starts_override
            or not set(t2r_fit_starts_override) <= {
                'f', 'phase', 'amp', 'initial_offset', 'final_offset'}
            or not all(isinstance(v, (int, float)) and not isinstance(v, bool)
                       and np.isfinite(v) for v in t2r_fit_starts_override.values())
            or t2r_fit_starts_override['f'] <= 0):
        raise ValueError('PUCQ4_T2R_FIT_STARTS requires one Ramsey row and finite '
                         'f>0, phase, amp, initial_offset, final_offset starts')
if os.environ.get('PUCQ4_QSPEC_PASSES'):
    # Short settling diagnostic: keep the bias applied across existing RR rounds.
    n = int(os.environ['PUCQ4_QSPEC_PASSES'])
    if (n not in (1, 2, 3) or len(Qs_to_look_at) != 1
            or {name for name, enabled in run_flags.items() if enabled} != {'q_spec'}):
        raise ValueError('PUCQ4_QSPEC_PASSES permits only 1-3 qspec-only passes on one row')
if (not run_flags['q_spec'] and not os.environ.get('PUCQ4_QFREQ_MHZ')
        and any(run_flags[name] for name in ('rabi', 'ss', 't1', 't2r', 't2e'))
        and any(q in Qs_to_look_at and row.get('requires_qspec_each_pass', False)
                for q, row in selected_flux_calibration.items())):
    raise ValueError('This bias has a varying qubit frequency: enable q_spec before '
                     'pulse measurements, or supply a freshly measured PUCQ4_QFREQ_MHZ '
                     'for an explicit diagnostic run.')

coherence_scaling = os.environ.get(
    'PUCQ4_COHERENCE_SCALING', 'true'
).strip().lower() in {'1', 'true', 'yes', 'on'}


# PUCQ4 readout settings. The 9 GHz band comes back ~30 dB weaker than the old
# 7.2 GHz chip (~11 dB of DAC sin(x)/x rolloff plus ~19 dB of analog loss), so
# these are much longer and louder than squill's 9 us / 0.25.
res_leng_vals = list(selected_flux_config["res_length_ge"])
res_gain = list(selected_flux_config["res_gain_ge"])
freq_offsets = list(selected_flux_config["res_freq_offset_ge"])
for _override_index in Qs_to_look_at:
    if os.environ.get('PUCQ4_READOUT_LENGTH_US'):
        res_leng_vals[_override_index] = float(os.environ['PUCQ4_READOUT_LENGTH_US'])
    if os.environ.get('PUCQ4_READOUT_GAIN'):
        res_gain[_override_index] = float(os.environ['PUCQ4_READOUT_GAIN'])
    if os.environ.get('PUCQ4_READOUT_OFFSET_MHZ'):
        freq_offsets[_override_index] = float(os.environ['PUCQ4_READOUT_OFFSET_MHZ'])

qubit_freqs_ef = [None]*6
increase_steps_to_ef = 600
ef_res_sample_number = 1
number_of_qubits = 6
figure_quality = 200
################################################ Data Saving Setup ##################################################
#Folders

data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Check the full timestamped paths before folders or hardware are touched.
_planned_root = os.path.join("M:/_Data/20250822 - Olivia", run_name,
                             device_name, study, sub_study, data_set)
_planned_paths = [
    os.path.join(_planned_root, "study_data", "Data_h5", "qspec_ge",
                 data_set + "_qspec_ge_results_batch_1_Num_per_batch1.h5"),
    os.path.join(_planned_root, "documentation", "qubit_spec_ge_plots",
                 "R_1_Q_6_" + data_set + "_qubit_spec_ge_q6.png"),
    os.path.join(_planned_root, "documentation", "ss_repeat_meas_ge", "Q6",
                 "R_1_Q_6_Readout_Optimization" + data_set + "__q6.png"),
]
if max(map(len, _planned_paths)) >= 260:
    raise ValueError("Shorten sub_study: generated Windows path reaches 260 characters")
if not Qs_to_look_at or len(set(Qs_to_look_at)) != len(Qs_to_look_at) or any(q not in range(6) for q in Qs_to_look_at):
    raise ValueError("Select distinct PUCQ4 Python indices from 0 through 5")

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

with open(os.path.join(studyDocumentationFolder, "acquisition_profile.json"), "w") as _profile_file:
    json.dump(dict(qubits=Qs_to_look_at, rounds=n, run_flags=run_flags,
                   yoko3_current_mA=YOKO3_CURRENT_MA, yoko4_current_mA=YOKO4_CURRENT_MA,
                   yoko_settle_seconds=YOKO_SETTLE_SECONDS,
                   system=selected_flux_config,
                   profiles={q: pucq4_round_robin_profile(q, selected_flux_calibration)
                             for q in Qs_to_look_at},
                   environment_overrides={k: v for k, v in os.environ.items() if k.startswith("PUCQ4_")},
                   longest_artifact_path=max(map(len, _planned_paths))), _profile_file, indent=2)


def return_yokos_to_zero():
    states = set_yoko_bias(0.0, 0.0)
    with open(os.path.join(studyDocumentationFolder, "yoko_final.json"), "w") as handle:
        json.dump(states, handle, indent=2)


atexit.register(return_yokos_to_zero)
_bias_states = set_yoko_bias(YOKO3_CURRENT_MA, YOKO4_CURRENT_MA)
with open(os.path.join(studyDocumentationFolder, "yoko_bias.json"), "w") as _bias_file:
    json.dump(_bias_states, _bias_file, indent=2)
for _remaining in range(YOKO_SETTLE_SECONDS, 0, -30):
    print("Yoko bias settling: %d seconds remaining" % _remaining, flush=True)
    time.sleep(min(30, _remaining))

for _flux_qubit_index in sorted(selected_flux_calibration):
    _flux_row = selected_flux_calibration[_flux_qubit_index]
    rr_logger.info(
        "Selected flux calibration: index=%d device=%s current_mA=%.1f "
        "qfreq_MHz=%.12g qspec_gain=%.12g qspec_length_us=%.12g "
        "sigma_us=%.12g gaussian_length_us=%.12g pi_amp=%.12g "
        "res_base_MHz=%.12g readout_frequency_MHz=%.12g "
        "readout_offset_MHz=%.12g readout_gain=%.12g readout_length_us=%.12g",
        _flux_qubit_index, _flux_row["device_key"], _flux_row["current_mA"],
        _flux_row["qfreq_MHz"], _flux_row["qspec_gain"],
        _flux_row["qspec_length_us"], _flux_row["sigma_us"],
        _flux_row["gaussian_length_us"], _flux_row["pi_amp"],
        _flux_row["res_base_MHz"], _flux_row["readout_frequency_MHz"],
        _flux_row["readout_offset_MHz"], _flux_row["readout_gain"],
        _flux_row["readout_length_us"],
    )

####################################################### RR #############################################################

def create_data_dict(keys, save_r, qs):
    return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}


def set_qubit_expt_value(experiment_name, key, qubit_index, value):
    """Apply a scalar override while preserving list-valued experiment config."""
    current = expt_cfg[experiment_name][key]
    if isinstance(current, (list, np.ndarray)):
        current[qubit_index] = value
    elif isinstance(current, tuple):
        current = list(current)
        current[qubit_index] = value
        expt_cfg[experiment_name][key] = current
    else:
        expt_cfg[experiment_name][key] = value


def calibrated_population(I, Q, ss_I_e, ss_Q_e, ss_I_g, ss_Q_g):
    """Return the signed SSF affine projection, with ground=0 and excited=1."""
    excited = np.mean(ss_I_e + 1j * ss_Q_e)
    ground = np.mean(ss_I_g + 1j * ss_Q_g)
    separation_squared = np.abs(excited - ground) ** 2
    if not np.isfinite(separation_squared) or separation_squared <= 0:
        raise RuntimeError('SSF calibration has zero/non-finite centroid separation')
    return np.real(
        ((I + 1j * Q) - ground) * np.conj(excited - ground)
    ) / separation_squared

# Define what to save to h5 files
tof_keys = ['Dates', 'iq_list','t', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
res_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
qspec_keys = ['Dates', 'I', 'Q', 'Population', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num','Recycled QFreq',
              'Exp Config', 'Syst Config', 'ss_Q_e', 'ss_Q_g', 'ss_I_e', 'ss_I_g']
rabi_keys = [
    'Dates', 'I', 'Q', 'Population', 'Gains', 'Fit', 'Round Num',
    'Batch Num', 'Exp Config', 'Syst Config', 'ss_Q_e', 'ss_Q_g',
    'ss_I_e', 'ss_I_g',
]
ss_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
ss_ef_keys = ['Fidelity', 'Angle', 'Dates', 'I_g', 'Q_g','I_e', 'Q_e', 'I_f', 'Q_f', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
t1_keys = ['T1', 'Errors', 'Dates', 'I', 'Q', 'Population', 'Delay Times', 'Fit',
           'Round Num', 'Batch Num', 'Exp Config', 'Syst Config', 'ss_Q_e', 'ss_Q_g',
           'ss_I_e', 'ss_I_g']
t2r_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Population', 'Delay Times', 'Fit', 'Fit Unseeded',
            'Round Num', 'Batch Num', 'Exp Config', 'Syst Config', 'ss_Q_e', 'ss_Q_g',
            'ss_I_e', 'ss_I_g']
t2r_correction_keys = ['T2', 'Errors', 'Dates', 'I', 'Q', 'Correction Freq', 'Delay Times', 'Fit', 'Round Num', 'Batch Num', 'Exp Config',
            'Syst Config']
t2e_keys = ['T2E', 'Errors', 'Dates', 'I', 'Q', 'Population', 'Delay Times', 'Fit',
            'Round Num', 'Batch Num', 'Exp Config', 'Syst Config', 'ss_Q_e', 'ss_Q_g',
            'ss_I_e', 'ss_I_g']
rabi_keys_ef_Qtemps = ['Dates', 'Qfreq_ge', 'I1', 'Q1', 'Gains1', 'Fit1', 'I2', 'Q2', 'Gains2', 'Fit2', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
ss_keys_gef = ['Fidelity', 'Angle_ef', 'Dates', 'I_g', 'Q_g', 'I_e', 'Q_e', 'I_f', 'Q_f', 'Round Num', 'Batch Num', 'Exp Config',
           'Syst Config']
res_keys_ef = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs', 'Round Num', 'Batch Num', 'Exp Config',
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
len_rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
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
res_data_fh = create_data_dict(res_keys_ef, save_r, list_of_all_qubits)

if pre_optimize:
    ################################################## Simple Optimization ###############################################

    def sweep_frequency_offset(experiment_opt, QubitIndex_opt, offset_values, n_loops=10, number_of_qubits=6,
                               outerFolder="", studyDocumentationFolder_opt="", optimizationFolder_opt="", j=0):
        baseline_freq = experiment_opt.readout_cfg['res_freq_ge'][QubitIndex]
        ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
        ssf_dict = {}
        for offset in offset_values:
            fids = []
            # repeat n times for each offset
            for i in range(n_loops):
                exp_copy = copy.deepcopy(experiment_opt)  # python is python, doesnt overwrite things properly

                res_freqs = exp_copy.readout_cfg['res_freq_ge']
                res_freqs[QubitIndex] = baseline_freq + offset
                exp_copy.readout_cfg['res_freq_ge'] = res_freqs

                ss = SingleShot(QubitIndex, number_of_qubits, outerFolder, j, save_figs, exp_copy)
                fid, angle, iq_list_g, iq_list_e, ss_config = ss.run()
                fids.append(fid)
                del exp_copy

                I_g = iq_list_g[QubitIndex][0].T[0]
                Q_g = iq_list_g[QubitIndex][0].T[1]
                I_e = iq_list_e[QubitIndex][0].T[0]
                Q_e = iq_list_e[QubitIndex][0].T[1]

                ss_data[QubitIndex]['Fidelity'][0] = fid
                ss_data[QubitIndex]['Angle'][0] = angle
                ss_data[QubitIndex]['Dates'][0] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                ss_data[QubitIndex]['I_g'][0] = I_g
                ss_data[QubitIndex]['Q_g'][0] = Q_g
                ss_data[QubitIndex]['I_e'][0] = I_e
                ss_data[QubitIndex]['Q_e'][0] = Q_e
                ss_data[QubitIndex]['Round Num'][0] = i
                ss_data[QubitIndex]['Batch Num'][0] = 0
                ss_data[QubitIndex]['Exp Config'][0] = copy.deepcopy(expt_cfg)
                ss_data[QubitIndex]['Syst Config'][0] = ss_config

                saver_ss = Data_H5(optimizationFolder_opt, ss_data, 0, save_r)
                saver_ss.save_to_h5('ss_ge')
                del saver_ss
                del ss_data

                ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)

            # find avg ssf
            avg_fid = np.mean(fids)
            ssf_dict[offset] = avg_fid
            if verbose:
                print(f"Offset: {offset} -> Average g-e SSF: {avg_fid:.4f}")
        import matplotlib.pyplot as plt
        plt.figure()
        offsets_sorted = sorted(ssf_dict.keys())
        ssf_values = [ssf_dict[offset] for offset in offsets_sorted]
        plt.plot(offsets_sorted, ssf_values, marker='o')
        plt.xlabel('Frequency Offset')
        plt.ylabel('Average g-e SSF')
        plt.title(f'g-e SSF vs Frequency Offset for Qubit {QubitIndex + 1}')
        plt.grid(True)
        if outerFolder:
            os.makedirs(outerFolder, exist_ok=True)
            plot_path = os.path.join(studyDocumentationFolder, f"ge_SSF_vs_offset_Q{QubitIndex + 1}.png")
            plt.savefig(plot_path)
            if verbose:
                print(f"Plot saved to {plot_path}")
        plt.close()

        # Determine the offset value that yielded the best (highest) average SSF.
        optimal_offset = max(ssf_dict, key=ssf_dict.get)
        if verbose:
            print(
                f"Optimal frequency offset for Qubit {QubitIndex}: {optimal_offset} (Avg g-e SSF: {ssf_dict[optimal_offset]:.4f})")

        return optimal_offset, ssf_dict
    ################################################## Simple Optimization ###############################################
    for Q in Qs_to_look_at:
        try:
            experiment = QICK_experiment(
                optimizationFolder,
                DAC_attenuator1=15,
                DAC_attenuator2=10,
                ADC_attenuator=17,
                fridge=FRIDGE,
                yoko3_current_mA=YOKO3_CURRENT_MA,
                yoko4_current_mA=YOKO4_CURRENT_MA,
            )
            # Set resonator configuration for this qubit
            experiment.readout_cfg['res_gain_ge'] = res_gain[Q]
            experiment.readout_cfg['res_length'] = res_leng_vals[Q]

            res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
            res_spec = ResonanceSpectroscopy(Q, tot_num_of_qubits, optimizationFolder, 0,
                                             save_figs=True, experiment=experiment, verbose=verbose,
                                             logger=rr_logger, qick_verbose=True)
            res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
            experiment.readout_cfg['res_freq_ge'] = res_freqs[Q]
            rr_logger.info(f"g-e ResSpec for qubit {Q}: {res_freqs}")

            res_data[Q]['Dates'][0] = (
                time.mktime(datetime.datetime.now().timetuple()))
            res_data[Q]['freq_pts'][0] = freq_pts
            res_data[Q]['freq_center'][0] = freq_center
            res_data[Q]['Amps'][0] = amps
            res_data[Q]['Found Freqs'][0] = res_freqs
            res_data[Q]['Batch Num'][0] = 0
            res_data[Q]['Exp Config'][0] = copy.deepcopy(expt_cfg)
            res_data[Q]['Syst Config'][0] = sys_config_rspec

            saver_res = Data_H5(optimizationFolder, res_data, 0, save_r)  # save
            saver_res.save_to_h5('res_ge')
            del saver_res
            del res_data
            res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)  # initialize again to a blank for saftey
            del res_spec
        except Exception as e:
            rr_logger.exception(f"g-e ResSpec error on qubit {Q} : {e}")
            continue
        ############ g-e Qubit Spec ##############
        qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
        # try:
        q_spec = QubitSpectroscopy(Q, tot_num_of_qubits, optimizationFolder, 0,
                                   signal, plot_fit=False, save_figs=True, experiment=experiment,
                                   live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                   qick_verbose=True, increase_reps=True, increase_reps_to=500)
        (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit,
         qubit_freq, sys_config_qspec) = q_spec.run()
        experiment.qubit_cfg['qubit_freq_ge'] = float(qubit_freq)
        stored_qspec = float(qubit_freq)
        rr_logger.info(f"Tune-up: g-e Qubit {Q + 1} frequency: {stored_qspec}")
        del q_spec

        qspec_data[Q]['Dates'][0] = (
            time.mktime(datetime.datetime.now().timetuple()))
        qspec_data[Q]['I'][0] = qspec_I
        qspec_data[Q]['Q'][0] = qspec_Q
        qspec_data[Q]['Frequencies'][0] = qspec_freqs
        qspec_data[Q]['I Fit'][0] = qspec_I_fit
        qspec_data[Q]['Q Fit'][0] = qspec_Q_fit
        qspec_data[Q]['Round Num'][0] = 0
        qspec_data[Q]['Batch Num'][0] = 0
        qspec_data[Q]['Recycled QFreq'][0] = False  # no rr so no recycling here
        qspec_data[Q]['Exp Config'][0] = copy.deepcopy(expt_cfg)
        qspec_data[Q]['Syst Config'][0] = sys_config_qspec

        saver_qspec = Data_H5(optimizationFolder, qspec_data, 0, save_r)
        saver_qspec.save_to_h5('qspec_ge')
        del saver_qspec
        del qspec_data

        # except Exception as e:
        #     rr_logger.exception(f"g-e QubitSpectroscopyGE error on qubit {Q}: {e}")
        #     continue

        ################### g-e amp rabi ################
        rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
        try:
            rabi = AmplitudeRabiExperiment(Q, tot_num_of_qubits, optimizationFolder, 0,
                                           signal, save_figs=True, experiment=experiment,
                                           live_plot=live_plot,
                                           increase_qubit_reps=increase_qubit_reps,
                                           qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                           multiply_qubit_reps_by=multiply_qubit_reps_by,
                                           verbose=verbose, logger=rr_logger,
                                           qick_verbose=True)
            (rabi_I, rabi_Q, rabi_gains, rabi_fit, stored_pi_amp, sys_config_rabi) = rabi.run()
            experiment.qubit_cfg['pi_amp'] = float(stored_pi_amp)
            rr_logger.info(f"Tune-up: g-e Pi amplitude for qubit {Q + 1}: {float(stored_pi_amp)}")
            with open(log_file, "a", encoding="utf-8") as file:
                file.write("\n" + f'g-e Pi Amplitude Used for optimization: {float(stored_pi_amp)}')

            if 0.61 < stored_pi_amp or stored_pi_amp < 0.55:
                rr_logger.info(f"Tune-up: g-e 0.61 < stored_pi_amp or stored_pi_amp < 0.55: {float(stored_pi_amp)}")
                with open(log_file, "a", encoding="utf-8") as file:
                    file.write("\n" + f'g-e 0.61 < stored_pi_amp or stored_pi_amp < 0.55: {float(stored_pi_amp)}')

                continue

            rabi_data[Q]['Dates'][0] = (
                time.mktime(datetime.datetime.now().timetuple()))
            rabi_data[Q]['I'][0] = rabi_I
            rabi_data[Q]['Q'][0] = rabi_Q
            rabi_data[Q]['Gains'][0] = rabi_gains
            rabi_data[Q]['Fit'][0] = rabi_fit
            rabi_data[Q]['Round Num'][0] = 0
            rabi_data[Q]['Batch Num'][0] = 0
            rabi_data[Q]['Exp Config'][0] = copy.deepcopy(expt_cfg)
            rabi_data[Q]['Syst Config'][0] = sys_config_rabi
            saver_rabi = Data_H5(optimizationFolder, rabi_data, 0, save_r)
            saver_rabi.save_to_h5('rabi_ge')
            del rabi
            del saver_rabi
            del rabi_data
        except Exception as e:
            rr_logger.exception(f"g-e Rabi error on qubit {Q}: {e}")
            continue

        ################################################ optimize ################################################

        freq_range = np.linspace(-0.5, 0.5, freq_offset_steps)

        optimal_offset, ssf_dict = sweep_frequency_offset(experiment, Q, freq_range, n_loops=ssf_avgs_per_opt_pt, number_of_qubits=6,
                                                          outerFolder=optimizationFolder, studyDocumentationFolder_opt=studyDocumentationFolder,
                                                          optimizationFolder_opt=optimizationFolder, j=0)

        freq_offsets[Q] = optimal_offset

        with open(log_file, "a", encoding="utf-8") as file:
            file.write("\n" + f'g-e Offset Frequency used for study: {optimal_offset}')

        del experiment


    # initialize a dictionary to store those values
    res_data = create_data_dict(res_keys, save_r, list_of_all_qubits)
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    ss_data = create_data_dict(ss_keys, save_r, list_of_all_qubits)
    t1_data = create_data_dict(t1_keys, save_r, list_of_all_qubits)
    t2r_data = create_data_dict(t2r_keys, save_r, list_of_all_qubits)
    t2e_data = create_data_dict(t2e_keys, save_r, list_of_all_qubits)

batch_num=0
j = 0
angles=[]
while j < n:
    j += 1
    for QubitIndex in Qs_to_look_at:
        recycled_qfreq = False
        row_profile = apply_pucq4_round_robin_profile(
            QubitIndex, selected_flux_calibration,
            selected_flux_config["qubit_freq_ge"][QubitIndex])
        if t2r_fit_starts_override is not None:
            row_profile["Ramsey_ge"].update(
                fit_frequency_guess_MHz=t2r_fit_starts_override['f'],
                fit_population_guess={k: v for k, v in t2r_fit_starts_override.items()
                                      if k != 'f'})
        rr_logger.info("M%d acquisition profile: %s", QubitIndex + 1, row_profile)

        #Get the config for this qubit
        experiment = QICK_experiment(optimizationFolder, DAC_attenuator1 = 10, DAC_attenuator2 = 15, qubit_DAC_attenuator1 = 5,
                                     qubit_DAC_attenuator2 = 4, ADC_attenuator = 17, fridge=FRIDGE,
                                     yoko3_current_mA=YOKO3_CURRENT_MA,
                                     yoko4_current_mA=YOKO4_CURRENT_MA) # ADC_attenuator MUST be above 16dB
        experiment.create_folder_if_not_exists(optimizationFolder)

        experiment.readout_cfg['res_gain_ge'] = res_gain[QubitIndex]
        experiment.readout_cfg['res_gain_ef'] = res_gain[QubitIndex]
        # build_state derives the program's res_length from res_length_ge.
        experiment.readout_cfg['res_length_ge'] = res_leng_vals[QubitIndex]
        experiment.readout_cfg['res_length'] = res_leng_vals[QubitIndex]
        offset_config = experiment.readout_cfg.get('res_freq_offset_ge')
        if isinstance(offset_config, (list, np.ndarray)):
            offset_config[QubitIndex] = freq_offsets[QubitIndex]
        elif isinstance(offset_config, tuple):
            offset_config = list(offset_config)
            offset_config[QubitIndex] = freq_offsets[QubitIndex]
            experiment.readout_cfg['res_freq_offset_ge'] = offset_config
        else:
            experiment.readout_cfg['res_freq_offset_ge'] = freq_offsets[QubitIndex]
        experiment.readout_cfg['res_freq_ge'] = experiment.readout_cfg['res_freq_ge'][QubitIndex]

        experiment.qubit_cfg['qubit_freq_ge'] = experiment.qubit_cfg['qubit_freq_ge'][QubitIndex]
        experiment.qubit_cfg['qubit_gain_ge'] = experiment.qubit_cfg['qubit_gain_ge'][QubitIndex]

        experiment.qubit_cfg['qubit_freq_fh'] = experiment.qubit_cfg['qubit_freq_fh'][QubitIndex]
        experiment.qubit_cfg['qubit_gain_fh'] = experiment.qubit_cfg['qubit_gain_fh'][QubitIndex]


        # NOTE: three hardcoded squill frequencies used to live here, left over
        # from a single-qubit TLS study (Qs_to_look_at = [5], study 'tls_qspec'):
        #     res_freq_ge = freq_offsets[QubitIndex] + 7287.6
        #     qubit_freq_ef = 2929.37
        #     qubit_freq_ge = 3095.44
        # They ran AFTER the per-qubit indexing above and silently overwrote it,
        # so every qubit got Q6's squill frequencies regardless of system_config.
        # Removed for PUCQ4. If you need to pin a frequency for a one-off study,
        # do it here and DELETE IT AFTERWARDS.
        if not run_flags['res_spec']:
            experiment.readout_cfg['res_freq_ge'] += freq_offsets[QubitIndex]
        experiment.qubit_cfg['qubit_freq_ef'] = experiment.qubit_cfg['qubit_freq_ef'][QubitIndex]
        experiment.qubit_cfg['qubit_gain_ef'] = experiment.qubit_cfg['qubit_gain_ef'][QubitIndex]
        experiment.qubit_cfg['pi_amp'] =  experiment.qubit_cfg['pi_amp'][QubitIndex]
        experiment.qubit_cfg['pi_ef_amp'] = experiment.qubit_cfg['pi_ef_amp'][QubitIndex]

        qfreq_override = os.environ.get('PUCQ4_QFREQ_MHZ')
        if qfreq_override is not None:
            experiment.qubit_cfg['qubit_freq_ge'] = float(qfreq_override)
        qgain_override = os.environ.get('PUCQ4_QSPEC_GAIN')
        if qgain_override is not None:
            experiment.qubit_cfg['qubit_gain_ge'] = float(qgain_override)
        sigma_override = os.environ.get('PUCQ4_SIGMA_US')
        if sigma_override is not None:
            sigma_cfg = experiment.qubit_cfg['sigma']
            if isinstance(sigma_cfg, (list, tuple, np.ndarray)):
                sigma_cfg[QubitIndex] = float(sigma_override)
            else:
                experiment.qubit_cfg['sigma'] = float(sigma_override)
        pi_override = os.environ.get('PUCQ4_PI_AMP')
        if pi_override is not None:
            experiment.qubit_cfg['pi_amp'] = float(pi_override)
        res_override = os.environ.get('PUCQ4_RES_FREQ_MHZ')
        if res_override is not None:
            experiment.readout_cfg['res_freq_ge'] = float(res_override)
        # ``PUCQ4_RES_FREQ_MHZ`` is the base/seed used by an enabled resonator
        # sweep.  Qspec-only map runs have no such sweep, so give them a
        # separate explicit override for the already offset, physical readout
        # tone.  Keeping these meanings distinct prevents a base-frequency
        # override from silently discarding the optimized readout offset.
        readout_frequency_override = os.environ.get('PUCQ4_READOUT_FREQ_MHZ')
        if readout_frequency_override is not None:
            experiment.readout_cfg['res_freq_ge'] = float(readout_frequency_override)

        if any(run_flags[name] for name in ('t1', 't2r', 't2e')):
            coherence_overrides = {
                'T1_ge': {
                    'start': ('PUCQ4_T1_START_US', float),
                    'stop': ('PUCQ4_T1_STOP_US', float),
                    'steps': ('PUCQ4_T1_STEPS', int),
                    'reps': ('PUCQ4_T1_REPS', int),
                    'rounds': ('PUCQ4_T1_ROUNDS', int),
                    'relax_delay': ('PUCQ4_T1_RELAX_DELAY_US', float),
                },
                'Ramsey_ge': {
                    'start': ('PUCQ4_T2R_START_US', float),
                    'stop': ('PUCQ4_T2R_STOP_US', float),
                    'steps': ('PUCQ4_T2R_STEPS', int),
                    'ramsey_freq': ('PUCQ4_T2R_DETUNING_MHZ', float),
                    'reps': ('PUCQ4_T2R_REPS', int),
                    'rounds': ('PUCQ4_T2R_ROUNDS', int),
                    'relax_delay': ('PUCQ4_T2R_RELAX_DELAY_US', float),
                },
                'SpinEcho_ge': {
                    'start': ('PUCQ4_T2E_START_US', float),
                    'stop': ('PUCQ4_T2E_STOP_US', float),
                    'steps': ('PUCQ4_T2E_STEPS', int),
                    'ramsey_freq': ('PUCQ4_T2E_DETUNING_MHZ', float),
                    'reps': ('PUCQ4_T2E_REPS', int),
                    'rounds': ('PUCQ4_T2E_ROUNDS', int),
                    'relax_delay': ('PUCQ4_T2E_RELAX_DELAY_US', float),
                },
            }
            for experiment_name, overrides in coherence_overrides.items():
                for config_key, (env_name, converter) in overrides.items():
                    if os.environ.get(env_name) is not None:
                        set_qubit_expt_value(
                            experiment_name, config_key, QubitIndex,
                            converter(os.environ[env_name]),
                        )
        ###################################################### TOF #####################################################
        if run_flags["tof"]:
            tof        = TOFExperiment(QubitIndex, studyDocumentationFolder, experiment, j, save_figs, unmasking_resgain = unmask)
            t, iq_list, tof_config = tof.run()
            del tof

        ################################################# g-e Res spec ####################################################
        if run_flags["res_spec"]:
            try:
                if os.environ.get('PUCQ4_RES_START_MHZ'):
                    expt_cfg['res_spec']['start'] = float(os.environ['PUCQ4_RES_START_MHZ'])
                if os.environ.get('PUCQ4_RES_STEP_MHZ'):
                    expt_cfg['res_spec']['step_size'] = float(os.environ['PUCQ4_RES_STEP_MHZ'])
                if os.environ.get('PUCQ4_RES_STEPS'):
                    expt_cfg['res_spec']['steps'] = int(os.environ['PUCQ4_RES_STEPS'])
                if os.environ.get('PUCQ4_RES_REPS'):
                    expt_cfg['res_spec']['reps'] = int(os.environ['PUCQ4_RES_REPS'])
                res_spec   = ResonanceSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs,
                                                   experiment = experiment, verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
                res_freqs, freq_pts, freq_center, amps, sys_config_rspec = res_spec.run()
                print(f'g-e Resonator {QubitIndex + 1} frequency: {res_freqs[0]}')

                # Collect completed spectra before a later fit can skip this row.
                # The normal end-of-round Data_H5 writer still owns persistence.
                if save_data_h5:
                    sample = j - batch_num * save_r - 1
                    res_data[QubitIndex]['Dates'][sample] = time.mktime(datetime.datetime.now().timetuple())
                    res_data[QubitIndex]['freq_pts'][sample] = freq_pts
                    res_data[QubitIndex]['freq_center'][sample] = freq_center
                    res_data[QubitIndex]['Amps'][sample] = amps
                    res_data[QubitIndex]['Found Freqs'][sample] = res_freqs
                    res_data[QubitIndex]['Round Num'][sample] = j
                    res_data[QubitIndex]['Batch Num'][sample] = batch_num
                    res_data[QubitIndex]['Exp Config'][sample] = copy.deepcopy(expt_cfg)
                    res_data[QubitIndex]['Syst Config'][sample] = sys_config_rspec

                offset = freq_offsets[QubitIndex]
                # Preserve the calibrated physical readout tone. A fresh dip
                # can differ from the reference used when optimizing its offset.
                # Explicit base/offset diagnostics retain their requested meaning.
                if not pre_optimize and not any(os.environ.get(name) for name in (
                        'PUCQ4_RES_FREQ_MHZ', 'PUCQ4_READOUT_OFFSET_MHZ')):
                    offset = selected_flux_config['readout_frequency_ge'][QubitIndex] - res_freqs[0]
                if readout_frequency_override is not None:
                    offset = float(readout_frequency_override) - res_freqs[0]
                offset_res_freqs = [r + offset for r in res_freqs]
                experiment.readout_cfg['res_freq_ge'] = offset_res_freqs[0]
                experiment.readout_cfg['res_freq_offset_ge'] = offset
                print('Calibrated readout: %.9f MHz; measured-dip offset: %+.9f MHz'
                      % (offset_res_freqs[0], offset))
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

        ################################################## g-e Qubit spec ##################################################
        if run_flags["q_spec"]:
            try:
                print(
                    "Qspec readout frequency: "
                    f"{float(experiment.readout_cfg['res_freq_ge']):.12g} MHz"
                )
                qspec_seed_frequency = float(experiment.qubit_cfg['qubit_freq_ge'])
                qspec_start = os.environ.get('PUCQ4_QSPEC_START_MHZ')
                qspec_stop = os.environ.get('PUCQ4_QSPEC_STOP_MHZ')
                if (qspec_start is None) != (qspec_stop is None):
                    raise ValueError(
                        'PUCQ4_QSPEC_START_MHZ and PUCQ4_QSPEC_STOP_MHZ must be supplied together'
                    )
                qspec_half_span = os.environ.get('PUCQ4_QSPEC_HALF_SPAN_MHZ')
                if qspec_start is not None:
                    expt_cfg['qubit_spec_ge']['start'][QubitIndex] = float(qspec_start)
                    expt_cfg['qubit_spec_ge']['stop'][QubitIndex] = float(qspec_stop)
                elif qspec_half_span is not None or qfreq_override is not None:
                    # The row profile is applied before the optional carrier
                    # override. Keep its span centered on that explicit seed.
                    half_span = float(qspec_half_span) if qspec_half_span is not None else float(
                        row_profile['qubit_spec_ge']['half_span_MHz'])
                    center = float(experiment.qubit_cfg['qubit_freq_ge'])
                    expt_cfg['qubit_spec_ge']['start'][QubitIndex] = center - half_span
                    expt_cfg['qubit_spec_ge']['stop'][QubitIndex] = center + half_span
                if os.environ.get('PUCQ4_QSPEC_STEPS'):
                    expt_cfg['qubit_spec_ge']['steps'] = int(os.environ['PUCQ4_QSPEC_STEPS'])
                if os.environ.get('PUCQ4_QSPEC_REPS'):
                    expt_cfg['qubit_spec_ge']['reps'] = int(os.environ['PUCQ4_QSPEC_REPS'])
                if os.environ.get('PUCQ4_QSPEC_SSF_SHOTS'):
                    expt_cfg['Readout_Optimization']['steps'] = int(
                        os.environ['PUCQ4_QSPEC_SSF_SHOTS']
                    )
                if os.environ.get('PUCQ4_QSPEC_SSF_RELAX_DELAY_US'):
                    expt_cfg['Readout_Optimization']['relax_delay'] = float(
                        os.environ['PUCQ4_QSPEC_SSF_RELAX_DELAY_US']
                    )
                if os.environ.get('PUCQ4_QSPEC_LENGTH_US'):
                    experiment.qubit_cfg['qubit_length_ge'] = float(os.environ['PUCQ4_QSPEC_LENGTH_US'])
                q_spec = QubitSpectroscopy(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j,
                                           signal, save_figs, plot_fit=True, experiment=experiment,
                                           live_plot=live_plot, verbose=verbose, logger=rr_logger,
                                           unmasking_resgain=unmask)
                qspec_scaling = os.environ.get('PUCQ4_QSPEC_SCALING', 'true').strip().lower() not in {
                    '0', 'false', 'no', 'off'
                }
                qspec_result = q_spec.run(return_fwhm=True, scaling=qspec_scaling)
                if qspec_scaling:
                    (qspec_I, qspec_Q, qspec_freqs, qspec_fit, qubit_freq, sys_config_qspec, qspec_fwhm,
                     ss_Q_e_qspec, ss_Q_g_qspec, ss_I_e_qspec, ss_I_g_qspec,
                     I_shots_qspec, Q_shots_qspec) = qspec_result
                    qspec_e = np.mean(ss_I_e_qspec + 1j * ss_Q_e_qspec)
                    qspec_g = np.mean(ss_I_g_qspec + 1j * ss_Q_g_qspec)
                    qspec_population = np.real(
                        ((qspec_I + 1j * qspec_Q) - qspec_g)
                        * np.conj(qspec_e - qspec_g)
                    ) / np.abs(qspec_e - qspec_g) ** 2
                    if os.environ.get('PUCQ4_QSPEC_LOG_CALIBRATION', '').strip().lower() in {
                            '1', 'true', 'yes', 'on'}:
                        print(
                            'Qspec SSF centroids: '
                            f'g_I={qspec_g.real:.12g}, g_Q={qspec_g.imag:.12g}, '
                            f'e_I={qspec_e.real:.12g}, e_Q={qspec_e.imag:.12g}'
                        )
                else:
                    (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit, qubit_freq,
                     sys_config_qspec, qspec_fwhm, I_shots_qspec, Q_shots_qspec) = qspec_result
                    qspec_fit = (qspec_I_fit, qspec_Q_fit)
                    ss_Q_e_qspec = ss_Q_g_qspec = ss_I_e_qspec = ss_I_g_qspec = None
                    qspec_population = None

                # Preserve acquired IQ and calibration even when no frequency
                # fit is accepted. None fits remain explicit diagnostic data.
                if save_data_h5:
                    sample = j - batch_num * save_r - 1
                    qspec_data[QubitIndex]['Dates'][sample] = time.mktime(datetime.datetime.now().timetuple())
                    qspec_data[QubitIndex]['I'][sample] = qspec_I
                    qspec_data[QubitIndex]['Q'][sample] = qspec_Q
                    qspec_data[QubitIndex]['Population'][sample] = qspec_population
                    qspec_data[QubitIndex]['Frequencies'][sample] = qspec_freqs
                    qspec_data[QubitIndex]['I Fit'][sample] = qspec_fit
                    qspec_data[QubitIndex]['Round Num'][sample] = j
                    qspec_data[QubitIndex]['Batch Num'][sample] = batch_num
                    qspec_data[QubitIndex]['Recycled QFreq'][sample] = recycled_qfreq
                    qspec_data[QubitIndex]['Exp Config'][sample] = copy.deepcopy(expt_cfg)
                    qspec_data[QubitIndex]['Syst Config'][sample] = sys_config_qspec
                    qspec_data[QubitIndex]['ss_Q_e'][sample] = ss_Q_e_qspec
                    qspec_data[QubitIndex]['ss_Q_g'][sample] = ss_Q_g_qspec
                    qspec_data[QubitIndex]['ss_I_e'][sample] = ss_I_e_qspec
                    qspec_data[QubitIndex]['ss_I_g'][sample] = ss_I_g_qspec

                if qubit_freq is None:
                    keep_seed = os.environ.get(
                        'PUCQ4_QSPEC_KEEP_SEED_ON_FIT_FAILURE', ''
                    ).strip().lower() in {'1', 'true', 'yes', 'on'}
                    if keep_seed:
                        qubit_freq = qspec_seed_frequency
                        recycled_qfreq = True
                        experiment.qubit_cfg['qubit_freq_ge'] = qubit_freq
                        stored_qspec_list[QubitIndex] = qubit_freq
                        rr_logger.warning(
                            f"Qspec fit failed; retaining explicitly supplied seed {qubit_freq} "
                            "so the acquired common-grid data can be saved"
                        )
                        if verbose:
                            print(f"Qspec fit failed; retaining supplied seed: {qubit_freq}")
                    elif stored_qspec_list[QubitIndex] is not None:
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

                    if qubit_freq is None:
                        continue
                else:
                    experiment.qubit_cfg['qubit_freq_ge'] = float(qubit_freq)
                    stored_qspec_list[QubitIndex] = float(qubit_freq)
                if save_data_h5:
                    qspec_data[QubitIndex]['Recycled QFreq'][sample] = recycled_qfreq
                rr_logger.info(f"g-e Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
                if verbose:
                    print(f"g-e Qubit {QubitIndex + 1} frequency: {float(qubit_freq)}")
                    if qspec_fwhm is not None:
                        print(f"g-e Qubit {QubitIndex + 1} FWHM: {float(qspec_fwhm)} MHz")
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
            # Apply axis overrides before construction. ``add_qubit_experiment``
            # creates the gain ``QickSweep1D`` inside the constructor, so
            # changing only ``rabi.config['start'/'stop']`` afterward changes
            # the metadata but leaves the hardware sweep on the old axis.
            rabi_reps_override = os.environ.get('PUCQ4_RABI_REPS')
            rabi_rounds_override = os.environ.get('PUCQ4_RABI_ROUNDS')
            rabi_relax_delay_override = os.environ.get(
                'PUCQ4_RABI_RELAX_DELAY_US'
            )
            rabi_steps_override = os.environ.get('PUCQ4_RABI_STEPS')
            rabi_start_override = os.environ.get('PUCQ4_RABI_START')
            rabi_stop_override = os.environ.get('PUCQ4_RABI_STOP')
            rabi_ssf_shots_override = os.environ.get(
                'PUCQ4_RABI_SSF_SHOTS'
            )
            rabi_ssf_relax_delay_override = os.environ.get(
                'PUCQ4_RABI_SSF_RELAX_DELAY_US'
            )
            if rabi_reps_override is not None:
                expt_cfg['power_rabi_ge']['reps'] = int(rabi_reps_override)
            if rabi_rounds_override is not None:
                expt_cfg['power_rabi_ge']['rounds'] = int(rabi_rounds_override)
            if rabi_relax_delay_override is not None:
                expt_cfg['power_rabi_ge']['relax_delay'] = float(
                    rabi_relax_delay_override
                )
            if rabi_steps_override is not None:
                expt_cfg['power_rabi_ge']['steps'] = int(rabi_steps_override)
            if rabi_start_override is not None:
                expt_cfg['power_rabi_ge']['start'] = float(rabi_start_override)
            if rabi_stop_override is not None:
                expt_cfg['power_rabi_ge']['stop'] = float(rabi_stop_override)
            if rabi_ssf_shots_override is not None:
                expt_cfg['Readout_Optimization']['steps'] = int(
                    rabi_ssf_shots_override
                )
            if rabi_ssf_relax_delay_override is not None:
                expt_cfg['Readout_Optimization']['relax_delay'] = float(
                    rabi_ssf_relax_delay_override
                )
            rabi = AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                                           save_figs=save_figs, save_shots=False,
                                           experiment=experiment, live_plot=live_plot,
                                           increase_qubit_reps=increase_qubit_reps,
                                           qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                           multiply_qubit_reps_by=multiply_qubit_reps_by,
                                           verbose=verbose, logger=rr_logger, unmasking_resgain=unmask)
            if rabi_reps_override is not None:
                rabi.config['reps'] = int(rabi_reps_override)
            if rabi_rounds_override is not None:
                rabi.config['rounds'] = int(rabi_rounds_override)
            if rabi_relax_delay_override is not None:
                rabi.config['relax_delay'] = float(rabi_relax_delay_override)
            if rabi_steps_override is not None:
                rabi.config['steps'] = int(rabi_steps_override)
            if rabi_start_override is not None:
                rabi.config['start'] = float(rabi_start_override)
            if rabi_stop_override is not None:
                rabi.config['stop'] = float(rabi_stop_override)
            print(
                "Rabi operating point: qubit_frequency_MHz="
                f"{float(experiment.qubit_cfg['qubit_freq_ge']):.12g}, "
                "readout_frequency_MHz="
                f"{float(experiment.readout_cfg['res_freq_ge']):.12g}, "
                f"sigma_us={float(rabi.config['sigma']):.12g}"
            )
            print(
                "Rabi sweep settings: "
                f"start={float(rabi.config['start']):.12g}, "
                f"stop={float(rabi.config['stop']):.12g}, "
                f"steps={int(rabi.config['steps'])}, "
                f"reps={int(rabi.config['reps'])}, "
                f"rounds={int(rabi.config['rounds'])}, "
                f"relax_delay_us={float(rabi.config['relax_delay']):.12g}, "
                "ssf_shots_per_state="
                f"{int(expt_cfg['Readout_Optimization']['steps'])}, "
                "ssf_relax_delay_us="
                f"{float(expt_cfg['Readout_Optimization']['relax_delay']):.12g}"
            )
            rabi_scaling = os.environ.get('PUCQ4_RABI_SCALING', 'true').strip().lower() not in {
                '0', 'false', 'no', 'off'
            }
            rabi_result = rabi.run(thresholding=thresholding, scaling=rabi_scaling)
            if rabi_scaling:
                (rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp,
                 sys_config_rabi, ss_Q_e, ss_Q_g, ss_I_e, ss_I_g,
                 rabi_I_shots, rabi_Q_shots) = rabi_result
                rabi_e = np.mean(ss_I_e + 1j * ss_Q_e)
                rabi_g = np.mean(ss_I_g + 1j * ss_Q_g)
                rabi_population = np.real(
                    ((rabi_I + 1j * rabi_Q) - rabi_g)
                    * np.conj(rabi_e - rabi_g)
                ) / np.abs(rabi_e - rabi_g) ** 2
                print(
                    "Rabi SSF centroids: "
                    f"g_I={float(np.mean(ss_I_g)):.12g}, "
                    f"g_Q={float(np.mean(ss_Q_g)):.12g}, "
                    f"e_I={float(np.mean(ss_I_e)):.12g}, "
                    f"e_Q={float(np.mean(ss_Q_e)):.12g}"
                )
                print(
                    "Rabi calibrated population range: "
                    f"min={float(np.min(rabi_population)):.12g}, "
                    f"max={float(np.max(rabi_population)):.12g}"
                )
            else:
                (rabi_I, rabi_Q, rabi_gains, rabi_fit, pi_amp,
                 sys_config_rabi, rabi_I_shots, rabi_Q_shots) = rabi_result
                ss_Q_e = ss_Q_g = ss_I_e = ss_I_g = None
                rabi_population = None

            # A mapping run still needs the measured row when the sinusoid fit
            # cannot identify a pi point inside its bounded gain interval.  The
            # explicit opt-in below retains the supplied calibration pulse only
            # for downstream metadata/SSF preparation; it never labels that
            # seed as a newly measured pi amplitude.
            rabi_fit_failed = rabi_fit is None and pi_amp is None
            if rabi_fit_failed:
                keep_seed = os.environ.get(
                    'PUCQ4_RABI_KEEP_SEED_ON_FIT_FAILURE', ''
                ).strip().lower() in {'1', 'true', 'yes', 'on'}
                if keep_seed:
                    pi_amp = float(experiment.qubit_cfg['pi_amp'])
                    rr_logger.warning(
                        'Rabi fit failed; retaining the explicitly supplied pi '
                        f'seed {pi_amp} so the acquired mapping row can be saved'
                    )
                    if verbose:
                        print(
                            'Rabi fit failed; retaining supplied pi seed for '
                            f'map persistence: {pi_amp}'
                        )
                else:
                    rr_logger.info(
                        'g-e Rabi fit didnt work, skipping the rest of this qubit'
                    )
                    if verbose:
                        print(
                            'g-e Rabi fit didnt work, skipping the rest of this qubit'
                        )
                    continue  # skip the rest of this qubit

            experiment.qubit_cfg['pi_amp'] = float(pi_amp)
            rr_logger.info(f'g-e Pi amplitude for qubit {QubitIndex + 1} is: {float(pi_amp)}')
            if verbose:
                print(
                    'g-e Pi amplitude for qubit ', QubitIndex + 1,
                    ' is: ', float(pi_amp)
                )
                if rabi_fit_failed:
                    print('Rabi pi amplitude status: retained seed; fit unavailable')
            del rabi

            # except Exception as e:
            #     if debug_mode:
            #         raise e # In debug mode, re-raise the exception immediately
            #     else:
            #         rr_logger.exception(f'Got the following error, continuing: {e}')
            #         if verbose: print(f'Got the following error, continuing: {e}')
            #         continue #skip the rest of this qubit
            ###################################################### g-e T2R_correction #####################################################
            if run_flags["t2r_correction"]:
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

                    experiment.qubit_cfg['qubit_freq_ge'] = experiment.qubit_cfg[
                                                                'qubit_freq_ge'] - ramsey_found_q_freq + \
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
                     sys_config_rabi_corrected, ss_Q_e2, ss_Q_g2, ss_I_e2, ss_I_g2, I_shots_rabi_corr,
                     Q_shots_rabi_corr) = rabi.run(thresholding=thresholding, scaling=True)

                    # if these are None, fit didnt work
                    if (rabi_fit_corrected is None and pi_amp_corrected is None):
                        rr_logger.info('g-e Rabi fit didnt work, skipping the rest of this qubit')
                        if verbose: print('g-e Rabi fit didnt work, skipping the rest of this qubit')
                        continue  # skip the rest of this qubit

                    experiment.qubit_cfg['pi_amp'] = float(pi_amp_corrected)
                    rr_logger.info(f'corrected g-e Pi amplitude for qubit {QubitIndex + 1} is: {float(pi_amp)}')
                    if verbose: print('corrected g-e Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(pi_amp))
                    del rabi
                except Exception as e:
                    if debug_mode:
                        raise e  # In debug mode, re-raise the exception immediately
                    else:
                        rr_logger.exception(f'Got the following error, continuing: {e}')
                        if verbose: print(f'Got the following error, continuing: {e}')
                        continue  # skip the rest of this qubit

        # The original Ramsey correction lived inside the Rabi branch.  A
        # current-sweep controller needs to seed it with an already validated
        # qspec frequency and pi pulse, persist that one correction trace, and
        # then launch the final Rabi map in a separate canonical run.  Keep the
        # established T2RMeasurement.adjust_qspec implementation and correction
        # formula, but allow that measurement to run independently.
        if run_flags["t2r_correction"] and not run_flags["rabi"]:
            try:
                correction_cfg = expt_cfg['Ramsey_ge_correction']
                correction_overrides = {
                    'PUCQ4_RAMSEY_CORRECTION_REPS': ('reps', int),
                    'PUCQ4_RAMSEY_CORRECTION_ROUNDS': ('rounds', int),
                    'PUCQ4_RAMSEY_CORRECTION_START_US': ('start', float),
                    'PUCQ4_RAMSEY_CORRECTION_STOP_US': ('stop', float),
                    'PUCQ4_RAMSEY_CORRECTION_STEPS': ('steps', int),
                    'PUCQ4_RAMSEY_CORRECTION_DETUNING_MHZ': ('ramsey_freq', float),
                    'PUCQ4_RAMSEY_CORRECTION_RELAX_DELAY_US': ('relax_delay', float),
                }
                for env_name, (config_name, converter) in correction_overrides.items():
                    if os.environ.get(env_name) is not None:
                        correction_cfg[config_name] = converter(os.environ[env_name])

                correction_seed_frequency = float(
                    experiment.qubit_cfg['qubit_freq_ge']
                )
                t2r = T2RMeasurement(
                    QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j,
                    signal, save_figs, experiment=experiment,
                    live_plot=live_plot, fit_data=True,
                    increase_qubit_reps=increase_qubit_reps,
                    qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                    multiply_qubit_reps_by=multiply_qubit_reps_by,
                    verbose=verbose, logger=rr_logger,
                    unmasking_resgain=unmask, correction=True,
                    correction_round=1,
                )
                print(
                    'Ramsey correction operating point: seed_frequency_MHz='
                    f'{correction_seed_frequency:.12g}, readout_frequency_MHz='
                    f'{float(experiment.readout_cfg["res_freq_ge"]):.12g}, '
                    f'sigma_us={float(t2r.config["sigma"]):.12g}, '
                    f'pi_amp={float(t2r.config["pi_amp"]):.12g}'
                )
                print(
                    'Ramsey correction sweep settings: '
                    f'start_us={float(t2r.config["start"]):.12g}, '
                    f'stop_us={float(t2r.config["stop"]):.12g}, '
                    f'steps={int(t2r.config["steps"])}, '
                    f'detuning_MHz={float(t2r.config["ramsey_freq"]):.12g}, '
                    f'reps={int(t2r.config["reps"])}, '
                    f'rounds={int(t2r.config["rounds"])}, '
                    f'relax_delay_us={float(t2r.config["relax_delay"]):.12g}'
                )
                (t2r_est_1, t2r_err_1, t2r_I_1, t2r_Q_1,
                 t2r_delay_times_1, fit_ramsey_1, sys_config_t2r_1,
                 ramsey_found_q_freq, t2r_I_shots_1,
                 t2r_Q_shots_1) = t2r.adjust_qspec(
                    thresholding=thresholding, correction=True
                )
                if (ramsey_found_q_freq is None
                        or not np.isfinite(ramsey_found_q_freq)):
                    raise RuntimeError(
                        'Ramsey correction did not return a finite fringe frequency'
                    )
                corrected_qubit_frequency = (
                    correction_seed_frequency - float(ramsey_found_q_freq)
                    + float(t2r.config['ramsey_freq'])
                )
                experiment.qubit_cfg['qubit_freq_ge'] = (
                    corrected_qubit_frequency
                )
                print(
                    'Ramsey correction result: seed_frequency_MHz='
                    f'{correction_seed_frequency:.12g}, fitted_fringe_MHz='
                    f'{float(ramsey_found_q_freq):.12g}, '
                    f'intentional_detuning_MHz='
                    f'{float(t2r.config["ramsey_freq"]):.12g}, '
                    f'corrected_frequency_MHz='
                    f'{corrected_qubit_frequency:.12g}'
                )
                del t2r
            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(
                    f'Ramsey frequency-correction error on qubit {QubitIndex}: {e}'
                )
                if verbose:
                    print(
                        'Ramsey frequency-correction error on qubit '
                        f'{QubitIndex}: {e}'
                    )
                continue

        ########################################## g-e Single Shot Measurements ############################################
        if run_flags["ss"]:
            if os.environ.get('PUCQ4_SSF_SHOTS'):
                set_qubit_expt_value('Readout_Optimization', 'steps', QubitIndex,
                                     int(os.environ['PUCQ4_SSF_SHOTS']))
            if os.environ.get('PUCQ4_SSF_RELAX_DELAY_US'):
                set_qubit_expt_value('Readout_Optimization', 'relax_delay', QubitIndex,
                                     float(os.environ['PUCQ4_SSF_RELAX_DELAY_US']))
            # try:
            ss = SingleShot(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, save_figs, experiment = experiment,
                            verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
            fid, angle, iq_list_g, iq_list_e, sys_config_ss = ss.run()
            I_g_ss_ge = iq_list_g[0][0].T[0]
            Q_g_ss_ge = iq_list_g[0][0].T[1]
            I_e_ss_ge = iq_list_e[0][0].T[0]
            Q_e_ss_ge = iq_list_e[0][0].T[1]

            # fid, threshold, angle, ig_new, ie_new = ss.hist_ssf(
            #     data=[I_g, Q_g, I_e, Q_e], cfg=ss.config, plot=save_figs)
            print(sys_config_ss)
            # except Exception as e:
            #     if debug_mode:
            #         raise  # In debug mode, re-raise the exception immediately
            #     else:
            #         rr_logger.exception(f'Got the following error, continuing: {e}')
            #         if verbose: print(f'Got the following error, continuing: {e}')
            #         continue #skip the rest of this qubit

        ############################################## res spec ef ####################################################
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

        ################################################ Qubit Spec EF ################################################
        if run_flags["ef_q_spec"]:
            try:
                rr_logger.info("----------------- Starting Qubit Spec EF  -----------------")
                if verbose:
                    print("----------------- Starting Qubit Spec EF  -----------------")


                ef_q_spec = EFQubitSpectroscopy(QubitIndex, number_of_qubits, studyDocumentationFolder, j, signal,
                               True, experiment, live_plot, unmasking_resgain = unmask)

                efqspec_I, efqspec_Q, efqspec_freqs, sys_config_qspec_ef, efqspec_I_fit, efqspec_Q_fit, efqubit_freq = ef_q_spec.run()

                qubit_freqs_ef[QubitIndex] = efqubit_freq
                experiment.qubit_cfg['qubit_freq_ef'] = float(efqubit_freq)

                # rr_logger.info(f"EF Qubit {QubitIndex + 1} frequency: {efqubit_freq}")
                if verbose:
                    print(f"EF Qubit {QubitIndex + 1} frequency: {efqubit_freq}")

                del ef_q_spec

            except Exception as e:
                if debug_mode:
                    raise  # In debug mode, re-raise the exception immediately
                rr_logger.exception(f"EF QubitSpectroscopyGE error on qubit {QubitIndex + 1}: {e}")
        ################################################### len rabi ###############################################
        if run_flags["len_rabi"]:
            try:

                from section_006p5_length_rabi_ge import LengthRabiExperiment

                rabi = LengthRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                                            save_figs=save_figs,
                                            experiment=experiment, live_plot=live_plot,
                                            increase_qubit_reps=increase_qubit_reps,
                                            qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                            multiply_qubit_reps_by=multiply_qubit_reps_by,
                                            verbose=verbose)
                (len_rabi_I, len_rabi_Q, len_rabi_gains, len_rabi_fit, len_pi_amp,
                 len_sys_config_rabi, len_ss_Q_e, len_ss_Q_g, len_ss_I_e, len_ss_I_g,
                 len_I_shots, len_Q_shots) = rabi.run(thresholding=thresholding)

                # if these are None, fit didnt work
                if (len_rabi_fit is None and len_pi_amp is None):
                    rr_logger.info('g-e Rabi fit didnt work, skipping the rest of this qubit')
                    if verbose: print('g-e Rabi fit didnt work, skipping the rest of this qubit')
                    continue  # skip the rest of this qubit

                del rabi
            except Exception as e:
                if debug_mode:
                    raise e  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
                    continue

        ################################################ e-f amp rabi pop meas. ################################################
        if run_flags["rabi_pop_meas"]:
            t0 = time.perf_counter()
            rr_logger.info(
                "----------------- Starting EF Amplitude Rabi Population Measurements -----------------")
            if verbose:
                print("----------------- EF Amplitude Rabi Population Measurements  -----------------")

            try:
                efAmprabi_Qtemps = Temps_EFAmpRabiExperiment(QubitIndex, number_of_qubits, list_of_all_qubits,
                                                             studyDocumentationFolder,
                                                             j,
                                                             signal, save_figs,
                                                             experiment, live_plot,
                                                             increase_qubit_reps, qubit_to_increase_reps_for,
                                                             multiply_qubit_reps_by, unmasking_resgain = unmask)
                (I1_qtemp, Q1_qtemp, gains1_qtemp, fit_cosine1_qtemp, pi_amp1_qtemp, A_amplitude1, amp_fit1,
                 I2_qtemp, Q2_qtemp, gains2_qtemp, fit_cosine2_qtemp, pi_amp2_qtemp, A_amplitude2, amp_fit2,
                 sysconfig_efrabi_Qtemps) = efAmprabi_Qtemps.run(experiment.soccfg, experiment.soc)

                # Just to quickly output the qubit temperature ---------------------
                # date = data_set
                # fit_saved = fit_data
                # outerFolder_save_plots = ""
                # unique_folder_path = ""
                # outerFolder = ""
                # qtempclass = PlotRR_noQick(date, figure_quality, save_figs, fit_saved, signal, run_name, number_of_qubits, outerFolder,
                #  outerFolder_save_plots, unique_folder_path)
                # T_K, T_mK, _, _ = qtempclass.Qubit_Temperature_Convert(A_amplitude1, A_amplitude2, qubit_freq)
                # print(f"Q{QubitIndex + 1} temperature: {T_mK} mK")
                #-------------------------------------------------------------------------

                rr_logger.info(f"RPM Amplitudes for qubit {QubitIndex + 1}: A1 = {float(A_amplitude1)}, A2 = {float(A_amplitude2)}")
                if verbose:
                    print(f"RPM Amplitudes for qubit {QubitIndex + 1}: A1 = {float(A_amplitude1)}, A2 = {float(A_amplitude2)}")

                del efAmprabi_Qtemps

            except Exception as e:
                if debug_mode:
                    raise  # In debug mode, re-raise the exception immediately
                rr_logger.exception(f"EF Rabi Population Measurements error on qubit {QubitIndex + 1}: {e}")

            t1 = time.perf_counter()
            print(f"RPM took {t1 - t0:.4f} seconds")

        # Apply coherence's calibration settings after Rabi/SSF, which can
        # independently override the shared Readout_Optimization settings.
        if any(run_flags[name] for name in ('t1', 't2r', 't2e')):
            if os.environ.get('PUCQ4_COHERENCE_SSF_SHOTS') is not None:
                set_qubit_expt_value(
                    'Readout_Optimization', 'steps', QubitIndex,
                    int(os.environ['PUCQ4_COHERENCE_SSF_SHOTS']),
                )
            if os.environ.get('PUCQ4_COHERENCE_SSF_RELAX_DELAY_US') is not None:
                set_qubit_expt_value(
                    'Readout_Optimization', 'relax_delay', QubitIndex,
                    float(os.environ['PUCQ4_COHERENCE_SSF_RELAX_DELAY_US']),
                )

        ###################################################### g-e T1 ######################################################
        if run_flags["t1"]:
            try:
                t1 = T1Measurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
                                   experiment = experiment,
                                   live_plot = live_plot, fit_data = fit_data,
                                   increase_qubit_reps = increase_qubit_reps,
                                   qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                   multiply_qubit_reps_by = multiply_qubit_reps_by,
                                   verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
                print(
                    'T1 operating point: qubit_frequency_MHz='
                    f'{float(experiment.qubit_cfg["qubit_freq_ge"]):.12g}, '
                    'readout_frequency_MHz='
                    f'{float(experiment.readout_cfg["res_freq_ge"]):.12g}, '
                    f'sigma_us={float(t1.config["sigma"]):.12g}, '
                    f'pi_amp={float(t1.config["pi_amp"]):.12g}'
                )
                print(
                    'T1 sweep settings: '
                    f'start_us={float(t1.config["start"]):.12g}, '
                    f'stop_us={float(t1.config["stop"]):.12g}, '
                    f'steps={int(t1.config["steps"])}, '
                    f'reps={int(t1.config["reps"])}, '
                    f'rounds={int(t1.config["rounds"])}, '
                    f'relax_delay_us={float(t1.config["relax_delay"]):.12g}, '
                    f'population_scaling={coherence_scaling}'
                )
                t1_result = t1.run(
                    thresholding=thresholding, scaling=coherence_scaling
                )
                if coherence_scaling:
                    (t1_est, t1_err, t1_I, t1_Q, t1_delay_times,
                     q1_fit_exponential, sys_config_t1, t1_ss_Q_e,
                     t1_ss_Q_g, t1_ss_I_e, t1_ss_I_g, _, _) = t1_result
                    t1_population = calibrated_population(
                        t1_I, t1_Q, t1_ss_I_e, t1_ss_Q_e,
                        t1_ss_I_g, t1_ss_Q_g,
                    )
                    print(
                        'T1 SSF centroids: '
                        f'g_I={float(np.mean(t1_ss_I_g)):.12g}, '
                        f'g_Q={float(np.mean(t1_ss_Q_g)):.12g}, '
                        f'e_I={float(np.mean(t1_ss_I_e)):.12g}, '
                        f'e_Q={float(np.mean(t1_ss_Q_e)):.12g}'
                    )
                    print(
                        'T1 calibrated population range: '
                        f'min={float(np.min(t1_population)):.12g}, '
                        f'max={float(np.max(t1_population)):.12g}'
                    )
                else:
                    (t1_est, t1_err, t1_I, t1_Q, t1_delay_times,
                     q1_fit_exponential, sys_config_t1) = t1_result
                    t1_population = None
                    t1_ss_Q_e = t1_ss_Q_g = None
                    t1_ss_I_e = t1_ss_I_g = None
                del t1

            except Exception as e:
                if debug_mode:
                    raise e # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
                    continue #skip the rest of this qubit

        ###################################################### g-e T2R #####################################################
        if run_flags["t2r"]:
            try:
                # Use the existing fitter's public frequency-guess input for
                # the measured slow fringe. Frequency and T2 remain free.
                t2r_fit_guess = row_profile["Ramsey_ge"].get("fit_frequency_guess_MHz")
                t2r_configured_fit = fit_data and t2r_fit_guess is not None
                t2r = T2RMeasurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal,
                                     save_figs and not t2r_configured_fit,
                                     experiment = experiment, live_plot = live_plot, fit_data = fit_data,
                                     increase_qubit_reps = increase_qubit_reps,
                                     qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                     multiply_qubit_reps_by = multiply_qubit_reps_by,
                                     verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
                print(
                    'T2R operating point: qubit_frequency_MHz='
                    f'{float(experiment.qubit_cfg["qubit_freq_ge"]):.12g}, '
                    'readout_frequency_MHz='
                    f'{float(experiment.readout_cfg["res_freq_ge"]):.12g}, '
                    f'sigma_us={float(t2r.config["sigma"]):.12g}, '
                    f'pi_amp={float(t2r.config["pi_amp"]):.12g}'
                )
                print(
                    'T2R sweep settings: '
                    f'start_us={float(t2r.config["start"]):.12g}, '
                    f'stop_us={float(t2r.config["stop"]):.12g}, '
                    f'steps={int(t2r.config["steps"])}, '
                    f'detuning_MHz={float(t2r.config["ramsey_freq"]):.12g}, '
                    f'reps={int(t2r.config["reps"])}, '
                    f'rounds={int(t2r.config["rounds"])}, '
                    f'relax_delay_us={float(t2r.config["relax_delay"]):.12g}, '
                    f'population_scaling={coherence_scaling}'
                )
                t2r_result = t2r.run(
                    thresholding=thresholding, scaling=coherence_scaling
                )
                if coherence_scaling:
                    (t2r_est, t2r_err, t2r_I, t2r_Q, t2r_delay_times,
                     fit_ramsey, sys_config_t2r, t2r_ss_Q_e, t2r_ss_Q_g,
                     t2r_ss_I_e, t2r_ss_I_g, _, _) = t2r_result
                    t2r_population = calibrated_population(
                        t2r_I, t2r_Q, t2r_ss_I_e, t2r_ss_Q_e,
                        t2r_ss_I_g, t2r_ss_Q_g,
                    )
                    print(
                        'T2R SSF centroids: '
                        f'g_I={float(np.mean(t2r_ss_I_g)):.12g}, '
                        f'g_Q={float(np.mean(t2r_ss_Q_g)):.12g}, '
                        f'e_I={float(np.mean(t2r_ss_I_e)):.12g}, '
                        f'e_Q={float(np.mean(t2r_ss_Q_e)):.12g}'
                    )
                    print(
                        'T2R calibrated population range: '
                        f'min={float(np.min(t2r_population)):.12g}, '
                        f'max={float(np.max(t2r_population)):.12g}'
                    )
                else:
                    (t2r_est, t2r_err, t2r_I, t2r_Q, t2r_delay_times,
                     fit_ramsey, sys_config_t2r) = t2r_result
                    t2r_population = None
                    t2r_ss_Q_e = t2r_ss_Q_g = None
                    t2r_ss_I_e = t2r_ss_I_g = None
                fit_ramsey_unseeded = copy.deepcopy(fit_ramsey)
                if t2r_configured_fit:
                    unseeded_tau, unseeded_error = t2r_est, t2r_err
                    t2r_fit_start = {"f": float(t2r_fit_guess)}
                    if coherence_scaling:
                        t2r_fit_start.update(row_profile["Ramsey_ge"].get("fit_population_guess") or {})
                    (fit_ramsey, t2r_est, t2r_err, t2r_plot_signal,
                     t2r_fitted_frequency) = t2r.t2_fit(
                        t2r_delay_times, t2r_I, t2r_Q,
                        guess=t2r_fit_start,
                        signal_data=t2r_population, return_freq=True,
                    )
                    sys_config_t2r = dict(
                        sys_config_t2r,
                        fit_frequency_guess_MHz=float(t2r_fit_guess),
                        fit_starting_values=copy.deepcopy(t2r_fit_start),
                        fitted_fringe_frequency_MHz=float(t2r_fitted_frequency),
                        unseeded_T2R_us=float(unseeded_tau),
                        unseeded_T2R_error_us=float(unseeded_error),
                    )
                    t2r.config.update(sys_config_t2r)
                    t2r.save_figs = save_figs
                    t2r.plot_results(
                        t2r_I, t2r_Q, t2r_delay_times, datetime.datetime.now(),
                        fit_ramsey, t2r_est, t2r_err, t2r_plot_signal,
                        scaling=coherence_scaling, Ie=t2r_ss_I_e, Ig=t2r_ss_I_g,
                        Qe=t2r_ss_Q_e, Qg=t2r_ss_Q_g,
                    )
                    print('T2R fit initialization: ' + str({
                        'starting_values': t2r_fit_start,
                        'fitted_frequency_MHz': t2r_fitted_frequency,
                        'unseeded_T2R_us': unseeded_tau, 'T2R_us': t2r_est,
                    }), flush=True)
                del t2r

            except Exception as e:
                if debug_mode:
                    raise e # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
                    continue #skip the rest of this qubit

        ##################################################### g-e T2E ######################################################
        if run_flags["t2e"]:
            try:
                t2e = T2EMeasurement(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs,
                                     experiment = experiment, live_plot = live_plot, fit_data = fit_data,
                                     increase_qubit_reps = increase_qubit_reps,
                                     qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                     multiply_qubit_reps_by = multiply_qubit_reps_by,
                                     verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
                print(
                    'T2E operating point: qubit_frequency_MHz='
                    f'{float(experiment.qubit_cfg["qubit_freq_ge"]):.12g}, '
                    'readout_frequency_MHz='
                    f'{float(experiment.readout_cfg["res_freq_ge"]):.12g}, '
                    f'sigma_us={float(t2e.config["sigma"]):.12g}, '
                    f'pi_amp={float(t2e.config["pi_amp"]):.12g}'
                )
                print(
                    'T2E sweep settings: '
                    f'start_us={float(t2e.config["start"]):.12g}, '
                    f'stop_us={float(t2e.config["stop"]):.12g}, '
                    f'steps={int(t2e.config["steps"])}, '
                    f'detuning_MHz={float(t2e.config["ramsey_freq"]):.12g}, '
                    f'reps={int(t2e.config["reps"])}, '
                    f'rounds={int(t2e.config["rounds"])}, '
                    f'relax_delay_us={float(t2e.config["relax_delay"]):.12g}, '
                    f'population_scaling={coherence_scaling}'
                )
                t2e_result = t2e.run(
                    thresholding=thresholding, scaling=coherence_scaling
                )
                if coherence_scaling:
                    (t2e_est, t2e_err, t2e_I, t2e_Q, t2e_delay_times,
                     fit_t2e, sys_config_t2e, t2e_ss_Q_e, t2e_ss_Q_g,
                     t2e_ss_I_e, t2e_ss_I_g, _, _) = t2e_result
                    t2e_population = calibrated_population(
                        t2e_I, t2e_Q, t2e_ss_I_e, t2e_ss_Q_e,
                        t2e_ss_I_g, t2e_ss_Q_g,
                    )
                    print(
                        'T2E SSF centroids: '
                        f'g_I={float(np.mean(t2e_ss_I_g)):.12g}, '
                        f'g_Q={float(np.mean(t2e_ss_Q_g)):.12g}, '
                        f'e_I={float(np.mean(t2e_ss_I_e)):.12g}, '
                        f'e_Q={float(np.mean(t2e_ss_Q_e)):.12g}'
                    )
                    print(
                        'T2E calibrated population range: '
                        f'min={float(np.min(t2e_population)):.12g}, '
                        f'max={float(np.max(t2e_population)):.12g}'
                    )
                else:
                    (t2e_est, t2e_err, t2e_I, t2e_Q, t2e_delay_times,
                     fit_t2e, sys_config_t2e) = t2e_result
                    t2e_population = None
                    t2e_ss_Q_e = t2e_ss_Q_g = None
                    t2e_ss_I_e = t2e_ss_I_g = None
                del t2e

            except Exception as e:
                if debug_mode:
                    raise e # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
                    continue #skip the rest of this qubit

        ################################################ e-f rabi ################################################
        if run_flags["ef_Rabi"]:
            increase_qubit_reps = False  # if you want to increase the reps for a qubit, set to True
            qubit_to_increase_reps_for = 0  # only has impact if previous line is True
            multiply_qubit_reps_by = 2  # only has impact if the line two above is True
            print('ef rabi')
            # try:
            # thresholding = False

            efrabi = EF_AmplitudeRabiExperiment(QubitIndex, number_of_qubits, studyDocumentationFolder, j, signal,
                                                False, experiment=experiment, live_plot=live_plot,
                                                increase_qubit_reps=increase_qubit_reps,
                                                qubit_to_increase_reps_for=qubit_to_increase_reps_for,
                                                multiply_qubit_reps_by=multiply_qubit_reps_by,
                                                unmasking_resgain=unmask)
            efrabi_I, efrabi_Q, efrabi_gains, efrabi_fit, efpi_amp, efsys_config_to_save = efrabi.run()
            # experiment.soccfg
            # if these are None, fit didnt work
            if (efrabi_fit is None and efpi_amp is None):
                print('Rabi fit didnt work, skipping the rest of this qubit')
                continue  # skip the rest of this qubit

            experiment.qubit_cfg['pi_ef_amp'] = float(efpi_amp)
            print('ef Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(efpi_amp))
            del efrabi
        ################################################ f res spec ################################################
        if run_flags["res_spec_fh"]:
            try:
                res_specFH = ResonanceSpectroscopyFH(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j,
                                                     save_figs,
                                                     experiment)
                res_freqs_fh, freq_pts_fh, freq_center_fh, amps_fh, sys_config_rspec_fh = res_specFH.run()
                experiment.readout_cfg['res_freq_fh'] = res_freqs_fh[0]
                print('Qubit ', QubitIndex + 1, ' f-h res freq: ', res_freqs_fh)

                del res_specFH

            except Exception as e:
                if debug_mode:
                    raise e
                rr_logger.exception(f"fh res spec error on qubit {QubitIndex}: {e}")
                if verbose:
                    print(f"fh res spec error on qubit {QubitIndex}: {e}")
                continue

        ########################################## e-f Single Shot Measurements ############################################
        if run_flags["ef_ss"]:
            try:

                ss_ef = SingleShot_ef(QubitIndex, number_of_qubits, studyDocumentationFolder, j,
                                                 save_figs, experiment=experiment, unmasking_resgain=unmask)

                iq_list_g, iq_list_e, iq_list_f, ie_new, if_new, theta_ef,  threshold_ef,  sys_config_ss_ef, fid, fid_ef = ss_ef.run_gef()
                # iq_list_g, iq_list_e, iq_list_f, ie_new, qe_new, if_new, qf_new, theta_ef, threshold_ef, self.config
                I_g_ss_gef = iq_list_g[0][0].T[0]
                Q_g_ss_gef = iq_list_g[0][0].T[1]
                I_e_ss_gef = iq_list_e[0][0].T[0]
                Q_e_ss_gef = iq_list_e[0][0].T[1]
                I_f_ss_gef = iq_list_f[0][0].T[0]
                Q_f_ss_gef = iq_list_f[0][0].T[1]

                # fid_ef, theta_ef, ie_new, if_new, threshold_ef = ss_ef.hist_ssf(
                #     data=[I_e, Q_e, I_f, Q_f], cfg=sys_config_ss_ef, plot=save_figs)
                # fid_ef, theta_ef, ie_new, qe_new, if_new, qf_new = ss_ef.fidelity_test()
                # fid, theta_ef, ie_new, qe_new, if_new, qf_new, threshold_ef
                print(sys_config_ss_ef)
            except Exception as e:
                if debug_mode:
                    raise  # In debug mode, re-raise the exception immediately
                else:
                    rr_logger.exception(f'Got the following error, continuing: {e}')
                    if verbose: print(f'Got the following error, continuing: {e}')
                    continue  # skip the rest of this qubit
        ################################################ Qubit Spec FH ################################################
        if run_flags["fh_q_spec"]:
            rr_logger.info("----------------- Starting Qubit Spec FH  -----------------")
            if verbose:
                print("----------------- Starting Qubit Spec FH  -----------------")

            try:
                increase_qubit_steps_fh = False
                # Qubit 4 needs more steps for e-f spec
                # if QubitIndex == 3:
                #     increase_qubit_steps_fh = True  # if you want to increase the steps for a qubit, set to True

                fh_q_spec = FHQubitSpectroscopy(QubitIndex, number_of_qubits, studyDocumentationFolder, j,
                                                signal,
                                                True, experiment, live_plot, unmasking_resgain=unmask)

                fhqspec_I, fhqspec_Q, fhqspec_freqs,  sys_config_qspec_fh = fh_q_spec.run()
                # fhqspec_I_fit, fhqspec_Q_fit, fhqubit_freq,
                # experiment.soccfg,
                # experiment.soc)
                # qubit_freqs_fh[QubitIndex] = fhqubit_freq
                experiment.qubit_cfg['qubit_freq_fh'][QubitIndex] = float(fhqspec_freqs)

                rr_logger.info(f"FH Qubit {QubitIndex + 1} frequency: {fhqspec_freqs}")
                if verbose:
                    print(f"FH Qubit {QubitIndex + 1} frequency: {fhqspec_freqs}")

                del fh_q_spec

            except Exception as e:
                if debug_mode:
                    raise  # In debug mode, re-raise the exception immediately
                rr_logger.exception(f"FH QubitSpectroscopyGE error on qubit {QubitIndex + 1}: {e}")

        ################################################ f-h rabi ################################################
        if run_flags["fh_rabi"]:
            increase_qubit_reps = False  # if you want to increase the reps for a qubit, set to True
            qubit_to_increase_reps_for = 0  # only has impact if previous line is True
            multiply_qubit_reps_by = 2  # only has impact if the line two above is True
            print('fh rabi')
            try:
                # thresholding = False

                fhrabi = FH_AmplitudeRabiExperiment(QubitIndex, tot_num_of_qubits, studyDocumentationFolder, j, signal, save_figs=save_figs,save_shots=False,
                                               experiment = experiment, live_plot = live_plot,
                                               increase_qubit_reps = increase_qubit_reps,
                                               qubit_to_increase_reps_for = qubit_to_increase_reps_for,
                                               multiply_qubit_reps_by = multiply_qubit_reps_by,
                                               verbose = verbose, logger = rr_logger, unmasking_resgain = unmask)
                fhrabi_I, fhrabi_Q, fhrabi_gains, fhrabi_fit, fhpi_amp, fhsys_config_to_save = fhrabi.run()
                # experiment.soccfg
                # if these are None, fit didnt work
                if (fhrabi_fit is None and fhpi_amp is None):
                    print('Rabi fit didnt work, skipping the rest of this qubit')
                    continue  # skip the rest of this qubit

                experiment.qubit_cfg['pi_fh_amp'] = float(fhpi_amp)
                print('fh Pi amplitude for qubit ', QubitIndex + 1, ' is: ', float(fhpi_amp))
                del fhrabi
            except Exception as e:
                if debug_mode:
                    raise  # In debug mode, re-raise the exception immediately
                rr_logger.exception(f"FH rabi error on qubit {QubitIndex + 1}: {e}")

        ########################################### g-e-f Single Shot Measurements ############################################

        if run_flags["ss_gef"]:
            try:
                ss = SingleShot_ef(QubitIndex, number_of_qubits, studyDocumentationFolder, j, save_figs, experiment)

                iq_list_g, iq_list_e, iq_list_f, ie_new,  if_new,  theta_ef, theta_fh, threshold_ef,threshold_fh , sys_config_ss_gef, fid, fid_fh = ss.run()

                I_g_ss_gef = iq_list_g[0][0].T[0]
                Q_g_ss_gef = iq_list_g[0][0].T[1]
                I_e_ss_gef = iq_list_e[0][0].T[0]
                Q_e_ss_gef = iq_list_e[0][0].T[1]
                I_f_ss_gef = iq_list_f[0][0].T[0]
                Q_f_ss_gef = iq_list_f[0][0].T[1]

                if run_flags["ss_gef"]:  # currently saves figs and h5 files every time this is run
                    provided_sigma_num = None  # de state circle radius = sigma_num * sigma. Set as None if you want the code to choose an appropriate one for you.
                    Analysis = False  # Keep as false, we are in RR mode here, not post-processing (analysis) mode
                    RR = True  # Keep as true, we are in RR mode here
                    date_analysis = None  # This only matters if you are in post-processing mode (for analysis purposes), keep as None here.
                    round_num = j
                    # analysis_gef_SSF = GEF_SSF_ANALYSIS(studyDocumentationFolder, QubitIndex, Analysis, RR,
                    #                                     date_analysis, round_num)
                    # (line_point1, line_point2, center_e, radius_e, T, v, f_outside, line_point1_rot, line_point2_rot,
                    #  center_e_rot, radius_e_rot, T_rot, v_rot, f_outside_rot
                    #  ) = analysis_gef_SSF.fstate_analysis_plot(I_g, Q_g, I_e, Q_e, I_f, Q_f, ig_new,  ie_new,
                    #
                    #                                            if_new,  theta_ef, threshold_ef, QubitIndex,
                    #                                            provided_sigma_num)
                    # # (line_point1, line_point2, center_e, radius_e, T, v, f_outside, line_point1_rot, line_point2_rot,
                    #  center_e_rot, radius_e_rot, T_rot, v_rot, f_outside_rot
                    #  ) = analysis_gef_SSF.fstate_analysis_plot(I_g, Q_g, I_e, Q_e, I_f, Q_f, ig_new, qg_new, ie_new,
                    #                                            qe_new,
                    #                                            if_new, qf_new, theta_ge, threshold_ge, QubitIndex,
                    #                                            provided_sigma_num)
                del ss
            except Exception as e:
                if debug_mode:
                    raise  # In debug mode, re-raise the exception immediately
                rr_logger.exception(f"gef ssf error on qubit {QubitIndex + 1}: {e}")

        ########################################## e-f Single Shot Measurements ############################################
        # if run_flags["test_act"]:
        #     # try:
        #
        #     act = Active_Reset_test(QubitIndex, number_of_qubits, studyDocumentationFolder, j,
        #                                      save_figs, experiment=experiment, unmasking_resgain=unmask)
        #     act_idata, act_qdata, no_act_idata, no_act_qdata, act_cfg = act.run()
        #         # iq_list_g, iq_list_e, iq_list_f, ie_new, qe_new, if_new, qf_new, theta_ef, threshold_ef, self.config
        #         # I_e = iq_list_e[QubitIndex][0].T[0]
        #         # Q_e = iq_list_e[QubitIndex][0].T[1]
        #         # I_f = iq_list_f[QubitIndex][0].T[0]
        #         # Q_f = iq_list_f[QubitIndex][0].T[1]
        #
        #         # fid_ef, theta_ef, ie_new, if_new, threshold_ef = ss_ef.hist_ssf(
        #         #     data=[I_e, Q_e, I_f, Q_f], cfg=sys_config_ss_ef, plot=save_figs)
        #         # idata, qdata = act.run()
        #         # fid, theta_ef, ie_new, qe_new, if_new, qf_new, threshold_ef
        #         # print(sys_config_ss_ef)
        #     # except Exception as e:
        #     #     if debug_mode:
        #     #         raise  # In debug mode, re-raise the exception immediately
        #     #     else:
        #     #         rr_logger.exception(f'Got the following error, continuing: {e}')
        #     #         if verbose: print(f'Got the following error, continuing: {e}')
        #     #         continue  # skip the rest of this qubit
        #     del act

        ############################################### Collect Results ################################################
        if save_data_h5:
            # ---------------------Collect g-e Res Spec Results----------------
            if run_flags["tof"]:
                tof_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                tof_data[QubitIndex]['t'][j - batch_num * save_r - 1] = t
                tof_data[QubitIndex]['iq_list'][j - batch_num * save_r - 1] = iq_list
                tof_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                tof_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                tof_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = copy.deepcopy(expt_cfg)
                tof_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = tof_config

            # Resonator/qspec results were collected immediately after acquisition
            # so a later rejected fit cannot discard their raw measurements.

            # ---------------------Collect g-e Rabi Results----------------
            if run_flags["rabi"]:
                rabi_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                rabi_data[QubitIndex]['I'][j - batch_num * save_r - 1] = rabi_I
                rabi_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = rabi_Q
                rabi_data[QubitIndex]['Population'][j - batch_num * save_r - 1] = rabi_population
                rabi_data[QubitIndex]['Gains'][j - batch_num * save_r - 1] = rabi_gains
                rabi_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = rabi_fit
                rabi_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                rabi_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                rabi_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = copy.deepcopy(expt_cfg)
                rabi_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rabi
                if rabi_scaling:
                    rabi_data[QubitIndex]['ss_Q_e'][j - batch_num * save_r - 1] = ss_Q_e
                    rabi_data[QubitIndex]['ss_Q_g'][j - batch_num * save_r - 1] = ss_Q_g
                    rabi_data[QubitIndex]['ss_I_e'][j - batch_num * save_r - 1] = ss_I_e
                    rabi_data[QubitIndex]['ss_I_g'][j - batch_num * save_r - 1] = ss_I_g
                else:
                    rabi_data[QubitIndex]['ss_Q_e'][j - batch_num * save_r - 1] = None
                    rabi_data[QubitIndex]['ss_Q_g'][j - batch_num * save_r - 1] = None
                    rabi_data[QubitIndex]['ss_I_e'][j - batch_num * save_r - 1] = None
                    rabi_data[QubitIndex]['ss_I_g'][j - batch_num * save_r - 1] = None
            # ---------------------Collect g-e Rabi Results----------------
            if run_flags["len_rabi"]:
                len_rabi_data[QubitIndex]['Dates'][0] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                len_rabi_data[QubitIndex]['I'][0] = len_rabi_I
                len_rabi_data[QubitIndex]['Q'][0] = len_rabi_Q
                len_rabi_data[QubitIndex]['Gains'][0] = len_rabi_gains
                len_rabi_data[QubitIndex]['Fit'][0] = len_rabi_fit
                len_rabi_data[QubitIndex]['Round Num'][0] = 0
                len_rabi_data[QubitIndex]['Batch Num'][0] = 0
                len_rabi_data[QubitIndex]['Exp Config'][0] = copy.deepcopy(expt_cfg)
                # len_rabi_data[QubitIndex]['Syst Config'][0] = len_sys_config_rabi
                # len_rabi_data[QubitIndex]['ss_Q_e'][0] = len_ss_Q_e
                # len_rabi_data[QubitIndex]['ss_Q_g'][0] = len_ss_Q_g
                # len_rabi_data[QubitIndex]['ss_I_e'][0] = len_ss_I_e
                # len_rabi_data[QubitIndex]['ss_I_g'][0] = len_ss_I_g

            # ---------------------Collect g-e Single Shot Results----------------
            if run_flags["ss"]:
                ss_data[QubitIndex]['Fidelity'][j - batch_num * save_r - 1] = fid
                ss_data[QubitIndex]['Angle'][j - batch_num * save_r - 1] = angle
                ss_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                ss_data[QubitIndex]['I_g'][j - batch_num * save_r - 1] = I_g_ss_ge
                ss_data[QubitIndex]['Q_g'][j - batch_num * save_r - 1] = Q_g_ss_ge
                ss_data[QubitIndex]['I_e'][j - batch_num * save_r - 1] = I_e_ss_ge
                ss_data[QubitIndex]['Q_e'][j - batch_num * save_r - 1] = Q_e_ss_ge
                ss_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                ss_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                ss_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = copy.deepcopy(expt_cfg)
                ss_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_ss

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
                ef_res_data[QubitIndex]['Exp Config'][0] = copy.deepcopy(expt_cfg)
                ef_res_data[QubitIndex]['Syst Config'][0] = sys_config_rspec_ef

            # ---------------------Collect f-h Res Spec Results----------------
            if run_flags["res_spec_fh"]:
                res_data_fh[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                res_data_fh[QubitIndex]['freq_pts'][j - batch_num * save_r - 1] = freq_pts_fh
                res_data_fh[QubitIndex]['freq_center'][j - batch_num * save_r - 1] = freq_center_fh
                res_data_fh[QubitIndex]['Amps'][j - batch_num * save_r - 1] = amps_fh
                res_data_fh[QubitIndex]['Found Freqs'][j - batch_num * save_r - 1] = res_freqs_fh
                res_data_fh[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                res_data_fh[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                res_data_fh[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = copy.deepcopy(expt_cfg)
                res_data_fh[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rspec_fh

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
                ef_qspec_data[QubitIndex]['Exp Config'][0] = copy.deepcopy(expt_cfg)
                ef_qspec_data[QubitIndex]['Syst Config'][0] = sys_config_qspec_ef

            if run_flags["fh_q_spec"]:
                fh_qspec_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                fh_qspec_data[QubitIndex]['I'][j - batch_num * save_r - 1] = fhqspec_I
                fh_qspec_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = fhqspec_Q
                fh_qspec_data[QubitIndex]['Frequencies'][j - batch_num * save_r - 1] = fhqspec_freqs
                fh_qspec_data[QubitIndex]['I Fit'][j - batch_num * save_r - 1] = None #fhqspec_I_fit
                fh_qspec_data[QubitIndex]['Q Fit'][j - batch_num * save_r - 1] = None #fhqspec_Q_fit
                fh_qspec_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                fh_qspec_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                fh_qspec_data[QubitIndex]['Recycled QFreq'][j - batch_num * save_r - 1] = False  # no rr so no recycling here
                fh_qspec_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = copy.deepcopy(expt_cfg)
                fh_qspec_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_qspec_fh

            # ----------Collect rabi population measurements (qubit temperature data) ----------------
            if run_flags["rabi_pop_meas"]:
                rabi_data_ef_Qtemps[QubitIndex]['Dates'][0] = (time.mktime(datetime.datetime.now().timetuple()))
                rabi_data_ef_Qtemps[QubitIndex]['Qfreq_ge'][0] = qubit_freq  # save the g-e qubit freq too for this qubit
                rabi_data_ef_Qtemps[QubitIndex]['I1'][0] = I1_qtemp
                rabi_data_ef_Qtemps[QubitIndex]['Q1'][0] = Q1_qtemp
                rabi_data_ef_Qtemps[QubitIndex]['Gains1'][0] = gains1_qtemp
                rabi_data_ef_Qtemps[QubitIndex]['Fit1'][0] = fit_cosine1_qtemp

                rabi_data_ef_Qtemps[QubitIndex]['I2'][0] = I2_qtemp
                rabi_data_ef_Qtemps[QubitIndex]['Q2'][0] = Q2_qtemp
                rabi_data_ef_Qtemps[QubitIndex]['Gains2'][0] = gains2_qtemp
                rabi_data_ef_Qtemps[QubitIndex]['Fit2'][0] = fit_cosine2_qtemp

                rabi_data_ef_Qtemps[QubitIndex]['Round Num'][0] = j
                rabi_data_ef_Qtemps[QubitIndex]['Batch Num'][0] = batch_num
                rabi_data_ef_Qtemps[QubitIndex]['Exp Config'][0] = copy.deepcopy(expt_cfg)
                rabi_data_ef_Qtemps[QubitIndex]['Syst Config'][0] = sysconfig_efrabi_Qtemps

            #---------------------Collect g-e T1 Results----------------
            if run_flags["t1"]:
                t1_data[QubitIndex]['T1'][j - batch_num*save_r - 1] = t1_est
                t1_data[QubitIndex]['Errors'][j - batch_num*save_r - 1] = t1_err
                t1_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t1_data[QubitIndex]['I'][j - batch_num*save_r - 1] = t1_I
                t1_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = t1_Q
                t1_data[QubitIndex]['Population'][j - batch_num*save_r - 1] = t1_population
                t1_data[QubitIndex]['Delay Times'][j - batch_num*save_r - 1] = t1_delay_times
                t1_data[QubitIndex]['Fit'][j - batch_num*save_r - 1] = q1_fit_exponential
                t1_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t1_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t1_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = copy.deepcopy(expt_cfg)
                t1_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t1
                t1_data[QubitIndex]['ss_Q_e'][j - batch_num * save_r - 1] = t1_ss_Q_e
                t1_data[QubitIndex]['ss_Q_g'][j - batch_num * save_r - 1] = t1_ss_Q_g
                t1_data[QubitIndex]['ss_I_e'][j - batch_num * save_r - 1] = t1_ss_I_e
                t1_data[QubitIndex]['ss_I_g'][j - batch_num * save_r - 1] = t1_ss_I_g

            # ---------------------Collect g-e T2 correction 1 Results----------------
            if run_flags["t2r_correction"]:
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
                t2r_correction_1_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = copy.deepcopy(expt_cfg)
                t2r_correction_1_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2r_1
                t2r_correction_1_data[QubitIndex]['Correction Freq'][j - batch_num * save_r - 1] = ramsey_found_q_freq

                if run_flags["rabi"]:
                    rabi_corrected_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                        time.mktime(datetime.datetime.now().timetuple()))
                    rabi_corrected_data[QubitIndex]['I'][j - batch_num * save_r - 1] = rabi_I_corrected
                    rabi_corrected_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = rabi_Q_corrected
                    rabi_corrected_data[QubitIndex]['Gains'][j - batch_num * save_r - 1] = rabi_gains_corrected
                    rabi_corrected_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = rabi_fit_corrected
                    rabi_corrected_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                    rabi_corrected_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                    rabi_corrected_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = copy.deepcopy(expt_cfg)
                    rabi_corrected_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_rabi_corrected
            #---------------------Collect g-e T2 Results----------------
            if run_flags["t2r"]:
                t2r_data[QubitIndex]['T2'][j - batch_num*save_r - 1] = t2r_est
                t2r_data[QubitIndex]['Errors'][j - batch_num*save_r - 1] = t2r_err
                t2r_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t2r_data[QubitIndex]['I'][j - batch_num*save_r - 1] = t2r_I
                t2r_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = t2r_Q
                t2r_data[QubitIndex]['Population'][j - batch_num*save_r - 1] = t2r_population
                t2r_data[QubitIndex]['Delay Times'][j - batch_num*save_r - 1] = t2r_delay_times
                t2r_data[QubitIndex]['Fit'][j - batch_num*save_r - 1] = fit_ramsey
                t2r_data[QubitIndex]['Fit Unseeded'][j - batch_num*save_r - 1] = fit_ramsey_unseeded
                t2r_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t2r_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t2r_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = copy.deepcopy(expt_cfg)
                t2r_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2r
                t2r_data[QubitIndex]['ss_Q_e'][j - batch_num * save_r - 1] = t2r_ss_Q_e
                t2r_data[QubitIndex]['ss_Q_g'][j - batch_num * save_r - 1] = t2r_ss_Q_g
                t2r_data[QubitIndex]['ss_I_e'][j - batch_num * save_r - 1] = t2r_ss_I_e
                t2r_data[QubitIndex]['ss_I_g'][j - batch_num * save_r - 1] = t2r_ss_I_g

            #---------------------Collect g-e T2E Results----------------
            if run_flags["t2e"]:
                t2e_data[QubitIndex]['T2E'][j - batch_num*save_r - 1] = t2e_est
                t2e_data[QubitIndex]['Errors'][j - batch_num*save_r - 1] = t2e_err
                t2e_data[QubitIndex]['Dates'][j - batch_num*save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                t2e_data[QubitIndex]['I'][j - batch_num*save_r - 1] = t2e_I
                t2e_data[QubitIndex]['Q'][j - batch_num*save_r - 1] = t2e_Q
                t2e_data[QubitIndex]['Population'][j - batch_num*save_r - 1] = t2e_population
                t2e_data[QubitIndex]['Delay Times'][j - batch_num*save_r - 1] = t2e_delay_times
                t2e_data[QubitIndex]['Fit'][j - batch_num*save_r - 1] = fit_t2e
                t2e_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                t2e_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                t2e_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = copy.deepcopy(expt_cfg)
                t2e_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_t2e
                t2e_data[QubitIndex]['ss_Q_e'][j - batch_num * save_r - 1] = t2e_ss_Q_e
                t2e_data[QubitIndex]['ss_Q_g'][j - batch_num * save_r - 1] = t2e_ss_Q_g
                t2e_data[QubitIndex]['ss_I_e'][j - batch_num * save_r - 1] = t2e_ss_I_e
                t2e_data[QubitIndex]['ss_I_g'][j - batch_num * save_r - 1] = t2e_ss_I_g

            #---------------------Collect e-f Single Shot Results----------------
            if run_flags["ef_ss"]:
                ef_ss_data[QubitIndex]['Fidelity'][j - batch_num * save_r - 1] = fid_ef
                ef_ss_data[QubitIndex]['Angle'][j - batch_num * save_r - 1] = theta_ef
                ef_ss_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                ef_ss_data[QubitIndex]['I_g'][j - batch_num * save_r - 1] = I_g_ss_gef
                ef_ss_data[QubitIndex]['Q_g'][j - batch_num * save_r - 1] = Q_g_ss_gef
                ef_ss_data[QubitIndex]['I_e'][j - batch_num * save_r - 1] = I_e_ss_gef
                ef_ss_data[QubitIndex]['Q_e'][j - batch_num * save_r - 1] = Q_e_ss_gef
                ef_ss_data[QubitIndex]['I_f'][j - batch_num * save_r - 1] = I_f_ss_gef
                ef_ss_data[QubitIndex]['Q_f'][j - batch_num * save_r - 1] = Q_f_ss_gef
                ef_ss_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                ef_ss_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                ef_ss_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = copy.deepcopy(expt_cfg)
                ef_ss_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_ss_ef
            # ---------------------Collect g-e-f Single Shot Results----------------
            if run_flags["ss_gef"]:
                # ss_data[QubitIndex]['Fidelity'][j - batch_num * save_r - 1] = fid
                ss_data_gef[QubitIndex]['Angle_ef'][j - batch_num * save_r - 1] = theta_ef
                ss_data_gef[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                ss_data_gef[QubitIndex]['I_g'][j - batch_num * save_r - 1] = I_g_ss_gef
                ss_data_gef[QubitIndex]['Q_g'][j - batch_num * save_r - 1] = Q_g_ss_gef
                ss_data_gef[QubitIndex]['I_e'][j - batch_num * save_r - 1] = I_e_ss_gef
                ss_data_gef[QubitIndex]['Q_e'][j - batch_num * save_r - 1] = Q_e_ss_gef
                ss_data_gef[QubitIndex]['I_f'][j - batch_num * save_r - 1] = I_f_ss_gef
                ss_data_gef[QubitIndex]['Q_f'][j - batch_num * save_r - 1] = Q_f_ss_gef
                ss_data_gef[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                ss_data_gef[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                ss_data_gef[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = copy.deepcopy(expt_cfg)
                # ss_data_gef[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = sys_config_ss_gef

            # ---------------------Collect g-e Single Shot Results----------------
            if run_flags["test_act"]:

                act_data[QubitIndex]['actI'][j - batch_num * save_r - 1] = act_idata
                act_data[QubitIndex]['actQ'][j - batch_num * save_r - 1] = act_qdata
                act_data[QubitIndex]['noactI'][j - batch_num * save_r - 1] = no_act_idata
                act_data[QubitIndex]['noactQ'][j - batch_num * save_r - 1] = no_act_qdata
                act_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = act_cfg

            # ---------------------Collect g-e Rabi Results----------------
            if run_flags["fh_rabi"]:
                rabi_data[QubitIndex]['Dates'][j - batch_num * save_r - 1] = (
                    time.mktime(datetime.datetime.now().timetuple()))
                rabi_data[QubitIndex]['I'][j - batch_num * save_r - 1] = fhrabi_I
                rabi_data[QubitIndex]['Q'][j - batch_num * save_r - 1] = fhrabi_Q
                rabi_data[QubitIndex]['Gains'][j - batch_num * save_r - 1] = fhrabi_gains
                rabi_data[QubitIndex]['Fit'][j - batch_num * save_r - 1] = fhrabi_fit
                rabi_data[QubitIndex]['Round Num'][j - batch_num * save_r - 1] = j
                rabi_data[QubitIndex]['Batch Num'][j - batch_num * save_r - 1] = batch_num
                rabi_data[QubitIndex]['Exp Config'][j - batch_num * save_r - 1] = copy.deepcopy(expt_cfg)
                rabi_data[QubitIndex]['Syst Config'][j - batch_num * save_r - 1] = fhsys_config_to_save

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

            # --------------------------save f-h Res Spec-----------------------
            if run_flags["res_spec_fh"]:
                saver_res = Data_H5(subStudyDataFolder, res_data_fh, batch_num, save_r)
                saver_res.save_to_h5('res_fh')
                del saver_res
                del res_data_fh
            # --------------------------save g-e Rabi-----------------------
            if run_flags["rabi"]:
                saver_rabi = Data_H5(subStudyDataFolder, rabi_data, batch_num, save_r)
                saver_rabi.save_to_h5('rabi_ge')
                del saver_rabi
                del rabi_data
            # --------------------------save g-e len Rabi-----------------------
            if run_flags["len_rabi"]:
                saver_rabi = Data_H5(optimizationFolder, len_rabi_data, 0, save_r)
                saver_rabi.save_to_h5('len_rabi')
                del saver_rabi
                del len_rabi_data
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

                if run_flags["rabi"]:
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
    ef_rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    qspec_data = create_data_dict(qspec_keys, save_r, list_of_all_qubits)
    rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
    len_rabi_data = create_data_dict(rabi_keys, save_r, list_of_all_qubits)
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
    res_data_fh = create_data_dict(res_keys_ef, save_r, list_of_all_qubits)

en=time.time()
print('timetaken=',en-st)
return_yokos_to_zero()
atexit.unregister(return_yokos_to_zero)
