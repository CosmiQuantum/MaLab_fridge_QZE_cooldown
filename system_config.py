from qick import *
import sys
import os
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from socProxy import makeProxy
import os
import datetime
import numpy as np


# Historical PUCQ4 fit candidates retained only for provenance.  Runtime code
# uses the current table below; inspect status for fresh/historical values. These
# predate the last Ramsey correction and sigma/coherence pass.
_PUCQ4_FLUX_CALIBRATION_HISTORY = {
    "Yoko3_Q4_M5": {
        -10.0: {
            "qfreq_MHz": 6037.585407899529,
            "qspec_fwhm_MHz": 0.20377584742238586,
            "qspec_gain": 0.001649267441664,
            "qspec_length_us": 40.0,
            "uniform_qspec_gain": 0.03,
            "sigma_us": 0.0165397052076522,
            "pi_amp": 0.9949337728132821,
            "res_base_MHz": 9010.49,
            "readout_frequency_MHz": 9011.44,
            "readout_length_us": 5.0,
            "readout_gain": 0.95,
            "readout_offset_MHz": 0.95,
            "SSF": 0.124,
            "T1_us": 17.59795456742065,
            "T2R_us": 1.6056569353905643,
            "T2E_us": 3.6289562499916426,
            "status": "complete_rabi_recalibration_deferred",
        },
        -7.5: {
            "qfreq_MHz": 6422.620534808272,
            "qspec_fwhm_MHz": 0.17031186889627797,
            "qspec_gain": 0.000839,
            "qspec_length_us": 20.0,
            "sigma_us": 0.0617065954067694,
            "pi_amp": 0.6287310016480498,
            "res_base_MHz": 9011.54,
            "readout_frequency_MHz": 9012.22,
            "readout_length_us": 7.0,
            "readout_gain": 0.95,
            "readout_offset_MHz": 0.68,
            "SSF": 0.6825,
            "T1_us": 16.892026340586487,
            "T2R_us": 1.6712887423637215,
            "T2E_us": 5.974627251966843,
            "status": "complete",
        },
        -5.0: {
            "qfreq_MHz": 6724.886451375884,
            "qspec_fwhm_MHz": 0.16478367561014154,
            "qspec_gain": 0.002,
            "qspec_length_us": 20.0,
            "sigma_us": 0.07923262394176339,
            "pi_amp": 0.6886101446621498,
            "res_base_MHz": 9012.79,
            "readout_frequency_MHz": 9012.87,
            "readout_length_us": 6.0,
            "readout_gain": 0.95,
            "readout_offset_MHz": 0.08,
            "SSF": 0.725,
            "T1_us": 15.792375911866403,
            "T2R_us": 2.3839576581898085,
            "T2E_us": 7.325418728160058,
            "status": "complete",
        },
        -2.5: {
            "qfreq_MHz": 6945.111991749794,
            "qspec_fwhm_MHz": 0.12021327827535877,
            "qspec_gain": 0.0025,
            "qspec_length_us": 20.0,
            "sigma_us": 0.05911146169340639,
            "pi_amp": 0.6486907159860831,
            "res_base_MHz": 9013.74,
            "readout_frequency_MHz": 9014.06,
            "readout_length_us": 6.0,
            "readout_gain": 0.73,
            "readout_offset_MHz": 0.32,
            "SSF": 0.6183,
            "T1_us": 11.948751384988029,
            "T2R_us": 3.6265382578259957,
            "T2E_us": 9.068963556255856,
            "status": "complete",
        },
        0.0: {
            "qfreq_MHz": 7084.565677917129,
            "qspec_fwhm_MHz": 0.10236154214941956,
            "qspec_gain": 0.003,
            "qspec_length_us": 20.0,
            "sigma_us": 0.05911146169340639,
            "pi_amp": 0.6287310016480498,
            "res_base_MHz": 9014.74,
            "readout_frequency_MHz": 9014.94,
            "readout_length_us": 6.0,
            "readout_gain": 0.65,
            "readout_offset_MHz": 0.20,
            "SSF": 0.581,
            "T1_us": 14.605931751988077,
            "T2R_us": 6.418563288818102,
            "T2E_us": 15.284901770309753,
            "status": "complete",
        },
        2.5: {
            "qfreq_MHz": 7143.8595708799985,
            "qspec_fwhm_MHz": 0.11536552555969569,
            "qspec_gain": 0.0045,
            "qspec_length_us": 20.0,
            "sigma_us": 0.0549176813895992,
            "pi_amp": 0.6287310016480498,
            "res_base_MHz": 9015.04,
            "readout_frequency_MHz": 9014.56,
            "readout_length_us": 5.0,
            "readout_gain": 0.93,
            "readout_offset_MHz": -0.48,
            "SSF": 0.615,
            "T1_us": 11.542920594344714,
            "T2R_us": 10.522589617668842,
            "T2E_us": 17.247786087599696,
            "status": "complete",
        },
        5.0: {
            "qfreq_MHz": 7121.580621960356,
            "qspec_fwhm_MHz": 0.8169228035068649,
            "qspec_gain": 0.03,
            "qspec_length_us": 40.0,
            "sigma_us": 0.08084894313680056,
            "pi_amp": 0.6786302874931331,
            "res_base_MHz": 9014.59,
            "readout_frequency_MHz": 9014.11,
            "readout_length_us": 6.0,
            "readout_gain": 0.92,
            "readout_offset_MHz": -0.48,
            "SSF": 0.512,
            "T1_us": 2.31231204747479,
            "T2R_us": 3.9874039658648956,
            "T2E_us": 1.9007638446966737,
            "status": "complete_population_optimal_power_broadened_qspec",
        },
        7.5: {
            "qfreq_MHz": 7022.121082210489,
            "qspec_fwhm_MHz": 0.11431904425743382,
            "qspec_gain": 0.002,
            "qspec_length_us": 20.0,
            "sigma_us": 0.05873041582061609,
            "pi_amp": 0.6087712873100165,
            "res_base_MHz": 9014.04,
            "readout_frequency_MHz": 9013.80,
            "readout_length_us": 10.0,
            "readout_gain": 0.95,
            "readout_offset_MHz": -0.24,
            "SSF": 0.773,
            "T1_us": 10.698919700363385,
            "T2R_us": 3.6309914943113784,
            "T2E_us": 10.80659010578854,
            "status": "complete",
        },
        10.0: {
            "qfreq_MHz": 6841.20173951502,
            "qspec_fwhm_MHz": 0.1404901326882709,
            "qspec_gain": 0.002,
            "qspec_length_us": 20.0,
            "sigma_us": 0.059226039930878716,
            "pi_amp": 0.6187511444790331,
            "res_base_MHz": 9013.09,
            "readout_frequency_MHz": 9012.97,
            "readout_length_us": 6.0,
            "readout_gain": 0.95,
            "readout_offset_MHz": -0.12,
            "SSF": 0.6773,
            "T1_us": 14.279709140001536,
            "T2R_us": 2.7180866745871834,
            "T2E_us": 8.314394050782358,
            "status": "complete",
        },
    },
    "Yoko4_Q6_M6": {
        -2.5: {"qfreq_MHz": 6103.7968, "sigma_us": 0.0605405, "pi_amp": 0.60877,
               "res_base_MHz": 9057.78, "readout_length_us": 12.0,
               "readout_gain": 0.7967, "readout_offset_MHz": 0.08, "SSF": 0.4270,
               "T1_us": 5.03, "T2R_us": 1.08, "T2E_us": 4.46},
        0.0: {"qfreq_MHz": 6522.5684, "sigma_us": 0.06200, "pi_amp": 0.65867,
              "res_base_MHz": 9059.26, "readout_length_us": 10.0,
              "readout_gain": 0.95, "readout_offset_MHz": -0.40, "SSF": 0.5030},
        2.5: {"qfreq_MHz": 6842.1315, "sigma_us": 0.05800, "pi_amp": 0.60877,
              "res_base_MHz": 9060.46, "readout_length_us": 12.0,
              "readout_gain": 0.7967, "readout_offset_MHz": -0.52, "SSF": 0.5753},
        5.0: {"qfreq_MHz": 7062.7712, "sigma_us": 0.0556693, "pi_amp": 0.61875,
              "res_base_MHz": 9060.98, "readout_length_us": 12.0,
              "readout_gain": 0.9167, "readout_offset_MHz": -0.76, "SSF": 0.4910,
              "T1_us": 5.48, "T2R_us": 2.77, "T2E_us": 8.00},
        7.5: {"qfreq_MHz": 7185.6595, "sigma_us": 0.058001, "pi_amp": 0.63871,
              "res_base_MHz": 9061.53, "readout_length_us": 11.0,
              "readout_gain": 0.8334, "readout_offset_MHz": -0.04, "SSF": 0.4740,
              "T1_us": 5.48, "T2R_us": 4.13, "T2E_us": 9.35},
        10.0: {"qfreq_MHz": 7211.4436, "sigma_us": 0.0511568, "pi_amp": 0.62873,
               "res_base_MHz": 9061.68, "readout_length_us": 12.0,
               "readout_gain": 0.8722, "readout_offset_MHz": -0.16, "SSF": 0.4710,
               "T1_us": 5.35, "T2R_us": 6.35, "T2E_us": 9.56},
    },
}

# Historical nonzero-current calibration from the 2026-09-07/08 campaign; zero-current
# M5/M6 zero-current rows refreshed 2026-10-29; inspect current quality notes. Accepted
# round-robin refreshes identified by calibration_run. Unrefreshed entries
# retain the prior measured qspec, readout and Ramsey/sigma/coherence settings.
# Every listed pi amplitude lies in the 0.625-0.675 calibration window.
# Gaussian drive length is always 4 * sigma_us in the experiment programs.
PUCQ4_FLUX_PULSE_CALIBRATION = {
    "Yoko3_Q4_M5": {
        -10.0: {"qfreq_MHz": 6038.019853160, "qspec_gain": 0.03,
                "qspec_length_us": 40.0, "sigma_us": 0.096997820,
                "pi_amp": 0.651681621, "res_base_MHz": 9010.49,
                "readout_frequency_MHz": 9011.44, "readout_length_us": 5.0,
                "readout_gain": 0.95, "readout_offset_MHz": 0.95,
                "T1_us": 23.727664, "T2R_us": 1.058108, "T2E_us": 4.659599,
                "status": "validated"},
        -7.5: {"qfreq_MHz": 6422.718837340, "qspec_gain": 0.000839,
               "qspec_length_us": 20.0, "sigma_us": 0.059236826,
               "pi_amp": 0.651681621, "res_base_MHz": 9011.54,
               "readout_frequency_MHz": 9012.22, "readout_length_us": 7.0,
               "readout_gain": 0.95, "readout_offset_MHz": 0.68,
               "T1_us": 16.639058, "T2R_us": 1.491644, "T2E_us": 5.920499,
               "status": "validated"},
        -5.0: {"qfreq_MHz": 6724.844813670, "qspec_gain": 0.002,
               "qspec_length_us": 20.0, "sigma_us": 0.084033637,
               "pi_amp": 0.651681621, "res_base_MHz": 9012.79,
               "readout_frequency_MHz": 9012.87, "readout_length_us": 6.0,
               "readout_gain": 0.95, "readout_offset_MHz": 0.08,
               "T1_us": 14.807042, "T2R_us": 2.110755, "T2E_us": 7.355631,
               "status": "validated"},
        -2.5: {"qfreq_MHz": 6945.1129971731425, "qspec_gain": 0.0025,
               "qspec_length_us": 20.0, "sigma_us": 0.058777626,
               "pi_amp": 0.6467069523286333, "res_base_MHz": 9013.7,
               "readout_frequency_MHz": 9014.06, "readout_length_us": 6.0,
               "readout_gain": 0.73, "readout_offset_MHz": 0.3599999999987631,
               "T1_us": 12.843022492027956, "T2R_us": 3.3305882611408713, "T2E_us": 9.688102646269144,
               "qspec_FWHM_MHz": 0.11948517573545973,
               "qspec_FWHM_error_MHz": 0.0017745327919961704,
               "qspec_reps": 5000,
               "qspec_rounds": 1,
               "qspec_steps": 401,
               "SSF": 0.5926,
               "calibration_run": 'rrv5_m025b/2026-09-09_16-25-00',
               # Earlier visits moved by more than a linewidth; acquire qspec first.
               "requires_qspec_each_pass": True,
               "status": "validated"},
        0.0: {'qfreq_MHz': 7065.147636121578,
              'sigma_us': 0.056940825,
              'pi_amp': 0.6268082768723677,
              'qspec_gain': 0.006,
              'qspec_length_us': 100.0,
              'res_base_MHz': 9013.4,
              'readout_frequency_MHz': 9013.46,
              'readout_gain': 0.95,
              'readout_length_us': 6.0,
              'readout_offset_MHz': 0.05999999999949068,
              'T1_us': 13.55516324198618,
              'T2R_us': 5.109309255027539,
              'T2E_us': 13.40683881982994,
              'qspec_FWHM_MHz': 0.15399179334227603,
              'qspec_FWHM_error_MHz': 0.002411508607846688,
              'qspec_peak_population': 0.459379454537664,
              'qspec_R2': 0.9913337999163058,
              'qspec_reps': 2000,
              'qspec_rounds': 1,
              'qspec_steps': 151,
              'SSF': 0.66616,
              'calibration_run': 'M:/_Data/20250822 - '
                                 'Olivia/pucq4_run_started_Oct_7/PUCQ4/find_resonators_sandbox/full5_a_1029/2026-10-29_19-54-47',
              'calibration_date': '2026-10-29',
              'status': 'validated_effective_fits',
              'requires_qspec_each_pass': True,
              'fresh_qubit_validated': True,
              'fresh_resonator_validated': True,
              'fresh_cooldown_validated': True,
              'nominal_targets_met': True,
              'quality_notes': 'Measured FWHM 0.1540 MHz; peak population 0.459; R2 0.991. Nominal qspec '
                               'targets met. Fresh T1 is well described by a single exponential. Ramsey has '
                               'structured residuals: a flexible two-frequency comparison improves R2 from '
                               '.9878 to .9971; retain normal RR T2R5.109us as an effective single-frequency '
                               'fit. No microscopic mechanism inferred.'},
        2.5: {"qfreq_MHz": 7143.883429322838, "qspec_gain": 0.0045,
              "qspec_length_us": 20.0, "sigma_us": 0.053267224,
              "pi_amp": 0.6467069523286333, "res_base_MHz": 9014.68,
              "readout_frequency_MHz": 9014.82, "readout_length_us": 4.0,
              "readout_gain": 0.95, "readout_offset_MHz": 0.14,
              "T1_us": 10.196045128561797, "T2R_us": 7.605553477536421, "T2E_us": 15.566037333880487,
              "qspec_FWHM_MHz": 0.11932360664032048,
              "qspec_FWHM_error_MHz": 0.001401703175142955,
              "qspec_reps": 5000,
              "qspec_rounds": 1,
              "qspec_steps": 401,
              "SSF": 0.5841,
              "calibration_run": 'rrv5_p025b/2026-09-09_01-21-43',
              "status": 'validated'},
        5.0: {"qfreq_MHz": 7122.257671006202, "qspec_gain": 0.03,
              "qspec_length_us": 40.0, "sigma_us": 0.058777626,
              "pi_amp": 0.6516816211926998, "res_base_MHz": 9014.59,
              "readout_frequency_MHz": 9014.11, "readout_length_us": 6.0,
              "readout_gain": 0.92, "readout_offset_MHz": -0.47999999999956344,
              "T1_us": 5.1744804110233265, "T2R_us": 1.1310400931072198, "T2E_us": 8.360387540413047,
              "qspec_FWHM_MHz": 0.6243436691610823,
              "qspec_FWHM_error_MHz": 0.005091870061888435,
              "qspec_reps": 5000,
              "qspec_rounds": 1,
              "qspec_steps": 401,
              "SSF": 0.5782,
              "calibration_run": 'rrv5_p050a/2026-09-09_03-22-35',
              "status": "validated"},
        7.5: {"qfreq_MHz": 7022.162967054066, "qspec_gain": 0.002,
              "qspec_length_us": 20.0, "sigma_us": 0.055104024,
              "pi_amp": 0.6516816211926998, "res_base_MHz": 9013.96,
              "readout_frequency_MHz": 9013.800000000001, "readout_length_us": 10.0,
              "readout_gain": 0.95, "readout_offset_MHz": -0.1599999999980355,
              "T1_us": 12.757370732627168, "T2R_us": 4.038153856007287, "T2E_us": 10.907325682518627,
              "qspec_FWHM_MHz": 0.1223185584832736,
              "qspec_FWHM_error_MHz": 0.0018248357161659372,
              "qspec_reps": 5000,
              "qspec_rounds": 1,
              "qspec_steps": 401,
              "SSF": 0.7133,
              "calibration_run": 'rrv5_p075b/2026-09-09_09-01-35',
              "status": "validated"},
        10.0: {"qfreq_MHz": 6841.484578285427, "qspec_gain": 0.002,
               "qspec_length_us": 20.0, "sigma_us": 0.056481625,
               "pi_amp": 0.6516816211926998, "res_base_MHz": 9013.11,
               "readout_frequency_MHz": 9012.97, "readout_length_us": 6.0,
               "readout_gain": 0.95, "readout_offset_MHz": -0.1400000000012369,
               "T1_us": 15.426027649345263, "T2R_us": 2.4383651838558325, "T2E_us": 8.333998671208258,
               "qspec_FWHM_MHz": 0.15045271857763148,
               "qspec_FWHM_error_MHz": 0.0024709538824901063,
               "qspec_reps": 5000,
               "qspec_rounds": 1,
               "qspec_steps": 401,
               "SSF": 0.6431,
               "calibration_run": 'rrv5_p100b/2026-09-09_11-02-54',
               "requires_qspec_each_pass": True,
               "qspec_frequency_min_MHz": 6841.457614725295,
               "qspec_frequency_max_MHz": 6841.583959579264,
               "status": 'validated_tracking'},
    },
    "Yoko4_Q6_M6": {
        -10.0: {"qfreq_MHz": 4375.850089710, "qspec_gain": 0.03,
                "qspec_length_us": 20.0, "sigma_us": 0.105432413,
                "pi_amp": 0.646706952, "res_base_MHz": 9055.31,
                "readout_frequency_MHz": 9054.91, "readout_length_us": 12.0,
                "readout_gain": 0.88, "readout_offset_MHz": -0.40,
                "T1_us": 18.946445, "T2R_us": 0.665104, "T2E_us": 4.035453,
                "status": "validated"},
        -7.5: {"qfreq_MHz": 4839.206595110, "qspec_gain": 0.03,
               "qspec_length_us": 20.0, "sigma_us": 0.132844841,
               "pi_amp": 0.646706952, "res_base_MHz": 9056.01,
               "readout_frequency_MHz": 9055.25, "readout_length_us": 12.0,
               "readout_gain": 0.7945, "readout_offset_MHz": -0.76,
               "T1_us": 10.897905, "T2R_us": 0.885355, "T2E_us": 3.780041,
               "status": "validated"},
        -5.0: {"qfreq_MHz": 5588.103711440, "qspec_gain": 0.03,
               "qspec_length_us": 20.0, "sigma_us": 0.070716831,
               "pi_amp": 0.656656290, "res_base_MHz": 9056.91,
               "readout_frequency_MHz": 9056.63, "readout_length_us": 12.0,
               "readout_gain": 0.8767, "readout_offset_MHz": -0.28,
               "T1_us": 6.463409, "T2R_us": 0.722576, "T2E_us": 3.461524,
               "status": "validated"},
        -2.5: {"qfreq_MHz": 6103.73104802091, "qspec_gain": 0.002,
               "qspec_length_us": 20.0, "sigma_us": 0.056022425,
               "pi_amp": 0.6516816211926998, "res_base_MHz": 9057.9,
               "readout_frequency_MHz": 9057.86, "readout_length_us": 10.0,
               "readout_gain": 0.8, "readout_offset_MHz": -0.039999999999054126,
               "T1_us": 6.143175070153691, "T2R_us": 1.0157470309560712, "T2E_us": 4.2917276961703585,
               "qspec_FWHM_MHz": 0.3281951157831831,
               "qspec_FWHM_error_MHz": 0.013409378770879705,
               "qspec_reps": 40000,
               "qspec_rounds": 1,
               "qspec_steps": 101,
               "SSF": 0.32474,
               "calibration_run": 'rrv6_m025e/2026-09-09_18-28-03',
               "status": "validated"},
        # Fresh zero-current full RR after warmup, 2026-10-29; nonzero rows remain historical.
        0.0: {'qfreq_MHz': 6404.639622421642,
              'sigma_us': 0.0555868301534722,
              'pi_amp': 0.6516816211926998,
              'qspec_gain': 0.006,
              'qspec_length_us': 100.0,
              'res_base_MHz': 9057.715,
              'readout_frequency_MHz': 9057.695,
              'readout_gain': 0.95,
              'readout_length_us': 45.0,
              'readout_offset_MHz': -0.020000000000436557,
              'T1_us': 4.010622456099054,
              'T2R_us': 1.6580185850858566,
              'T2E_us': 4.397388046801702,
              'qspec_FWHM_MHz': 0.26458772655685375,
              'qspec_FWHM_error_MHz': 0.004268932872899609,
              'qspec_peak_population': 0.32013473539687687,
              'qspec_R2': 0.9913278257439754,
              'qspec_reps': 6000,
              'qspec_rounds': 1,
              'qspec_steps': 151,
              'SSF': 0.78504,
              'calibration_run': 'M:/_Data/20250822 - '
                                 'Olivia/pucq4_run_started_Oct_7/PUCQ4/find_resonators_sandbox/full6_a_1029/2026-10-29_18-39-02',
              'calibration_date': '2026-10-29',
              'status': 'measured_with_target_warning',
              'requires_qspec_each_pass': True,
              'fresh_qubit_validated': True,
              'fresh_resonator_validated': True,
              'fresh_cooldown_validated': True,
              'nominal_targets_met': False,
              'quality_notes': 'Clean repeatable line R2.991; FWHM.2646MHz exceeds preferred.1-.2MHz. '
                               'Population.320 meets target at.006/100us; lower.0045 gave width.2359/pop.222. '
                               'No nominal-linewidth pass claimed.  Earlier coarse1us-step T1control3.11us; '
                               'latest fine-sampled normal fullRR4.01us selected. Conditional fit errors do not '
                               'include between-run/protocol variation.'},
        2.5: {"qfreq_MHz": 6841.775497163998, "qspec_gain": 0.006,
              "qspec_length_us": 20.0, "sigma_us": 0.053726424,
              "pi_amp": 0.6516816211926998, "res_base_MHz": 9060.08,
              "readout_frequency_MHz": 9059.98, "readout_length_us": 22.0,
              "readout_gain": 0.80, "readout_offset_MHz": -0.1000000000003638,
              "T1_us": 6.79621105927428, "T2R_us": 2.320281858189406, "T2E_us": 6.533448775271403,
              "qspec_FWHM_MHz": 0.2052599851296157,
              "qspec_FWHM_error_MHz": 0.0020320259945249414,
              "qspec_reps": 5000,
              "qspec_rounds": 1,
              "qspec_steps": 401,
              "SSF": 0.6575,
              "calibration_run": 'rrv6_p025b/2026-09-09_02-56-53',
              "status": 'validated'},
        5.0: {"qfreq_MHz": 7062.612211354989, "qspec_gain": 0.006,
              "qspec_length_us": 20.0, "sigma_us": 0.053267224,
              "pi_amp": 0.6516816211926998, "res_base_MHz": 9060.78,
              "readout_frequency_MHz": 9060.22, "readout_length_us": 12.0,
              "readout_gain": 0.9167, "readout_offset_MHz": -0.5600000000013097,
              "T1_us": 7.120905647510946, "T2R_us": 2.4521516277292186, "T2E_us": 7.949307821673911,
              "qspec_FWHM_MHz": 0.17117461028187456,
              "qspec_FWHM_error_MHz": 0.002246002837143701,
              "qspec_reps": 5000,
              "qspec_rounds": 1,
              "qspec_steps": 401,
              "SSF": 0.4262,
              "calibration_run": 'rrv6_p050a/2026-09-09_03-40-19',
              "status": "validated"},
        7.5: {"qfreq_MHz": 7185.513939668361, "qspec_gain": 0.006,
              "qspec_length_us": 20.0, "sigma_us": 0.057859226,
              "pi_amp": 0.6467069523286333, "res_base_MHz": 9061.47,
              "readout_frequency_MHz": 9061.49, "readout_length_us": 11.0,
              "readout_gain": 0.8334, "readout_offset_MHz": 0.020000000000436557,
              "T1_us": 5.45179527299386, "T2R_us": 5.7231990564627235, "T2E_us": 9.238789592754308,
              "frequency_seed_run": "rrv6p75_qr/2026-09-09_09-46-23",
              "qspec_FWHM_MHz": 0.13651071888404784,
              "qspec_FWHM_error_MHz": 0.0029425361220824365,
              "qspec_reps": 5000,
              "qspec_rounds": 1,
              "qspec_steps": 401,
              "SSF": 0.433,
              "calibration_run": 'rrv6_p075b/2026-09-09_09-54-43',
              "status": 'validated'},
        10.0: {"qfreq_MHz": 7211.413079440341, "qspec_gain": 0.006,
               "qspec_length_us": 20.0, "sigma_us": 0.049593622,
               "pi_amp": 0.6417322834645669, "res_base_MHz": 9061.54,
               "readout_frequency_MHz": 9061.52, "readout_length_us": 12.0,
               "readout_gain": 0.8722, "readout_offset_MHz": -0.020000000000436557,
               "T1_us": 6.67982963731644, "T2R_us": 6.114422314936588, "T2E_us": 9.700775749042306,
               "qspec_FWHM_MHz": 0.17805935715636778,
               "qspec_FWHM_error_MHz": 0.0028538062848888603,
               "qspec_reps": 5000,
               "qspec_rounds": 1,
               "qspec_steps": 401,
               "SSF": 0.4358,
               "calibration_run": 'rrv6_p100e/2026-09-09_13-59-51',
               "status": "validated"},
    },
}

# Public runtime calibration.  The older candidates remain available under the
# private ``_PUCQ4_FLUX_CALIBRATION_HISTORY`` name for forensic comparison only.
PUCQ4_FLUX_CALIBRATION = PUCQ4_FLUX_PULSE_CALIBRATION
PUCQ4_FLUX_CALIBRATION_SOURCES = (
    "PUCQ4_round_robin_validation_20260908_artifacts/validation_manifest.json",
    "yoko_flux_campaign/ramsey_sigma_coherence_manifest.json",
    "yoko_flux_campaign/single_sweep_qspec_manifest.json",
    "line_6d/2026-09-08_18-51-46/documentation/zero_audit.json",
)
PUCQ4_FLUX_CURRENT_GRID_MA = (-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0)
PUCQ4_FLUX_DEVICE_BY_QUBIT_INDEX = {
    4: "Yoko3_Q4_M5",
    5: "Yoko4_Q6_M6",
}

_PUCQ4_REQUIRED_CALIBRATION_KEYS = (
    "qfreq_MHz", "qspec_gain", "qspec_length_us", "sigma_us", "pi_amp",
    "res_base_MHz", "readout_frequency_MHz", "readout_length_us",
    "readout_gain", "readout_offset_MHz",
)

# Rows 0-3 remain fixed-frequency.  Rows 4-5 are deliberately populated only
# by ``resolve_pucq4_flux_config`` so there is one authoritative copy of every
# current-dependent number.
# Fresh zero-current full RR, 2026-10-29.
# M1/M2/M4 readout values are drift-limited working pulses; see the run manifest
# for independent SSF examples, observed variation, and fit qualifications.
# M2 carrier is an effective single-fit center of a repeatable spectral doublet.
# M1 T1 is mildly model dependent; M2 lifetimes are time-specific effective fits.
# Fresh M4 lifetime checks pass; earlier nonexponential qualifiers are historical.
# Q2/M2/index1: measured low-power RR qspec/Rabi/readout update, 2026-11-04.
# Remaining all-six coherence/report work is deferred during 24h SSF.
# Q2 Nov20 measured working readout; modest contrast, qualified gain ceiling .24.
_PUCQ4_FIXED_CONFIG_ROWS = {'res_freq_ge': (8918.795, 8950.964390000001, 8974.11, 8999.07, None, None),
 'res_gain_ge': (0.9, 0.24, 0.95, 0.95, None, None),
 'res_length_ge': (6.0, 7.0, 13.0, 5.0, None, None),
 'res_freq_offset_ge': (-0.42000000000007276, -0.6393129999996745, -0.7800000000006548, 0.18000000000029104, None, None),
 'qubit_freq_ge': (5515.343391325498, 5438.856540689225, 5551.361090349251, 5594.644240774248, None, None),
 'qubit_gain_ge': (0.0045, 0.009, 0.009, 0.01, None, None),
 'qubit_length_ge': (100.0, 100.0, 100.0, 20.0, None, None),
 'sigma': (0.05, 0.101, 0.124, 0.0695, None, None),
 'pi_amp': (0.6865043032411646, 0.6765549655130318, 0.6566562900567662, 0.6467069523286333, None, None)}

# All six zero-current pulse rows are measured after the warmup; inspect fit qualifications.
PUCQ4_ZERO_CURRENT_PULSE_MEASURED_ROWS = (True, True, True, True, True, True)

# Filled only after the all-six low-power campaign has been reviewed. These
# limits apply at the recorded zero-current attenuation configuration only.
# Filled 2026-11-25 from the 24 Nov low-power campaign (30 kHz resonance-shift
# tolerance). round_robin_benchmark.py refuses a zero-current run above these
# gains. It does not check the readout length, and no other script checks them.
# Each value is the lowest ceiling known for the row at its stored length and at
# its 24 Nov working length. M1 0.25 at 8 us (none measured at the stored 6 us).
# M2 0.24 at 7 us. M3 0.20 at 13 us. M4 0.20 at the stored 5 us (0.40 is qualified
# at 4 and 8 us). M5 0.10 at 8 us (the stored 6 us was rejected). M6 0.10 at 45 us.
# Set PUCQ4_READOUT_GAIN and PUCQ4_READOUT_LENGTH_US for the row.
# Details in harness_engineering/STATE.md.
# Previous value (None, 0.24, None, None, None, None).
PUCQ4_ZERO_CURRENT_READOUT_GAIN_MAX = (0.25, 0.24, 0.20, 0.20, 0.10, 0.10)


def normalize_pucq4_flux_current(current_mA):
    """Return the exact supported current-grid value or fail before hardware use."""
    try:
        requested = float(current_mA)
    except (TypeError, ValueError):
        raise ValueError("PUCQ4 flux current must be numeric in mA")
    if not np.isfinite(requested):
        raise ValueError("PUCQ4 flux current must be finite in mA")
    for supported in PUCQ4_FLUX_CURRENT_GRID_MA:
        if np.isclose(requested, supported, rtol=0.0, atol=1e-9):
            return supported
    raise ValueError(
        "Unsupported PUCQ4 flux current %.12g mA; choose one of %s"
        % (requested, list(PUCQ4_FLUX_CURRENT_GRID_MA))
    )


def get_pucq4_flux_calibration(yoko3_current_mA=0.0, yoko4_current_mA=0.0):
    """Return current-table rows (indices 4/5); inspect status for fresh validation."""
    requested_currents = {
        4: normalize_pucq4_flux_current(yoko3_current_mA),
        5: normalize_pucq4_flux_current(yoko4_current_mA),
    }
    selected = {}
    for qubit_index, current_mA in requested_currents.items():
        device_key = PUCQ4_FLUX_DEVICE_BY_QUBIT_INDEX[qubit_index]
        row = dict(PUCQ4_FLUX_CALIBRATION[device_key][current_mA])
        missing = [key for key in _PUCQ4_REQUIRED_CALIBRATION_KEYS if key not in row]
        if missing:
            raise RuntimeError(
                "%s at %s mA is missing calibration fields: %s"
                % (device_key, current_mA, ", ".join(missing))
            )
        expected_readout = row["res_base_MHz"] + row["readout_offset_MHz"]
        if not np.isclose(expected_readout, row["readout_frequency_MHz"],
                          rtol=0.0, atol=1e-6):
            raise RuntimeError(
                "%s at %s mA has inconsistent base/offset/readout frequencies"
                % (device_key, current_mA)
            )
        if not 0.625 <= row["pi_amp"] <= 0.675:
            raise RuntimeError(
                "%s at %s mA has pi_amp outside the validated 0.625-0.675 window"
                % (device_key, current_mA)
            )
        row.update({
            "current_mA": current_mA,
            "device_key": device_key,
            "qubit_index": qubit_index,
            "gaussian_length_us": 4.0 * row["sigma_us"],
        })
        selected[qubit_index] = row
    return selected


def resolve_pucq4_flux_config(yoko3_current_mA=0.0, yoko4_current_mA=0.0):
    """Build ordinary six-qubit config lists for the selected Yoko currents."""
    resolved = {
        key: list(values) for key, values in _PUCQ4_FIXED_CONFIG_ROWS.items()
    }
    calibration_by_qubit = get_pucq4_flux_calibration(
        yoko3_current_mA=yoko3_current_mA,
        yoko4_current_mA=yoko4_current_mA,
    )
    field_map = {
        "res_freq_ge": "res_base_MHz",
        "res_gain_ge": "readout_gain",
        "res_length_ge": "readout_length_us",
        "res_freq_offset_ge": "readout_offset_MHz",
        "qubit_freq_ge": "qfreq_MHz",
        "qubit_gain_ge": "qspec_gain",
        "qubit_length_ge": "qspec_length_us",
        "sigma": "sigma_us",
        "pi_amp": "pi_amp",
    }
    for qubit_index, calibration in calibration_by_qubit.items():
        for config_key, calibration_key in field_map.items():
            resolved[config_key][qubit_index] = float(calibration[calibration_key])
    for config_key, values in resolved.items():
        if len(values) != 6 or any(value is None for value in values):
            raise RuntimeError("Invalid six-qubit PUCQ4 config for %s" % config_key)
    resolved["readout_frequency_ge"] = [
        base + offset for base, offset in zip(
            resolved["res_freq_ge"], resolved["res_freq_offset_ge"]
        )
    ]
    resolved["gaussian_length_ge"] = [4.0 * sigma for sigma in resolved["sigma"]]
    return resolved

# ADC_attenuator MUST be above 16dB
#DAC_attenuator1 and 2 are for the resonators
#qubit_DAC_attenuator1 and 2 are for the qubits
class QICK_experiment:
    def __init__(self, folder, DAC_attenuator1 = 10, DAC_attenuator2 = 15, qubit_DAC_attenuator1 = 5,
                                     qubit_DAC_attenuator2 = 4, ADC_attenuator = 17, fridge = None,
                                     yoko3_current_mA = 0.0, yoko4_current_mA = 0.0):

        # Resolve and validate the selected bias rows before creating folders or
        # connecting to the QICK.  The config dictionaries below remain normal
        # six-element per-qubit lists; no nested current tables reach QICK code.
        self.yoko3_current_mA = normalize_pucq4_flux_current(yoko3_current_mA)
        self.yoko4_current_mA = normalize_pucq4_flux_current(yoko4_current_mA)
        self.pucq4_flux_calibration = get_pucq4_flux_calibration(
            yoko3_current_mA=self.yoko3_current_mA,
            yoko4_current_mA=self.yoko4_current_mA,
        )
        self.pucq4_flux_config = resolve_pucq4_flux_config(
            yoko3_current_mA=self.yoko3_current_mA,
            yoko4_current_mA=self.yoko4_current_mA,
        )

        # Where do you want to save data
        self.outerFolder = folder
        self.create_folder_if_not_exists(self.outerFolder)

        # attenuation settings
        self.DAC_attenuator1 = DAC_attenuator1
        self.DAC_attenuator2 = DAC_attenuator2
        self.ADC_attenuator = ADC_attenuator
        self.qubit_DAC_attenuator1 = qubit_DAC_attenuator1
        self.qubit_DAC_attenuator2 = qubit_DAC_attenuator2

        # Make proxy to the QICK
        self.soc, self.soccfg = makeProxy()
        print(self.soccfg)

        self.FSGEN_CH = 1
        self.FSGEN_AMPL_CH = 2 # not used on 4x2
        self.MIXMUXGEN_CH  = 0 # Readout resonator DAC channel
        self.MUXRO_CH = 0

        # ### NEW for the RF board
        # self.qubit_center_freq = 4225 #4400  # To be in the middle of the qubit freqs.
        # self.res_center_freq   = 6330  # To be in the middle of the res freqs. 3000-5000 see nothing,6000 and 7000 see something, 8000+ see nothing
        # self.soc.rfb_set_gen_filter(self.MIXMUXGEN_CH, fc=self.res_center_freq / 1000, ftype='bandpass', bw=1.0)
        # self.soc.rfb_set_gen_filter(self.FSGEN_CH, fc=self.qubit_center_freq / 1000, ftype='bandpass', bw=1.6) # change to 2 in futrue tests
        # self.soc.rfb_set_ro_filter(self.MUXRO_CH[0], fc=self.res_center_freq / 1000, ftype='bandpass', bw=1.0) #readout ADC
        # # Set attenuator on DAC.
        # self.soc.rfb_set_gen_rf(self.MIXMUXGEN_CH, self.DAC_attenuator1, self.DAC_attenuator2)  # Verified 30->25 see increased gain in loopback
        # self.soc.rfb_set_gen_rf(self.FSGEN_CH, self.qubit_DAC_attenuator1, self.qubit_DAC_attenuator2)  # Verified 30->25 see increased gain in loopback
        # # Set attenuator on ADC.
        # ### IMPORTANT: set this to 30 and you get 60 dB of warm gain. Set to 0 and you get 90 dB of warm gain
        # self.soc.rfb_set_ro_rf(self.MUXRO_CH[0], self.ADC_attenuator)  # Verified 30->25 see increased gain in loopback


        # Qubit you want to work with
        self.QUBIT_INDEX = 5

        # Hardware Configuration
        self.hw_cfg = {
            # DAC
            "qubit_ch": self.FSGEN_CH,  # Qubit Channel Port, Full-speed DAC
            "qubit_ampl_ch": self.FSGEN_AMPL_CH ,
            "res_ch": self.MIXMUXGEN_CH ,  # Single Tone Readout Port, MUX DAC
            "qubit_ch_ef": self.FSGEN_CH, # Qubit ef Channel, Full-speed DAC
            # PUCQ4: qubits are 5411-7036 MHz, which is Nyquist zone 2 on the
            # full-speed DAC (was zone 1 for the old 2.7-3.1 GHz chip).
            # Confirm against soccfg with pucq4_00_check_board.py.
            "nqz_qubit": 2,  # PUCQ4. Was 1 for squill.
            "nqz_res": 2,
            # ADC
            "ro_ch": self.MUXRO_CH,  # MUX readout channel
            "list_of_all_qubits": [0, 1, 2, 3, 4, 5]
        }

        # Readout Configuration
        self.readout_cfg = {
            "trig_time": 0.4,  # [Clock ticks] - get this value from TOF experiment (updated by Sara July 22 2025 QICK box)

            # Changes related to the resonator output channel
            #"mixer_freq": 6000,  # [MHz] squill
            "mixer_freq": 8990,  # [MHz] PUCQ4, centred on the 8920-9059 comb
            #"res_freq_ge": [6217, 6276, 6335, 6407, 6476, 6538],  # MHz, run 5
            #'res_freq_ge': [6217.011, 6275.7973, 6335.1068, 6407.052, 6476.1091, 6538], # Arianna 3/27/
            #'res_freq_ge': [6216.811, 6275.9373, 6335, 6407.0338, 6475.8835, 6538], #Joyce 3/11
            #'res_freq_ge': [7149,7171,7204,7228.9, 7264.22, 7287.54],#[7148.588, 7170.546, 7203.351, 7228.059, 7263.744 ,7286.719], #updated by Kester for run 7, Qick board, 6418.4 R5
            # Fixed rows M1-M4 plus the selected validated current rows for
            # Yoko3/Q4/M5 and Yoko4/Q6/M6.
            'res_freq_ge': list(self.pucq4_flux_config["res_freq_ge"]),
            #'res_freq_ge': [6219.097, 6284.55, 6343.95, 6414.934, 6418.4, 6547.25],  # updated by Kester for run 7, QICK box

            # "res_freq_ge": [6191.419, 6216.1, 6292.361, 6405.77, 6432.759, 6468.481],  # MHz, run 4a
            # "res_gain_ge": [1] + [0]*5,
            "res_gain_ge": list(self.pucq4_flux_config["res_gain_ge"]),
            "res_length_ge": list(self.pucq4_flux_config["res_length_ge"]),  # [us]
            "res_freq_offset_ge": list(self.pucq4_flux_config["res_freq_offset_ge"]),  # [MHz]
            #"res_gain_ge": [0.96, 1, 0.76, 0.58, 0.75, 0.57], # Joyce 04/07 DAC 0
            # set_res_gain_ge(QUBIT_INDEX), #utomatically sets all gains to zero except for the qubit you are observing
            # "res_gain_ge": [1,1,0.7,0.7,0.7,1], #[0.4287450656184295, 0.4903077560386716, 0.4903077560386716, 0.3941941738241592, 0.3941941738241592, 0.4903077560386716],  # DAC units
            # "res_freq_ef": [7149.44, 0, 0, 0, 0, 0], # [MHz]
            # "res_gain_ef": [0.6, 0, 0, 0, 0, 0], # [DAC units]
            #"res_freq_ef": [6223.016, 6284.544, 6343.861, 6414.893, 7264.4, 7287.45],  # [MHz] updated by arianna for run 7
            "res_freq_ef": [8919.85, 8951.65, 8975.00, 8999.95, 9014.95, 9059.45],
            "res_gain_ef": [0.95,0.9,0.95,0.55,0.55,0.95],  # [DAC units]
            #"res_freq_fh": [6223.016, 6284.544, 6343.861, 6414.893, 6414.893, 6546.754],  # [MHz]
            "res_freq_fh": [8919.85, 8951.65, 8975.00, 8999.95, 9014.95, 9059.45],
            "res_gain_fh": [0.95,0.9,0.95,0.55,0.55,0.95],  # [DAC units]
            "res_length": 4.0,  # [us] (1.0 for res spec)
            "res_phase": 0,#[ -180+((1.281174-2.6703) * 180/np.pi), -10, 85,
                        #   0, 150,
                        # -90], #Joyce 3/11
            #"res_phase": [(0.19-0.38) * 180/np.pi, (2.07-3.12-1.16) * 180/np.pi, (-0.35+2.28) * 180/np.pi,
               #           (-1.36+1.68+1.1) * 180/np.pi, (-2.4-1.5) * 180/np.pi, (-0.56+1.18) * 180/np.pi],
            # [-0.1006 *360/np.pi, -2.412527*360/np.pi, -1.821284*360/np.pi, -1.90962*360/np.pi, -0.566479*360/np.pi, -0.5941687*360/np.pi], # Rotation Angle From QICK Function, is the ang of 10 ss angles per qubit
            # "res_phase": [0]*6,#[-0.1006 *360/np.pi, -2.412527*360/np.pi, -1.821284*360/np.pi, -1.90962*360/np.pi, -0.566479*360/np.pi, -0.5941687*360/np.pi], # Rotation Angle From QICK Function, is the ang of 10 ss angles per qubit
            "ro_phase": 0,  # Rotation Angle From QICK Function
            "n_resets": 3,
            "g_center":0,
            "e_center":0,
            "threshold": 0, #Joyce 3/11
            #"threshold": [7.3961, -12.5812, 4.8613, -7.5323, 7.0689, 4.6805], # Threshold for Distinguish g/e, from QICK Function
            "res_ring_up_time": 4,  # Olivia May 17th
            "qubit_is_in_g_threshold": 100000, #100000,#-8837,
            "edge_of_e_state_threshold": -26046,
        }

        # Qubit Configuration
        self.qubit_cfg = {
            #"qubit_freq_ge": [2764, 2980, 2876, 3096, 3043.32, 3095.65],#[2766, 2980, 2873, 3096, 3043, 3093],  # Joyce 3/11
            # Row order is M1..M6. M4 remains unassigned; physical Q4 is M5
            # (index 4/Yoko 3) and physical Q6 is M6 (index 5/Yoko 4).
            "qubit_freq_ge": list(self.pucq4_flux_config["qubit_freq_ge"]),
            "qubit_freq_chevron_detuned_ge": [4189.7582, 3820.4723, 4161.3726, 4463.15226, 4471.43854, 4997.86], # Olivia May 17
            "qubit_freq_ge_starked": [4189.737678, 3820.4723, 4161.3726, 4463.15226, 4471.4469, 4997.86], # Olivia 4/04 for zeno/stark tone
            "fwhm_w01_starked": None, #for err bars
            "fwhm_w01": None, #for err bars

            # Current-specific spectroscopy gain and rectangular pulse length.
            "qubit_gain_ge": list(self.pucq4_flux_config["qubit_gain_ge"]),
            "qubit_ampl_gain_ge": 0.025,
            "qubit_pi_len": 0.11, # Olivia May 17th
            # [0.4287450656184295, 0.4287450656184295, 0.4903077560386716, 0.6, 0.4903077560386716, 0.4287450656184295], # For spec pulse
            "qubit_length_ge": list(self.pucq4_flux_config["qubit_length_ge"]),  # [us] qspec pulse
            #"qubit_freq_ef": [2764, 2980, 2876, 3096, 3043.32, 3095.65], #Q4 not fixed, looks like it shifted quite a lot
            # PUCQ4: f_ef = f_ge - alpha, with alpha from the deck
            # (220, 210, 218, 218, 214, 216 MHz). Cross-checks against the
            # measured f_gf/2 column to <0.5 MHz on every row.
            "qubit_freq_ef": [5287, 5201, 5269, 5357, 6822, 6185],  # PUCQ4
            # [MHz] Freqs of Qubit e/f Transition
            #"qubit_freq_fh": [4016.3, 3450.8, 3988.44, 4292.73, 4292.73, 4833.17],
            # PUCQ4: rough guess only, f_fh ~ f_ef - alpha. The f-h anharmonicity
            # is larger than alpha, so expect these to be off by tens of MHz.
            "qubit_freq_fh": [5067, 4991, 5051, 5139, 6608, 5969],  # PUCQ4
            "qubit_freq_ftores": [4016.3, 3644.76, 3988.44, 4292.73, 4303.18, 4833.17],
            "qubit_gain_ef":  [0.003,0.002,0.002,0.0005,0.002, 0.004],# [0.03, 0.14, 0.04, 0.1, 0.15, 0.08],#
            "qubit_gain_fh": [0.001, 0.015, 0.0075, 0.1, 0.15, 0.006],
            "corrected_qspec_freq":[],
            'qubit_gain_ftores': 1,#[0.2, 0.14, 0.04, 0.17, 0.13, 0.08],
            # [0.01, 0.05, 0.05, 0.05, 0.01, 0.5], # [DAC units] Pulse Gain
            "qubit_length_ef": 25, #longer pulse and lower gain here
            "qubit_length_fh": 30,  # longer pulse and lower gain here
            "qubit_length_ftores": 22,  # 25.0,
            "qubit_phase": 0,  # [deg]
            #"sigma": [0.15]*6,  # [us] for Gaussian Pulse (5+10 DAC atten for qubit)
            "sigma_ampl": [0.03, 0.03, 0.05, 0.04, 0.05, 0.05], #DAC 0 04/07
            # Final 0-1 Gaussian widths; pulse programs use length = 4*sigma.
            "sigma": list(self.pucq4_flux_config["sigma"]),  # [us]
            "active_reset_test_sigma": 0.19*1.8,
            #"sigma": [0.05, 0.09, 0.07, 0.065, 0.09, 0.3],  # Goal: cut sigma in half [us] for Gaussian Pulse (5+4 DAC atten for qubit)
            # "pi_amp": [0.92, 0.87, 0.75, 0.73, 0.77, 0.78], # old RR values
            "sigma_ef": [0.28, 0.25, 0.22, 0.25, 0.15, 0.09],  # [us] for Gaussian Pulse, #Arianna 3/27
            "sigma_fh": [0.15, 0.21, 0.25, 0.29, 0.28, 0.15],  # [us] for Gaussian Pulse, #Arianna 3/27
            # Final pi gains paired with the selected sigma values above.
            "pi_amp": list(self.pucq4_flux_config["pi_amp"]),
            "pi_amp_ampl": [0.5942, 0.634499, 0.76542, 0.7754, 0.55393, 0.9], # Joyce 04/07 DAC 0
            #"pi_amp": [1.0, 0.93, 0.77, 0.8, 0.81, 0.9], # Eyeballed by Sara today (5+10 DAC atten for qubit)
            #"pi_amp": [0.7, 0.95, 0.75, 0.78, 0.77, 0.8],  # With shorter sigma (5+4 DAC instead of 5+5 DAC atten for qubit)
            "pi_ef_amp": [0.563, 0.673, 0.511, 0.7018, 0.6751, 0.758], # Arianna 3/27
            "pi_fh_amp": [0.563, 0.8023, 0.511, 0.7018, 0.6751, 0.595],  # 0.589, 0.61, .6
            #"qubit_mixer_freq": 4300,  # [MHz] squill
            "qubit_mixer_freq": 6200,  # [MHz] PUCQ4, centred on 5411-7036

        }


    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def mask_gain_res(self, QUBIT_INDEX, IndexGain = 1, num_qubits=6):
        """Sets the gain for the selected qubit to 1, others to 0."""
        filtered_gain_ge = [0] * num_qubits  # Initialize all gains to 0
        if 0 <= QUBIT_INDEX < num_qubits: #makes sure you are within the range of options
            filtered_gain_ge[QUBIT_INDEX] = IndexGain  # Set the gain for the selected qubit
        return filtered_gain_ge




