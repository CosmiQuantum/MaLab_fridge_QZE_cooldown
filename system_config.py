from qick import *
import sys
import os
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))
from socProxy import makeProxy
import os
import datetime
import numpy as np


# Accepted PUCQ4 flux calibrations, keyed by physical flux line and current in mA.
# These supplement (and do not overwrite) the zero-current six-row arrays below.
PUCQ4_FLUX_CALIBRATION = {
    "Yoko3_Q4_M5": {
        0.0: {"qfreq_MHz": 7084.7785, "sigma_us": 0.05996, "pi_amp": 0.63871,
              "T1_us": 5.96, "T2R_us": 4.33, "T2E_us": 11.35},
        -2.5: {"qfreq_MHz": 6945.5141, "sigma_us": 0.05996, "pi_amp": 0.63871,
               "readout_length_us": 8.0, "readout_gain": 0.57, "readout_offset_MHz": 0.08,
               "SSF": 0.552, "T1_us": 11.22, "T2R_us": 3.48, "T2E_us": 9.75},
        -5.0: {"qfreq_MHz": 6725.3975, "sigma_us": 0.08530, "pi_amp": 0.64869,
               "readout_length_us": 7.0, "readout_gain": 0.89, "readout_offset_MHz": 0.20,
               "SSF": 0.572, "T1_us": 15.53, "T2R_us": 2.40, "T2E_us": 7.74},
        -7.5: {"qfreq_MHz": 6423.2222, "sigma_us": 0.05911, "pi_amp": 0.64869,
               "res_base_MHz": 9012.49, "readout_length_us": 7.0,
               "readout_gain": 0.9267, "readout_offset_MHz": -0.16,
               "SSF": 0.666, "T1_us": 19.09, "T2R_us": 1.86, "T2E_us": 6.01},
        2.5: {"qfreq_MHz": 7143.8491, "sigma_us": 0.05264, "pi_amp": 0.65867,
              "res_base_MHz": 9015.30, "readout_length_us": 5.0,
              "readout_gain": 0.95, "readout_offset_MHz": -0.84, "SSF": 0.6296},
        5.0: {"qfreq_MHz": 7121.7063, "sigma_us": 0.07321, "pi_amp": 0.66865,
              "res_base_MHz": 9014.95, "readout_length_us": 7.0,
              "readout_gain": 0.65, "readout_offset_MHz": -0.24, "SSF": 0.3305},
        7.5: {"qfreq_MHz": 7022.3193, "sigma_us": 0.05445, "pi_amp": 0.65867,
              "res_base_MHz": 9014.65, "readout_length_us": 6.0,
              "readout_gain": 0.9033, "readout_offset_MHz": -0.24, "SSF": 0.6913},
        10.0: {"qfreq_MHz": 6841.5394, "sigma_us": 0.05911, "pi_amp": 0.62873,
               "res_base_MHz": 9013.50, "readout_length_us": 6.0,
               "readout_gain": 0.9261, "readout_offset_MHz": -0.60, "SSF": 0.6960},
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

# Current-specific pulse seeds are separate from the zero-current ``sigma`` and
# ``pi_amp`` arrays below. Entries marked ``revalidation_required`` produced an
# in-window raw-I/Q Rabi result but did not pass the complete post-readout chain;
# callers must not silently substitute them for the 0 mA device calibration.
PUCQ4_FLUX_PULSE_CALIBRATION = {
    "Yoko3_Q4_M5": {
        -10.0: {
            "sigma_us": 0.04519688543145721,
            "raw_iq_pi_amp": 0.6985900018311665,
            "raw_iq_qfreq_MHz": 5984.071123039789,
            "status": "revalidation_required",
            "note": (
                "M5 uses a resonator peak near 9012.87 MHz at -10 mA. "
                "Low-power qspec passed at 5984.07112 MHz with 0.17386 MHz "
                "FWHM. Raw-I/Q Rabi entered the pi window, but its trace is "
                "noisy and post-pulse readout rescue fidelity remained 0.077; "
                "do not use for coherence until full revalidation passes."
            ),
        },
    },
    "Yoko4_Q6_M6": {
        -7.5: {
            "sigma_us": 0.12225276618925736,
            "raw_iq_pi_amp": 0.6985900018311665,
            "raw_iq_qfreq_MHz": 4840.0,
            "status": "revalidation_required",
            "note": (
                "Corrected line replaces the rejected 5061.75 MHz alias. "
                "Raw-I/Q Rabi visibly resolves just over half an oscillation "
                "and meets the 0.6-0.7 pi window; complete full validation."
            ),
        },
        -5.0: {
            "sigma_us": 0.06714287589206007,
            "raw_iq_pi_amp": 0.6586705731550998,
            "raw_iq_qfreq_MHz": 5587.9,
            "status": "revalidation_required",
            "note": (
                "Corrected-readout raw-I/Q Rabi resolves just over half an "
                "oscillation in I and meets the 0.6-0.7 pi window; complete "
                "post-readout/coherence validation."
            ),
        },
        -10.0: {
            "sigma_us": 0.09862908781821704,
            "raw_iq_pi_amp": 0.6886101446621498,
            "raw_iq_qfreq_MHz": 4375.392561188658,
            "status": "revalidation_required",
            "note": (
                "Raw-I/Q Rabi met the 0.6-0.7 pi window. Final qspec FWHM was "
                "0.81095 MHz (limit 0.8 MHz) and optimized-readout Rabi returned "
                "the 0.84829 boundary fit; revalidate before coherence use."
            ),
        },
        5.0: {
            "sigma_us": 0.055669293746329056,
            "pi_amp": 0.6187511444790331,
            "qfreq_MHz": 7062.7712444396775,
            "status": "validated",
            "note": (
                "Full post-readout chain passed at +5 mA: qspec FWHM 0.17556 MHz, "
                "pi amplitude 0.61875, SSF 0.4910, T1 5.48 us, T2R 2.77 us, "
                "and T2E 8.00 us."
            ),
        },
        7.5: {
            "sigma_us": 0.058001,
            "pi_amp": 0.6387108588170665,
            "qfreq_MHz": 7185.659478471124,
            "status": "validated",
            "note": (
                "Full post-readout chain passed at +7.5 mA: qspec FWHM "
                "0.14571 MHz, pi amplitude 0.63871, SSF 0.4740, T1 5.48 us, "
                "T2R 4.13 us, and T2E 9.35 us."
            ),
        },
        10.0: {
            "sigma_us": 0.05115680589458381,
            "pi_amp": 0.6287310016480498,
            "qfreq_MHz": 7211.443553644731,
            "status": "validated",
            "note": (
                "Full post-readout chain passed at +10 mA: qspec FWHM 0.11949 "
                "MHz, pi amplitude 0.62873, SSF 0.4710, T1 5.35 us, T2R "
                "6.35 us, and T2E 9.56 us."
            ),
        },
        -2.5: {
            "sigma_us": 0.060540487104986655,
            "pi_amp": 0.6087712873100165,
            "qfreq_MHz": 6103.796752687653,
            "status": "validated",
            "note": (
                "Full post-readout chain passed at -2.5 mA: qspec FWHM "
                "0.24612 MHz, pi amplitude 0.60877, SSF 0.4270, T1 5.03 us, "
                "T2R 1.08 us, and T2E 4.46 us."
            ),
        },
    },
}

# ADC_attenuator MUST be above 16dB
#DAC_attenuator1 and 2 are for the resonators
#qubit_DAC_attenuator1 and 2 are for the qubits
class QICK_experiment:
    def __init__(self, folder, DAC_attenuator1 = 10, DAC_attenuator2 = 15, qubit_DAC_attenuator1 = 5,
                                     qubit_DAC_attenuator2 = 4, ADC_attenuator = 17, fridge = None):

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
            # PUCQ4 from "PUCQ4 Initial Characterization.pptx" (M1..M6).
            # These are the HIGH-POWER VNA values -- replace with the punched
            # out frequencies once you have run a power sweep.
            # PUCQ4 coarse high-power centers measured 2026-08-27 at gain 0.9,
            # 0.2 MHz steps. Refine at low power before calling them dressed.
            'res_freq_ge': [8919.85, 8951.65, 8975.00, 8999.95, 9014.95, 9059.45],
            #'res_freq_ge': [6219.097, 6284.55, 6343.95, 6414.934, 6418.4, 6547.25],  # updated by Kester for run 7, QICK box

            # "res_freq_ge": [6191.419, 6216.1, 6292.361, 6405.77, 6432.759, 6468.481],  # MHz, run 4a
            # "res_gain_ge": [1] + [0]*5,
            "res_gain_ge": [0.48, 0.56, 0.41, 0.30, 0.33, 0.60],
            "res_length_ge": [10.0, 5.0, 12.0, 10.0, 5.0, 3.0],  # [us]
            "res_freq_offset_ge": [-0.52, -0.40, -0.08, 0.40, 0.44, -0.04],  # [MHz]
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
            # PUCQ4. *** FLUX TUNABLE -- these are only valid at the DC bias the
            # VNA characterization was taken at. Q4 moves 5850-7110 MHz across
            # +/-10 mA, so at the wrong bias these numbers are meaningless. ***
            # Also note the resonator<->qubit mapping is NOT settled: the deck
            # says M5->Q4 and M6->Q6, but the avoided-crossing data suggests
            # Q4->M1 and Q6->M2. Row order here follows the deck's table.
            # Row order is M1..M6. M4 has no assigned physical qubit and remains
            # a deck estimate. The deck maps physical Q4 to M5 and Q6 to M6.
            "qubit_freq_ge": [5517.69373, 5438.71351, 5557.33447, 5575.00,
                              7084.77785, 6522.54018],
            "qubit_freq_chevron_detuned_ge": [4189.7582, 3820.4723, 4161.3726, 4463.15226, 4471.43854, 4997.86], # Olivia May 17
            "qubit_freq_ge_starked": [4189.737678, 3820.4723, 4161.3726, 4463.15226, 4471.4469, 4997.86], # Olivia 4/04 for zeno/stark tone
            "fwhm_w01_starked": None, #for err bars
            "fwhm_w01": None, #for err bars

            # Weak continuous-wave-like spectroscopy drive. The former
            # 0.03-0.08 gains produced coherent square-pulse sidelobes.
            "qubit_gain_ge": [0.001, 0.007, 0.002, 0.010, 0.004, 0.006],
            "qubit_ampl_gain_ge": 0.025,
            "qubit_pi_len": 0.11, # Olivia May 17th
            # [0.4287450656184295, 0.4287450656184295, 0.4903077560386716, 0.6, 0.4903077560386716, 0.4287450656184295], # For spec pulse
            "qubit_length_ge": 20,  # 5 [us] for spec Pulse
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
            # PUCQ4 starting point for the 0-1 amplitude-Rabi calibration.
            "sigma": [0.050, 0.101, 0.124, 0.10, 0.055, 0.062],  # [us]
            "active_reset_test_sigma": 0.19*1.8,
            #"sigma": [0.05, 0.09, 0.07, 0.065, 0.09, 0.3],  # Goal: cut sigma in half [us] for Gaussian Pulse (5+4 DAC atten for qubit)
            # "pi_amp": [0.92, 0.87, 0.75, 0.73, 0.77, 0.78], # old RR values
            "sigma_ef": [0.28, 0.25, 0.22, 0.25, 0.15, 0.09],  # [us] for Gaussian Pulse, #Arianna 3/27
            "sigma_fh": [0.15, 0.21, 0.25, 0.29, 0.28, 0.15],  # [us] for Gaussian Pulse, #Arianna 3/27
            # PUCQ4, with the per-mode sigma values above, measured 2026-08-27.
            # M4 remains uncalibrated; physical Q4 is the calibrated M5 row.
            "pi_amp": [0.6786303, 0.6486907, 0.6586706, 0.7,
                       0.6486907, 0.6486907],
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




