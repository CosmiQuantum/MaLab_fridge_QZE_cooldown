import numpy as np

FRIDGE = "BOB"

if FRIDGE == "QUIET" or FRIDGE == "BOB":
    VNA_res = np.array([7149,7171,7204,7228.9, 7264.22,7287.5])#[7148.588, 7170.546, 7203.351, 7228.059, 7263.744 ,7286.719])#*1000  # run 5
    VNA_qubit = np.array([2780, 2980, 2885, 3096, 3043.32, 3093]) #[2766, 2980, 2873, 3096, 3043, 3093] # Freqs of Qubit g/e Transition
    ef_freqs = np.array([2616, 2830, 2723, 2946, 2893, 2943]) # Freqs of Qubit e/f Transition, updated for run 7
    fh_freqs = np.array([2466, 2680, 2573, 2796, 2743, 2793])
    # Set this for your experiment
    tot_num_of_qubits = 6

    gain_start = 0.000001
    gain_stop = 0.01
    gain_steps= 50

    list_of_all_qubits = list(range(tot_num_of_qubits))

    expt_cfg = {
        "tof": {
            "reps": 1, #reps doesnt make a difference here, leave it at 1
            "soft_avgs": 400,
            "relax_delay": 0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "res_spec": {
            "reps": 1, #shots at one freq
            "rounds": 200, #sweeps through each freq and average
            "start": -0.7, #[MHz]
            "step_size": 0.01,  # [MHz]
            "steps": 150,#,200,#70
            "relax_delay": 5,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "res_spec_ef": {
            "reps": 1,#shots at one freq
            "rounds": 100, #sweeps through each freq and average
            "start": -0.7, #[MHz]
            "step_size": 0.01,  # [MHz]
            "steps": 150,#,200,#70
            "relax_delay": 5,  # [us]
        },

        "qubit_spec_ge": {
            "reps": 250,
            "rounds": 5,
            "start": list(VNA_qubit-30), # [MHz] -30
            "stop": list(VNA_qubit+15), # [MHz] +15
            "steps": 500,
            "relax_delay": 10, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "qubit_spec_ge_zeno": {
            "reps": 250,
            "rounds": 5,
            "start": list(VNA_qubit - 15),  # [MHz] -40
            "stop": list(VNA_qubit + 10),  # [MHz] +40
            "steps": 500,
            "relax_delay": 200,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
            "gain_start": gain_start,
            "gain_stop": gain_stop,
            "gain_steps": gain_steps,
        },

        "qubit_spec_ge_extended": {
            "reps": 500,  # 300
            "rounds": 1,  # 10
            "start": list(VNA_qubit - 250),  # [MHz]
            "stop": list(VNA_qubit + 100),  # [MHz]
            "steps": 1000,  # 100
            "relax_delay": 10,  # 1000, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "qubit_spec_ge_high_gain": {
            "reps": 50,  # 300
            "rounds": 500,  # 10
            "start": list(VNA_qubit - 30),  # [MHz]
            "stop": list(VNA_qubit + 30),  # [MHz]
            "steps": 600,  # 100
            "relax_delay": 10,  # 1000, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "qubit_spec_ge_zeno_stark": {
            "reps": 2500,  # 300
            "rounds": 1,  # 10
            "start": list(VNA_qubit - 25),  # [MHz] #-300 #-15
            "stop": list(VNA_qubit + 5),  # [MHz] #+15
            # "start": list(VNA_qubit - 100),  # [MHz] #-300 #-15
            # "stop": list(VNA_qubit + 100),  # [MHz] #+15
            "steps": 200,  # 100
            "relax_delay": 700,  # 1000, # [us]
            "list_of_all_qubits": list_of_all_qubits,
            "qze_mask": [],
        },

        "qubit_spec_ef": {
            "reps": 100,  # 300
            "rounds": 1000,  # 10
            "start": list(ef_freqs - 0.2),# 0.2),  # [MHz] #-300 #-6
            "stop":  list(ef_freqs + 0.2),#0.2),  # [MHz] #6
            "steps": 220,  # 1000 #450
            "relax_delay": 500, #1000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "qubit_spec_fh": {
            "reps": 10,  # 300
            "rounds": 5000,  # 10
            "start": list(fh_freqs - 0.3), # [MHz] #-300 #-6
            "stop": list(fh_freqs +  0.2),  # [MHz] #6
            "steps": 180,#450,  # 1000 #450
            "relax_delay": 1000,  # 1000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "qubit_spec_ftores": {
            "reps": 10000,  # 300
            "rounds": 1,  # 10
            "start": list((VNA_qubit + ef_freqs) - VNA_res - 200),  # [MHz] #-300
            "stop":  list((VNA_qubit + ef_freqs) - VNA_res + 200),  # [MHz]
            "steps": 1000,  # 1000
            "relax_delay": 0.5,  # 1000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "bias_qubit_spec_ge": {
            "reps": 700,  # 100
            "rounds": 1,  # 10
            "start": list(VNA_qubit - 70),  # [MHz]
            "stop": list(VNA_qubit + 70),  # [MHz]
            "steps": 300,
            "relax_delay": 0.5,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "power_rabi_ge": {
            "reps": 20,
            "rounds": 3,
            "start": 0, # [DAC units]
            "stop": 0.085,  # [DAC units]
            "steps": 70,
            "relax_delay": 500,# [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "power_rabi_ge_zeno": {
            "reps": 50,
            "rounds": 5,
            "start": 0,  # [DAC units]
            "stop": 0.085,  # [DAC units]
            "steps": 70,
            "relax_delay": 500,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "power_rabi_vs_gain": {
            "start_gain": 0.00001,
            "end_gain": 0.1,
            "gain_steps": 3,
            "reps": 50,
            "rounds": 5,
            "start": 0,  # [DAC units]
            "stop": 0.085*6,  # [DAC units] get many fringes, 6/2
            "steps": 70,
            "relax_delay": 500,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "length_rabi_ge": {
            "reps": 20,#500,
            "rounds": 5,
            "start": 0.01,  # [us]
            "stop": 15,#[0.7] * 6,  # [us]
            "steps": 500,
            "relax_delay": 500,# [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "length_rabi_Qtemps": {
            "reps": 10,#500,
            "rounds": 400,
            "start": 0.02,  # [us]
            "stop": 2,#[0.7] * 6,  # [us]
            "steps": 200,
            "relax_delay": 500,# [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "length_rabi_ge_pi_len": { #for the zeno tests, bare qubit frequency rabi
            "reps": 10,  # 500,
            "rounds": 300,
            "start": 0.01,  # [us]
            "stop": 4,  # [0.7] * 6,  # [us]
            "steps": 60,
            "relax_delay": 500,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "length_rabi_ge_qze": {
            "reps": 500,
            "rounds": 1, #600
            "start": 0.01,  # [us]
            "stop": 10,   #[3] * 6,   # [us]
            "steps": 150,  # 140,
            "relax_delay": 500,# [us]
            "list_of_all_qubits": list_of_all_qubits,
            "qze_mask": [],
            "zeno_pulse_width": 0.007,
            "zeno_pulse_period": 0.10,
        },

        "power_rabi_ef": {
            "reps": 20,
            "reps2": 850, #this is only used for the experiment that uses e-f rabi to calculate qubit temperatures.
            "rounds": 400,
            "start": 0.0,  # [DAC units]
            "stop": 1.0,  # [DAC units]
            "steps": 155,
            "relax_delay": 500,  # [us]
        },
        "power_rabi_fh": {
            "reps": 20,
            "reps2": 850,  # this is only used for the experiment that uses e-f rabi to calculate qubit temperatures.
            "rounds": 400,
            "start": 0.0,  # [DAC units]
            "stop": 1.0,  # [DAC units]
            "steps": 155,
            "relax_delay": 500,  # [us]
        },
        "T1_ge": {
            "reps": 100,#,50, #300
            "rounds": 1, #1
            "start":  0.01,  # [us]
            "stop": 500,  # [us] ### Should be ~10x T1! Should change this per qubit.
            "steps": 200,
            "relax_delay": 500,  # [us] ### Should be >10x T1!
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "T1_IBM_qze": {
            "reps": 100,
            "rounds": 1,#500,
            "start": 30,  # [us]
            "stop": 31,  # [250.0] * 6,  # [us] ### Should be ~10x T1! Should change this per qubit.
            "steps": 1,
            "relax_delay": 500,  # [us] ### Should be >10x T1!
            "wait_time": 30,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
            "gain_start": gain_start,
            "gain_stop": gain_stop,
            "gain_steps": gain_steps,
        },

        "T1_IBM_qze_loop": {
            "reps": 100,
            "rounds": 1,#20,
            "start": 0.01,  # [us]
            "stop": 100,  # [250.0] * 6,  # [us] ### Should be ~10x T1! Should change this per qubit.
            "steps": 200,
            "relax_delay": 500,  # [us] ### Should be >10x T1!
            "wait_time": 30,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
            "gain_start": gain_start,
            "gain_stop": gain_stop,
            "gain_steps": gain_steps,
        },

        "T1_fe": {
            "reps": 20,  # 300
            "rounds": 300,  # 1
            "start": 0.0,  # [us]
            "stop": 200,  # [250.0] * 6,  # [us] ### Should be ~10x T1! Should change this per qubit.
            "steps": 150,
            "relax_delay": 500,  # [us] ### Should be >10x T1!
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "T1_fg": {
            "reps": 20,  # 300
            "rounds": 300,  # 1
            "start": 0.0,  # [us]
            "stop": 200,  # [250.0] * 6,  # [us] ### Should be ~10x T1! Should change this per qubit.
            "steps": 150,
            "relax_delay": 500,  # [us] ### Should be >10x T1!
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "ckp_nbar_calibration":{
            "reps": 20,
            "rounds": 300,
            "list_of_all_qubits": list_of_all_qubits,
            "qubit_pulse_delay": 3,  # [us] time to wait for qubit pulse after stark tone is sent
            "stark_length": 19,  # [us] stark tone length for 2D scan, overlaps qubit pulse
            "gain_steps": 20,
            "start_gain": 0.00,
            "end_gain": 1.0,  # res gain values between -1 and 1
            "qubit_pulse_steps": 500,
            "res_freq_steps": 20,
            "res_freq_start": 0.00,
            "res_freq_stop": 1.0,
            "start_freq": -55,  # -35, #[MHz] from qubit freq
            "end_freq": 4,  # 3, # [MHz] from qubit freq
            "readout_pulse_delay": 2,  # 2/kappa
            "ckp_gain": [],
            "res_freq_ckp": [],
            "res_phase_ckp": [],
            "ckp_mask": [],
        },
        "res_stark_shift_2D": {
            "reps": 30,
            "rounds": 1500,
            "list_of_all_qubits": list_of_all_qubits,
            "qubit_pulse_delay": 3, #[us] time to wait for qubit pulse after stark tone is sent
            "stark_length": 19,  # [us] stark tone length for 2D scan, overlaps qubit pulse
            "gain_steps": 20,
            "start_gain": 0.00,
            "end_gain": 0.1, #res gain values between -1 and 1
            "qubit_pulse_steps": 500,
            "start_freq": -65, #[MHz] from qubit freq
            "end_freq":  20, # [MHz] from qubit freq
            "readout_pulse_delay": 5, #2/kappa
            "stark_gain": [],
            "res_freq_stark": [],
            "res_phase_stark": [],
            "stark_mask": [],
        },

        "stark_shift_2D": {
            "reps": 10,
            "rounds": 500,
            "list_of_all_qubits": list_of_all_qubits,
            "qubit_pulse_delay": 0.5,  # [us] time to wait for qubit pulse after stark tone is sent
            "stark_length": 16,  # [us] stark tone length for 2D scan, overlaps qubit pulse
            "gain_steps": 20,
            "start_gain": 0.00,
            "end_gain": 1.0,  # res gain values between -1 and 1
            "qubit_pulse_steps": 400,
            "min_freq": 3,  # [MHz] from qubit freq
            "max_freq": 15,  # [MHz] from qubit freq
            "start_freq": 0,
            "end_freq": 0,
            "readout_pulse_delay": 2,  # 2/kappa
            "detuning": [-20, -10, -10, -10, -15, -10],  # [MHz]
            "stark_gain": [],
        },

        "stark_shift_spec": {
            "reps": 20,
            "rounds": 300,
            "list_of_all_qubits": list_of_all_qubits,
            "stark_length": 25, # [us] stark tone length for TLS spectroscopy, try 10-30% of T1
            "max_shift": 20, # [MHz] specify frequency range of stark shift
            "duffing_constant": [500000,500000,50000,50000,500000,500000], #constant for duffing oscillator model from stark ramsey measurement, update for each qubit
            "gain_steps": 100, #for each branch of pos,neg detuning stark and for entire res stark
            "start_gain": 0.0,
            "end_gain": 1.0, #res gain values between -1 and 1, convert to qubit freq shift w/ stark ramsey
            "readout_pulse_delay": 2, #2/kappa
            "relax_delay": 500, #[us]
            "detuning": [-20, -10, -10, -10, -15, -10], #[MHz] start w/negative detuning, script flips to positive halfway thru scan
            "stark_sigma": 0.01,  # [us] 10 ns
            "stark_gain": [],
            "anharmonicity": [172.34, 176.38, 167.13, 172.57, 172.03, 161.14],
            "res_gain_stark": [],
            "res_freq_stark": [],
            "res_phase_stark": [],
            "stark_mask": [],
        },

        "Ramsey_stark": {
            "reps": 300,
            "rounds": 3,
            "start": 0.02,  # [us]
            "stop": 60,  # [us]
            "steps": 1500,
            "start_gain": 0.0,
            "end_gain": 0.012,
            "gain_steps": 100,
            "ramsey_freq": 1.5, #1.5 [MHz]
            "relax_delay": 2000,
            "wait_time": 0.0,  # [us]
            "stark_gain": 0.0,
            "detuning": -20, # [MHz]
            "stark_sigma": 0.01, # [us] 10 ns
            "list_of_all_qubits": list_of_all_qubits,
            "anharmonicity": [150]*6,
            'chi': [x/2 for x in [-0.264, -0.272, -0.266, -0.274, -0.274, -0.234]],
        },

        "FastRelEx":{
            "reps": 500000,
            "rounds": 1,
            "list_of_all_qubits": list_of_all_qubits,
            "relax_delay": 10, #[us], keep short to do post-processing
            "meas_wait": 2, #[us], a fast delay after qubit pi pulse
            "readout_pulse_delay": 0.1, #for resonator to ring down, may be ok to set to zero
            "pre_stark_delay": 0.1, #could match to pi pulse length or set to zero
            "stark_sigma": 0.01, #10 ns following Carrol paper
            "stark_length": 2, #try 1-5 us
        },

        "Ramsey_ge": {
            "reps": 100,
            "rounds": 1,
            "start": 0.0, # [us]
            "stop":  20, # [us]
            "steps": 200,
            "ramsey_freq": 0.6,  # [MHz]
            "relax_delay": 500, # [us] the time to wait to let the qubit to relax to gnd again after exciting it (make it way above T1)
            "wait_time": 0.0, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "Ramsey_ge_zeno": {
            "reps": 100,
            "rounds": 1,
            "start": 0.01,  # [us]
            "stop": 20,  # [us]
            "steps": 200,
            "ramsey_freq": 0.6,  #.15 [MHz]
            "relax_delay": 500,
            # [us] the time to wait to let the qubit to relax to gnd again after exciting it (make it way above T1)
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
            "gain_start": gain_start,
            "gain_stop": gain_stop,
            "gain_steps": gain_steps,
        },

        "Ramsey_ge_correction": {
            "reps": 20,#20,  # 300
            "rounds": 2,  # 10
            "start": 0.0,  # [us]
            "stop": 1,  # [us]
            "steps": 50,
            "ramsey_freq": 4,  # [MHz]
            "relax_delay": 500,
            # [us] the time to wait to let the qubit to relax to gnd again after exciting it (make it way above T1)
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "Ramsey_ge_zeno_correction": {
            "reps": 20,  # 20,  # 300
            "rounds": 2,  # 10
            "start": 0.0,  # [us]
            "stop": 1,  # [us]
            "steps": 100,
            "ramsey_freq": 4,  # [MHz]
            "relax_delay": 500,
            # [us] the time to wait to let the qubit to relax to gnd again after exciting it (make it way above T1)
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "SpinEcho_ge": {
            "reps": 100,#20,
            "rounds": 1,
            "start": 0.01, # [us]
            "stop":  20, # [us]
            "steps": 200,
            "ramsey_freq": 0.6,  #0.12 [MHz]
            "relax_delay": 500, # [us]
            "wait_time": 0.0, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "SpinEcho_ge_zeno": {
            "reps": 100,  # 20,
            "rounds": 1,
            "start": 0.01,  # [us]
            "stop": 20,  # [us]
            "steps": 200,
            "ramsey_freq": 0.6,  # 0.12 [MHz]
            "relax_delay": 500,  # [us]
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
            "gain_start": gain_start,
            "gain_stop": gain_stop,
            "gain_steps": gain_steps,
        },


    #
    #     "res_spec_ef": {
    #         "reps": 100,
    #         "py_avg": 10,
    #         "start": [7148, 0, 7202, 0, 0, 0], # [MHz]
    #         "stop":  [7151, 0, 7207, 0, 0, 0], # [MHz]
    #         "steps": 200,
    #         "relax_delay": 1000, # [us]
    #         "list_of_all_qubits": list_of_all_qubits,
    #     },
    #
    #     "qubit_spec_ef": {
    #         "reps": 100,
    #         "py_avg": 10,
    #         "start": [2750, 0, 0, 0, 0, 0], # [MHz]
    #         "stop":  [2850, 0, 0, 0, 0, 0], # [MHz]
    #         "steps": 500,
    #         "relax_delay": 1000, # [us]
        #         "list_of_all_qubits": list_of_all_qubits,
    #     },
    #
        # "qubit_temp": { #this is for Santi's length rabi qubit temperature script (not working fully)
        #     "reps": 500,
        #     "py_avg": 1, #this is rounds, change after u get script working
        #     "start": [20]*6, # [us]
        #     "expts":  [200] * 6, #points
        #     "step": (22 - 20) / (200 - 1), # [us], step = (stop - start) / (expts - 1)
        #     "relax_delay": 800, # [us]
        #     "list_of_all_qubits": list_of_all_qubits,
        # },


    #
    #     "power_rabi_ef": {
    #         "reps": 1000,
    #         "py_avg": 10,
    #         "start": [0.0] * 6, # [DAC units]
    #         "stop":  [1.0] * 6, # [DAC units]
    #         "steps": 100,
    #         "relax_delay": 1000, # [us]
    #         "list_of_all_qubits": list_of_all_qubits,
    #     },
    #
    #     "Ramsey_ef": {
    #         "reps": 100,
    #         "py_avg": 10,
    #         "start": [0.0] * 6, # [us]
    #         "stop":  [100] * 6, # [us]
    #         "steps": 100,
    #         "ramsey_freq": 0.05,  # [MHz]
    #         "relax_delay": 1000, # [us]
    #         "wait_time": 0.0, # [us]
    #         "list_of_all_qubits": list_of_all_qubits,
    #     },
    #
        "IQ_plot":{
            "steps": 5000, # shots
            "py_avg": 1,
            "reps": 1,
            "relax_delay": 1000, # [us]
            "SS_ONLY": False,
            "list_of_all_qubits": list_of_all_qubits,
        },
    # #

        "Readout_Optimization":{
            "steps": 3000, # shots
            "py_avg": 1,
            "gain_start" : [0, 0, 0, 0],
            "gain_stop" : [1, 0, 0, 0],
            "gain_step" : 0.1,
            "freq_start" : [6176.0, 0, 0, 0],
            "freq_stop" : [6178.0, 0, 0, 0],
            "freq_step" : 0.1,
            "relax_delay": 500,#600, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "Readout_Optimization_300k": {
            "steps": 300000,  # shots
            "py_avg": 1,
            "gain_start": [0, 0, 0, 0],
            "gain_stop": [1, 0, 0, 0],
            "gain_step": 0.1,
            "freq_start": [6176.0, 0, 0, 0],
            "freq_stop": [6178.0, 0, 0, 0],
            "freq_step": 0.1,
            "relax_delay": 500,  # 600, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

    }


else:
    print('Please set variable FRIDGE to QUIET, or configure for your fridge as needed')
