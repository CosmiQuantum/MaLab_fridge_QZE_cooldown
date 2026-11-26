# Q2/M2/index1 qubit/readout values updated from measured RR, 2026-11-04; other rows retained.
import numpy as np

FRIDGE = "BOB"

if FRIDGE == "QUIET" or FRIDGE == "BOB":
    # ---- squill (previous chip), kept for reference -------------------------
    #VNA_res = np.array([7149,7171,7204,7228.9, 7264.4,7287.54])#[7148.588, 7170.546, 7203.351, 7228.059, 7263.744 ,7286.719])#*1000  # run 5
    #VNA_qubit = np.array([2764, 2980, 2876, 3096, 3043.32, 3095.44]) #[2766, 2980, 2873, 3096, 3043, 3093] # Freqs of Qubit g/e Transition
    #ef_freqs = np.array([2616, 2830, 2723, 2946, 2893, 2929.37]) # Freqs of Qubit e/f Transition, updated for run 7
    #fh_freqs = np.array([2466, 2680, 2573, 2796, 2900, 2744])

    # ---- PUCQ4 --------------------------------------------------------------
    # From "PUCQ4 Initial Characterization.pptx", rows M1..M6.
    # FLUX TUNABLE: the qubit numbers are only valid at the DC bias the VNA
    # sweep used. The resonators move ~7 MHz across +/-10 mA; the qubits move
    # GHz. See YOKOGS200.py for the current-source driver.
    # All six rows: fresh zero-current full RR, 2026-10-29; inspect fit qualifications.
    VNA_res = np.array([8918.795, 8950.964390000001, 8974.11, 8999.07, 9013.4, 9057.715])
    # Row order is M1..M6; all rows use the October 29 zero-current round robin;
    # M1-M4 physical numbers remain unassigned; physical Q4 is M5 and Q6 is M6.
    # M6 carrier and Gaussian pulse are now confirmed by repeated qspec and Rabi.
    # Nonzero currents are selected by apply_pucq4_flux_experiment_config below.
    # Q2 working carrier revalidated by normal RR spectroscopy/Rabi, 2026-11-20.
    VNA_qubit = np.array([5515.343391325498, 5438.856540689225, 5551.361090349251, 5594.644240774248, 7065.147636121578, 6404.639622421642])
    ef_freqs = np.array([5287, 5201, 5269, 5357, 6822, 6185])  # f_ge - alpha
    fh_freqs = np.array([5067, 4991, 5051, 5139, 6608, 5969])  # rough guess
    # Set this for your experiment
    tot_num_of_qubits = 6

    # Wide flags retain the historical rediscovery mode. All six rows now have
    # measured zero-current lines; fresh profiles select the acquired settings.
    # Historical deck linewidth estimates are not fresh QICK linewidths.
    # Fresh tracking bounds/averaging come from the round-robin profile below.
    WIDE_RES_SCAN = False    # resonators located; track them now
    WIDE_QUBIT_SCAN = False  # tracking all six measured zero-current carriers
    QSPEC_HALF_SPAN = np.array([0.8, 0.8, 0.8, 2.0, 0.8, 0.8])  # [MHz]

    gain_start = 0.000001
    gain_stop = 0.04
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
            # *** reps, NOT rounds. ***
            # res_spec loops frequency in PYTHON, so every point is a separate
            # program upload. 'reps' averages on the tProc for one Pyro round
            # trip; 'rounds' averages in QICK's acquire() at one round trip
            # EACH (~8 ms of network latency). The original reps=1/rounds=600
            # meant 600 round trips per point: with 150 points x 6 qubits that
            # is over an hour of pure latency for ~30 s of measurement.
            # Same averaging, same SNR, ~600x less waiting.
            "reps": 500,
            "rounds": 1,
            # Wide covers +/-15 MHz, comfortably more than the ~7 MHz the
            # resonators move across the full +/-10 mA flux range, so it finds
            # them at characterized biases. Narrow tracks a measured frequency.
            # Tracking settings previously used by the accepted RR profile.
            "start": -15.0 if WIDE_RES_SCAN else -1.5,  # [MHz]
            "step_size": 0.1 if WIDE_RES_SCAN else 0.02,  # [MHz]
            "steps": 300 if WIDE_RES_SCAN else 151,
            "relax_delay": 100,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "res_spec_ef": {
            "reps": 1,#shots at one freq
            "rounds": 600, #sweeps through each freq and average
            "start": -0.7, #[MHz]
            "step_size": 0.01,  # [MHz]
            "steps": 150,#,200,#70
            "relax_delay": 2000,  # [us]
        },
        "res_spec_fh": {
            "reps": 1,
            "rounds": 600,
            "start": -2,  # [MHz]
            "step_size": 0.02,  # [MHz]
            "steps": 150,
            "relax_delay": 4000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        # "qubit_spec_ge": {
        #     "reps": 200,
        #     "rounds": 20,#3
        #     "start": list(VNA_qubit-15), # [MHz] -30
        #     "stop": list(VNA_qubit+15), # [MHz] +15
        #     "steps": 200,
        #     "relax_delay": 60, # [us]
        #     "list_of_all_qubits": list_of_all_qubits,
        # },
        "qubit_spec_ge": {
            # qubit spec sweeps frequency in HARDWARE (add_loop 'freqloop'), so
            # rounds here costs only a few round trips for the whole sweep --
            # unlike res_spec above, this one is fine as written.
            "reps": 100,
            "rounds": 4 if WIDE_QUBIT_SCAN else 1,  # 3
            # Historical wide-search mode; fresh tracking uses the RR profile.
            # +/-500 MHz because VNA_qubit is only valid at the bias the VNA
            # sweep used, and the flux lines are now ramped to 0 mA. Reading Q4
            # off the student's flux scan it sits ~6900-7000 MHz at zero bias
            # vs 7036 in the table, so zero is close -- but the other qubits'
            # tuning curves were never characterized, hence the wide window.
            # 500 MHz / 400 steps = 2.5 MHz per point, still fine against a
            # 20-40 MHz linewidth.
            # NOTE: 0 mA is NOT a sweet spot. Q4's is around +4 mA / 7110 MHz.
            # Move there before measuring T1/T2 -- see the note in the deck.
            "start": list(VNA_qubit - (500 if WIDE_QUBIT_SCAN else QSPEC_HALF_SPAN)),  # [MHz]
            "stop": list(VNA_qubit + (500 if WIDE_QUBIT_SCAN else QSPEC_HALF_SPAN)),  # [MHz]
            "steps": 100,
            "relax_delay": 10,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "qubit_spec_ef": {
            "reps": 200,
            "rounds": 40,  # 3
            "start": list(ef_freqs - 0.5),  # [MHz] -30
            "stop": list(ef_freqs + 0.5),  # [MHz] +15
            "steps": 100,
            "relax_delay": 4000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "qubit_spec_fh": {
            "reps": 300,  # 300
            "rounds": 20,  # 10
            "start": list(fh_freqs - 4),  # [MHz] #-300 #-6
            "stop": list(fh_freqs + 4),  # [MHz] #6
            "steps": 100,  # 450,  # 1000 #450
            "relax_delay": 4000,  # 4000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "qubit_spec_ge_zeno": {
            "reps": 20,
            "rounds": 20,
            "start": list(VNA_qubit - 8),  # [MHz] -40
            "stop": list(VNA_qubit + 1),  # [MHz] +40
            "steps": 500,
            "relax_delay": 200,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
            "gain_start": gain_start,
            "gain_stop": gain_stop,
            "gain_steps": gain_steps,
            "readout_pulse_delay": 5,
        },
        # "qubit_spec_ge_zeno": {
        #     "reps": 100,
        #     "rounds": 10,#20,
        #     "start": list(VNA_qubit - 4),#8),  # [MHz] -40
        #     "stop": list(VNA_qubit + 1),  # [MHz] +40
        #     "steps": 500,#500,
        #     "relax_delay": 200,  # [us]
        #     "list_of_all_qubits": list_of_all_qubits,
        #     "gain_start": gain_start,
        #     "gain_stop": gain_stop,
        #     "gain_steps": int(round(gain_steps/4)),
        #     "readout_pulse_delay": 5,
        # },

        "qubit_spec_ge_dual_stark_drive": {
            "reps": 1000,
            "rounds": 1,
            "start": list(VNA_qubit - 1),  # [MHz] -40
            "stop": list(VNA_qubit + 1),  # [MHz] +40
            "steps": 100,#500
            "relax_delay": 100,  # [us]
            "qubit_stark_pulse_length":20,
            "qubit_pulse_length": 5,
            "qubit_pulse_gain": 0.005,
            "qubit_stark_freq": 3095.43+2,
            "start_qubit_stark_gain": 0,
            "stop_qubit_stark_gain": 0.5,#0.01,
            "stark_gain_steps": 3,
            "tof_readout_length": 30,  # [us] decimated capture window for the TOF stark-pulse check
                                       # (spans qubit pulse + 25us stark pulse + readout). The decimated
                                       # ADC buffer is limited (~16384 samples / ~29us), so this is
                                       # auto-clamped to the buffer max at runtime; the full stark pulse
                                       # still fits. Lower it if you want a shorter capture.
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
            "steps": 4000,  # 100
            "relax_delay": 10,  # 4000, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "qubit_spec_ge_high_gain": {
            "reps": 50,  # 300
            "rounds": 500,  # 10
            "start": list(VNA_qubit - 30),  # [MHz]
            "stop": list(VNA_qubit + 30),  # [MHz]
            "steps": 600,  # 100
            "relax_delay": 10,  # 4000, # [us]
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
            "relax_delay": 700,  # 4000, # [us]
            "list_of_all_qubits": list_of_all_qubits,
            "qze_mask": [],
        },

        "qubit_spec_ftores": {
            "reps": 10000,  # 300
            "rounds": 1,  # 10
            "start": list((VNA_qubit + ef_freqs) - VNA_res - 200),  # [MHz] #-300
            "stop":  list((VNA_qubit + ef_freqs) - VNA_res + 200),  # [MHz]
            "steps": 4000,  # 1000
            "relax_delay": 0.5,  # 4000,  # [us]
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
            "reps": 200,  # 300,
            "rounds": 1,  # 10,
            "start": 0,  # [DAC units]
            "stop": 0.85,  # [DAC units], just beyond the expected pi point
            "steps": 86,
            "relax_delay": 4000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },


        "power_rabi_ge_chevron": {
            "reps": 200,
            "rounds": 1,
            "start": 0,  # [DAC units]
            "stop": 1,  # [DAC units]
            "steps": 70,
            "relax_delay": 4000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
            "start_freq": VNA_qubit[5] -4,
            "end_freq": VNA_qubit[5] + 4,
            "freq_steps": 25
        },

        "power_rabi_ge_zeno": {
            "reps": 50,
            "rounds": 5,
            "start": 0,  # [DAC units]
            "stop": 0.085,  # [DAC units]
            "steps": 70,
            "relax_delay": 4000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "power_rabi_vs_gain": {
            "start_gain": 0.000001,
            "end_gain": 0.1,
            "gain_steps": 3,
            "reps": 50,
            "rounds": 5,
            "start": 0,  # [DAC units]
            "stop": 0.085*6,  # [DAC units] get many fringes, 6/2
            "steps": 70,
            "relax_delay": 4000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "length_rabi_ge": {
            "reps": 200,
            "rounds": 1,
            "start": 0.01,  # [us]
            "stop": 20,#[0.7] * 6,  # [us]
            "steps": 50,
            "relax_delay": 4000,# [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "length_rabi_ge_chevron": {
            "reps": 100,
            "rounds": 3,
            "start": 0.01,  # [us]
            "stop": 2,  # [0.7] * 6,  # [us]
            "steps": 100,
            "relax_delay": 4000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
            "start_freq":VNA_qubit[5] - 15,
            "end_freq":VNA_qubit[5] + 15,
            "freq_steps":100
        },
        "length_rabi_vs_gain": {
            "reps": 300,
            "rounds": 5,
            "start": 0.01,  # [us]
            "stop": 2,  # [0.7] * 6,  # [us]
            "steps": 100,
            "relax_delay": 4000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
            "start_gain":0.01,
            "end_gain":0.5,
            "gain_steps":20
        },

        "length_rabi_Qtemps": {
            "reps": 10,#500,
            "rounds": 400,
            "start": 0.02,  # [us]
            "stop": 2,#[0.7] * 6,  # [us]
            "steps": 200,
            "relax_delay": 4000,# [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "length_rabi_ge_pi_len": { #for the zeno tests, bare qubit frequency rabi
            "reps": 10,  # 500,
            "rounds": 300,
            "start": 0.01,  # [us]
            "stop": 4,  # [0.7] * 6,  # [us]
            "steps": 60,
            "relax_delay": 4000,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "length_rabi_ge_qze": {
            "reps": 500,
            "rounds": 1, #600
            "start": 0.01,  # [us]
            "stop": 10,   #[3] * 6,   # [us]
            "steps": 150,  # 140,
            "relax_delay": 4000,# [us]
            "list_of_all_qubits": list_of_all_qubits,
            "qze_mask": [],
            "zeno_pulse_width": 0.007,
            "zeno_pulse_period": 0.10,
        },

        "power_rabi_ef": {
            "reps": 200,
            "reps2": 850, #this is only used for the experiment that uses e-f rabi to calculate qubit temperatures.
            "rounds": 20,
            "start": 0.0,  # [DAC units]
            "stop": 1.0,  # [DAC units]
            "steps": 30,
            "relax_delay": 4000,  # [us]
        },
        "power_rabi_fh": {
            "reps": 200,
            "reps2": 850,  # this is only used for the experiment that uses e-f rabi to calculate qubit temperatures.
            "rounds": 1,
            "start": 0.0,  # [DAC units]
            "stop": 1.0,  # [DAC units]
            "steps": 30,
            "relax_delay": 4000,  # [us]
        },
        "T1_ge": {
            "reps": 800,
            "rounds": 1, #1
            "start":  0.01,  # [us]
            "stop": [100, 60, 150, 50, 40, 15],  # [us], M1..M6
            "steps": [201, 121, 301, 101, 161, 61],  # <=0.5 us spacing
            "relax_delay": 250,  # [us], >7x the longest measured PUCQ4 T1
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "T1_IBM_qze": {
            "reps": 300,
            "rounds": 1,#500,
            "start": 30,  # [us]
            "stop": 31,  # [250.0] * 6,  # [us] ### Should be ~10x T1! Should change this per qubit.
            "steps": 1,
            "relax_delay": 4000,  # [us] ### Should be >10x T1!
            "wait_time": 30,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
            "gain_start": gain_start,
            "gain_stop": gain_stop,
            "gain_steps": gain_steps,
        },

        "T1_IBM_qze_loop": {
            "reps": 600,#100,
            "rounds": 1,#20,
            "start": 0.01,  # [us]
            "stop": 100,  # [250.0] * 6,  # [us] ### Should be ~10x T1! Should change this per qubit.
            "steps": 30,#200
            "relax_delay": 4000,  # [us] ### Should be >10x T1!
            "wait_time": 30,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
            "gain_start": gain_start,
            "gain_stop": gain_stop,
            "gain_steps": gain_steps,
            "readout_pulse_delay": 5,
        },

        "T1_fe": {
            "reps": 20,  # 300
            "rounds": 300,  # 1
            "start": 0.0,  # [us]
            "stop": 200,  # [250.0] * 6,  # [us] ### Should be ~10x T1! Should change this per qubit.
            "steps": 150,
            "relax_delay": 4000,  # [us] ### Should be >10x T1!
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "T1_fg": {
            "reps": 20,  # 300
            "rounds": 300,  # 1
            "start": 0.0,  # [us]
            "stop": 200,  # [250.0] * 6,  # [us] ### Should be ~10x T1! Should change this per qubit.
            "steps": 150,
            "relax_delay": 4000,  # [us] ### Should be >10x T1!
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "ckp_nbar_calibration": {
            "reps": 2000,
            "rounds": 1,
            "list_of_all_qubits": list_of_all_qubits,

            "gain_steps": 30,
            "start_gain": 0,
            "end_gain": 0.04,

            "res_freq_steps": 100,
            "res_freq_start": 7287.57+0.5,
            "res_freq_stop": 7287.57-0.7,

            "start_freq": -10, #-6.5,
            "end_freq": 5,
            "qubit_pulse_steps": 100,


            "qubit_pulse_delay": 6,
            "readout_pulse_delay": 5,

            "ckp_gain": [],
            "res_freq_ckp": [],
            "res_phase_ckp": [],
            "ckp_mask": [],

            "relax_delay": 2000,
        },
        "res_stark_shift_2D": {
            "reps": 30,
            "rounds": 150,
            "list_of_all_qubits": list_of_all_qubits,
            "qubit_pulse_delay": 5, #[us] time to wait for qubit pulse after stark tone is sent
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
            "qubit_pulse_delay": 5,  # [us] time to wait for qubit pulse after stark tone is sent
            "stark_length": 16,  # [us] stark tone length for 2D scan, overlaps qubit pulse
            "gain_steps": 20,
            "start_gain": 0.00,
            "end_gain": 1.0,  # res gain values between -1 and 1
            "qubit_pulse_steps": 400,
            "min_freq": 3,  # [MHz] from qubit freq
            "max_freq": 15,  # [MHz] from qubit freq
            "start_freq": 0,
            "end_freq": 0,
            "readout_pulse_delay": 5,  # 2/kappa
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
            "readout_pulse_delay": 5, #2/kappa
            "relax_delay": 4000, #[us]
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
            "steps": 100,
            "start_gain": 0.0,
            "end_gain": 0.012,
            "gain_steps": 3,
            "ramsey_freq": 1.5, #1.5 [MHz]
            "relax_delay": 2000,
            "wait_time": 0.0,  # [us]
            "stark_gain": 0.0,
            "detuning": -20, # [MHz]
            "stark_sigma": 0.01, # [us] 10 ns
            "list_of_all_qubits": list_of_all_qubits,
            "readout_pulse_delay":5,
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
            "reps": 1000,
            "rounds": 1,
            "start": 0.0, # [us]
            "stop": [60, 40, 80, 50, 15, 10],  # [us], M1..M6
            "steps": [301, 201, 401, 251, 151, 101],  # >=10 points/fringe
            "ramsey_freq": 0.5,  # [MHz]
            "relax_delay": 250, # [us], >7x the longest measured PUCQ4 T1
            "wait_time": 0.0, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "Ramsey_ge_zeno": {
            "reps": 100,
            "rounds": 1,
            "start": 0.01,  # [us]
            "stop": 20,  # [us]
            "steps": 200,
            "ramsey_freq": 0.5,  #.15 [MHz]
            "relax_delay": 4000,
            # [us] the time to wait to let the qubit to relax to gnd again after exciting it (make it way above T1)
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
            "gain_start": gain_start,
            "gain_stop": gain_stop,
            "gain_steps": gain_steps,
            "readout_pulse_delay": 5,
        },

        "Ramsey_ge_correction": {
            "reps": 20,#20,  # 300
            "rounds": 2,  # 10
            "start": 0.0,  # [us]
            "stop": 1,  # [us]
            "steps": 50,
            "ramsey_freq": 4,  # [MHz]
            "relax_delay": 4000,
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
            "relax_delay": 4000,
            # [us] the time to wait to let the qubit to relax to gnd again after exciting it (make it way above T1)
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

        "SpinEcho_ge": {
            "reps": 1000,
            "rounds": 1,
            "start": 0.01, # [us]
            "stop": [100, 60, 150, 50, 30, 15],  # [us], M1..M6
            "steps": [501, 301, 751, 251, 301, 151],  # >=10 points/fringe
            "ramsey_freq": 0.5,  # [MHz]
            "relax_delay": 250, # [us], >7x the longest measured PUCQ4 T1
            "wait_time": 0.0, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },
        "SpinEcho_ge_zeno": {
            "reps": 100,  # 20,
            "rounds": 1,
            "start": 0.01,  # [us]
            "stop": 20,  # [us]
            "steps": 100,
            "ramsey_freq": 0.5,  # 0.12 [MHz]
            "relax_delay": 4000,  # [us]
            "wait_time": 0.0,  # [us]
            "list_of_all_qubits": list_of_all_qubits,
            "gain_start": gain_start,
            "gain_stop": gain_stop,
            "gain_steps": gain_steps,
            "readout_pulse_delay": 5,
        },


    #
    #     "res_spec_ef": {
    #         "reps": 100,
    #         "py_avg": 10,
    #         "start": [7148, 0, 7202, 0, 0, 0], # [MHz]
    #         "stop":  [7151, 0, 7207, 0, 0, 0], # [MHz]
    #         "steps": 200,
    #         "relax_delay": 4000, # [us]
    #         "list_of_all_qubits": list_of_all_qubits,
    #     },
    #
    #     "qubit_spec_ef": {
    #         "reps": 100,
    #         "py_avg": 10,
    #         "start": [2750, 0, 0, 0, 0, 0], # [MHz]
    #         "stop":  [2850, 0, 0, 0, 0, 0], # [MHz]
    #         "steps": 500,
    #         "relax_delay": 4000, # [us]
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
    #         "reps": 4000,
    #         "py_avg": 10,
    #         "start": [0.0] * 6, # [DAC units]
    #         "stop":  [1.0] * 6, # [DAC units]
    #         "steps": 100,
    #         "relax_delay": 4000, # [us]
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
    #         "relax_delay": 4000, # [us]
    #         "wait_time": 0.0, # [us]
    #         "list_of_all_qubits": list_of_all_qubits,
    #     },
    #
        "IQ_plot":{
            "steps": 5000, # shots
            "py_avg": 1,
            "reps": 1,
            "relax_delay": 4000, # [us]
            "SS_ONLY": False,
            "list_of_all_qubits": list_of_all_qubits,
        },
    # #

        "Readout_Optimization":{
            "steps": 1000, # shots
            "py_avg": 1,
            "gain_start" : [0, 0, 0, 0],
            "gain_stop" : [1, 0, 0, 0],
            "gain_step" : 0.1,
            "freq_start" : [6176.0, 0, 0, 0],
            "freq_stop" : [6178.0, 0, 0, 0],
            "freq_step" : 0.1,
            "relax_delay": 250,  # [us], >5x the longest measured PUCQ4 T1
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
            "relax_delay": 4000,  # 600, # [us]
            "list_of_all_qubits": list_of_all_qubits,
        },

    }


else:
    print('Please set variable FRIDGE to QUIET, or configure for your fridge as needed')


def _set_pucq4_indexed_bounds(experiment_name, qubit_index, start_mhz, stop_mhz):
    """Replace one row of a six-qubit frequency sweep without changing shape."""
    config = expt_cfg[experiment_name]
    starts = list(config["start"])
    stops = list(config["stop"])
    if len(starts) != tot_num_of_qubits or len(stops) != tot_num_of_qubits:
        raise RuntimeError("%s does not contain six-qubit frequency bounds" % experiment_name)
    starts[qubit_index] = float(start_mhz)
    stops[qubit_index] = float(stop_mhz)
    config["start"] = starts
    config["stop"] = stops


def apply_pucq4_flux_experiment_config(calibration_by_qubit):
    """Recenter all f_ge experiment sweeps for selected Q4/Q6 bias rows.

    ``calibration_by_qubit`` is the mapping returned by
    ``system_config.get_pucq4_flux_calibration``.  Keeping this function in
    expt_config avoids importing hardware-facing system_config here.
    """
    validated = {}
    for raw_index, calibration in calibration_by_qubit.items():
        qubit_index = int(raw_index)
        if qubit_index not in (4, 5):
            raise ValueError("Only PUCQ4 indices 4 (Yoko 3) and 5 (Yoko 4) are flux tunable")
        qfreq_mhz = float(calibration["qfreq_MHz"])
        res_base_mhz = float(calibration["res_base_MHz"])
        if not np.isfinite(qfreq_mhz) or not np.isfinite(res_base_mhz):
            raise ValueError("PUCQ4 experiment frequencies must be finite")
        validated[qubit_index] = (qfreq_mhz, res_base_mhz)

    for qubit_index, (qfreq_mhz, res_base_mhz) in validated.items():
        VNA_qubit[qubit_index] = qfreq_mhz
        VNA_res[qubit_index] = res_base_mhz

        tracking_half_span = (
            500.0 if WIDE_QUBIT_SCAN else float(QSPEC_HALF_SPAN[qubit_index])
        )
        _set_pucq4_indexed_bounds(
            "qubit_spec_ge", qubit_index,
            qfreq_mhz - tracking_half_span, qfreq_mhz + tracking_half_span,
        )
        for experiment_name, lower_offset, upper_offset in (
                ("qubit_spec_ge_zeno", -8.0, 1.0),
                ("qubit_spec_ge_dual_stark_drive", -1.0, 1.0),
                ("qubit_spec_ge_extended", -250.0, 100.0),
                ("qubit_spec_ge_high_gain", -30.0, 30.0),
                ("qubit_spec_ge_zeno_stark", -25.0, 5.0),
                ("bias_qubit_spec_ge", -70.0, 70.0)):
            _set_pucq4_indexed_bounds(
                experiment_name, qubit_index,
                qfreq_mhz + lower_offset, qfreq_mhz + upper_offset,
            )

        ftores_center_mhz = qfreq_mhz + float(ef_freqs[qubit_index]) - res_base_mhz
        _set_pucq4_indexed_bounds(
            "qubit_spec_ftores", qubit_index,
            ftores_center_mhz - 200.0, ftores_center_mhz + 200.0,
        )

        # These two legacy chevron configs are scalar and explicitly target M6.
        if qubit_index == 5:
            expt_cfg["power_rabi_ge_chevron"]["start_freq"] = qfreq_mhz - 4.0
            expt_cfg["power_rabi_ge_chevron"]["end_freq"] = qfreq_mhz + 4.0
            expt_cfg["length_rabi_ge_chevron"]["start_freq"] = qfreq_mhz - 15.0
            expt_cfg["length_rabi_ge_chevron"]["end_freq"] = qfreq_mhz + 15.0

    return {
        "qubit_freq_ge": VNA_qubit.tolist(),
        "res_freq_ge": VNA_res.tolist(),
    }


def pucq4_round_robin_profile(qubit_index, calibration_by_qubit):
    """Persistent acquisition settings from the accepted six-row/flux runs.

    Pulse/readout values remain in system_config. These are the previously
    missing averaging and sampling settings, in us/MHz/DAC gain. Return fresh
    scalar dictionaries so one row cannot inherit the preceding row's sweep.

    Resonator spectroscopy uses expt_cfg["res_spec"] directly. Do not include
    it here: applying this profile before each row would overwrite user edits
    to its sweep and averaging settings (including WIDE_RES_SCAN).
    """
    if qubit_index not in range(6):
        raise ValueError("PUCQ4 row index must be 0 through 5")
    i = qubit_index
    profile = {
        "qubit_spec_ge": dict(half_span_MHz=0.8, steps=301, reps=5000,
                              rounds=1, relax_delay=100.0),
        "power_rabi_ge": dict(start=0.0, stop=0.85, steps=171, reps=1200,
                              rounds=1, relax_delay=250.0),
        "Readout_Optimization": dict(steps=10000, relax_delay=250.0),
        "T1_ge": dict(start=0.01, stop=(100, 60, 150, 5, 60, 30)[i],
                      steps=(201, 121, 301, 201, 241, 201)[i],
                      reps=1200, rounds=1, relax_delay=250.0),
        "Ramsey_ge": dict(start=0.0, stop=(60, 40, 80, 3, 30, 10)[i],
                          steps=(301, 201, 401, 201, 301, 201)[i],
                          reps=1200, rounds=1, relax_delay=250.0,
                          ramsey_freq=0.5, fit_frequency_guess_MHz=None,
                          fit_population_guess=None),
        "SpinEcho_ge": dict(start=0.01, stop=(100, 60, 150, 5, 60, 30)[i],
                            steps=(501, 301, 751, 201, 301, 301)[i],
                            reps=1200, rounds=1, relax_delay=250.0,
                            ramsey_freq=0.5),
    }
    if i == 2:
        # Historical M3 profile; fresh zero-current settings override it below.
        profile["qubit_spec_ge"].update(half_span_MHz=0.25, steps=201, reps=10000)
        profile["Readout_Optimization"].update(steps=20000)
    if i in (4, 5):
        current = calibration_by_qubit[i]["current_mA"]
        profile["qubit_spec_ge"].update(half_span_MHz=1.0, steps=401)
        if i == 4 and current == 5.0:
            # The prior 0.869 MHz line needs baseline on both sides of the peak.
            profile["qubit_spec_ge"].update(half_span_MHz=2.0)
        if i == 5 and current == -7.5:
            # The tentative archived line is 0.61 MHz above the pulse seed.
            profile["qubit_spec_ge"].update(half_span_MHz=2.0)
        for name in ("power_rabi_ge", "T1_ge", "Ramsey_ge", "SpinEcho_ge"):
            profile[name].update(reps=1200, rounds=2)
        if current != 0.0:
            profile["T1_ge"].update(stop=60.0, steps=241)
            profile["Ramsey_ge"].update(stop=30.0, steps=301)
            profile["SpinEcho_ge"].update(stop=60.0, steps=301)
        if i == 5 and current == -2.5:
            # Full RR plus exact SSF repeat, September 9: retained lower contrast.
            profile["qubit_spec_ge"].update(steps=101, reps=40000)
            profile["Readout_Optimization"].update(steps=50000)
            for name in ("Ramsey_ge", "SpinEcho_ge"):
                profile[name].update(reps=2400)
        if i == 5 and current == 10.0:
            # Measured midpoint frame: full-window Ramsey diagnostic passes.
            profile["Ramsey_ge"].update(ramsey_freq=0.015398310214337,
                                         fit_frequency_guess_MHz=0.04,
                                         fit_population_guess=dict(phase=0.0, amp=1.0,
                                                                   initial_offset=0.5, final_offset=0.5))
        if i == 5 and current == -10.0:
            # The archived 500-rep spectrum did not resolve this line reliably.
            profile["qubit_spec_ge"].update(reps=20000)
            for name in ("T1_ge", "Ramsey_ge", "SpinEcho_ge"):
                profile[name].update(reps=2000, rounds=4)
    # Historical September profiles; fresh all-six zero-current block overrides below.
    if i == 3 or (i == 5 and calibration_by_qubit[i]["current_mA"] == 0.0):
        profile["qubit_spec_ge"].update(
            half_span_MHz=1.2 if i == 3 else 0.5,
            reps=128000 if i == 3 else 64000, steps=97 if i == 3 else 101)
        profile["Readout_Optimization"].update(steps=50000, relax_delay=100.0)
        profile["power_rabi_ge"].update(reps=5000, rounds=1, relax_delay=100.0)
        for name in ("T1_ge", "Ramsey_ge", "SpinEcho_ge"):
            profile[name].update(reps=(6000 if name == "T1_ge" else 8000)
                                 if i == 3 else 5000,
                                 rounds=1, relax_delay=100.0)
            if name != "T1_ge":
                profile[name]["ramsey_freq"] = 2.0 if i == 3 else 1.0
    # Fresh zero-current full RR settings, 2026-10-29; preserve nonzero profiles.
    if i < 4 or calibration_by_qubit[i]["current_mA"] == 0.0:
        profile["qubit_spec_ge"].update(half_span_MHz=(0.45, 0.45, 0.45, 0.6, 0.6, 0.6)[i])
        profile["qubit_spec_ge"].update(steps=(151, 151, 151, 241, 151, 151)[i])
        profile["qubit_spec_ge"].update(reps=(2000, 2000, 2000, 5000, 2000, 6000)[i])
        profile["qubit_spec_ge"].update(rounds=1)
        profile["qubit_spec_ge"].update(relax_delay=100.0)
        profile["power_rabi_ge"].update(start=0.0)
        profile["power_rabi_ge"].update(stop=0.85)
        profile["power_rabi_ge"].update(steps=171)
        profile["power_rabi_ge"].update(reps=2000)
        profile["power_rabi_ge"].update(rounds=1)
        profile["power_rabi_ge"].update(relax_delay=(250.0, 100.0, 250.0, 100.0, 100.0, 100.0)[i])
        profile["Readout_Optimization"].update(steps=50000)
        profile["Readout_Optimization"].update(relax_delay=(250.0, 100.0, 250.0, 100.0, 100.0, 100.0)[i])
        profile["T1_ge"].update(start=0.01)
        profile["T1_ge"].update(stop=(100.0, 40.0, 150.0, 30.0, 50.0, 25.0)[i])
        profile["T1_ge"].update(steps=(201, 161, 201, 121, 201, 161)[i])
        profile["T1_ge"].update(reps=(1200, 2400, 1200, 1200, 2400, 2400)[i])
        profile["T1_ge"].update(rounds=(1, 1, 1, 1, 2, 1)[i])
        profile["T1_ge"].update(relax_delay=(250.0, 100.0, 250.0, 250.0, 100.0, 150.0)[i])
        profile["Ramsey_ge"].update(start=0.0)
        profile["Ramsey_ge"].update(stop=(60.0, 25.0, 80.0, 40.0, 12.0, 15.0)[i])
        # M1 and M3 steps raised 2026-11-25 (were 201 and 268) so that every row
        # has at least eight acquired points per fringe (0.2 us step at 0.5 MHz).
        profile["Ramsey_ge"].update(steps=(301, 251, 401, 201, 241, 301)[i])
        profile["Ramsey_ge"].update(reps=(1200, 2400, 1200, 1200, 2400, 2400)[i])
        profile["Ramsey_ge"].update(rounds=(1, 1, 1, 1, 2, 1)[i])
        profile["Ramsey_ge"].update(relax_delay=(250.0, 100.0, 250.0, 250.0, 100.0, 150.0)[i])
        profile["Ramsey_ge"].update(ramsey_freq=(0.5, 1.0, 0.5, 0.5, 2.0, 1.0)[i])
        profile["SpinEcho_ge"].update(start=0.01)
        profile["SpinEcho_ge"].update(stop=(100.0, 50.0, 150.0, 60.0, 40.0, 30.0)[i])
        # M1 and M3 steps raised 2026-11-25 (were 251 and 376), same reason.
        profile["SpinEcho_ge"].update(steps=(501, 251, 751, 301, 401, 301)[i])
        profile["SpinEcho_ge"].update(reps=(1200, 2400, 1200, 1200, 2400, 2400)[i])
        profile["SpinEcho_ge"].update(rounds=(1, 1, 1, 1, 2, 1)[i])
        profile["SpinEcho_ge"].update(relax_delay=(250.0, 100.0, 250.0, 250.0, 100.0, 150.0)[i])
        profile["SpinEcho_ge"].update(ramsey_freq=(0.5, 0.5, 0.5, 0.5, 1.0, 1.0)[i])
    # User-requested Q2 SSF tracking: shorter snapshots at the measured pulse.
    # Explicit calibration runs can still request larger SSF reference samples.
    if i == 1:
        profile["Readout_Optimization"].update(steps=2000)
    return profile


def apply_pucq4_round_robin_profile(qubit_index, calibration_by_qubit, qfreq_mhz):
    """Apply a row before constructing any experiment; no hardware calls."""
    profile = pucq4_round_robin_profile(qubit_index, calibration_by_qubit)
    for name, settings in profile.items():
        for key, value in settings.items():
            if key == "half_span_MHz":
                _set_pucq4_indexed_bounds(name, qubit_index,
                                          qfreq_mhz - value, qfreq_mhz + value)
            elif isinstance(expt_cfg[name].get(key), (list, tuple, np.ndarray)):
                values = list(expt_cfg[name][key])
                values[qubit_index] = value
                expt_cfg[name][key] = values
            else:
                expt_cfg[name][key] = value
    return profile
