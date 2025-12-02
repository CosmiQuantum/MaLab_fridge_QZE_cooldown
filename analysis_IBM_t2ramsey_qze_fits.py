from analysis_006_T1_vs_time_plots import T1VsTime
from analysis_007_T2R_vs_time_plots import T2rVsTime
from analysis_008_T2E_vs_time_plots import T2eVsTime
from analysis_003_q_freqs_vs_time_plots import QubitFreqsVsTime

# Configuration
save_figs = True
save_individual_qspec = True  # Set to False to skip saving individual qspec plots
nbar_from_ramsey = True
figure_quality = 100
final_figure_quality = 200
FRIDGE = "QUIET"
qubits = [4]
path = '2d_less_reps_more_qspec_steps'
ramsey_nbar_path='M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/ramsey_n_bar_calibration/debug/2025-12-01_13-51-16'

for qubit in qubits:
    run_name = f'bob_run_started_Aug_23/squill/{path}/all_qubits/'
    top_folder_dates = [f'qubit_{qubit}round{round}' for round in range(10)]
    
    # QSpec analysis
    q_vs_time = QubitFreqsVsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs,
                                 False, 'None', run_name, fridge=FRIDGE, exp_name='ge', qubit=qubit)
    _, _, amps_qspec, gains_qspec, rounds_qspec, freqs_qspec = q_vs_time.run_q_sweep_new(exp_extension='_ge', scaling=True)

    # Calculate nbar (no plotting)
    if nbar_from_ramsey:
        from section_008_save_data_to_h5 import Data_H5
        from T2R_stark import starkT2RMeasurement
        import glob
        import os
        import numpy as np
        import re
        import datetime
        
        # Define helper to process h5 data (copied/adapted from T2rVsTime)
        def process_h5_data(data):
            if isinstance(data, bytes):
                data_str = data.decode()
            elif isinstance(data, str):
                data_str = data
            else:
                return []
            cleaned_data = ''.join(c for c in data_str if c.isdigit() or c in ['-', '.', ' ', 'e', '+'])
            numbers = []
            for x in cleaned_data.split():
                if x:
                    try:
                        numbers.append(float(x))
                    except ValueError:
                        continue
            return numbers

        # Load data
        outerFolder_expt = os.path.join(ramsey_nbar_path, "Data_h5", "StarkRamsey")
        h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
        
        # Sort files by time
        TS = re.compile(r'(\d{4})[-_\.]?(\d{2})[-_\.]?(\d{2})[ Tt_-]?(\d{2})[-_\.]?(\d{2})[-_\.]?(\d{2})')
        def dt_from_name(path):
            name = os.path.basename(path)
            m = TS.search(name)
            if not m:
                return datetime.datetime.min
            y, mo, d, h, mi, s = map(int, m.groups())
            return datetime.datetime(y, mo, d, h, mi, s)
        h5_files = sorted(h5_files, key=dt_from_name)

        f_est_list = []
        f_err_list = []
        gains_list = []
        
        # Initialize T2R measurement for fitting
        # We need a dummy instance to access t2_fit and plot_stark_shift
        t2r_instance = starkT2RMeasurement(qubit, 6, ramsey_nbar_path, 0, 'None', True)
        
        for h5_file in h5_files:
            try:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                if '(' in save_round:
                    save_round = save_round.split('(')[0]
                
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type='StarkRamsey', save_r=int(save_round))
                
                q_key = f'Q{qubit}'
                if q_key not in load_data['StarkRamsey']:
                    continue
                    
                # Iterate through datasets in the file
                dates = load_data['StarkRamsey'][q_key].get('Dates', [])[0]
                for i in range(len(dates)):
                    if 'nan' in str(dates[i]):
                        continue
                        
                    I = process_h5_data(load_data['StarkRamsey'][q_key].get('I', [])[0][i].decode())
                    Q = process_h5_data(load_data['StarkRamsey'][q_key].get('Q', [])[0][i].decode())
                    delay_times = process_h5_data(load_data['StarkRamsey'][q_key].get('Delay Times', [])[0][i].decode())
                    
                    # Get gain from config
                    try:
                        syst_config = load_data['StarkRamsey'][q_key].get('Syst Config', [])[0][i].decode()
                        match = re.search(r"'stark_gain':\s*([\d\.]+)", syst_config)
                        if match:
                            gain = float(match.group(1))
                        else:
                            match = re.search(r"'res_gain_qze':\s*([\d\.]+)", syst_config)
                            if match:
                                gain = float(match.group(1))
                            else:
                                continue 
                    except:
                        continue

                    # Fit Ramsey
                    if len(I) > 0 and len(delay_times) > 0:
                        _, _, _, f_est, f_err, _ = t2r_instance.t2_fit(delay_times, I, Q, verbose=False)
                        f_est_list.append(f_est)
                        f_err_list.append(f_err)
                        gains_list.append(gain)
                        
            except Exception as e:
                print(f"Error processing {h5_file}: {e}")
                continue

        # Convert to arrays and sort
        gains_arr = np.array(gains_list)
        f_est_arr = np.array(f_est_list)
        f_err_arr = np.array(f_err_list)
        
        sort_inds = np.argsort(gains_arr)
        gains_arr = gains_arr[sort_inds]
        f_est_arr = f_est_arr[sort_inds]
        f_err_arr = f_err_arr[sort_inds]
        
        # Setup config for plot_stark_shift
        t2r_instance.config = {
            'chi': {qubit: -0.137}, 
            'anharmonicity': {qubit: -200}, 
            'detuning': 0,
            'qubit_freq_ge': 0
        }
        
        # Calculate nbar calibration curve
        nbar_per_gain = t2r_instance.plot_stark_shift(gains_arr, f_est_arr, f_err_arr)
        
        # Map gains to nbar for the main experiment
        n_bars = {}
        for q in gains_qspec:
            n_bars[q] = []
            for gain_list in gains_qspec[q]:
                # Handle nested lists or single values
                if isinstance(gain_list, (list, np.ndarray)):
                    vals = np.array(gain_list)
                    n_bars[q].append(np.interp(vals, gains_arr, nbar_per_gain).tolist())
                else:
                    val = float(gain_list)
                    n_bars[q].append(float(np.interp(val, gains_arr, nbar_per_gain)))

    else:
        n_bars = q_vs_time.calculate_nbar(amps_qspec, gains_qspec, rounds_qspec, freqs_qspec, chi_MHz=-0.137)

    q_vs_time.plot_all_q_heatmaps_new_format(amps_qspec, gains_qspec, rounds_qspec, freqs_qspec,
                                              f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/',
                                              n_bar=n_bars, save_individual_plots=save_individual_qspec)

    # T1 analysis
    t1_vs_time = T1VsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs, False,
                         'None', run_name, FRIDGE, exp_name='ge', qubit=qubit, t1_slice='10us')
    _, _, amps_t1, gains_t1, rounds_t1, delay_times_t1 = t1_vs_time.run_t1_sweep_new(exp_extension='_ge', scaling=True, weighted_mean=True)

    t1_vs_time.fit_and_save_t1_slices_new_format(amps_t1, gains_t1, rounds_t1, delay_times_t1,
                                                  f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/', 
                                                  n_bar=n_bars)
    t1_vs_time.plot_all_t1_heatmaps_new_format(amps_t1, gains_t1, rounds_t1, delay_times_t1,
                                                f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/', 
                                                n_bar=n_bars, use_linear_x=False)

    # T2R analysis
    t2_vs_time = T2rVsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs, False,
                           'None', run_name, fridge=FRIDGE, exp_name='ge', qubit=qubit)
    _, _, amps_t2, gains_t2, rounds_t2, delay_times_t2 = t2_vs_time.run_t2_sweep_new(exp_extension='_ge', scaling=True)

    t2_vs_time.plot_all_t2_heatmaps_new_format(amps_t2, gains_t2, rounds_t2, delay_times_t2,
                                                f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/', 
                                                n_bar=n_bars, use_linear_x=False)

    t2_vs_time.plot_all_t2_curves(amps_t2, gains_t2, rounds_t2, delay_times_t2,
                                   f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/', 
                                   n_bar=n_bars)

    # T2E analysis
    t2e_vs_time = T2eVsTime(figure_quality, final_figure_quality, 6, top_folder_dates, save_figs,
                           False, 'None', run_name, fridge=FRIDGE, exp_name='ge', qubit=qubit)
    _, _, amps_t2e, gains_t2e, rounds_t2e, delay_times_t2e = t2e_vs_time.run_t2_sweep_new(exp_extension='_ge', scaling=True)
    
    t2e_vs_time.plot_all_t2_heatmaps_new_format(amps_t2e, gains_t2e, rounds_t2e, delay_times_t2e,
                                               f'M:/_Data/20250822 - Olivia/bob_run_started_Aug_23/squill/{path}/all_qubits/analysis/', 
                                               n_bar=n_bars, save_individual_plots=save_individual_qspec)
