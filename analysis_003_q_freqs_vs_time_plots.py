import numpy as np
import os
import sys
sys.path.append(os.path.abspath("/home/qubituser/Documents/GitHub/tprocv2_demos/qick_tprocv2_experiments_mux/"))

from section_002_res_spec_ge_mux import ResonanceSpectroscopy
from section_004_qubit_spec_ge import QubitSpectroscopy
from section_006_amp_rabi_ge import AmplitudeRabiExperiment
from section_007_T1_ge import T1Measurement
from section_008_save_data_to_h5 import Data_H5
from section_009_T2R_ge import T2RMeasurement
from section_010_T2E_ge import T2EMeasurement
#from expt_config import *
import glob
import re
import datetime
import ast
import os
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter
from scipy.stats import norm
from scipy.optimize import curve_fit

class QubitFreqsVsTime:
    def __init__(self, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name,  fridge, exp_name='ge', qubit=0):
        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.qubit=qubit
        self.exp_name=exp_name
        self.signal = signal
        self.figure_quality = figure_quality
        self.run_name = run_name
        self.number_of_qubits = number_of_qubits
        self.final_figure_quality = final_figure_quality
        self.top_folder_dates = top_folder_dates
        self.fridge = fridge

    def datetime_to_unix(self, dt):
        # Convert to Unix timestamp
        unix_timestamp = int(dt.timestamp())
        return unix_timestamp

    def unix_to_datetime(self, unix_timestamp):
        # Convert the Unix timestamp to a datetime object
        dt = datetime.fromtimestamp(unix_timestamp)
        return dt

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def exponential(self, x, a, b, c, d):
        return a * np.exp(-(x - b) / c) + d

    def optimal_bins(self, data):
        n = len(data)
        if n == 0:
            return {}
        # Sturges' Rule
        sturges_bins = int(np.ceil(np.log2(n) + 1))
        return sturges_bins

    def process_string_of_nested_lists(self, data):
        # Remove extra whitespace and non-numeric characters.
        data = re.sub(r'\s*\[(\s*.*?\s*)\]\s*', r'[\1]', data)
        data = data.replace('[ ', '[')
        data = data.replace('[ ', '[')
        data = data.replace('[ ', '[')

        cleaned_data = ''.join(c for c in data if c.isdigit() or c in ['-', '.', ' ', 'e', '[', ']'])
        pattern = r'\[(.*?)\]'  # Regular expression to match data within brackets
        matches = re.findall(pattern, cleaned_data)
        result = []
        for match in matches:
            numbers = [float(x.strip('[').strip(']').replace("'", "").replace(" ", "").replace("  ", "")) for x in match.split()] # Convert strings to integers
            result.append(numbers)

        return result

    def process_h5_data(self, data):
        if isinstance(data, bytes):
            data_str = data.decode()
        elif isinstance(data, str):
            data_str = data
        else:
            raise ValueError("Unsupported data type. Data should be bytes or string.")

        # Keep characters for scientific notation
        cleaned_data = ''.join(c for c in data_str if c.isdigit() or c in ['-', '.', ' ', 'e', '+'])
        
        # Split and convert, skipping any invalid tokens
        numbers = []
        for x in cleaned_data.split():
            if x:
                try:
                    numbers.append(float(x))
                except ValueError:
                    continue
        return numbers

    def string_to_float_list(self, input_string):
        try:
            # Remove 'np.float64()' parts
            cleaned_string = input_string.replace('np.float64(', '').replace(')', '')

            # Use ast.literal_eval for safe evaluation
            float_list = ast.literal_eval(cleaned_string)

            # Check if all elements are floats (or can be converted to floats)
            return [float(x) for x in float_list]
        except (ValueError, SyntaxError, TypeError):
            print("Error: Invalid input string format.  It should be a string representation of a list of numbers.")
            return None
    def run_q_sweep(self, exp_extension='', scaling=False, return_calibration_data=False):
        import datetime

        # ----------Load/get data------------------------
        Is = {i: [] for i in range(self.number_of_qubits)}
        Qs = {i: [] for i in range(self.number_of_qubits)}
        Ig_calibration = {i: [] for i in range(self.number_of_qubits)}
        Ie_calibration = {i: [] for i in range(self.number_of_qubits)}
        Qg_calibration = {i: [] for i in range(self.number_of_qubits)}
        Qe_calibration = {i: [] for i in range(self.number_of_qubits)}
        amps = {i: [] for i in range(self.number_of_qubits)}
        gains = {i: [] for i in range(self.number_of_qubits)}
        rounds_completed = {i: [] for i in range(self.number_of_qubits)}
        reps = []
        steps=0
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        delay_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}

        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                outerFolder = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date + "/study_data"
                outerFolder_save_plots = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date + "_plots/"
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # ------------------------------------------------Load/Plot/Save T1----------------------------------------------
            if '_' in exp_extension:
                outerFolder_expt = outerFolder + f"/Data_h5/QSpec_zeno/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/QSpec_zeno/"

            round_we_are_on=outerFolder_expt.split(f'qubit_{self.qubit}round')[-1].split('/')[0].split('_')[0]
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

            TS = re.compile(
                r'(\d{4})[-_\.]?(\d{2})[-_\.]?(\d{2})[ Tt_-]?(\d{2})[-_\.]?(\d{2})[-_\.]?(\d{2})'
            )
            import datetime as dt
            def dt_from_name(path):
                name = os.path.basename(path)
                m = TS.search(name)
                if not m:
                    return dt.datetime.min  # or dt.datetime.max to push unknowns to the end
                y, mo, d, h, mi, s = map(int, m.groups())
                return dt.datetime(y, mo, d, h, mi, s)

            h5_files = sorted(h5_files, key=dt_from_name)

            for h5_file in h5_files:

                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type=f'QSpec_zeno', save_r=int(save_round), scaling=scaling)
                # H5_class_instance.print_h5_contents(h5_file)
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data[f'QSpec_zeno']:
                    for dataset in range(len(load_data[f'QSpec_zeno'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'QSpec_zeno'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T1 = load_data['T1'][q_key].get('T1', [])[0][dataset]
                        # errors = load_data['T1'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data[f'QSpec_zeno'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue
                        delays = self.process_h5_data(
                            load_data[f'QSpec_zeno'][q_key].get('Frequencies', [])[0][dataset].decode())
                        try:
                            I = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('I', [])[0][dataset].decode())
                            Q = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('Q', [])[0][dataset].decode())
                            if scaling:

                                Ie = self.process_h5_data(
                                    load_data[f'T1{exp_extension}'][q_key].get('ss_I_e', [])[0][dataset].decode())

                                Ig = self.process_h5_data(
                                    load_data[f'T1{exp_extension}'][q_key].get('ss_I_g', [])[0][dataset].decode())
                                Qe = self.process_h5_data(
                                    load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                                Qg = self.process_h5_data(
                                    load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_Q_g', [])[0][dataset].decode())
                        except:
                            I = self.process_h5_data(load_data[f'QSpec_zeno'][q_key].get('I', [])[0][dataset].decode())
                            Q = self.process_h5_data(load_data[f'QSpec_zeno'][q_key].get('Q', [])[0][dataset].decode())

                            if scaling:
                                Ie = self.process_h5_data(load_data[f'QSpec_zeno'][q_key].get('ss_I_e', [])[0][dataset].decode())
                                Ig = self.process_h5_data(load_data[f'QSpec_zeno'][q_key].get('ss_I_g', [])[0][dataset].decode())
                                Qe = self.process_h5_data(load_data[f'QSpec_zeno'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                                Qg = self.process_h5_data(load_data[f'QSpec_zeno'][q_key].get('ss_Q_g', [])[0][dataset].decode())
                            round_num = load_data[f'QSpec_zeno'][q_key].get('Round Num', [])[0][dataset]
                        try:
                            batch_num = load_data[f'QSpec_zeno'][q_key].get('Batch Num', [])[0][dataset]
                            syst_config = load_data[f'QSpec_zeno'][q_key].get('Syst Config', [])[0][dataset].decode()
                            exp_config = load_data[f'QSpec_zeno'][q_key].get('Exp Config', [])[0][dataset].decode()
                            #print(exp_config)
                            #safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                            #exp_config = eval(exp_config, safe_globals)
                        except:
                            exp_config =None

                        if len(I) > 0:
                            steps = round(
                                float(
                                    exp_config.split('Readout_Optimization\': ')[-1].split('steps\': ')[-1].split(',')[
                                        0]), 6)

                            Is[q_key].append(I)
                            Qs[q_key].append(Q)

                            gain = round(
                                float(syst_config.split('res_gain_qze\': ')[-1].split(',')[0]), 6)

                            gains[q_key].append(gain)
                            rounds_completed[q_key].append(round_we_are_on)
                            if scaling:
                                I = np.asarray(I, dtype=float)
                                Q = np.asarray(Q, dtype=float)
                                Ie = np.asarray(Ie, dtype=float)
                                Qe = np.asarray(Qe, dtype=float)
                                Ig = np.asarray(Ig, dtype=float)
                                Qg = np.asarray(Qg, dtype=float)
                                e = np.mean((Ie + 1j * Qe))
                                g = np.mean((Ig + 1j * Qg))
                                ### Normalization ###
                                pop_norm = abs(((I + 1j * Q) - g) * (e - g) / abs(e - g) ** 2)
                                amp = pop_norm
                                Ig_calibration[q_key].append(Ig)
                                Ie_calibration[q_key].append(Ie)
                                Qg_calibration[q_key].append(Qg)
                                Qe_calibration[q_key].append(Qe)
                            else:
                                amp=np.hypot(I, Q)
                            amps[q_key].append(amp.tolist())
                            delay_times[q_key].append(delays)
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                del H5_class_instance
        if return_calibration_data:
            return Is, Qs, amps, gains, rounds_completed, delay_times, Ig_calibration, Ie_calibration, Qe_calibration, Qg_calibration, steps
        else:
            return Is,Qs,amps, gains, rounds_completed, delay_times
    def robust_center(self,z, c=5.5, iters=100, eps=1e-12):
        """
        z: complex array of IQ samples (I + 1j*Q)
        c: Tukey biweight tuning constant (~4.685 gives ~95% efficiency for Gaussian)
        iters: small fixed number of IRLS steps
        returns complex robust location estimate
        """
        I = np.real(z);
        Q = np.imag(z)
        # start from median (very robust) to avoid bias from tails
        mu_I, mu_Q = np.median(I), np.median(Q)

        for _ in range(iters):
            d = np.hypot(I - mu_I, Q - mu_Q)
            # robust scale via MAD of distances
            s = 1.4826 * np.median(np.abs(d - np.median(d))) + eps
            u = d / (c * s + eps)
            # Tukey biweight weights (points past u>=1 get weight 0)
            w = (1 - u ** 2) ** 2
            w[u >= 1] = 0.0
            # if all weights vanished (e.g., tiny cluster), fall back to equal weights
            if np.all(w == 0):
                w = np.ones_like(d)
            # weighted means
            mu_I = np.sum(w * I) / (np.sum(w) + eps)
            mu_Q = np.sum(w * Q) / (np.sum(w) + eps)

        return mu_I + 1j * mu_Q
    def run_q_sweep_new(self, exp_extension='', scaling=False, return_calibration_data=False, weighted_mean=True, gain=''):
        import datetime
        import glob, os, re
        import numpy as np

        # Initialize data containers
        amps = {i: [] for i in range(self.number_of_qubits)}
        gains = {i: [] for i in range(self.number_of_qubits)}
        rounds_completed = {i: [] for i in range(self.number_of_qubits)}
        delay_times = {i: [] for i in range(self.number_of_qubits)}

        for folder_date in self.top_folder_dates:
            outerFolder = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date + "/study_data"
            outerFolder_expt = outerFolder + f"/Data_h5/QSpec_zeno{gain}/"
            
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
            TS = re.compile(r'(\d{4})[-_\.]?(\d{2})[-_\.]?(\d{2})[ Tt_-]?(\d{2})[-_\.]?(\d{2})[-_\.]?(\d{2})')
            
            def dt_from_name(path):
                name = os.path.basename(path)
                m = TS.search(name)
                if not m:
                    return datetime.datetime.min
                y, mo, d, h, mi, s = map(int, m.groups())
                return datetime.datetime(y, mo, d, h, mi, s)

            h5_files = sorted(h5_files, key=dt_from_name)

            for h5_file in h5_files:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type=f'QSpec_zeno{gain}', save_r=int(save_round), scaling=scaling)

                for q_key in load_data[f'QSpec_zeno{gain}']:
                    for dataset in range(len(load_data[f'QSpec_zeno{gain}'][q_key].get('Dates', [])[0])):
                        
                        delays = self.process_h5_data(
                            load_data[f'QSpec_zeno{gain}'][q_key].get('Frequencies', [])[0][dataset].decode())
                        
                        I = self.process_string_of_nested_lists(
                            load_data[f'QSpec_zeno{gain}'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_string_of_nested_lists(
                            load_data[f'QSpec_zeno{gain}'][q_key].get('Q', [])[0][dataset].decode())
                        
                        gains_swept = self.process_h5_data(
                            load_data[f'QSpec_zeno{gain}'][q_key].get('Gains', [])[0][dataset].decode())
                        
                        if scaling:
                            Ie = self.process_string_of_nested_lists(
                                load_data[f'QSpec_zeno{gain}'][q_key].get('ss_I_e', [])[0][dataset].decode())
                            Ig = self.process_string_of_nested_lists(
                                load_data[f'QSpec_zeno{gain}'][q_key].get('ss_I_g', [])[0][dataset].decode())
                            Qe = self.process_string_of_nested_lists(
                                load_data[f'QSpec_zeno{gain}'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                            Qg = self.process_string_of_nested_lists(
                                load_data[f'QSpec_zeno{gain}'][q_key].get('ss_Q_g', [])[0][dataset].decode())

                        if len(I) > 0:
                            round_current = int(folder_date.split('round')[-1])
                            rounds_completed[q_key].append(round_current)
                            gains[q_key].append(gains_swept)
                            
                            # Assume I and Q are list-of-lists (one per gain), single calibration per round
                            if scaling:
                                # Single calibration for all gains
                                Ie_cal = np.asarray(Ie[0], dtype=float)
                                Ig_cal = np.asarray(Ig[0], dtype=float)
                                Qe_cal = np.asarray(Qe[0], dtype=float)
                                Qg_cal = np.asarray(Qg[0], dtype=float)
                                
                                e = np.mean(Ie_cal + 1j * Qe_cal)
                                g = np.mean(Ig_cal + 1j * Qg_cal)
                                
                                # Apply calibration to each gain's data
                                calibrated_sublists = []
                                for sub_I, sub_Q in zip(I, Q):
                                    sub_I = np.asarray(sub_I, dtype=float)
                                    sub_Q = np.asarray(sub_Q, dtype=float)
                                    pop_norm = np.abs(((sub_I + 1j * sub_Q) - g) * (e - g) / (np.abs(e - g) ** 2))
                                    calibrated_sublists.append(pop_norm.tolist())
                                
                                # Flatten for gain sweep
                                flat_amps = [item for sublist in calibrated_sublists for item in sublist]
                                amps[q_key].append(flat_amps)
                                
                                # Expand gains and delays to match flattened structure
                                expanded_gains = np.repeat(gains_swept, len(delays))
                                gains[q_key][-1] = expanded_gains.tolist()
                                
                                expanded_delays = np.tile(delays, len(gains_swept))
                                delay_times[q_key].append(expanded_delays.tolist())
                            else:
                                # No scaling - just compute amplitude from I/Q
                                amp_sublists = []
                                for sub_I, sub_Q in zip(I, Q):
                                    sub_I = np.asarray(sub_I, dtype=float)
                                    sub_Q = np.asarray(sub_Q, dtype=float)
                                    amp_sublists.append(np.hypot(sub_I, sub_Q).tolist())
                                
                                # Average over gains
                                amp_avg = np.mean(np.array(amp_sublists), axis=0)
                                amps[q_key].append(amp_avg.tolist())
                                delay_times[q_key].append(delays)

                del H5_class_instance

        return None, None, amps, gains, rounds_completed, delay_times

    def run_q_sweep_single_gain(self, exp_extension='', scaling=False, return_calibration_data=False, weighted_mean=True,
                        gain=''):
        import datetime
        import glob, os, re
        import numpy as np

        # Initialize data containers
        # We will first aggregate data into temporary lists per qubit
        agg_amps = {i: [] for i in range(self.number_of_qubits)}
        agg_dates = {i: [] for i in range(self.number_of_qubits)}
        agg_delays = {i: [] for i in range(self.number_of_qubits)}
        
        # Final containers to return
        amps = {i: [] for i in range(self.number_of_qubits)}
        dates = {i: [] for i in range(self.number_of_qubits)}
        rounds_completed = {i: [] for i in range(self.number_of_qubits)}
        delay_times = {i: [] for i in range(self.number_of_qubits)}

        # We need to sort the folders/dates to ensure the sweep is ordered (if desired, though not strictly necessary for scatter plots)
        # The existing code iterates self.top_folder_dates. We assume this is the order we want.

        for folder_date in self.top_folder_dates:
            outerFolder = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date + "/study_data"
            outerFolder_expt = outerFolder + f"/Data_h5/QSpec_zeno{gain}/"

            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
            if not h5_files:
                continue

            TS = re.compile(r'(\d{4})[-_\.]?(\d{2})[-_\.]?(\d{2})[ Tt_-]?(\d{2})[-_\.]?(\d{2})[-_\.]?(\d{2})')

            def dt_from_name(path):
                name = os.path.basename(path)
                m = TS.search(name)
                if not m:
                    return datetime.datetime.min
                y, mo, d, h, mi, s = map(int, m.groups())
                return datetime.datetime(y, mo, d, h, mi, s)

            h5_files = sorted(h5_files, key=dt_from_name)

            for h5_file in h5_files:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type=f'QSpec_zeno{gain}', save_r=int(save_round),
                                                           scaling=scaling)
                # H5_class_instance.print_h5_contents(h5_file)
                for q_key in load_data[f'QSpec_zeno{gain}']:
                    for dataset in range(len(load_data[f'QSpec_zeno{gain}'][q_key].get('Dates', [])[0])):

                        delays = self.process_h5_data(
                            load_data[f'QSpec_zeno{gain}'][q_key].get('Frequencies', [])[0][dataset].decode())
                        
                        # I and Q are 1D lists here (frequencies corresponding to the single gain)
                        I = self.process_h5_data(
                            load_data[f'QSpec_zeno{gain}'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(
                            load_data[f'QSpec_zeno{gain}'][q_key].get('Q', [])[0][dataset].decode())

                        try:
                            date = datetime.datetime.fromtimestamp(
                                load_data[f'QSpec_zeno{gain}'][q_key].get('Dates', [])[0][dataset])
                        except:
                            continue

                        if scaling:
                            Ie = self.process_h5_data(
                                load_data[f'QSpec_zeno{gain}'][q_key].get('ss_I_e', [])[0][dataset].decode())
                            Ig = self.process_h5_data(
                                load_data[f'QSpec_zeno{gain}'][q_key].get('ss_I_g', [])[0][dataset].decode())
                            Qe = self.process_h5_data(
                                load_data[f'QSpec_zeno{gain}'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                            Qg = self.process_h5_data(
                                load_data[f'QSpec_zeno{gain}'][q_key].get('ss_Q_g', [])[0][dataset].decode())

                        if len(I) > 0:
                            # Calibration
                            if scaling:
                                I = np.asarray(I, dtype=float)
                                Q = np.asarray(Q, dtype=float)
                                Ie = np.asarray(Ie, dtype=float)
                                Qe = np.asarray(Qe, dtype=float)
                                Ig = np.asarray(Ig, dtype=float)
                                Qg = np.asarray(Qg, dtype=float)
                                
                                e = np.mean(Ie + 1j * Qe)
                                g = np.mean(Ig + 1j * Qg)
                                
                                pop_norm = np.abs(((I + 1j * Q) - g) * (e - g) / (np.abs(e - g) ** 2))
                                amp_data = pop_norm.tolist()
                            else:
                                I = np.asarray(I, dtype=float)
                                Q = np.asarray(Q, dtype=float)
                                amp_data = np.hypot(I, Q).tolist()

                            # Append to aggregation lists
                            agg_amps[q_key].extend(amp_data)
                            
                            # Expand gain to match the number of frequency points
                            expanded_date = [date] * len(amp_data)
                            agg_dates[q_key].extend(expanded_date)
                            
                            agg_delays[q_key].extend(delays)

                del H5_class_instance

        # After processing all dates, package the aggregated data as a single "round"
        for q_key in range(self.number_of_qubits):
            if agg_amps[q_key]:
                amps[q_key].append(agg_amps[q_key])
                dates[q_key].append(agg_dates[q_key])
                delay_times[q_key].append(agg_delays[q_key])
                # Use a placeholder for the round name, or the first date
                rounds_completed[q_key].append("combined_sweep")

        return None, None, amps, dates, rounds_completed, delay_times

    def run(self,exp_extension=''):
        import datetime

        qubit_frequencies = {i: [] for i in range(self.number_of_qubits)}
        qspec_fit_errs= {i: [] for i in range(self.number_of_qubits)}
        rounds = []
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}
        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                outerFolder = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date+ "/study_data" + "/"
                outerFolder_save_plots = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date + "_plots/"
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date+ "/study_data" + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # ------------------------------------------Load/Plot/Save Q Spec------------------------------------
            if '_' in exp_extension:
                outerFolder_expt = outerFolder + f"/Data_h5/qspec{exp_extension}/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/qspec_ge/"

            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

            for h5_file in h5_files:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]

                H5_class_instance = Data_H5(h5_file)
                #H5_class_instance.print_h5_contents(h5_file)
                #sometimes you get '1(1)' when redownloading the h5 files for some reason
                load_data = H5_class_instance.load_from_h5(data_type=f'QSpec{exp_extension}', save_r=int(save_round.split('(')[0]))

                # Define specific days to exclude
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data[f'QSpec{exp_extension}']:
                    for dataset in range(len(load_data[f'QSpec{exp_extension}'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'QSpec{exp_extension}'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        date = datetime.datetime.fromtimestamp(load_data[f'QSpec{exp_extension}'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        I = self.process_h5_data(load_data[f'QSpec{exp_extension}'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data[f'QSpec{exp_extension}'][q_key].get('Q', [])[0][dataset].decode())
                        # I_fit = load_data['QSpec'][q_key].get('I Fit', [])[0][dataset]
                        # Q_fit = load_data['QSpec'][q_key].get('Q Fit', [])[0][dataset]
                        freqs = self.process_h5_data(load_data[f'QSpec{exp_extension}'][q_key].get('Frequencies', [])[0][dataset].decode())
                        round_num = load_data[f'QSpec{exp_extension}'][q_key].get('Round Num', [])[0][dataset]
                        batch_num = load_data[f'QSpec{exp_extension}'][q_key].get('Batch Num', [])[0][dataset]
                        try:
                            syst_config = load_data[f'QSpec{exp_extension}'][q_key].get('Syst Config', [])[0][dataset].decode()
                            exp_config = load_data[f'QSpec{exp_extension}'][q_key].get('Exp Config', [])[0][dataset].decode()
                            safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                            exp_config = eval(exp_config, safe_globals)
                        except:
                            exp_config =None

                        if len(I) > 0:
                            qspec_class_instance = QubitSpectroscopy(q_key, self.number_of_qubits, outerFolder_save_plots, round_num, self.signal,
                                                                     self.save_figs)
                            # if '_' in exp_extension:
                            #     q_spec_cfg = exp_config[f'qubit_spec{exp_extension}']
                            # else:
                            #     q_spec_cfg = exp_config['qubit_spec_ge']
                            largest_amp_curve_mean, I_fit, Q_fit, qspec_fit_err = qspec_class_instance.get_results(I, Q, freqs)
                            if qspec_fit_err is not None and qspec_fit_err < 1: #above 1 MHz fit err is probably not a good fit
                                qubit_frequencies[q_key].extend([largest_amp_curve_mean])
                                qspec_fit_errs[q_key].extend([qspec_fit_err])
                                date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                            del qspec_class_instance

                del H5_class_instance
        return date_times, qubit_frequencies, qspec_fit_errs

    def plot_all_q_heatmaps_new_format(
            self,
            amps,
            gains,
            rounds,
            delay_times,
            save_path,
            max_ylabels=6,
            individual_subfolder="individual_specs",
            # NEW:
            n_bar=None,  # list/np.array OR dict[str]->(list or {"lorentzian"/"gaussian"})
            save_individual_plots=True,  # Set to False to skip saving individual spec plots
    ):
        """
        NEW FORMAT ONLY + per-gain individual spec plots

        - Keeps your heatmaps exactly as before (fixed global z-scale).
        - ALSO saves, for each round & gain, a 1D "calibrated spec" plot:
            x = delay (MHz), y = averaged amplitude at that delay for this gain.
          Files go to: {save_path}/{individual_subfolder}/
          Filenames include the gain value.

        If n_bar is provided, the heatmap x-axis shows n̄ instead of gain:
          - n_bar can be a single list/array aligned to the sorted gains for that round,
          - or a dict mapping round_id -> list, or -> {"gains":..., "lorentzian":..., "gaussian":...}
            (prefers 'lorentzian' if present, else 'gaussian').
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        import matplotlib.ticker as mticker
        from scipy.optimize import least_squares
        from scipy.signal import savgol_filter

        # ---------- Fitting Helpers (Robust Lorentzian - Flat Baseline) ----------
        def lorentz_flat(x, f0, gamma, A, c0):
            return A * (gamma ** 2) / ((x - f0) ** 2 + gamma ** 2) + c0

        def _initial_guesses(x, y):
            x = np.asarray(x); y = np.asarray(y)
            # Flat baseline guess: median of the edges
            n = len(x)
            edge_ids = np.r_[np.arange(int(0.15 * n), int(0.25 * n)), np.arange(int(0.75 * n), int(0.85 * n))]
            if len(edge_ids) < 4: edge_ids = np.r_[0, 1, n - 2, n - 1]
            c0_0 = np.median(y[edge_ids])

            # Peak finding on smoothed data
            y_s = savgol_filter(y, max(5, (len(y) // 25) * 2 + 1), 2, mode='interp') if len(y) >= 11 else y
            imax, imin = np.argmax(y_s), np.argmin(y_s)
            prom_max = y_s[imax] - np.median(np.r_[y_s[:max(1, imax - 20)], y_s[min(len(y_s), imax + 20):]])
            prom_min = np.median(np.r_[y_s[:max(1, imin - 20)], y_s[min(len(y_s), imin + 20):]]) - y_s[imin]
            is_peak = prom_max >= prom_min
            idx0 = imax if is_peak else imin
            f0_0 = x[idx0]

            # Amplitude and Width
            A_0 = (y[idx0] - c0_0)
            if not is_peak and A_0 > 0: A_0 = -abs(A_0)
            y_half = c0_0 + 0.5 * A_0
            left, right = idx0, idx0
            while left > 1 and ((y[left] > y_half) if is_peak else (y[left] < y_half)): left -= 1
            while right < n - 2 and ((y[right] > y_half) if is_peak else (y[right] < y_half)): right += 1
            fwhm_0 = max((x[right] - x[left]), (x.max() - x.min()) / 50) if right > left else (x.max() - x.min()) / 20
            gamma_0 = max(fwhm_0 / 2.0, (x[1] - x[0]) * 1.5)
            return f0_0, gamma_0, A_0, c0_0

        def fit_slice(x, y):
            x = np.asarray(x, float); y = np.asarray(y, float)
            try:
                f0_0, g0, A0, c0_0 = _initial_guesses(x, y)
                p0 = np.array([f0_0, g0, A0, c0_0])
                span = x.max() - x.min()
                # Bounds: f0 within extended range, gamma positive, A unbounded, c0 unbounded
                lb = [x.min() - 0.1 * span, (x[1] - x[0]) * 0.2, -np.inf, -np.inf]
                ub = [x.max() + 0.1 * span, span, np.inf, np.inf]
                res = least_squares(lambda p: lorentz_flat(x, *p) - y, p0, bounds=(lb, ub), loss='soft_l1', f_scale=1.0)
                return res.x[0] if res.success else np.nan
            except:
                return np.nan

        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        # Basic presence & length checks
        n = min(len(amps_q), len(gains_q), len(rounds_q), len(delay_q))
        if n == 0 or not (len(amps_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
            print(f"No usable data for qubit {q} (missing lists or length mismatch). Skipping.")
            return

        # ---------- Flatten to points: (round_id, gain, delay, amp) ----------
        all_points = []
        for i in range(n):
            r_id = str(rounds_q[i])

            a_samples = np.asarray(amps_q[i], dtype=float).ravel()
            if a_samples.size == 0:
                continue

            # gain can be scalar or per-sample
            g_i = gains_q[i]
            g_arr = np.asarray(g_i, dtype=float).ravel() if isinstance(g_i, (list, tuple, np.ndarray)) else None
            if g_arr is None or g_arr.size == 1:
                try:
                    g_scalar = float(g_i)
                except Exception:
                    continue
                g_arr = np.full(a_samples.shape, g_scalar, dtype=float)
            elif g_arr.size != a_samples.size:
                continue

            # delay can be scalar or per-sample
            d_i = delay_q[i]
            d_arr = np.asarray(d_i, dtype=float).ravel() if isinstance(d_i, (list, tuple, np.ndarray)) else None
            if d_arr is None or d_arr.size == 1:
                try:
                    d_scalar = float(d_i)
                except Exception:
                    continue
                d_arr = np.full(a_samples.shape, d_scalar, dtype=float)
            elif d_arr.size != a_samples.size:
                continue

            # keep only finite triples
            mask = np.isfinite(a_samples) & np.isfinite(g_arr) & np.isfinite(d_arr)
            if not np.any(mask):
                continue

            for g, d, a in zip(g_arr[mask], d_arr[mask], a_samples[mask]):
                all_points.append((r_id, float(g), float(d), float(a)))

        if not all_points:
            print(f"No numeric points for qubit {q}. Skipping.")
            return

        # Unique rounds present
        unique_rounds = sorted({r for (r, _, __, ___) in all_points})

        # Ensure save folder(s) exist
        self.create_folder_if_not_exists(save_path)
        if save_individual_plots:
            indiv_root = os.path.join(save_path, individual_subfolder)
            self.create_folder_if_not_exists(indiv_root)

        # Helper: centers -> bin edges for pcolormesh
        def centers_to_edges(centers):
            centers = np.asarray(sorted(np.unique(centers)), dtype=float)
            if centers.size == 1:
                d = 1.0
                return np.array([centers[0] - d / 2, centers[0] + d / 2])
            mids = (centers[:-1] + centers[1:]) / 2.0
            first = centers[0] - (centers[1] - centers[0]) / 2.0
            last = centers[-1] + (centers[-1] - centers[-2]) / 2.0
            return np.concatenate([[first], mids, [last]])

        # ---------- Global color scale limits (z) ----------
        all_amps = np.array([a for (_, _, _, a) in all_points], dtype=float)
        global_vmin = float(np.nanmin(all_amps))
        global_vmax = float(np.nanmax(all_amps))

        # ---------------------------------------------------

        # --- helper to pull the n̄ vector for a given round (if provided) ---
        def get_nbar_for_round(r_id_str, gains_sorted):
            if n_bar is None:
                return None

            # direct list/array: assume aligned to sorted gains for this round
            if isinstance(n_bar, (list, tuple, np.ndarray)):
                nb = np.asarray(n_bar, float).ravel()
                return nb if nb.size == len(gains_sorted) else None

            # dict-like
            if isinstance(n_bar, dict):
                # keys might be strings or numbers; normalize to string
                key = r_id_str if r_id_str in n_bar else str(r_id_str)
                entry = n_bar.get(key, None)
                if entry is None:
                    return None
                # earlier return format: {"gains":[...], "lorentzian":[...], "gaussian":[...]}
                if isinstance(entry, dict):
                    candidate = entry.get("lorentzian") or entry.get("gaussian") or entry.get("nbar") or entry.get(
                        "values")
                    if candidate is None:
                        return None
                    nb = np.asarray(candidate, float).ravel()
                    # if "gains" present and mismatched order, we could align, but we assume ascending/aligned per spec
                    return nb if nb.size == len(gains_sorted) else None
                # maybe it's already the list
                nb = np.asarray(entry, float).ravel()
                return nb if nb.size == len(gains_sorted) else None

            return None

        # ---------------------------------------------------

        # For each round, build a 2D heatmap
        for r_id in unique_rounds:
            pts = [(gv, dv, av) for (rr, gv, dv, av) in all_points if rr == r_id]
            if not pts:
                continue

            # Sort gains; build 2D grid
            gains_r = sorted(set(g for (g, _, __) in pts))
            delays_r = sorted(set(d for (_, d, __) in pts))
            Ny = len(delays_r)
            Nx = len(gains_r)
            if Ny == 0 or Nx == 0:
                continue

            # delay -> index
            d_map = {d: i for i, d in enumerate(delays_r)}
            # gain -> index
            gi_map = {g: i for i, g in enumerate(gains_r)}

            # Build C: shape (Ny, Nx), each cell avg of data from (gain, delay)
            cell_vals = defaultdict(list)
            for (gg, dd, aa) in pts:
                ix = gi_map[gg]
                iy = d_map[dd]
                cell_vals[(iy, ix)].append(aa)
            C = np.full((Ny, Nx), np.nan, dtype=float)
            for (iy, ix), vals in cell_vals.items():
                C[iy, ix] = float(np.nanmean(vals))

            # ---------- Fit Centers (Robust) ----------
            centers_L = []
            x_freqs = np.array(delays_r, float)
            for ix in range(Nx):
                y_col = C[:, ix]
                if np.isfinite(y_col).any() and len(x_freqs) >= 5:
                    m = np.isfinite(y_col)
                    c = fit_slice(x_freqs[m], y_col[m])
                    centers_L.append(c)
                else:
                    centers_L.append(np.nan)
            centers_L = np.array(centers_L)

            # Decide X-axis: gains or n̄
            nbar_vec = get_nbar_for_round(r_id, gains_r)
            if nbar_vec is not None:
                x_vals = nbar_vec
                x_label = "Photon Number n̄"
            else:
                x_vals = gains_r
                x_label = "Gain (a.u.)"

            idx_sorted = np.argsort(x_vals)
            x_vals_sorted = np.array(x_vals, dtype=float)[idx_sorted]
            C_sorted = C[:, idx_sorted]

            # Bin edges for pcolormesh
            x_edges = centers_to_edges(x_vals_sorted)
            y_edges = centers_to_edges(delays_r)

            # ---------- Heatmap ----------
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            mesh = ax.pcolormesh(
                x_edges, y_edges, C_sorted, shading='flat',
                vmin=global_vmin, vmax=global_vmax
            )
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label("Qubit Population")

            # Overlay fits
            if np.any(np.isfinite(centers_L)):
                # x_vals_sorted corresponds to C_sorted columns
                # We need to match centers_L (indexed by original gains_r) to x_vals_sorted
                # x_vals are aligned with gains_r (by index if nbar list, or identity if gains)
                # So centers_L[i] corresponds to x_vals[i]
                # But C_sorted is sorted by x_vals.
                # We should plot (x_vals, centers_L) directly, no need to sort for scatter
                ax.plot(x_vals, centers_L, 'o', ms=4, mfc='none', mec='w', mew=1.5, label='Fits')
                ax.plot(x_vals, centers_L, '.', ms=2, color='k')
                ax.legend(loc='best')

            ax.set_title(f"Qubit {self.qubit + 1} — Round {r_id}")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Frequency (MHz)")

            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7, prune=None))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # Label a subset of delay times on Y
            if Ny > 0:
                if Ny <= max_ylabels:
                    yticks_idx = list(range(Ny))
                else:
                    yticks_idx = np.linspace(0, Ny - 1, num=max_ylabels, dtype=int).tolist()
                    yticks_idx = sorted(set(yticks_idx))
                yticks_vals = [delays_r[i] for i in yticks_idx]
                ax.set_yticks(yticks_vals)
                ax.set_yticklabels([f"{v:.2f}" for v in yticks_vals])

            fig.tight_layout()
            if n_bar is not None:
                outfile = (save_path + f"qspec_heatmap_q{self.qubit}_round{r_id}_nbar.png")
            else:
                outfile = (save_path + f"qspec_heatmap_q{self.qubit}_round{r_id}.png")


            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved heatmap for round {r_id} to: {outfile}")

            # ---------- Per-gain individual calibrated spec plots ----------
            if save_individual_plots:
                round_folder = os.path.join(indiv_root, f"round_{r_id}")
                self.create_folder_if_not_exists(round_folder)

                for g in gains_r:
                    # Collect amplitudes grouped by delay for this gain
                    d_to_vals = defaultdict(list)
                    for (gg, dd, aa) in pts:
                        if gg == g and np.isfinite(dd) and np.isfinite(aa):
                            d_to_vals[dd].append(aa)

                    if not d_to_vals:
                        continue

                    # Sort by delay and average amplitudes per delay
                    delays_sorted = np.array(sorted(d_to_vals.keys()), dtype=float)
                    amps_avg = np.array([float(np.nanmean(d_to_vals[d])) for d in delays_sorted], dtype=float)

                    fig2, ax2 = plt.subplots(figsize=(6.5, 4.0))
                    ax2.plot(delays_sorted, amps_avg, marker='o', linewidth=1.5)
                    # If n̄ is provided, add it in the title for clarity
                    title_suffix = ""
                    if nbar_vec is not None:
                        # n̄ value corresponding to this gain index
                        idx = gi_map[g]
                        try:
                            nbar_val = float(nbar_vec[idx])
                            title_suffix = f" — n̄ {nbar_val:g}"
                        except Exception:
                            title_suffix = ""
                    ax2.set_title(f"Qubit {self.qubit + 1} — Round {r_id} — Gain {g:g}{title_suffix}")
                    ax2.set_xlabel("Frequency (MHz)")
                    ax2.set_ylabel("Qubit Population")
                    ax2.grid(True, alpha=0.3)
                    fig2.tight_layout()

                    gain_str = f"{g:.6g}".replace("/", "_")
                    out_indiv = os.path.join(
                        round_folder,
                        f"qspec_q{self.qubit}_round{r_id}_gain{gain_str}.png"
                    )
                    fig2.savefig(out_indiv, transparent=False, dpi=self.final_figure_quality)
                    plt.close(fig2)
                    print(f"Saved individual spec for round {r_id}, gain {g:g} to: {out_indiv}")
    def plot_all_q_heatmaps_single_gain(
            self,
            amps,
            date_times,
            rounds,
            delay_times,
            save_path,
            max_ylabels=6,
            individual_subfolder="individual_specs",
            save_individual_plots=True,
    ):
        """
        Updated to plot Heatmap with Time (Date) on X-axis and Frequency on Y-axis.

        Data model (per qubit q):
          - amps[q]         : list of lists; amps[q][i] is a list of amplitude samples for dataset i
          - date_times[q]   : list; date_times[q][i] is the datetime string for dataset i
          - rounds[q]       : list; rounds[q][i] is the round label for dataset i
          - delay_times[q]  : list; delay_times[q][i] is the delay/frequency (scalar) for dataset i
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        from collections import defaultdict
        import matplotlib.ticker as mticker
        from scipy.optimize import least_squares
        from scipy.signal import savgol_filter

        # ---------- Fitting Helpers (Robust Lorentzian - Flat Baseline) ----------
        def lorentz_flat(x, f0, gamma, A, c0):
            return A * (gamma ** 2) / ((x - f0) ** 2 + gamma ** 2) + c0

        def _initial_guesses(x, y):
            x = np.asarray(x); y = np.asarray(y)
            # Flat baseline guess: median of the edges
            n = len(x)
            edge_ids = np.r_[np.arange(int(0.15 * n), int(0.25 * n)), np.arange(int(0.75 * n), int(0.85 * n))]
            if len(edge_ids) < 4: edge_ids = np.r_[0, 1, n - 2, n - 1]
            c0_0 = np.median(y[edge_ids])

            # Peak finding on smoothed data
            y_s = savgol_filter(y, max(5, (len(y) // 25) * 2 + 1), 2, mode='interp') if len(y) >= 11 else y
            imax, imin = np.argmax(y_s), np.argmin(y_s)
            prom_max = y_s[imax] - np.median(np.r_[y_s[:max(1, imax - 20)], y_s[min(len(y_s), imax + 20):]])
            prom_min = np.median(np.r_[y_s[:max(1, imin - 20)], y_s[min(len(y_s), imin + 20):]]) - y_s[imin]
            is_peak = prom_max >= prom_min
            idx0 = imax if is_peak else imin
            f0_0 = x[idx0]

            # Amplitude and Width
            A_0 = (y[idx0] - c0_0)
            if not is_peak and A_0 > 0: A_0 = -abs(A_0)
            y_half = c0_0 + 0.5 * A_0
            left, right = idx0, idx0
            while left > 1 and ((y[left] > y_half) if is_peak else (y[left] < y_half)): left -= 1
            while right < n - 2 and ((y[right] > y_half) if is_peak else (y[right] < y_half)): right += 1
            fwhm_0 = max((x[right] - x[left]), (x.max() - x.min()) / 50) if right > left else (x.max() - x.min()) / 20
            gamma_0 = max(fwhm_0 / 2.0, (x[1] - x[0]) * 1.5)
            return f0_0, gamma_0, A_0, c0_0

        def fit_slice(x, y):
            x = np.asarray(x, float); y = np.asarray(y, float)
            try:
                f0_0, g0, A0, c0_0 = _initial_guesses(x, y)
                p0 = np.array([f0_0, g0, A0, c0_0])
                span = x.max() - x.min()
                # Bounds: f0 within extended range, gamma positive, A unbounded, c0 unbounded
                lb = [x.min() - 0.1 * span, (x[1] - x[0]) * 0.2, -np.inf, -np.inf]
                ub = [x.max() + 0.1 * span, span, np.inf, np.inf]
                res = least_squares(lambda p: lorentz_flat(x, *p) - y, p0, bounds=(lb, ub), loss='soft_l1', f_scale=1.0)
                return res.x[0] if res.success else np.nan
            except:
                return np.nan

        q = self.qubit
        dates_q = date_times.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        # Basic presence & length checks
        n = min(len(amps_q), len(dates_q), len(rounds_q), len(delay_q))
        if n == 0 or not (len(amps_q) == len(dates_q) == len(rounds_q) == len(delay_q)):
            print(f"No usable data for qubit {q} (missing lists or length mismatch). Skipping.")
            return

        # ---------- Flatten to points: (round_id, date_num, delay, amp) ----------
        all_points = []
        for i in range(n):
            r_id = str(rounds_q[i])

            a_samples = np.asarray(amps_q[i], dtype=float).ravel()
            if a_samples.size == 0:
                continue

            # Date handling
            d_raw = dates_q[i]

            def to_num(d):
                if isinstance(d, datetime):
                    return mdates.date2num(d)
                try:
                    return mdates.date2num(datetime.strptime(str(d), "%Y-%m-%d %H:%M:%S"))
                except:
                    return np.nan

            is_seq = isinstance(d_raw, (list, tuple, np.ndarray))
            # If d_raw is a sequence of the same length as a_samples, we assume 1-to-1 mapping
            if is_seq and len(d_raw) == len(a_samples):
                dt_arr = np.array([to_num(d) for d in d_raw], dtype=float)
            else:
                # Otherwise, broadcast the single date (or first element)
                d_single = d_raw[0] if is_seq else d_raw
                dt_arr = np.full(a_samples.shape, to_num(d_single), dtype=float)

            # delay can be scalar or per-sample
            d_i = delay_q[i]
            d_arr = np.asarray(d_i, dtype=float).ravel() if isinstance(d_i, (list, tuple, np.ndarray)) else None
            if d_arr is None or d_arr.size == 1:
                try:
                    d_scalar = float(d_i)
                except Exception:
                    continue
                d_arr = np.full(a_samples.shape, d_scalar, dtype=float)
            elif d_arr.size != a_samples.size:
                continue

            # keep only finite triples
            mask = np.isfinite(a_samples) & np.isfinite(dt_arr) & np.isfinite(d_arr)
            if not np.any(mask):
                continue

            for dt, d, a in zip(dt_arr[mask], d_arr[mask], a_samples[mask]):
                all_points.append((r_id, float(dt), float(d), float(a)))

        if not all_points:
            print(f"No numeric points for qubit {q}. Skipping.")
            return

        # Unique rounds present
        unique_rounds = sorted({r for (r, _, __, ___) in all_points})

        # Ensure save folder(s) exist
        self.create_folder_if_not_exists(save_path)
        if save_individual_plots:
            indiv_root = os.path.join(save_path, individual_subfolder)
            self.create_folder_if_not_exists(indiv_root)

        # Helper: centers -> bin edges for pcolormesh
        def centers_to_edges(centers):
            centers = np.asarray(sorted(np.unique(centers)), dtype=float)
            if centers.size == 1:
                d = 1.0/24.0 # 1 hour default width
                return np.array([centers[0] - d / 2, centers[0] + d / 2])
            mids = (centers[:-1] + centers[1:]) / 2.0
            first = centers[0] - (centers[1] - centers[0]) / 2.0
            last = centers[-1] + (centers[-1] - centers[-2]) / 2.0
            return np.concatenate([[first], mids, [last]])

        # ---------- Global color scale limits (z) ----------
        all_amps = np.array([a for (_, _, _, a) in all_points], dtype=float)
        global_vmin = float(np.nanmin(all_amps))
        global_vmax = float(np.nanmax(all_amps))

        # For each round, build a 2D heatmap
        for r_id in unique_rounds:
            pts = [(dt, dv, av) for (rr, dt, dv, av) in all_points if rr == r_id]
            if not pts:
                continue

            # Sort dates; build 2D grid
            dates_r = sorted(set(dt for (dt, _, __) in pts))
            delays_r = sorted(set(d for (_, d, __) in pts))
            Ny = len(delays_r)
            Nx = len(dates_r)
            if Ny == 0 or Nx == 0:
                continue

            # delay -> index
            d_map = {d: i for i, d in enumerate(delays_r)}
            # date -> index
            dt_map = {dt: i for i, dt in enumerate(dates_r)}

            # Build C: shape (Ny, Nx), each cell avg of data from (date, delay)
            cell_vals = defaultdict(list)
            for (dt, dd, aa) in pts:
                ix = dt_map[dt]
                iy = d_map[dd]
                cell_vals[(iy, ix)].append(aa)
            C = np.full((Ny, Nx), np.nan, dtype=float)
            for (iy, ix), vals in cell_vals.items():
                C[iy, ix] = float(np.nanmean(vals))

            # ---------- Fit Centers (Robust) ----------
            centers_L = []
            x_freqs = np.array(delays_r, float)
            for ix in range(Nx):
                y_col = C[:, ix]
                if np.isfinite(y_col).any() and len(x_freqs) >= 5:
                    m = np.isfinite(y_col)
                    c = fit_slice(x_freqs[m], y_col[m])
                    centers_L.append(c)
                else:
                    centers_L.append(np.nan)
            centers_L = np.array(centers_L)

            x_vals = dates_r
            x_label = "Time"

            idx_sorted = np.argsort(x_vals)
            x_vals_sorted = np.array(x_vals, dtype=float)[idx_sorted]
            C_sorted = C[:, idx_sorted]

            # Bin edges for pcolormesh
            x_edges = centers_to_edges(x_vals_sorted)
            y_edges = centers_to_edges(delays_r)

            # ---------- Heatmap ----------
            fig, ax = plt.subplots(figsize=(10, 6))
            mesh = ax.pcolormesh(
                x_edges, y_edges, C_sorted, shading='flat',
                vmin=global_vmin, vmax=global_vmax
            )
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label("Qubit Population")

            # Overlay fits
            if np.any(np.isfinite(centers_L)):
                ax.plot(x_vals_sorted, centers_L, 'o', ms=4, mfc='none', mec='w', mew=1.5, label='Fits')
                ax.plot(x_vals_sorted, centers_L, '.', ms=2, color='k')
                ax.legend(loc='best')

            ax.set_title(f"Qubit {self.qubit + 1} — Round {r_id}")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Frequency (MHz)")

            # Format X-axis as dates
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # Label a subset of delay times on Y
            if Ny > 0:
                if Ny <= max_ylabels:
                    yticks_idx = list(range(Ny))
                else:
                    yticks_idx = np.linspace(0, Ny - 1, num=max_ylabels, dtype=int).tolist()
                    yticks_idx = sorted(set(yticks_idx))
                yticks_vals = [delays_r[i] for i in yticks_idx]
                ax.set_yticks(yticks_vals)
                ax.set_yticklabels([f"{v:.2f}" for v in yticks_vals])

            fig.tight_layout()
            outfile = (save_path + f"qspec_heatmap_q{self.qubit}_round{r_id}.png")

            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved heatmap for round {r_id} to: {outfile}")

            # ---------- Per-date individual calibrated spec plots ----------
            if save_individual_plots:
                round_folder = os.path.join(indiv_root, f"round_{r_id}")
                self.create_folder_if_not_exists(round_folder)

                for dt in dates_r:
                    # Collect amplitudes grouped by delay for this date
                    d_to_vals = defaultdict(list)
                    for (ddt, dd, aa) in pts:
                        if ddt == dt and np.isfinite(dd) and np.isfinite(aa):
                            d_to_vals[dd].append(aa)

                    if not d_to_vals:
                        continue

                    # Sort by delay and average amplitudes per delay
                    delays_sorted = np.array(sorted(d_to_vals.keys()), dtype=float)
                    amps_avg = np.array([float(np.nanmean(d_to_vals[d])) for d in delays_sorted], dtype=float)

                    fig2, ax2 = plt.subplots(figsize=(6.5, 4.0))
                    ax2.plot(delays_sorted, amps_avg, marker='o', linewidth=1.5)

                    # Convert dt back to string for title/filename
                    dt_obj = mdates.num2date(dt)
                    dt_str = dt_obj.strftime("%Y-%m-%d %H:%M:%S")
                    dt_file_str = dt_obj.strftime("%Y%m%d_%H%M%S")

                    ax2.set_title(f"Qubit {self.qubit + 1} | Round {r_id} | {dt_str}")
                    ax2.set_xlabel("Frequency (MHz)")
                    ax2.set_ylabel("Population")
                    ax2.set_ylim(-0.05, 1.05)
                    ax2.grid(True, alpha=0.3)

                    # Fit overlay
                    center = fit_slice(delays_sorted, amps_avg)
                    if np.isfinite(center):
                        ax2.axvline(center, color='r', linestyle='--', alpha=0.7, label=f"Fit: {center:.4f} MHz")
                        ax2.legend()

                    fig2.tight_layout()
                    out_indiv = os.path.join(round_folder, f"spec_q{self.qubit}_{dt_file_str}.png")
                    fig2.savefig(out_indiv, transparent=False, dpi=self.final_figure_quality)
                    plt.close(fig2)
                    print(f"Saved individual spec for round {r_id}, date {dt_str} to: {out_indiv}")

    def calculate_nbar(self, amps, gains, rounds, delay_times, chi_MHz=0.274, fit_gaussian=True, fit_lorentzian=True):
        """
        Calculate nbar values without plotting.
        
        Returns:
            dict {<round_id_str>: {"gains": [...], "gaussian": [...], "lorentzian": [...]}}
        """
        import numpy as np
        from scipy.optimize import least_squares
        from scipy.stats import linregress
        from scipy.signal import savgol_filter
        from collections import defaultdict
        
        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])
        
        n = min(len(amps_q), len(gains_q), len(rounds_q), len(delay_q))
        if n == 0 or not (len(amps_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
            return {}
        
        # Flatten to points
        all_points = []
        for i in range(n):
            r_id = str(rounds_q[i])
            a_samples = np.asarray(amps_q[i], dtype=float).ravel()
            if a_samples.size == 0:
                continue
            
            g_i = gains_q[i]
            g_arr = np.asarray(g_i, dtype=float).ravel() if isinstance(g_i, (list, tuple, np.ndarray)) else None
            if g_arr is None or g_arr.size == 1:
                try:
                    g_scalar = float(g_i)
                except:
                    continue
                g_arr = np.full(a_samples.shape, g_scalar, dtype=float)
            elif g_arr.size != a_samples.size:
                continue
            
            d_i = delay_q[i]
            d_arr = np.asarray(d_i, dtype=float).ravel() if isinstance(d_i, (list, tuple, np.ndarray)) else None
            if d_arr is None or d_arr.size == 1:
                try:
                    d_scalar = float(d_i)
                except:
                    continue
                d_arr = np.full(a_samples.shape, d_scalar, dtype=float)
            elif d_arr.size != a_samples.size:
                continue
            
            mask = np.isfinite(a_samples) & np.isfinite(g_arr) & np.isfinite(d_arr)
            if not np.any(mask):
                continue
            
            for g, d, a in zip(g_arr[mask], d_arr[mask], a_samples[mask]):
                all_points.append((r_id, float(g), float(d), float(a)))
        
        if not all_points:
            return {}
        
        unique_rounds = sorted({r for (r, _, __, ___) in all_points})
        nbar_return = {}
        
        # Lorentzian fit function (linear baseline)
        def lorentz_lin(x, f0, gamma, A, c0, c1, xbar):
            return A * (gamma ** 2) / ((x - f0) ** 2 + gamma ** 2) + (c0 + c1 * (x - xbar))
        
        def fit_slice_simple(x, y):
            """Simple robust Lorentzian fit"""
            x = np.asarray(x, float)
            y = np.asarray(y, float)
            
            # Initial guesses
            xbar = x.mean()
            y_s = savgol_filter(y, max(5, (len(y) // 25) * 2 + 1), 2, mode='interp') if len(y) >= 11 else y
            idx0 = np.argmax(y_s)
            f0_0 = x[idx0]
            
            # Linear baseline from edges
            n = len(x)
            edge_ids = np.r_[np.arange(int(0.15 * n), int(0.25 * n)), np.arange(int(0.75 * n), int(0.85 * n))]
            if len(edge_ids) < 4:
                edge_ids = np.r_[0, 1, n - 2, n - 1]
            c1_0, c0_0 = np.polyfit(x[edge_ids] - xbar, y[edge_ids], 1)
            
            y_base0 = c0_0 + c1_0 * (x[idx0] - xbar)
            A_0 = y[idx0] - y_base0
            gamma_0 = (x.max() - x.min()) / 20
            
            p0 = np.array([f0_0, gamma_0, A_0, c0_0, c1_0])
            
            def residuals(p):
                return lorentz_lin(x, p[0], p[1], p[2], p[3], p[4], xbar) - y
            
            try:
                res = least_squares(residuals, p0, loss='soft_l1', f_scale=0.5)
                if res.success:
                    f0_fit, gamma_fit = res.x[0], res.x[1]
                    return f0_fit, 2 * gamma_fit
            except:
                pass
            
            return np.nan, np.nan
        
        # Process each round
        for r_id in unique_rounds:
            pts = [(gv, dv, av) for (rr, gv, dv, av) in all_points if rr == r_id]
            if not pts:
                continue
            
            gains_r = sorted(set(g for (g, _, __) in pts))
            
            centers_L = []
            gains_used = []
            
            for g in gains_r:
                d_to_vals = defaultdict(list)
                for (gg, dd, aa) in pts:
                    if gg == g and np.isfinite(dd) and np.isfinite(aa):
                        d_to_vals[dd].append(aa)
                
                if not d_to_vals:
                    continue
                
                delays_sorted = np.array(sorted(d_to_vals.keys()), dtype=float)
                amps_avg = np.array([float(np.nanmean(d_to_vals[d])) for d in delays_sorted], dtype=float)
                
                if len(delays_sorted) >= 5:
                    center, _ = fit_slice_simple(delays_sorted, amps_avg)
                    centers_L.append(center)
                    gains_used.append(g)
            
            if len(gains_used) >= 2:
                gains_used = np.array(gains_used, float)
                pwr = gains_used ** 2
                centers_L = np.array(centers_L, float)
                
                ok = np.isfinite(centers_L) & np.isfinite(pwr)
                if ok.sum() >= 2:
                    slope, intercept, _, _, _ = linregress(pwr[ok], centers_L[ok])
                    wq = slope * pwr + intercept
                    nbar = (wq - intercept) / (2.0 * chi_MHz)
                    
                    nbar_return[r_id] = {
                        "gains": list(map(float, gains_used.tolist())),
                        "gaussian": None,
                        "lorentzian": list(map(float, nbar.tolist())),
                    }
        
        return nbar_return

    def plot_all_q_heatmaps_nbar(
            self,
            amps,
            gains,
            rounds,
            delay_times,
            save_path,
            max_ylabels=6,
            individual_subfolder="individual_specs",
            # physics
            chi_MHz=0.25,
            fit_gaussian=True,  # unchanged
            fit_lorentzian=True,  # keep True to run the good fitter
            lorentz_fit_folder="lorentz_fits",
            # --- NEW controls for the better fitter ---
            robust_loss='soft_l1',  # 'soft_l1' or 'huber'
            window_factor=3.0,  # fit range ±window_factor*gamma_guess around peak
            allow_quadratic_baseline=False,  # set True if baseline is curved
            # --- NEW: opt-in return of nbar values ---
            return_nbar=False
    ):
        """
        Heatmaps + per-gain specs + nbar extraction.
        Uses robust Lorentzian + baseline fitting for vastly better centers/FWHM.

        Frequency axis must be MHz. chi_MHz is χ/2π in MHz.

        If return_nbar=True, returns:
            {
              <round_id_str>: {
                "gains": [g1, g2, ...] (ascending),
                "gaussian":   [nbar_at_g1, ...] or None,
                "lorentzian": [nbar_at_g1, ...] or None,
              },
              ...
            }
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from collections import defaultdict
        from scipy.optimize import least_squares
        from scipy.optimize import curve_fit
        from scipy.stats import linregress
        from scipy.signal import savgol_filter

        # ---------- shapes ----------
        def gauss(x, H, A, x0, sigma):
            return H + A * np.exp(-(x - x0) ** 2 / (2 * sigma ** 2))

        # baseline-aware lorentzian
        def lorentz_lin(x, f0, gamma, A, c0, c1, xbar):
            return A * (gamma ** 2) / ((x - f0) ** 2 + gamma ** 2) + (c0 + c1 * (x - xbar))

        def lorentz_quad(x, f0, gamma, A, c0, c1, c2, xbar):
            dx = (x - xbar)
            return A * (gamma ** 2) / ((x - f0) ** 2 + gamma ** 2) + (c0 + c1 * dx + c2 * dx * dx)

        def centers_to_edges(centers):
            centers = np.asarray(sorted(np.unique(centers)), dtype=float)
            if centers.size == 1:
                d = 1.0
                return np.array([centers[0] - d / 2, centers[0] + d / 2])
            mids = (centers[:-1] + centers[1:]) / 2.0
            first = centers[0] - (centers[1] - centers[0]) / 2.0
            last = centers[-1] + (centers[-1] - centers[-2]) / 2.0
            return np.concatenate([[first], mids, [last]])

        # ---------- NEW: robust fitter ----------
        def _initial_guesses(x, y):
            x = np.asarray(x);
            y = np.asarray(y)
            xbar = x.mean()
            y_s = savgol_filter(y, max(5, (len(y) // 25) * 2 + 1), 2, mode='interp') if len(y) >= 11 else y
            imax, imin = np.argmax(y_s), np.argmin(y_s)
            prom_max = y_s[imax] - np.median(np.r_[y_s[:max(1, imax - 20)], y_s[min(len(y_s), imax + 20):]])
            prom_min = np.median(np.r_[y_s[:max(1, imin - 20)], y_s[min(len(y_s), imin + 20):]]) - y_s[imin]
            is_peak = prom_max >= prom_min
            idx0 = imax if is_peak else imin
            f0_0 = x[idx0]

            # linear baseline from edges
            n = len(x)
            edge_ids = np.r_[np.arange(int(0.15 * n), int(0.25 * n)),
            np.arange(int(0.75 * n), int(0.85 * n))]
            if len(edge_ids) < 4:
                edge_ids = np.r_[0, 1, n - 2, n - 1]
            c1_0, c0_0 = np.polyfit(x[edge_ids] - xbar, y[edge_ids], 1)

            # amplitude guess relative to baseline at center
            y_base0 = c0_0 + c1_0 * (x[idx0] - xbar)
            A_0 = (y[idx0] - y_base0)
            if not is_peak and A_0 > 0:
                A_0 = -abs(A_0)

            # width from half-height crossings (fallback to 1/20 span)
            y_half = y_base0 + 0.5 * A_0
            left, right = idx0, idx0
            while left > 1 and ((y[left] > y_half) if is_peak else (y[left] < y_half)): left -= 1
            while right < n - 2 and ((y[right] > y_half) if is_peak else (y[right] < y_half)): right += 1
            if right > left:
                fwhm_0 = max((x[right] - x[left]), (x.max() - x.min()) / 50)
            else:
                fwhm_0 = (x.max() - x.min()) / 20
            gamma_0 = max(fwhm_0 / 2.0, (x[1] - x[0]) * 1.5)
            return f0_0, gamma_0, A_0, c0_0, c1_0, xbar

        def fit_slice(x, y):
            """Return dict {f0, FWHM, params, yfit, rms, success} with linear (or quadratic) baseline."""
            x = np.asarray(x, float);
            y = np.asarray(y, float)
            f0_0, g0, A0, c0_0, c1_0, xbar = _initial_guesses(x, y)

            # local window
            dx = window_factor * g0
            m = (x >= f0_0 - dx) & (x <= f0_0 + dx)
            if m.sum() < 8: m = np.ones_like(x, bool)
            xs, ys = x[m], y[m]

            span = x.max() - x.min()
            if allow_quadratic_baseline:
                # params = [f0, gamma, A, c0, c1, c2]
                p0 = np.array([f0_0, g0, A0, c0_0, c1_0, 0.0])
                lb = [x.min() - 0.1 * span, (x[1] - x[0]) * 0.2, -np.inf, -np.inf, -np.inf, -np.inf]
                ub = [x.max() + 0.1 * span, span, np.inf, np.inf, np.inf, np.inf]

                def resid(p):
                    f0, g, A, c0, c1, c2 = p
                    return lorentz_quad(xs, f0, g, A, c0, c1, c2, xbar) - ys

                res = least_squares(resid, p0, bounds=(lb, ub), loss=robust_loss, f_scale=1.0, max_nfev=20000)
                f0, g, A, c0, c1, c2 = res.x
                yfit = lorentz_quad(x, f0, g, A, c0, c1, c2, xbar)
            else:
                # params = [f0, gamma, A, c0, c1]
                p0 = np.array([f0_0, g0, A0, c0_0, c1_0])
                lb = [x.min() - 0.1 * span, (x[1] - x[0]) * 0.2, -np.inf, -np.inf, -np.inf]
                ub = [x.max() + 0.1 * span, span, np.inf, np.inf, np.inf]

                def resid(p):
                    f0, g, A, c0, c1 = p
                    return lorentz_lin(xs, f0, g, A, c0, c1, xbar) - ys

                res = least_squares(resid, p0, bounds=(lb, ub), loss=robust_loss, f_scale=1.0, max_nfev=20000)
                f0, g, A, c0, c1 = res.x
                yfit = lorentz_lin(x, f0, g, A, c0, c1, xbar)

            fwhm = 2.0 * abs(g)
            rms = float(np.sqrt(np.mean((yfit - y) ** 2)))
            return dict(f0=float(f0), FWHM=float(fwhm), params=res.x, yfit=yfit, rms=rms, success=bool(res.success))

        two_chi_MHz = 2.0 * float(chi_MHz)

        # ---------- flatten inputs ----------
        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        n = min(len(amps_q), len(gains_q), len(rounds_q), len(delay_q))
        if n == 0 or not (len(amps_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
            print(f"No usable data for qubit {q} (missing lists or length mismatch). Skipping.")
            return None if return_nbar else None

        all_points = []
        for i in range(n):
            r_id = str(rounds_q[i])
            a_samples = np.asarray(amps_q[i], float).ravel()
            if a_samples.size == 0: continue

            g_i = gains_q[i]
            g_arr = np.asarray(g_i, float).ravel() if isinstance(g_i, (list, tuple, np.ndarray)) else None
            if g_arr is None or g_arr.size == 1:
                try:
                    g_scalar = float(g_i)
                except Exception:
                    continue
                g_arr = np.full(a_samples.shape, g_scalar, float)
            elif g_arr.size != a_samples.size:
                continue

            d_i = delay_q[i]
            d_arr = np.asarray(d_i, float).ravel() if isinstance(d_i, (list, tuple, np.ndarray)) else None
            if d_arr is None or d_arr.size == 1:
                try:
                    d_scalar = float(d_i)
                except Exception:
                    continue
                d_arr = np.full(a_samples.shape, d_scalar, float)
            elif d_arr.size != a_samples.size:
                continue

            mask = np.isfinite(a_samples) & np.isfinite(g_arr) & np.isfinite(d_arr)
            if not np.any(mask): continue
            for gval, dval, aval in zip(g_arr[mask], d_arr[mask], a_samples[mask]):
                all_points.append((r_id, float(gval), float(dval), float(aval)))

        if not all_points:
            print(f"No numeric points for qubit {q}. Skipping.")
            return None if return_nbar else None

        unique_rounds = sorted({r for (r, _, __, ___) in all_points})

        # folders
        self.create_folder_if_not_exists(save_path)
        indiv_root = os.path.join(save_path, individual_subfolder);
        self.create_folder_if_not_exists(indiv_root)
        lorentz_root = os.path.join(save_path, lorentz_fit_folder);
        self.create_folder_if_not_exists(lorentz_root)

        # global color scale
        all_amps = np.array([a for (_, _, _, a) in all_points], float)
        global_vmin = float(np.nanmin(all_amps));
        global_vmax = float(np.nanmax(all_amps))

        # --- NEW: where we collect outputs if requested ---
        nbar_return = {}  # { round_id: {"gains": [...], "gaussian": [...]/None, "lorentzian": [...]/None} }

        for r_id in unique_rounds:
            pts = [(g, d, a) for (r, g, d, a) in all_points if r == r_id]
            if not pts: continue

            gains_r = sorted({g for (g, _, _) in pts})
            delays_r = sorted({d for (_, d, _) in pts})
            gi_map = {g: i for i, g in enumerate(gains_r)}
            di_map = {d: i for i, d in enumerate(delays_r)}

            # grid C[delay, gain]
            bucket = defaultdict(list)
            for g, d, a in pts:
                bucket[(di_map[d], gi_map[g])].append(a)
            Ny, Nx = len(delays_r), len(gains_r)
            C = np.full((Ny, Nx), np.nan)
            for (iy, ix), vals in bucket.items():
                C[iy, ix] = float(np.nanmean(vals))

            # per-gain fits
            round_folder = os.path.join(indiv_root, f"round_{r_id}");
            self.create_folder_if_not_exists(round_folder)
            round_lorentz_folder = os.path.join(lorentz_root, f"round_{r_id}");
            self.create_folder_if_not_exists(round_lorentz_folder)

            x_all = np.array(delays_r, float)  # MHz
            centers_G, widths_G = [], []
            centers_L, widths_L, rms_L = [], [], []
            gains_used = []

            for gi, g in enumerate(gains_r):
                y = C[:, gi]
                if not np.isfinite(y).any():
                    centers_G.append(np.nan);
                    widths_G.append(np.nan)
                    centers_L.append(np.nan);
                    widths_L.append(np.nan);
                    rms_L.append(np.nan)
                    continue

                m = np.isfinite(x_all) & np.isfinite(y)
                x = x_all[m];
                yy = y[m]
                if x.size < 5:
                    centers_G.append(np.nan);
                    widths_G.append(np.nan)
                    centers_L.append(np.nan);
                    widths_L.append(np.nan);
                    rms_L.append(np.nan)
                    continue

                # quick guesses (Gaussian)
                i0 = np.argmin(yy) if np.ptp(yy) > 0 else np.argmax(yy)
                x0g = x[i0];
                H = float(np.median(yy));
                Ag = float(yy[i0] - H)
                sig = max((x.max() - x.min()) / 10.0, 1e-3)

                fig2, ax2 = plt.subplots(figsize=(6.5, 4.0))
                ax2.plot(x, yy, 'o', ms=3, label='data')

                if fit_gaussian:
                    try:
                        pg, _ = curve_fit(gauss, x, yy, p0=[H, Ag, x0g, sig], maxfev=10000)
                        centers_G.append(pg[2]);
                        widths_G.append(2.355 * abs(pg[3]))
                        ax2.plot(x, gauss(x, *pg), '--', label='Gaussian')
                    except Exception:
                        centers_G.append(np.nan);
                        widths_G.append(np.nan)

                # robust Lorentzian + baseline
                if fit_lorentzian:
                    fit = fit_slice(x, yy)
                    centers_L.append(fit["f0"]);
                    widths_L.append(fit["FWHM"]);
                    rms_L.append(fit["rms"])
                    ax2.plot(x_all, fit["yfit"], '-', lw=2, label='Lorentz+baseline')
                    ax2.axvline(fit["f0"], color='k', ls=':', lw=1)
                    ax2.text(0.02, 0.95,
                             f"f0={fit['f0']:.2f} MHz\nFWHM={fit['FWHM']:.2f} MHz\nRMS={fit['rms']:.3g}",
                             transform=ax2.transAxes, va='top', ha='left', fontsize=9)

                    # save artifacts (PNG + NPZ)
                    gain_str = f"{g:.6g}".replace("/", "_")
                    out_png = os.path.join(round_lorentz_folder,
                                           f"lorentz_q{self.qubit}_round{r_id}_gain{gain_str}.png")
                    ax2.set_title(f"Q{self.qubit + 1} — Round {r_id} — Gain {g:g}")
                    ax2.set_xlabel("Frequency (MHz)");
                    ax2.set_ylabel("Qubit Population")
                    ax2.grid(True, alpha=0.3);
                    ax2.legend()
                    fig2.tight_layout();
                    fig2.savefig(out_png, dpi=self.final_figure_quality)

                    out_npz = os.path.join(round_lorentz_folder,
                                           f"lorentz_q{self.qubit}_round{r_id}_gain{gain_str}.npz")
                    np.savez_compressed(out_npz,
                                        gain=float(g), x_data=x, y_data=yy,
                                        x_full=x_all, y_fit_full=fit["yfit"],
                                        f0=fit["f0"], FWHM=fit["FWHM"], rms=fit["rms"],
                                        params=np.asarray(fit["params"]))
                # also save generic per-gain plot
                gain_str = f"{g:.6g}".replace("/", "_")
                out_indiv = os.path.join(round_folder, f"qspec_q{self.qubit}_round_{r_id}_gain{gain_str}.png")
                fig2.savefig(out_indiv, dpi=self.final_figure_quality)
                plt.close(fig2)

                gains_used.append(g)

            gains_used = np.array(gains_used, float)  # ascending because gains_r was sorted
            pwr = gains_used ** 2

            # heatmap + overlay centers
            x_edges = centers_to_edges(gains_r);
            y_edges = centers_to_edges(delays_r)
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            mesh = ax.pcolormesh(x_edges, y_edges, C, shading='flat', vmin=global_vmin, vmax=global_vmax)
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02);
            cbar.set_label("Qubit Population")

            centers_overlay = np.array(centers_L if fit_lorentzian else centers_G, float)
            ok_overlay = np.isfinite(centers_overlay) & np.isfinite(gains_used)
            if ok_overlay.any():
                ax.plot(gains_used[ok_overlay], centers_overlay[ok_overlay],
                        'o', ms=4, mfc='none', mec='w', mew=1.5, label='fit centers')
                ax.plot(gains_used[ok_overlay], centers_overlay[ok_overlay], '.', ms=2, color='k')
                ax.legend(loc='best')

            ax.set_title(f"Qubit {self.qubit + 1} — Round {r_id}")
            ax.set_xlabel("Pulse gain (a.u.)");
            ax.set_ylabel("Frequency (MHz)")
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            if Ny > 0:
                yt = list(range(Ny)) if Ny <= max_ylabels else sorted(
                    set(np.linspace(0, Ny - 1, max_ylabels, dtype=int)))
                ax.set_yticks([delays_r[i] for i in yt])
                ax.set_yticklabels([f"{delays_r[i]:.2f}" for i in yt])
            fig.tight_layout()
            out_hm = os.path.join(save_path, f"qspec_heatmap_q{self.qubit}_round{r_id}.png")
            fig.savefig(out_hm, dpi=self.final_figure_quality);
            plt.close(fig)
            print(f"Saved heatmap for round {r_id} to: {out_hm}")

            # --- helper to regress & plot + (NEW) capture nbar series
            # Will return the nbar array (aligned with gains_used) or None.
            def do_regress_and_plot(label, centers, widths):
                centers = np.array(centers, float)
                ok = np.isfinite(centers) & np.isfinite(pwr)
                if ok.sum() < 2:
                    # Not enough points to fit; still return an array of NaNs aligned to gains_used
                    nbar_series = np.full_like(gains_used, np.nan, dtype=float)
                    return nbar_series

                slope, intercept, r, pval, stderr = linregress(pwr[ok], centers[ok])
                wq = slope * pwr + intercept
                nbar = (wq - intercept) / (2.0 * chi_MHz)  # sign convention unchanged

                fig3, ax3 = plt.subplots(1, 3, figsize=(15, 4.2))
                ax3[0].plot(pwr, centers, 'o', ms=4, label=f'{label} centers')
                ax3[0].plot(pwr, wq, '-', label=f'fit (r={r:.3f})')
                ax3[0].legend();
                ax3[0].grid(alpha=0.3)
                ax3[0].set_xlabel('Gain^2 (a.u.)');
                ax3[0].set_ylabel('Center freq (MHz)')

                ax3[1].plot(gains_used, nbar, '-', label=f'{label} n̄')
                ax3[1].legend();
                ax3[1].grid(alpha=0.3)
                ax3[1].set_xlabel('Gain (a.u.)');
                ax3[1].set_ylabel('n̄')

                if np.isfinite(widths).any():
                    ax3[2].plot(gains_used, widths, '-', label=f'{label} FWHM')
                    ax3[2].legend();
                    ax3[2].grid(alpha=0.3)
                    ax3[2].set_xlabel('Gain (a.u.)');
                    ax3[2].set_ylabel('FWHM (MHz)')

                fig3.suptitle(
                    f"Q{self.qubit + 1} Round {r_id} — {label}: slope={slope:.4g} MHz/a.u.^2, q0={intercept:.4g} MHz")
                fig3.tight_layout()
                out_sum = os.path.join(save_path, f"q{self.qubit}_round{r_id}_{label}_centers_nbar.png")
                fig3.savefig(out_sum, dpi=self.final_figure_quality);
                plt.close(fig3)

                return nbar

            # compute & collect nbar (aligned with ascending gains_used)
            gaussian_nbar = None
            lorentzian_nbar = None

            if fit_gaussian and len(centers_G):
                gaussian_nbar = do_regress_and_plot("gaussian", centers_G, np.array(widths_G, float))
            if fit_lorentzian and len(centers_L):
                lorentzian_nbar = do_regress_and_plot("lorentzian", centers_L, np.array(widths_L, float))

            if return_nbar:
                # Ensure lists and ascending gains order
                nbar_return[r_id] =  nbar_return[r_id] = {
                        "gains": list(map(float, gains_used.tolist())),
                        "gaussian": None if gaussian_nbar is None else list(map(float, np.asarray(gaussian_nbar, float))),
                        "lorentzian": None if lorentzian_nbar is None else list(map(float, np.asarray(lorentzian_nbar, float))),
                    }

        # NEW: return results if requested; otherwise, keep legacy behavior (no return)
        if return_nbar:
            return nbar_return

    def plot_all_q_heatmaps_with_singular_ssf_plotting(
            self,
            amps,
            gains,
            rounds,
            delay_times,
            save_path,
            max_ylabels=6,
            # --- optional SSD inputs (same contract as the T1/T2 versions) ---
            ss_class_instance=None,
            ss_cfg=None,
            Ig_calibration=None,
            Ie_calibration=None,
            Qg_calibration=None,
            Qe_calibration=None,
            # --- experiment single-shot sources (same format as amps) ---
            I_experiment=None,
            Q_experiment=None,
    ):
        """
        NEW FORMAT ONLY

        Q-spectrum heatmaps with:
          - fixed global color scale across rounds (global min/max amplitude)
          - Y-axis shows at most `max_ylabels` frequency tick labels (evenly spaced)
          - OPTIONAL: per-dataset SSF plot using provided calibrations, with a single
            (I,Q) experiment point overlaid if available.

        Data model (per qubit q):
          - amps[q]         : list of lists/arrays; amps[q][i] are amplitude samples for dataset i
          - gains[q]        : list; gains[q][i] is scalar or array aligned with amps[q][i]
          - rounds[q]       : list; rounds[q][i] is the round label for dataset i
          - delay_times[q]  : list; delay_times[q][i] scalar or array aligned with amps[q][i]
                              (here they represent frequency points in MHz for the y-axis)
          - I_experiment[q] : list; I_experiment[q][i] is scalar or array of I-shots for dataset i
          - Q_experiment[q] : list; Q_experiment[q][i] is scalar or array of Q-shots for dataset i
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        import matplotlib.ticker as mticker

        # --- helpers (mirrors your T1/T2 versions) ---
        def fmt_p(x, sig=4):
            try:
                s = f"{float(x):.{sig}g}"
            except Exception:
                return "nan"
            return s.replace(".", "p")

        def as_1d_array(x):
            if isinstance(x, (list, tuple, np.ndarray)):
                arr = np.asarray(x).ravel()
                return arr if arr.size > 0 else None
            try:
                return np.asarray([float(x)], dtype=float)
            except Exception:
                return None

        def first_scalar_or_nan(x):
            a = as_1d_array(x)
            if a is None or not np.all(np.isfinite(a)):
                return np.nan
            return float(np.median(a))

        def centers_to_edges(centers):
            centers = np.asarray(sorted(np.unique(centers)), dtype=float)
            if centers.size == 1:
                d = 1.0
                return np.array([centers[0] - d / 2, centers[0] + d / 2])
            mids = (centers[:-1] + centers[1:]) / 2.0
            first = centers[0] - (centers[1] - centers[0]) / 2.0
            last = centers[-1] + (centers[-1] - centers[-2]) / 2.0
            return np.concatenate([[first], mids, [last]])

        def pick_single_shot(x, k=0):
            a = as_1d_array(x)
            if a is None:
                return np.nan
            finite_idx = np.flatnonzero(np.isfinite(a))
            if finite_idx.size == 0:
                return np.nan
            return float(a[finite_idx[0]])

        # --- extract per-qubit lists ---
        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])  # treated as frequency (MHz) here

        # optional experiment I/Q
        Iexp_q = (I_experiment or {}).get(q, [])
        Qexp_q = (Q_experiment or {}).get(q, [])

        # presence & length checks
        n = min(len(amps_q), len(gains_q), len(rounds_q), len(delay_q))
        if n == 0 or not (len(amps_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
            print(f"No usable data for qubit {q} (missing lists or length mismatch). Skipping.")
            return

        # --- OPTIONAL: SSD enablement ---
        do_ssd = (
                ss_class_instance is not None and
                ss_cfg is not None and
                Ig_calibration is not None and
                Ie_calibration is not None and
                Qg_calibration is not None and
                Qe_calibration is not None
        )
        if do_ssd:
            Ig_q = Ig_calibration.get(q, [])
            Ie_q = Ie_calibration.get(q, [])
            Qg_q = Qg_calibration.get(q, [])
            Qe_q = Qe_calibration.get(q, [])
            n_ss = min(len(Ig_q), len(Ie_q), len(Qg_q), len(Qe_q), n)
            if n_ss == 0:
                print("SSD requested but no calibration arrays found; skipping SSD.")
                do_ssd = False

        # --- collect points for heatmaps: (round_id, gain, freq, amp) ---
        all_points = []
        for i in range(n):
            r_id = str(rounds_q[i])

            a_samples = np.asarray(amps_q[i], dtype=float).ravel()
            if a_samples.size == 0:
                continue

            g_i = gains_q[i]
            g_arr = np.asarray(g_i, dtype=float).ravel() if isinstance(g_i, (list, tuple, np.ndarray)) else None
            if g_arr is None or g_arr.size == 1:
                try:
                    g_scalar = float(g_i)
                except Exception:
                    continue
                g_arr = np.full(a_samples.shape, g_scalar, dtype=float)
            elif g_arr.size != a_samples.size:
                continue

            f_i = delay_q[i]  # "delay" list is frequency here
            f_arr = np.asarray(f_i, dtype=float).ravel() if isinstance(f_i, (list, tuple, np.ndarray)) else None
            if f_arr is None or f_arr.size == 1:
                try:
                    f_scalar = float(f_i)
                except Exception:
                    continue
                f_arr = np.full(a_samples.shape, f_scalar, dtype=float)
            elif f_arr.size != a_samples.size:
                continue

            mask = np.isfinite(a_samples) & np.isfinite(g_arr) & np.isfinite(f_arr)
            if not np.any(mask):
                continue

            for g, f, a in zip(g_arr[mask], f_arr[mask], a_samples[mask]):
                all_points.append((r_id, float(g), float(f), float(a)))

        if not all_points:
            print(f"No numeric points for qubit {q}. Skipping.")
            return

        unique_rounds = sorted({r for (r, _, __, ___) in all_points})

        # ensure save root exists
        self.create_folder_if_not_exists(save_path)

        # --- global z scale ---
        all_amps = np.array([a for (_, _, _, a) in all_points], dtype=float)
        global_vmin = float(np.nanmin(all_amps))
        global_vmax = float(np.nanmax(all_amps))

        # --- per-dataset SSF emission (once per Q-spec curve) ---
        if do_ssd:
            analysis_root = os.path.join(save_path, "analysis")
            self.create_folder_if_not_exists(analysis_root)

            _old_outer = getattr(ss_class_instance, "outerFolder", None)
            _old_qidx = getattr(ss_class_instance, "QubitIndex", None)
            _old_rnum = getattr(ss_class_instance, "round_num", None)
            _old_name = getattr(ss_class_instance, "expt_name", None)

            try:
                ss_class_instance.outerFolder = analysis_root
                ss_class_instance.QubitIndex = getattr(self, "qubit", 0)

                for i in range(n_ss):
                    r_id = str(rounds_q[i])
                    g_lbl = first_scalar_or_nan(gains_q[i])
                    f_lbl = first_scalar_or_nan(delay_q[i])

                    ss_class_instance.round_num = r_id
                    gain_tag = f"gain_{fmt_p(g_lbl)}_"
                    freq_tag = f"freqMHz_{fmt_p(f_lbl)}"
                    ss_class_instance.expt_name = f"{gain_tag}{freq_tag}_qspec_ssd"

                    # pull calibration vectors
                    try:
                        I_g = np.asarray(Ig_q[i]).ravel()
                        Q_g = np.asarray(Qg_q[i]).ravel()
                        I_e = np.asarray(Ie_q[i]).ravel()
                        Q_e = np.asarray(Qe_q[i]).ravel()
                        if min(I_g.size, Q_g.size, I_e.size, Q_e.size) == 0:
                            print(f"[SSD] Skipping dataset {i}: empty calibration vectors.")
                            continue
                    except Exception as e:
                        print(f"[SSD] Failed to read calibration for dataset {i}: {e}")
                        continue

                    # pick ONE (I,Q) from experiment dicts
                    I_single = np.nan
                    Q_single = np.nan
                    if i < len(Iexp_q):
                        I_single = pick_single_shot(Iexp_q[i])
                    if i < len(Qexp_q):
                        Q_single = pick_single_shot(Qexp_q[i])

                    kwargs_meas = {}
                    if np.isfinite(I_single) and np.isfinite(Q_single):
                        kwargs_meas = {"I_meas": I_single, "Q_meas": Q_single}

                    try:
                        ss_class_instance.hist_ssf_with_annotations(
                            data=[I_g, Q_g, I_e, Q_e],
                            cfg=ss_cfg,
                            plot=True, path_ext='_qspec',
                            **kwargs_meas
                        )

                    except Exception as e:
                        print(f"[SSD] Failed on dataset {i} (round {r_id}): {e}")

            finally:
                if _old_outer is not None:
                    ss_class_instance.outerFolder = _old_outer
                if _old_qidx is not None:
                    ss_class_instance.QubitIndex = _old_qidx
                if _old_rnum is not None:
                    ss_class_instance.round_num = _old_rnum
                if _old_name is not None:
                    ss_class_instance.expt_name = _old_name

        # --- per-round heatmaps (fixed z across rounds) ---
        for r_id in unique_rounds:
            pts = [(g, f, a) for (r, g, f, a) in all_points if r == r_id]
            if not pts:
                continue

            gains_r = sorted({g for (g, _, _) in pts})
            freqs_r = sorted({f for (_, f, _) in pts})

            bucket = defaultdict(list)
            gi_map = {g: i for i, g in enumerate(gains_r)}
            fi_map = {f: i for i, f in enumerate(freqs_r)}
            for g, f, a in pts:
                bucket[(fi_map[f], gi_map[g])].append(a)

            Ny, Nx = len(freqs_r), len(gains_r)
            C = np.full((Ny, Nx), np.nan, dtype=float)
            for (iy, ix), vals in bucket.items():
                C[iy, ix] = float(np.nanmean(vals))

            x_edges = centers_to_edges(gains_r)
            y_edges = centers_to_edges(freqs_r)

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            mesh = ax.pcolormesh(
                x_edges, y_edges, C, shading='flat',
                vmin=global_vmin, vmax=global_vmax
            )
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label("Qubit Population")

            ax.set_title(f"Qubit {self.qubit + 1} — Round {r_id}")
            ax.set_xlabel("Pulse gain (a.u.)")
            ax.set_ylabel("Frequency (MHz)")

            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7, prune=None))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            if Ny > 0:
                if Ny <= max_ylabels:
                    yticks_idx = list(range(Ny))
                else:
                    yticks_idx = np.linspace(0, Ny - 1, num=max_ylabels, dtype=int).tolist()
                    yticks_idx = sorted(set(yticks_idx))
                yticks_vals = [freqs_r[i] for i in yticks_idx]
                ax.set_yticks(yticks_vals)
                ax.set_yticklabels([f"{v:.2f}" for v in yticks_vals])

            fig.tight_layout()
            outfile = (save_path + f"qspec_heatmap_q{self.qubit}_round{r_id}.png")
            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved heatmap for round {r_id} to: {outfile}")

    def plot_all_q_heatmaps_single_calibration(self,Is, Qs, Ig_calibration1, \
        Ie_calibration1, Qe_calibration1, Qg_calibration1, gains, rounds, delay_times, save_path, max_ylabels=6):
        """
        NEW FORMAT ONLY

        Changes vs your original:
          - Y-axis now shows at most `max_ylabels` delay_time tick labels (evenly spaced).
          - Color scale (z) is fixed across rounds using the global min/max amplitude.

        Data model (per qubit q):
          - amps[q]         : list of lists; amps[q][i] is a list of amplitude samples for dataset i
          - gains[q]        : list; gains[q][i] is the gain for dataset i
          - rounds[q]       : list; rounds[q][i] is the round label for dataset i
          - delay_times[q]  : list; delay_times[q][i] is the delay (scalar) for dataset i
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict

        q = self.qubit
        gains_q = gains.get(q, [])
        I_q = Is.get(q, [])
        Q_q = Qs.get(q, [])

        g = np.mean(Ig_calibration1 + 1j * Qg_calibration1)
        e = np.mean(Ie_calibration1 + 1j * Qe_calibration1)
        denom = np.abs(e - g) ** 2
        if denom <= 0 or not np.isfinite(denom):
            raise ValueError("Best calibration is degenerate (e ≈ g); cannot normalize.")

        # Build amps_q for ALL datasets, using the SAME e/g
        amps_q = []
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])
        n_data = min(len(I_q), len(Q_q), len(gains_q), len(rounds_q), len(delay_q))
        for i in range(n_data):
            I = np.asarray(I_q[i], dtype=float)
            Q = np.asarray(Q_q[i], dtype=float)
            z = I + 1j * Q
            amp_i = np.abs((z - g) * (e - g) / denom)  # == pop_norm
            amps_q.append(np.asarray(amp_i, dtype=float))


        # Basic presence & length checks
        n = min(len(amps_q), len(gains_q), len(rounds_q), len(delay_q))
        if n == 0 or not (len(amps_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
            print(f"No usable data for qubit {q} (missing lists or length mismatch). Skipping.")
            return

        # Flatten to points: (round_id, gain, delay, amp)
        all_points = []
        for i in range(n):
            r_id = str(rounds_q[i])

            # amplitudes (required)
            a_samples = np.asarray(amps_q[i], dtype=float).ravel()
            if a_samples.size == 0:
                continue

            # gain can be scalar or per-sample
            g_i = gains_q[i]
            g_arr = np.asarray(g_i, dtype=float).ravel() if isinstance(g_i, (list, tuple, np.ndarray)) else None
            if g_arr is None or g_arr.size == 1:
                try:
                    g_scalar = float(g_i)
                except Exception:
                    continue
                g_arr = np.full(a_samples.shape, g_scalar, dtype=float)
            elif g_arr.size != a_samples.size:
                continue

            # delay can be scalar or per-sample
            d_i = delay_q[i]
            d_arr = np.asarray(d_i, dtype=float).ravel() if isinstance(d_i, (list, tuple, np.ndarray)) else None
            if d_arr is None or d_arr.size == 1:
                try:
                    d_scalar = float(d_i)
                except Exception:
                    continue
                d_arr = np.full(a_samples.shape, d_scalar, dtype=float)
            elif d_arr.size != a_samples.size:
                continue

            # keep only finite triples
            mask = np.isfinite(a_samples) & np.isfinite(g_arr) & np.isfinite(d_arr)
            if not np.any(mask):
                continue

            for g, d, a in zip(g_arr[mask], d_arr[mask], a_samples[mask]):
                all_points.append((r_id, float(g), float(d), float(a)))

        if not all_points:
            print(f"No numeric points for qubit {q}. Skipping.")
            return

        # Unique rounds present
        unique_rounds = sorted({r for (r, _, __, ___) in all_points})

        # Ensure save folder exists
        self.create_folder_if_not_exists(save_path)

        # Helper to convert centers -> bin edges for pcolormesh
        def centers_to_edges(centers):
            centers = np.asarray(sorted(np.unique(centers)), dtype=float)
            if centers.size == 1:
                d = 1.0
                return np.array([centers[0] - d / 2, centers[0] + d / 2])
            mids = (centers[:-1] + centers[1:]) / 2.0
            first = centers[0] - (centers[1] - centers[0]) / 2.0
            last = centers[-1] + (centers[-1] - centers[-2]) / 2.0
            return np.concatenate([[first], mids, [last]])

        # ---------- NEW: compute global color scale limits (z) ----------
        all_amps = np.array([a for (_, _, _, a) in all_points], dtype=float)
        global_vmin = float(np.nanmin(all_amps))
        global_vmax = float(np.nanmax(all_amps))
        # ----------------------------------------------------------------

        for r_id in unique_rounds:
            # Collect this round's points
            pts = [(g, d, a) for (r, g, d, a) in all_points if r == r_id]
            if not pts:
                continue

            gains_r = sorted({g for (g, _, _) in pts})
            delays_r = sorted({d for (_, d, _) in pts})

            # Map (delay_idx, gain_idx) -> list of amplitudes
            from collections import defaultdict
            bucket = defaultdict(list)
            gi_map = {g: i for i, g in enumerate(gains_r)}
            di_map = {d: i for i, d in enumerate(delays_r)}
            for g, d, a in pts:
                bucket[(di_map[d], gi_map[g])].append(a)

            # Grid of average amplitudes
            Ny, Nx = len(delays_r), len(gains_r)
            C = np.full((Ny, Nx), np.nan, dtype=float)
            for (iy, ix), vals in bucket.items():
                C[iy, ix] = float(np.nanmean(vals))

            # Bin edges for pcolormesh
            x_edges = centers_to_edges(gains_r)
            y_edges = centers_to_edges(delays_r)

            # Plot
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            mesh = ax.pcolormesh(
                x_edges, y_edges, C, shading='flat',
                vmin=global_vmin, vmax=global_vmax  # <-- fixed z scale
            )
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label("Qubit Population")

            ax.set_title(f"Qubit {self.qubit + 1} — Round {r_id}")
            ax.set_xlabel("Pulse gain (a.u.)")
            ax.set_ylabel("Frequency (MHz)")

            # X ticks at actual centers
            import matplotlib.ticker as mticker
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7, prune=None))

            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # ---------- NEW: only label a subset of delay times on Y ----------
            if Ny > 0:
                if Ny <= max_ylabels:
                    # small: show all
                    yticks_idx = list(range(Ny))
                else:
                    # large: pick evenly spaced indices
                    yticks_idx = np.linspace(0, Ny - 1, num=max_ylabels, dtype=int).tolist()
                    # ensure uniqueness/monotonic
                    yticks_idx = sorted(set(yticks_idx))

                yticks_vals = [delays_r[i] for i in yticks_idx]
                ax.set_yticks(yticks_vals)
                ax.set_yticklabels([f"{v:.2f}" for v in yticks_vals])
            # -------------------------------------------------------------------

            fig.tight_layout()
            outfile = (save_path + f"qspec_heatmap_q{self.qubit}_round{r_id}.png")
            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved heatmap for round {r_id} to: {outfile}")

    def plot_all_q_heatmaps_single_calibration_IQ(self, Is, Qs, Ig_calibration1,
                                               Ie_calibration1, Qe_calibration1, Qg_calibration1, gains, rounds,
                                               delay_times, save_path, max_ylabels=6):
        """
        NEW FORMAT ONLY — now plots 4 heatmaps per round (I, Q, |IQ|, Calibrated)

        Changes vs previous:
          - Adds subplots for I, Q, and raw amplitude sqrt(I^2 + Q^2) alongside calibrated amplitude.
          - Keeps fixed color scales (per-metric) across rounds.
          - Y-axis shows at most `max_ylabels` frequency tick labels (evenly spaced).

        Data model (per qubit q):
          - Is[q][i], Qs[q][i] : 1D arrays of samples for dataset i
          - gains[q][i]        : scalar or per-sample array
          - rounds[q][i]       : round label for dataset i (str/int)
          - delay_times[q][i]  : scalar or per-sample array (here: frequency in MHz)
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        import matplotlib.ticker as mticker

        q = self.qubit
        gains_q = gains.get(q, [])
        rounds_q = rounds.get(q, [])
        freq_q = delay_times.get(q, [])  # frequency (MHz)
        I_q = Is.get(q, [])
        Q_q = Qs.get(q, [])

        # --- Calibration (shared for all datasets) ---
        g = np.mean(Ig_calibration1 + 1j * Qg_calibration1)
        e = np.mean(Ie_calibration1 + 1j * Qe_calibration1)
        denom = np.abs(e - g) ** 2
        if denom <= 0 or not np.isfinite(denom):
            raise ValueError("Best calibration is degenerate (e ≈ g); cannot normalize.")

        # Ensure save folder exists
        self.create_folder_if_not_exists(save_path)

        def centers_to_edges(centers):
            centers = np.asarray(sorted(np.unique(centers)), dtype=float)
            if centers.size == 1:
                d = 1.0
                return np.array([centers[0] - d / 2, centers[0] + d / 2])
            mids = (centers[:-1] + centers[1:]) / 2.0
            first = centers[0] - (centers[1] - centers[0]) / 2.0
            last = centers[-1] + (centers[-1] - centers[-2]) / 2.0
            return np.concatenate([[first], mids, [last]])

        # Build per-sample points for each metric
        n_data = min(len(I_q), len(Q_q), len(gains_q), len(rounds_q), len(freq_q))
        if n_data == 0:
            print(f"No usable data for qubit {q} (missing lists). Skipping.")
            return

        # all_points_<metric>: list of tuples (round_id, gain, freq, value)
        all_points_I = []
        all_points_Q = []
        all_points_abs = []
        all_points_cal = []

        for i in range(n_data):
            I = np.asarray(I_q[i], dtype=float).ravel()
            Q = np.asarray(Q_q[i], dtype=float).ravel()
            if I.size == 0 or Q.size == 0 or I.size != Q.size:
                continue

            z = I + 1j * Q
            amp_cal = np.abs((z - g) * (e - g) / denom)  # calibrated (population-like)
            amp_raw = np.sqrt(I ** 2 + Q ** 2)  # raw amplitude

            # gain can be scalar or per-sample
            g_i = gains_q[i]
            if isinstance(g_i, (list, tuple, np.ndarray)):
                g_arr = np.asarray(g_i, dtype=float).ravel()
                if g_arr.size not in (1, I.size):
                    continue
                if g_arr.size == 1:
                    g_arr = np.full(I.shape, float(g_arr[0]), dtype=float)
            else:
                try:
                    g_arr = np.full(I.shape, float(g_i), dtype=float)
                except Exception:
                    continue

            # frequency can be scalar or per-sample
            f_i = freq_q[i]
            if isinstance(f_i, (list, tuple, np.ndarray)):
                f_arr = np.asarray(f_i, dtype=float).ravel()
                if f_arr.size not in (1, I.size):
                    continue
                if f_arr.size == 1:
                    f_arr = np.full(I.shape, float(f_arr[0]), dtype=float)
            else:
                try:
                    f_arr = np.full(I.shape, float(f_i), dtype=float)
                except Exception:
                    continue

            r_id = str(rounds_q[i])

            # keep only finite entries
            mask = (np.isfinite(I) & np.isfinite(Q) &
                    np.isfinite(amp_raw) & np.isfinite(amp_cal) &
                    np.isfinite(g_arr) & np.isfinite(f_arr))
            if not np.any(mask):
                continue

            for g_val, f_val, iv, qv, av, cv in zip(g_arr[mask], f_arr[mask], I[mask], Q[mask], amp_raw[mask],
                                                    amp_cal[mask]):
                g_val = float(g_val);
                f_val = float(f_val)
                all_points_I.append((r_id, g_val, f_val, float(iv)))
                all_points_Q.append((r_id, g_val, f_val, float(qv)))
                all_points_abs.append((r_id, g_val, f_val, float(av)))
                all_points_cal.append((r_id, g_val, f_val, float(cv)))

        # If nothing collected, bail
        if not (all_points_I and all_points_Q and all_points_abs and all_points_cal):
            print(f"No numeric points for qubit {q}. Skipping.")
            return

        unique_rounds = sorted({r for (r, _, __, ___) in all_points_cal})

        # Compute per-metric global color scales (fixed across rounds)
        def global_minmax(points):
            vals = np.array([v for (_, _, __, v) in points], dtype=float)
            return float(np.nanmin(vals)), float(np.nanmax(vals))

        vmin_I, vmax_I = global_minmax(all_points_I)
        vmin_Q, vmax_Q = global_minmax(all_points_Q)
        vmin_abs, vmax_abs = global_minmax(all_points_abs)
        vmin_cal, vmax_cal = global_minmax(all_points_cal)

        # Helper: grid and heatmap data for a given round and point-list
        def grid_for_round(r_id, points):
            pts = [(g, f, a) for (r, g, f, a) in points if r == r_id]
            if not pts:
                return None
            gains_r = sorted({g for (g, _, _) in pts})
            freqs_r = sorted({f for (_, f, _) in pts})
            gi_map = {g: i for i, g in enumerate(gains_r)}
            fi_map = {f: i for i, f in enumerate(freqs_r)}
            bucket = defaultdict(list)
            for g, f, a in pts:
                bucket[(fi_map[f], gi_map[g])].append(a)
            Ny, Nx = len(freqs_r), len(gains_r)
            C = np.full((Ny, Nx), np.nan, dtype=float)
            for (iy, ix), vals in bucket.items():
                C[iy, ix] = float(np.nanmean(vals))
            x_edges = centers_to_edges(gains_r)
            y_edges = centers_to_edges(freqs_r)
            return (x_edges, y_edges, C, gains_r, freqs_r)

        for r_id in unique_rounds:
            gi = grid_for_round(r_id, all_points_I)
            gq = grid_for_round(r_id, all_points_Q)
            ga = grid_for_round(r_id, all_points_abs)
            gc = grid_for_round(r_id, all_points_cal)
            if not all([gi, gq, ga, gc]):
                print(f"Round {r_id}: incomplete metric grids; skipping.")
                continue

            (xI, yI, CI, gains_Ir, freqs_Ir) = gi
            (xQ, yQ, CQ, gains_Qr, freqs_Qr) = gq
            (xA, yA, CA, gains_Ar, freqs_Ar) = ga
            (xC, yC, CC, gains_Cr, freqs_Cr) = gc

            # Plot 2x2 subplots
            fig, axes = plt.subplots(2, 2, figsize=(11, 8))
            axI, axQ = axes[0]
            axA, axC = axes[1]

            meshI = axI.pcolormesh(xI, yI, CI, shading='flat', vmin=vmin_I, vmax=vmax_I)
            meshQ = axQ.pcolormesh(xQ, yQ, CQ, shading='flat', vmin=vmin_Q, vmax=vmax_Q)
            meshA = axA.pcolormesh(xA, yA, CA, shading='flat', vmin=vmin_abs, vmax=vmax_abs)
            meshC = axC.pcolormesh(xC, yC, CC, shading='flat', vmin=vmin_cal, vmax=vmax_cal)

            # Titles
            axI.set_title("I")
            axQ.set_title("Q")
            axA.set_title("|IQ| = sqrt(I^2 + Q^2)")
            axC.set_title("Calibrated amplitude")

            # Axis labels & ticks
            for ax in (axI, axQ, axA, axC):
                ax.set_xlabel("Pulse gain (a.u.)")
                ax.set_ylabel("Frequency (MHz)")
                ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # Limit number of Y tick labels (per axis, MHz formatting)
            def set_y_ticks(ax, freq_centers):
                Ny = len(freq_centers)
                if Ny == 0:
                    return
                if Ny <= max_ylabels:
                    idx = list(range(Ny))
                else:
                    idx = sorted(set(np.linspace(0, Ny - 1, num=max_ylabels, dtype=int).tolist()))
                vals = [freq_centers[i] for i in idx]
                ax.set_yticks(vals)
                ax.set_yticklabels([f"{v:.2f}" for v in vals])

            set_y_ticks(axI, freqs_Ir)
            set_y_ticks(axQ, freqs_Qr)
            set_y_ticks(axA, freqs_Ar)
            set_y_ticks(axC, freqs_Cr)

            # Colorbars (independent per panel)
            cI = fig.colorbar(meshI, ax=axI, pad=0.02);
            cI.set_label("I (a.u.)")
            cQ = fig.colorbar(meshQ, ax=axQ, pad=0.02);
            cQ.set_label("Q (a.u.)")
            cA = fig.colorbar(meshA, ax=axA, pad=0.02);
            cA.set_label("|IQ| (a.u.)")
            cCal = fig.colorbar(meshC, ax=axC, pad=0.02);
            cCal.set_label("Qubit Population")

            fig.suptitle(f"Qubit {self.qubit + 1} — Round {r_id}", y=0.98)
            fig.tight_layout(rect=[0, 0, 1, 0.96])

            outfile = (save_path + f"qspec_heatmap_quad_q{self.qubit}_round{r_id}.png")
            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved 4-panel heatmaps for round {r_id} to: {outfile}")

    def plot_without_errs(self, date_times, qubit_frequencies, show_legends):
        # ---------------------------------plot-----------------------------------------------------
        if self.fridge.upper() == 'QUIET':
            analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        elif self.fridge.upper() == 'NEXUS':
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

        # ----------------To Plot a specific timeframe------------------
        from datetime import datetime
        year = 2025
        month = 1
        day1 = 24  # Start date
        day2 = 25  # End date
        hour_start = 0  # Start hour
        hour_end = 12  # End hour
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 0)
        # -----------------------------------------------------------------

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.title('Qubit Frequencies vs Time', fontsize=font)
        axes = axes.flatten()

        from datetime import datetime
        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:  # If we have fewer qubits than subplots, stop plotting and hide the rest
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            x = date_times[i]
            y = qubit_frequencies[i]

            # Convert strings to datetime objects.
            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            # Combine datetime objects and y values into a list of tuples and sort by datetime.
            combined = list(zip(datetime_objects, y))
            combined.sort(reverse=True, key=lambda x: x[0])

            if len(combined) == 0:
                # If this qubit has no data, just skip
                ax.set_visible(False)
                continue

            # Unpack them back into separate lists, in order from latest to most recent.
            sorted_x, sorted_y = zip(*combined)
            ax.scatter(sorted_x, sorted_y, color=colors[i])

            # Set x-axis limits for the specific timeframe
            #ax.set_xlim(start_time, end_time)

            #ax.set_ylim(sorted_y[0] - 2.0, sorted_y[0] + 2.0)

            sorted_x = np.asarray(sorted(x))

            num_points = 5
            indices = np.linspace(0, len(sorted_x) - 1, num_points, dtype=int)

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically choose good tick locations
            # ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))  # Format as month-day
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))  # Show day and time
            ax.tick_params(axis='x', rotation=45)  # Rotate ticks for better readability

            # Disable scientific notation and format y-ticks
            ax.ticklabel_format(style="plain", axis="y")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))  # 2 decimal places

            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time', fontsize=font - 2)
            ax.set_ylabel('Qubit Frequency (MHz)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'Q_Freqs_no_errs.pdf', transparent=True, dpi=self.final_figure_quality)
        plt.close()

    def plot_hist(self,  qubit_frequencies, show_legends):
        # ---------------------------------Setup Analysis Folder-----------------------------------------------------
        if self.fridge.upper() == 'QUIET':
            analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        elif self.fridge.upper() == 'NEXUS':
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

        # ----------------Histogram Plotting of Qubit Frequencies------------------
        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.suptitle('Histogram of Qubit Frequencies', fontsize=font)
        axes = axes.flatten()

        means = []
        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            # Ignore the date_times; only use qubit_frequencies.
            y = qubit_frequencies[i]

            if len(y) == 0:
                # If this qubit has no data, hide the subplot.
                ax.set_visible(False)
                continue
            y = self.remove_none_values_1D(y)
            # Plot histogram of the frequency data.
            ax.hist(y, bins=50, color=colors[i], edgecolor='black', alpha=0.7)
            means.append(np.mean(y))
            if show_legends:
                ax.legend([f"Freq Data Qubit {i + 1}"], edgecolor='black')
            ax.set_xlabel('Qubit Frequency (MHz)', fontsize=font - 2)
            ax.set_ylabel('Count', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'Q_Freqs_no_errs.pdf', transparent=True, dpi=self.final_figure_quality)
        plt.close()
        return means

    def remove_none_values(self,list1, list2, list3):
        """Removes None values from list1 and their corresponding indices in list2 and list3."""
        if not (len(list1) == len(list2) == len(list3)):
            raise ValueError("All lists must have the same length")

        # Filter out None values and their corresponding elements in list2 and list3
        filtered_data = [(x, y, z) for x, y, z in zip(list1, list2, list3) if x is not None]

        # Unzip to separate the lists
        filtered_list1, filtered_list2, filtered_list3 = zip(*filtered_data) if filtered_data else ([], [], [])

        return list(filtered_list1), list(filtered_list2), list(filtered_list3)

    def remove_none_values_1D(self,list1):
        """Removes None values from list1 and their corresponding indices in list2 and list3."""

        # Filter out None values and their corresponding elements in list2 and list3
        filtered_data = [x for x in list1 if x is not None]

        return filtered_data
    def plot_with_errs(self, date_times, qubit_frequencies, qspec_fit_err, show_legends, exp_extension=''):
        #---------------------------------plot-----------------------------------------------------
        if self.fridge.upper() == 'QUIET':
            analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        elif self.fridge.upper() == 'NEXUS':
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

        # ----------------To Plot a specific timeframe------------------
        from datetime import datetime
        year = 2025
        month = 1
        day1 = 24  # Start date
        day2 = 25  # End date
        hour_start = 0  # Start hour
        hour_end = 12  # End hour
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 0)
        # -----------------------------------------------------------------

        font = 14
        titles = [f"Qubit {i+1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        ext = exp_extension.split('_')[0]
        plt.suptitle(f'Qubit Frequencies vs Time {ext}', fontsize=font)
        axes = axes.flatten()

        from datetime import datetime  # (if not already imported)
        # Loop over each qubit’s data.
        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:  # Hide extra subplots.
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            x = date_times[i]       # list of date strings
            y = qubit_frequencies[i]
            err = qspec_fit_err[i]  # corresponding error bars

            # Convert date strings to datetime objects.
            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            # Combine datetime objects, frequencies, and error values, then sort in ascending order.
            combined = list(zip(datetime_objects, y, err))
            combined.sort(key=lambda tup: tup[0])  # sort by time (oldest first)

            if len(combined) == 0:
                # Skip if there is no data for this qubit.
                ax.set_visible(False)
                continue

            # Unpack the sorted data.
            sorted_x, sorted_y, sorted_err = zip(*combined)
            sorted_x = np.array(sorted_x)
            sorted_y, sorted_x,sorted_err = self.remove_none_values(sorted_y,sorted_x,sorted_err)
            #try:
            ax.errorbar(
                sorted_x, sorted_y, yerr=sorted_err,
                fmt='none',
                ecolor=colors[i],
                elinewidth=1,
                capsize=0
            )
            #except:
            #    print(sorted_x,sorted_y)

            ax.scatter(
                sorted_x, sorted_y,
                s=10,
                color=colors[i],
                alpha=0.5
            )

            num_points = 5
            indices = np.linspace(0, len(sorted_x) - 1, num_points, dtype=int)
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            ax.tick_params(axis='x', rotation=45)

            ax.ticklabel_format(style="plain", axis="y")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))

            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time', fontsize=font-2)
            ax.set_ylabel('Qubit Frequency (MHz)', fontsize=font-2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + f'Q_Freqs{exp_extension}.pdf', transparent=True, dpi=self.final_figure_quality)
        plt.close()

    def plot_with_errs_single_plot(self, date_times, qubit_frequencies, qspec_fit_err, show_legends):
        # ---------------------------------folder setup-----------------------------------------------------
        if self.fridge.upper() == 'QUIET':
            analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        elif self.fridge.upper() == 'NEXUS':
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/"
            self.create_folder_if_not_exists(analysis_folder)
            analysis_folder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
            self.create_folder_if_not_exists(analysis_folder)
        else:
            raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

        from datetime import datetime
        year = 2025
        month = 1
        day1 = 24  # Start date
        day2 = 25  # End date
        hour_start = 0  # Start hour
        hour_end = 12  # End hour
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 0)

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('Qubit Frequencies vs Time', fontsize=font)

        for i in range(self.number_of_qubits):
            x = date_times[i]
            y = qubit_frequencies[i]
            err = qspec_fit_err[i]

            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            combined = list(zip(datetime_objects, y, err))
            combined.sort(key=lambda tup: tup[0])

            if len(combined) == 0:
                continue

            sorted_x, sorted_y, sorted_err = zip(*combined)
            sorted_x = np.array(sorted_x)

            ax.errorbar(
                sorted_x, sorted_y, yerr=sorted_err,
                fmt='none',
                ecolor=colors[i],
                elinewidth=1,
                capsize=0,
                label=titles[i] if show_legends else None
            )
            ax.scatter(
                sorted_x, sorted_y,
                s=10,
                color=colors[i],
                alpha=0.5
            )

        import matplotlib.dates as mdates
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis='x', rotation=45)

        ax.ticklabel_format(style="plain", axis="y")
        from matplotlib.ticker import StrMethodFormatter
        ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))

        if show_legends:
            ax.legend(edgecolor='black')

        ax.set_xlabel('Time', fontsize=font - 2)
        ax.set_ylabel('Qubit Frequency (MHz)', fontsize=font - 2)
        ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'Q_Freqs_single_plot.pdf', transparent=True, dpi=self.final_figure_quality)
        plt.close()

