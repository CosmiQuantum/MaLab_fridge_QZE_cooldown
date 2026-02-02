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
import math
import ast
import os
import matplotlib.pyplot as plt
import allantools
from scipy.stats import norm
from scipy.optimize import curve_fit
import matplotlib.dates as mdates
from matplotlib.ticker import StrMethodFormatter

class T1VsTime:
    def __init__(self, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, fridge, exp_name = 'ge', qubit=0,t1_slice='10us'):
        self.save_figs = save_figs
        self.qubit=qubit
        self.t1_slice=t1_slice
        self.fit_saved = fit_saved
        self.signal = signal
        self.figure_quality = figure_quality
        self.run_name = run_name
        self.number_of_qubits = number_of_qubits
        self.final_figure_quality = final_figure_quality
        self.top_folder_dates = top_folder_dates
        self.fridge = fridge
        self.exp_name = exp_name

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

    def _is_list_of_lists(self,x):
        return isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple))

    def _flatten_all_sublists(self,arr):
        """Flatten list or list-of-lists to 1D np.array(float)."""
        import numpy as np
        if self._is_list_of_lists(arr):
            return np.asarray([v for sub in arr for v in sub], dtype=float)
        return np.asarray(arr, dtype=float)

    def compute_ssf_discriminant(self, Ig_list, Qg_list, Ie_list, Qe_list, c=5.5):
        import numpy as np
        Ig = self._flatten_all_sublists(Ig_list);
        Qg = self._flatten_all_sublists(Qg_list)
        Ie = self._flatten_all_sublists(Ie_list);
        Qe = self._flatten_all_sublists(Qe_list)

        zg = Ig + 1j * Qg
        ze = Ie + 1j * Qe

        mu_g = self.robust_center(zg, c=c)
        mu_e = self.robust_center(ze, c=c)

        v = mu_e - mu_g
        norm2 = (v.real * v.real + v.imag * v.imag)
        return {"mu_g": mu_g, "mu_e": mu_e, "v": v, "norm2": norm2}

    def project_population(self, z, disc):
        import numpy as np
        mu_g = disc["mu_g"];
        v = disc["v"];
        norm2 = disc["norm2"]
        t = np.real((z - mu_g) * np.conj(v)) / (norm2 + 1e-12)
        return np.clip(t, 0.0, 1.0)

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

    def run(self, return_errs = False, exp_extension='', just_data=False):
        import datetime

        # ----------Load/get data------------------------
        t1_vals = {i: [] for i in range(self.number_of_qubits)}
        t1_errs = {i: [] for i in range(self.number_of_qubits)}
        scaled_amps = {i: [] for i in range(self.number_of_qubits)}
        rounds = []
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}
        #print(self.top_folder_dates)
        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                outerFolder = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date + "/study_data"+ "/"
                outerFolder_save_plots = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date + "_plots/"
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date+ "/study_data" + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # ------------------------------------------------Load/Plot/Save T1----------------------------------------------
            if '_' in exp_extension:
                outerFolder_expt = outerFolder + f"/Data_h5/T1{exp_extension}/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/T1_ge/"
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

            for h5_file in h5_files:

                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type=f'T1{exp_extension}', save_r=int(save_round),
                                                           scaling=just_data)

                #H5_class_instance.print_h5_contents(h5_file)
                # if '01-27' in outerFolder_expt:
                #     print(load_data)
                # Define specific days to exclude
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data[f'T1{exp_extension}']:
                    for dataset in range(len(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        print(load_data[f'T1{exp_extension}'][q_key])
                        # T1 = load_data['T1'][q_key].get('T1', [])[0][dataset]
                        # errors = load_data['T1'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue
                        try:
                            I = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('I', [])[0][dataset].decode())
                            Q = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('Q', [])[0][dataset].decode())
                            delay_times = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('Delay Times', [])[0][dataset].decode())

                        except:
                            I = load_data[f'T1{exp_extension}'][q_key].get('I', [])[0]
                            Q = load_data[f'T1{exp_extension}'][q_key].get('Q', [])[0]
                            delay_times = load_data[f'T1{exp_extension}'][q_key].get('Delay Times', [])[0]

                        if just_data:

                            Ie = self.process_h5_data(
                                load_data[f'T1{exp_extension}'][q_key].get('ss_I_e', [])[0][dataset].decode())
                            Ig = self.process_h5_data(
                                load_data[f'T1{exp_extension}'][q_key].get('ss_I_g', [])[0][dataset].decode())
                            Qe = self.process_h5_data(
                                load_data[f'T1{exp_extension}'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                            Qg = self.process_h5_data(
                                load_data[f'T1{exp_extension}'][q_key].get('ss_Q_g', [])[0][dataset].decode())
                        # fit = load_data['T1'][q_key].get('Fit', [])[0][dataset]
                        round_num = load_data[f'T1{exp_extension}'][q_key].get('Round Num', [])[0][dataset]
                        try:
                            batch_num = load_data[f'T1{exp_extension}'][q_key].get('Batch Num', [])[0][dataset]
                            syst_config = load_data[f'T1{exp_extension}'][q_key].get('Syst Config', [])[0][dataset].decode()
                            exp_config = load_data[f'T1{exp_extension}'][q_key].get('Exp Config', [])[0][dataset].decode()
                            safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                            exp_config = eval(exp_config, safe_globals)
                        except:
                            exp_config =None

                        if len(I) > 0:

                            T1_class_instance = T1Measurement(q_key, self.number_of_qubits, outerFolder_save_plots, round_num, self.signal, self.save_figs,
                                                              fit_data=True)
                            #T1_spec_cfg = exp_config['T1_ge']
                            q1_fit_exponential, T1_err, T1_est, plot_sig = T1_class_instance.t1_fit(I, Q, delay_times)
                            if T1_est < 0:
                                print("The value is negative, continuing...")
                                continue
                            if T1_est > 1000:
                                print("The value is above 1000 us, this is a bad fit, continuing...")
                                continue
                            if T1_err >= 0.8 * T1_est:
                                print(
                                    f"Skipping T1 = {T1_est:.3f} µs because its error {T1_err:.3f} µs is >= 80% of its value.")
                                continue

                            t1_vals[q_key].extend([T1_est])
                            t1_errs[q_key].extend([T1_err])
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])
                            if just_data:

                                I_arr = np.asarray(I, dtype=float)
                                Q_arr = np.asarray(Q, dtype=float)
                                Ie = np.asarray(Ie, dtype=float)
                                Qe = np.asarray(Qe, dtype=float)
                                Ig = np.asarray(Ig, dtype=float)
                                Qg = np.asarray(Qg, dtype=float)
                                print(len(Qe))
                                e = np.mean((Ie + 1j * Qe))
                                g = np.mean((Ig + 1j * Qg))
                                ### Normalization ###
                                pop_norm = abs(((I_arr + 1j * Q_arr) - g) * (e - g) / abs(e - g) ** 2)
                                scaled_amps[q_key].extend([pop_norm])
                            del T1_class_instance
                del H5_class_instance
        if just_data:
            return delay_times, scaled_amps
        elif return_errs:
            return date_times, t1_vals, t1_errs
        else:
            return date_times, t1_vals


    def run_IBM_qze(self, exp_extension=''):
        import datetime

        # ----------Load/get data------------------------
        Is = {i: [] for i in range(self.number_of_qubits)}
        Qs = {i: [] for i in range(self.number_of_qubits)}
        amps = {i: [] for i in range(self.number_of_qubits)}
        gains = {i: [] for i in range(self.number_of_qubits)}
        rounds = []
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}
        #print(self.top_folder_dates)
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
                outerFolder_expt = outerFolder + f"/Data_h5/T1{exp_extension}/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/T1_ge/"
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
            #print(outerFolder_expt)
            for h5_file in h5_files:

                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type=f'T1{exp_extension}', save_r=int(save_round))
                # if '01-27' in outerFolder_expt:
                #     print(load_data)
                # Define specific days to exclude
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data[f'T1{exp_extension}']:
                    for dataset in range(len(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T1 = load_data['T1'][q_key].get('T1', [])[0][dataset]
                        # errors = load_data['T1'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        I = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('Q', [])[0][dataset].decode())
                        #delay_times = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('Delay Times', [])[0][dataset])
                        # fit = load_data['T1'][q_key].get('Fit', [])[0][dataset]
                        round_num = load_data[f'T1{exp_extension}'][q_key].get('Round Num', [])[0][dataset]
                        try:
                            batch_num = load_data[f'T1{exp_extension}'][q_key].get('Batch Num', [])[0][dataset]
                            syst_config = load_data[f'T1{exp_extension}'][q_key].get('Syst Config', [])[0][dataset].decode()
                            exp_config = load_data[f'T1{exp_extension}'][q_key].get('Exp Config', [])[0][dataset].decode()
                            #print(exp_config)
                            safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                            exp_config = eval(exp_config, safe_globals)
                        except:
                            exp_config =None

                        if len(I) > 0:
                            Is[q_key].extend(I)
                            Qs[q_key].extend(Q)
                            gain = round(
                                float(syst_config.split('res_gain_qze\': [')[-1].split(']')[0].split(',')[-1].split('(')[-1].replace(')','')), 6)

                            gains[q_key].append(gain)
                            amp=np.hypot(I, Q)
                            amps[q_key].extend(amp)
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])


                del H5_class_instance
        return Is,Qs,amps, gains

    def run_IBM_qze_rounds(self, exp_extension='', scaling=False):
        import datetime

        # ----------Load/get data------------------------
        Is = {i: [] for i in range(self.number_of_qubits)}
        Qs = {i: [] for i in range(self.number_of_qubits)}
        amps = {i: [] for i in range(self.number_of_qubits)}
        gains = {i: [] for i in range(self.number_of_qubits)}
        rounds_completed = {i: [] for i in range(self.number_of_qubits)}
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}
        #print(self.top_folder_dates)
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
                outerFolder_expt = outerFolder + f"/Data_h5/T1{exp_extension}/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/T1_ge/"
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
                load_data = H5_class_instance.load_from_h5(data_type=f'T1{exp_extension}', save_r=int(save_round), scaling=scaling)
                # if '01-27' in outerFolder_expt:
                #     print(load_data)
                # Define specific days to exclude
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data[f'T1{exp_extension}']:
                    for dataset in range(len(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T1 = load_data['T1'][q_key].get('T1', [])[0][dataset]
                        # errors = load_data['T1'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        try:
                            I = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('I', [])[0][dataset].decode())
                            Q = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('Q', [])[0][dataset].decode())
                            if scaling:

                                Ie = self.process_h5_data(
                                    load_data[f'T1{exp_extension}'][q_key].get('ss_I_e', [])[0][dataset].decode())

                                Ig = self.process_h5_data(
                                    load_data[f'T1{exp_extension}'][q_key].get('ss_I_g', [])[0][dataset].decode())
                                Qe = self.process_h5_data(
                                    load_data[f'T1{exp_extension}'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                                Qg = self.process_h5_data(
                                    load_data[f'T1{exp_extension}'][q_key].get('ss_Q_g', [])[0][dataset].decode())
                        except:
                            I = load_data[f'T1{exp_extension}'][q_key].get('I', [])[0]
                            Q = load_data[f'T1{exp_extension}'][q_key].get('Q', [])[0]

                            if scaling:
                                Ie = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('ss_I_e', [])[0][dataset].decode())
                                Ig = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('ss_I_g', [])[0][dataset].decode())
                                Qe = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                                Qg = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('ss_Q_g', [])[0][dataset].decode())
                        round_num = load_data[f'T1{exp_extension}'][q_key].get('Round Num', [])[0][dataset]
                        try:
                            batch_num = load_data[f'T1{exp_extension}'][q_key].get('Batch Num', [])[0][dataset]
                            syst_config = load_data[f'T1{exp_extension}'][q_key].get('Syst Config', [])[0][dataset].decode()
                            exp_config = load_data[f'T1{exp_extension}'][q_key].get('Exp Config', [])[0][dataset].decode()
                            #print(exp_config)
                            safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                            exp_config = eval(exp_config, safe_globals)
                        except:
                            exp_config =None

                        if len(I) > 0:
                            Is[q_key].extend(I)
                            Qs[q_key].extend(Q)

                            gain = round(
                                float(syst_config.split('res_gain_qze\': ')[-1].split(',')[0]), 6)

                            gains[q_key].append(gain)
                            rounds_completed[q_key].append(round_we_are_on)
                            if scaling:
                                Ie = np.asarray(Ie, dtype=float)
                                Qe = np.asarray(Qe, dtype=float)
                                Ig = np.asarray(Ig, dtype=float)
                                Qg = np.asarray(Qg, dtype=float)
                                e = np.mean((Ie + 1j * Qe))
                                g = np.mean((Ig + 1j * Qg))
                                ### Normalization ###
                                pop_norm = abs(((I + 1j * Q) - g) * (e - g) / abs(e - g) ** 2)
                                amp = pop_norm
                            else:
                                amp=np.hypot(I, Q)
                            amps[q_key].extend(amp)
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                del H5_class_instance
        return Is,Qs,amps, gains, rounds_completed


    def _fd_bins(self,x):
        # Freedman–Diaconis rule → reasonable, data-driven bin count
        x = np.asarray(x)
        iqr = np.subtract(*np.percentile(x, [75, 25]))
        if iqr == 0:
            return 50  # fallback
        bw = 2 * iqr / np.cbrt(len(x))
        rng = np.ptp(x)
        if rng == 0:
            return 50
        return int(np.clip(np.ceil(rng / bw), 30, 200))  # keep it sane

    def mode_center(self,I, Q, bins=None, smooth_sigma=1.0):
        """
        Estimate cluster center as the highest-density point (mode) of a 2D histogram.
        - bins: int or (int, int). If None, uses Freedman–Diaconis per axis.
        - smooth_sigma: Gaussian sigma (in bins) to de-noise the histogram a bit.
        Returns a complex number x + 1j*y.
        """
        from scipy.ndimage import gaussian_filter

        I = np.asarray(I);
        Q = np.asarray(Q)
        if bins is None:
            bins = (self._fd_bins(I), self._fd_bins(Q))

        H, xedges, yedges = np.histogram2d(I, Q, bins=bins)
        if smooth_sigma and smooth_sigma > 0:
            H = gaussian_filter(H, smooth_sigma, mode='nearest')

        idx = np.argmax(H)
        ix, iy = np.unravel_index(idx, H.shape)

        # bin centers
        x0 = 0.5 * (xedges[ix] + xedges[ix + 1])
        y0 = 0.5 * (yedges[iy] + yedges[iy + 1])
        return x0 + 1j * y0

    def run_t1_sweep_single_gain(self, exp_extension='', scaling=False, return_calibration_data=False, weighted_mean=True,
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

        for folder_date in self.top_folder_dates:
            outerFolder = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date + "/study_data"
            # Assuming T1 data structure similar to run_t1_sweep_new
            outerFolder_expt = outerFolder + f"/Data_h5/T1{exp_extension}_zeno/"

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
                # Load T1 data
                load_data = H5_class_instance.load_from_h5(data_type=f'T1{exp_extension}', save_r=int(save_round),
                                                           scaling=scaling)
                
                for q_key in load_data[f'T1{exp_extension}']:
                    for dataset in range(len(load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0])):

                        try:
                            # Load I, Q, Delays
                            I = self.process_h5_data(
                                load_data[f'T1{exp_extension}'][q_key].get('I', [])[0][dataset].decode())
                            Q = self.process_h5_data(
                                load_data[f'T1{exp_extension}'][q_key].get('Q', [])[0][dataset].decode())
                            delays = self.process_h5_data(
                                load_data[f'T1{exp_extension}'][q_key].get('Delay Times', [])[0][dataset].decode())
                            
                            date = datetime.datetime.fromtimestamp(
                                load_data[f'T1{exp_extension}'][q_key].get('Dates', [])[0][dataset])
                        except:
                            continue

                        if scaling:
                            try:
                                Ie = self.process_h5_data(
                                    load_data[f'T1{exp_extension}'][q_key].get('ss_I_e', [])[0][dataset].decode())
                                Ig = self.process_h5_data(
                                    load_data[f'T1{exp_extension}'][q_key].get('ss_I_g', [])[0][dataset].decode())
                                Qe = self.process_h5_data(
                                    load_data[f'T1{exp_extension}'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                                Qg = self.process_h5_data(
                                    load_data[f'T1{exp_extension}'][q_key].get('ss_Q_g', [])[0][dataset].decode())
                            except:
                                # Fallback if scaling requested but data missing
                                scaling = False

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
                            
                            # Expand date to match the number of delay points
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
                # Use a placeholder for the round name
                rounds_completed[q_key].append("combined_sweep")

        return None, None, amps, dates, rounds_completed, delay_times

    def plot_all_t1_heatmaps_single_gain(
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
        Updated to plot Heatmap with Time (Date) on X-axis and Delay Time on Y-axis.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.dates as mdates
        from datetime import datetime
        from collections import defaultdict
        import matplotlib.ticker as mticker
        from scipy.optimize import curve_fit
        from scipy.signal import savgol_filter

        # ---------- Fitting Helpers (Exponential Decay) ----------
        def exponential_decay(x, a, b, c):
            return a * np.exp(-x / b) + c

        def fit_slice(x, y):
            x = np.asarray(x, float); y = np.asarray(y, float)
            try:
                # Initial guesses
                c_guess = np.min(y)
                a_guess = np.max(y) - c_guess
                b_guess = np.mean(x) if np.mean(x) > 0 else 10.0 # simple guess for decay constant
                
                p0 = [a_guess, b_guess, c_guess]
                # Bounds: a>0, b>0, c unbounded
                lb = [0, 0, -np.inf]
                ub = [np.inf, np.inf, np.inf]
                
                popt, pcov = curve_fit(exponential_decay, x, y, p0=p0, bounds=(lb, ub), maxfev=1000)
                return popt[1] # Return decay constant T1
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

            # ---------- Fit Centers (T1 Decay) ----------
            # Note: For T1, we don't plot "centers" like freq shift, but maybe T1 value itself?
            # The previous code plotted "centers_L" which was the Lorentzian center.
            # Here we fit T1 for each time slice (column).
            t1_values = []
            x_delays = np.array(delays_r, float)
            for ix in range(Nx):
                y_col = C[:, ix]
                if np.isfinite(y_col).any() and len(x_delays) >= 5:
                    m = np.isfinite(y_col)
                    t1 = fit_slice(x_delays[m], y_col[m])
                    t1_values.append(t1)
                else:
                    t1_values.append(np.nan)
            t1_values = np.array(t1_values)

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

            # Overlay fits (T1 values) - T1 values are in time units (us), same as Y axis? 
            # Wait, Y axis is Delay Time. T1 is a time constant. 
            # If we plot T1 on this graph, it might be on the same scale, or might not.
            # Typically T1 is within the delay range. So plotting it might make sense.
            if np.any(np.isfinite(t1_values)):
                ax.plot(x_vals_sorted, t1_values, 'o', ms=4, mfc='none', mec='w', mew=1.5, label='T1 Fit')
                ax.plot(x_vals_sorted, t1_values, '.', ms=2, color='k')
                ax.legend(loc='best')

            ax.set_title(f"Qubit {self.qubit + 1} — Round {r_id}")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Delay Time (us)")

            # Format X-axis as dates
            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            fig.tight_layout()
            outfile = (save_path + f"t1_heatmap_q{self.qubit}_round{r_id}.png")

            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved heatmap for round {r_id} to: {outfile}")

    def run_t1_sweep_new(self, exp_extension='', scaling=False, return_calibration_data=False, weighted_mean=True):
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
            outerFolder_expt = outerFolder + f"/Data_h5/T1{exp_extension}_zeno/"
            
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
                load_data = H5_class_instance.load_from_h5(data_type=f'T1{exp_extension}_zeno', save_r=int(save_round), scaling=scaling)

                for q_key in load_data[f'T1{exp_extension}_zeno']:
                    for dataset in range(len(load_data[f'T1{exp_extension}_zeno'][q_key].get('Dates', [])[0])):
                        delays = self.process_h5_data(
                            load_data[f'T1{exp_extension}_zeno'][q_key].get('Delay Times', [])[0][dataset].decode())

                        I = self.process_string_of_nested_lists(
                            load_data[f'T1{exp_extension}_zeno'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_string_of_nested_lists(
                            load_data[f'T1{exp_extension}_zeno'][q_key].get('Q', [])[0][dataset].decode())
                        gains_swept = self.process_h5_data(
                            load_data[f'T1{exp_extension}_zeno'][q_key].get('Gains', [])[0][dataset].decode())

                        if scaling:
                            Ie = self.process_string_of_nested_lists(
                                load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_I_e', [])[0][dataset].decode())
                            Ig = self.process_string_of_nested_lists(
                                load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_I_g', [])[0][dataset].decode())
                            Qe = self.process_string_of_nested_lists(
                                load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                            Qg = self.process_string_of_nested_lists(
                                load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_Q_g', [])[0][dataset].decode())

                        if len(I) > 0:
                            round_current = int(folder_date.split('round')[-1])
                            rounds_completed[q_key].append(round_current)
                            
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
                                
                                # Check if this is a gain sweep (multiple gains matching multiple sublists)
                                is_sweep = len(gains_swept) > 1 and len(gains_swept) == len(I)
                                
                                if is_sweep:
                                    # Flatten the list of lists
                                    flat_amps = [item for sublist in calibrated_sublists for item in sublist]
                                    amps[q_key].append(flat_amps)
                                    
                                    # Expand gains to match the flattened structure
                                    expanded_gains = np.repeat(gains_swept, len(delays))
                                    gains[q_key].append(expanded_gains.tolist())
                                    
                                    # Expand delays to match
                                    expanded_delays = np.tile(delays, len(gains_swept))
                                    delay_times[q_key].append(expanded_delays.tolist())
                                else:
                                    # Standard averaging over repetitions (single gain)
                                    amp_avg = np.mean(np.array(calibrated_sublists), axis=0)
                                    amps[q_key].append(amp_avg.tolist())
                                    gains[q_key].append(gains_swept)
                                    delay_times[q_key].append(delays)
                            else:
                                # No scaling - just compute amplitude from I/Q
                                amp_sublists = []
                                for sub_I, sub_Q in zip(I, Q):
                                    sub_I = np.asarray(sub_I, dtype=float)
                                    sub_Q = np.asarray(sub_Q, dtype=float)
                                    amp_sublists.append(np.hypot(sub_I, sub_Q).tolist())
                                
                                # Check if this is a gain sweep
                                is_sweep = len(gains_swept) > 1 and len(gains_swept) == len(I)
                                
                                if is_sweep:
                                    # Flatten for gain sweep
                                    flat_amps = [item for sublist in amp_sublists for item in sublist]
                                    amps[q_key].append(flat_amps)
                                    
                                    expanded_gains = np.repeat(gains_swept, len(delays))
                                    gains[q_key].append(expanded_gains.tolist())
                                    
                                    expanded_delays = np.tile(delays, len(gains_swept))
                                    delay_times[q_key].append(expanded_delays.tolist())
                                else:
                                    # Average over repetitions
                                    amp_avg = np.mean(np.array(amp_sublists), axis=0)
                                    amps[q_key].append(amp_avg.tolist())
                                    gains[q_key].append(gains_swept)
                                    delay_times[q_key].append(delays)

                del H5_class_instance

        return None, None, amps, gains, rounds_completed, delay_times

    def _ema_update_disc_with_ssf(self, disc, Ig_list, Qg_list, Ie_list, Qe_list,
                                  alpha=0.05, mode='translate', c=5.5):
        """
        Update the frozen discriminant using the small SSF taken for THIS main dataset.
        mode:
          - 'translate' (default): estimate drift from g; translate μg and μe together; keep axis v fixed.
          - 'full': EMA μg and μe separately from observed per-main μ's; axis v slowly adapts.
        """
        import numpy as np
        # If we don't have references, do nothing
        if (Ig_list is None) or (Qg_list is None) or (Ie_list is None) or (Qe_list is None):
            return disc

        Ig = self._flatten_all_sublists(Ig_list);
        Qg = self._flatten_all_sublists(Qg_list)
        Ie = self._flatten_all_sublists(Ie_list);
        Qe = self._flatten_all_sublists(Qe_list)
        if Ig.size == 0 or Qg.size == 0 or Ie.size == 0 or Qe.size == 0:
            return disc

        mu_g_obs = self.robust_center(Ig + 1j * Qg, c=c)
        mu_e_obs = self.robust_center(Ie + 1j * Qe, c=c)

        mu_g = disc["mu_g"];
        mu_e = disc["mu_e"];
        v = disc["v"];
        norm2 = disc["norm2"]

        if mode == 'translate':
            # common-mode drift from g only (most stable)
            d = mu_g_obs - mu_g
            mu_g = mu_g + alpha * d
            mu_e = mu_e + alpha * d
            # keep v fixed (direction & scale unchanged)
            # (if you want a tiny scale adjust, uncomment below)
            # s = np.abs(mu_e_obs - mu_g_obs) / (np.sqrt(norm2) + 1e-12)
            # v = v * (1 - alpha + alpha * s); norm2 = (v.real*v.real + v.imag*v.imag)
        else:  # 'full'
            mu_g = (1 - alpha) * mu_g + alpha * mu_g_obs
            mu_e = (1 - alpha) * mu_e + alpha * mu_e_obs
            v = mu_e - mu_g
            norm2 = (v.real * v.real + v.imag * v.imag)

        return {"mu_g": mu_g, "mu_e": mu_e, "v": v, "norm2": norm2}

    def run_t1_sweep_combine_ssf_rounds(self, exp_extension='', scaling=False,
                                        freeze_discriminant=True, return_calibration_data=False,
                                        tukey_c=5.5, drift_alpha=0.05, drift_mode='translate'):
        import datetime
        import glob, os, re
        import numpy as np

        # ----------Load/get data------------------------
        steps = 0
        frozen_disc_by_q_gain = {}  # base discriminant per (q_key, gain)
        Ig_calibration = {i: [] for i in range(self.number_of_qubits)}
        Ie_calibration = {i: [] for i in range(self.number_of_qubits)}
        Qg_calibration = {i: [] for i in range(self.number_of_qubits)}
        Qe_calibration = {i: [] for i in range(self.number_of_qubits)}
        Is = {i: [] for i in range(self.number_of_qubits)}
        Qs = {i: [] for i in range(self.number_of_qubits)}
        amps = {i: [] for i in range(self.number_of_qubits)}
        gains = {i: [] for i in range(self.number_of_qubits)}
        rounds_completed = {i: [] for i in range(self.number_of_qubits)}
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        delay_times = {i: [] for i in range(self.number_of_qubits)}
        I_avgs = {i: [] for i in range(self.number_of_qubits)}
        Q_avgs = {i: [] for i in range(self.number_of_qubits)}

        def _is_list_of_lists(x):
            return isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple))

        def _avg_over_sublists(list_of_lists):
            arr = np.array(list_of_lists, dtype=float)
            return np.mean(arr, axis=0)

        # --- existing folder/file discovery code unchanged ---

        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                outerFolder = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date + "/study_data"
                outerFolder_save_plots = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date + "_plots/"
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            if '_' in exp_extension:
                outerFolder_expt = outerFolder + f"/Data_h5/T1{exp_extension}_zeno/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/T1_ge_zeno/"
            round_we_are_on = outerFolder_expt.split(f'qubit_{self.qubit}round')[-1].split('/')[0].split('_')[0]
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))
            TS = re.compile(r'(\d{4})[-_\.]?(\d{2})[-_\.]?(\d{2})[ Tt_-]?(\d{2})[-_\.]?(\d{2})[-_\.]?(\d{2})')
            import datetime as dt
            def dt_from_name(path):
                name = os.path.basename(path)
                m = TS.search(name)
                if not m:
                    return dt.datetime.min
                y, mo, d, h, mi, s = map(int, m.groups())
                return dt.datetime(y, mo, d, h, mi, s)

            h5_files = sorted(h5_files, key=dt_from_name)

            for h5_file in h5_files:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type=f'T1{exp_extension}_zeno', save_r=int(save_round),
                                                           scaling=scaling)

                exclude_dates = {
                    datetime.date(2025, 1, 26),
                    datetime.date(2025, 1, 29),
                    datetime.date(2025, 1, 30),
                    datetime.date(2025, 1, 31)
                }

                for q_key in load_data[f'T1{exp_extension}_zeno']:
                    for dataset in range(len(load_data[f'T1{exp_extension}_zeno'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'T1{exp_extension}_zeno'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        date = datetime.datetime.fromtimestamp(
                            load_data[f'T1{exp_extension}_zeno'][q_key].get('Dates', [])[0][dataset])
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        delays = self.process_h5_data(
                            load_data[f'T1{exp_extension}_zeno'][q_key].get('Delay Times', [])[0][dataset].decode())
                        I = self.process_string_of_nested_lists(
                            load_data[f'T1{exp_extension}_zeno'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_string_of_nested_lists(
                            load_data[f'T1{exp_extension}_zeno'][q_key].get('Q', [])[0][dataset].decode())

                        if scaling:
                            Ie = self.process_string_of_nested_lists(
                                load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_I_e', [])[0][dataset].decode())
                            Ig = self.process_string_of_nested_lists(
                                load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_I_g', [])[0][dataset].decode())
                            Qe = self.process_string_of_nested_lists(
                                load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                            Qg = self.process_string_of_nested_lists(
                                load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_Q_g', [])[0][dataset].decode())

                        round_num = load_data[f'T1{exp_extension}_zeno'][q_key].get('Round Num', [])[0][dataset]
                        try:
                            batch_num = load_data[f'T1{exp_extension}_zeno'][q_key].get('Batch Num', [])[0][dataset]
                            syst_config = load_data[f'T1{exp_extension}_zeno'][q_key].get('Syst Config', [])[0][
                                dataset].decode()
                            exp_config = load_data[f'T1{exp_extension}_zeno'][q_key].get('Exp Config', [])[0][
                                dataset].decode()
                        except:
                            exp_config = None

                        if len(I) > 0:
                            Is[q_key].append(I);
                            Qs[q_key].append(Q)

                            if exp_config is not None:
                                gain = round(float(syst_config.split("res_gain_qze': ")[-1].split(',')[0]), 6)
                                steps = round(float(
                                    exp_config.split("Readout_Optimization': ")[-1].split("steps': ")[-1].split(',')[
                                        0]), 6)
                                gains[q_key].append(gain)
                            rounds_completed[q_key].append(round_we_are_on)

                            if scaling:
                                # wrap to list-of-lists
                                I_nested = I if _is_list_of_lists(I) else [I]
                                Q_nested = Q if _is_list_of_lists(Q) else [Q]
                                Ie_nested = Ie if _is_list_of_lists(Ie) else [Ie]
                                Ig_nested = Ig if _is_list_of_lists(Ig) else [Ig]
                                Qe_nested = Qe if _is_list_of_lists(Qe) else [Qe]
                                Qg_nested = Qg if _is_list_of_lists(Qg) else [Qg]

                                # ---------- ONE base SSF per (q,gain), then drift tracking per dataset ----------
                                gain_key = gains[q_key][-1] if len(gains[q_key]) > 0 else None
                                cache_key = (q_key, gain_key)

                                # --- base discriminant (combine ALL SSF sublists, only once) ---
                                if freeze_discriminant and cache_key not in frozen_disc_by_q_gain:
                                    base_disc = self.compute_ssf_discriminant(
                                        Ig_nested, Qg_nested, Ie_nested, Qe_nested, c=tukey_c
                                    )
                                    frozen_disc_by_q_gain[cache_key] = base_disc

                                # start from the cached base
                                current_disc = frozen_disc_by_q_gain.get(cache_key, None)

                                # --- NEW: drift update using THIS dataset's SSF (small EMA) ---
                                if freeze_discriminant and current_disc is not None:
                                    current_disc = self._ema_update_disc_with_ssf(
                                        current_disc, Ig_nested, Qg_nested, Ie_nested, Qe_nested,
                                        alpha=drift_alpha, mode=drift_mode, c=tukey_c
                                    )
                                    # keep the updated disc so next dataset starts from here
                                    frozen_disc_by_q_gain[cache_key] = current_disc

                                calibrated_sublists = []
                                I_sublists, Q_sublists = [], []

                                for sub_I, sub_Q in zip(
                                        I_nested, Q_nested
                                ):
                                    sub_I = np.asarray(sub_I, dtype=float)
                                    sub_Q = np.asarray(sub_Q, dtype=float)
                                    z = sub_I + 1j * sub_Q

                                    if freeze_discriminant and current_disc is not None:
                                        pop = self.project_population(z, current_disc)
                                    else:
                                        # fallback: per-sublist fit (shouldn't happen with freeze_discriminant=True)
                                        e = self.robust_center(
                                            _flatten_all_sublists(Ie_nested) + 1j * _flatten_all_sublists(Qe_nested),
                                            c=tukey_c)
                                        g = self.robust_center(
                                            _flatten_all_sublists(Ig_nested) + 1j * _flatten_all_sublists(Qg_nested),
                                            c=tukey_c)
                                        v = e - g;
                                        norm2 = (v.real * v.real + v.imag * v.imag)
                                        pop = np.clip(np.real((z - g) * np.conj(v)) / (norm2 + 1e-12), 0.0, 1.0)

                                    calibrated_sublists.append(pop.tolist())
                                    I_sublists.append(sub_I.tolist())
                                    Q_sublists.append(sub_Q.tolist())

                                amp_avg = _avg_over_sublists(calibrated_sublists)
                                I_avg = _avg_over_sublists(I_sublists)
                                Q_avg = _avg_over_sublists(Q_sublists)

                                amps[q_key].append(amp_avg.tolist())
                                I_avgs[q_key].append(I_avg.tolist())
                                Q_avgs[q_key].append(Q_avg.tolist())

                                Ig_calibration[q_key].append(Ig_nested)
                                Ie_calibration[q_key].append(Ie_nested)
                                Qg_calibration[q_key].append(Qg_nested)
                                Qe_calibration[q_key].append(Qe_nested)

                            else:
                                I_nested = I if _is_list_of_lists(I) else [I]
                                Q_nested = Q if _is_list_of_lists(Q) else [Q]
                                amp_sublists = []
                                I_sublists = []
                                Q_sublists = []
                                for sub_I, sub_Q in zip(I_nested, Q_nested):
                                    sub_I = np.asarray(sub_I, dtype=float)
                                    sub_Q = np.asarray(sub_Q, dtype=float)
                                    amp_sublists.append(np.hypot(sub_I, sub_Q).tolist())
                                    I_sublists.append(sub_I.tolist())
                                    Q_sublists.append(sub_Q.tolist())

                                amp_avg = _avg_over_sublists(amp_sublists)
                                I_avg = _avg_over_sublists(I_sublists)
                                Q_avg = _avg_over_sublists(Q_sublists)

                                amps[q_key].append(amp_avg.tolist())
                                I_avgs[q_key].append(I_avg.tolist())
                                Q_avgs[q_key].append(Q_avg.tolist())

                            delay_times[q_key].append(delays)
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                del H5_class_instance

        if return_calibration_data:
            return (I_avgs, Q_avgs, amps, gains, rounds_completed, delay_times,
                    Ig_calibration, Ie_calibration, Qe_calibration, Qg_calibration, steps)
        else:
            return I_avgs, Q_avgs, amps, gains, rounds_completed, delay_times

    def run_t1_sweep(self, exp_extension='', scaling=False, return_calibration_data=False):
        import datetime

        # ----------Load/get data------------------------
        steps=0
        Ig_calibration = {i: [] for i in range(self.number_of_qubits)}
        Ie_calibration = {i: [] for i in range(self.number_of_qubits)}
        Qg_calibration = {i: [] for i in range(self.number_of_qubits)}
        Qe_calibration = {i: [] for i in range(self.number_of_qubits)}
        Is = {i: [] for i in range(self.number_of_qubits)}
        Qs = {i: [] for i in range(self.number_of_qubits)}
        amps = {i: [] for i in range(self.number_of_qubits)}
        gains = {i: [] for i in range(self.number_of_qubits)}
        rounds_completed = {i: [] for i in range(self.number_of_qubits)}
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        delay_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}
        #print(self.top_folder_dates)
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
                outerFolder_expt = outerFolder + f"/Data_h5/T1{exp_extension}_zeno/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/T1_ge_zeno/"
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
                load_data = H5_class_instance.load_from_h5(data_type=f'T1{exp_extension}_zeno', save_r=int(save_round), scaling=scaling)
                # H5_class_instance.print_h5_contents(h5_file)
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data[f'T1{exp_extension}_zeno']:
                    for dataset in range(len(load_data[f'T1{exp_extension}_zeno'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'T1{exp_extension}_zeno'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T1 = load_data['T1'][q_key].get('T1', [])[0][dataset]
                        # errors = load_data['T1'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data[f'T1{exp_extension}_zeno'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue
                        delays = self.process_h5_data(
                            load_data[f'T1{exp_extension}_zeno'][q_key].get('Delay Times', [])[0][dataset].decode())

                        I = self.process_h5_data(load_data[f'T1{exp_extension}_zeno'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data[f'T1{exp_extension}_zeno'][q_key].get('Q', [])[0][dataset].decode())

                        if scaling:
                            Ie = self.process_h5_data(load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_I_e', [])[0][dataset].decode())
                            Ig = self.process_h5_data(load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_I_g', [])[0][dataset].decode())
                            Qe = self.process_h5_data(load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                            Qg = self.process_h5_data(load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_Q_g', [])[0][dataset].decode())
                        round_num = load_data[f'T1{exp_extension}_zeno'][q_key].get('Round Num', [])[0][dataset]
                        try:
                            batch_num = load_data[f'T1{exp_extension}_zeno'][q_key].get('Batch Num', [])[0][dataset]
                            syst_config = load_data[f'T1{exp_extension}_zeno'][q_key].get('Syst Config', [])[0][dataset].decode()
                            exp_config = load_data[f'T1{exp_extension}_zeno'][q_key].get('Exp Config', [])[0][dataset].decode()

                        except:
                            exp_config =None

                        if len(I) > 0:
                            Is[q_key].append(I)
                            Qs[q_key].append(Q)

                            gain = round(
                                float(syst_config.split('res_gain_qze\': ')[-1].split(',')[0]), 6)

                            steps = round(
                                float(exp_config.split('Readout_Optimization\': ')[-1].split('steps\': ')[-1].split(',')[0]), 6)

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

    def plot_IBM_qze(self,amps,gains, save_path):

        qubit_ids = sorted(amps.keys())  # → [0, 1, 2, 3, 4, 5]
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=False, sharey=False)
        axes = axes.flatten()

        for idx, q in enumerate(qubit_ids):
            ax = axes[idx]
            #print(len(gains[q]), len(amps[q]))
            ax.plot(gains[q], [1/n for n in amps[q]], marker='o', linewidth=1)
            ax.set_title(f"Qubit {q}")
            ax.set_xlabel("Pulse gain (a.u.)")
            ax.set_ylabel("1/ T1 Signal amplitude (a.u.)")

        for j in range(len(qubit_ids), len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        self.create_folder_if_not_exists(save_path)
        fig.savefig(save_path + 'gamma.png', transparent=False, dpi=self.final_figure_quality)

        print('Plot saved to: ', save_path)
    def plot_IBM_qze_normal(self,amps,gains, save_path):

        qubit_ids = sorted(amps.keys())  # → [0, 1, 2, 3, 4, 5]
        fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=False, sharey=False)
        axes = axes.flatten()

        for idx, q in enumerate(qubit_ids):
            ax = axes[idx]
            #print(len(gains[q]), len(amps[q]))
            ax.plot(gains[q], amps[q], marker='o', linewidth=1)
            ax.set_title(f"Qubit {q}")
            ax.set_xlabel("Pulse gain (a.u.)")
            ax.set_ylabel("T1 Signal amplitude (a.u.)")

        for j in range(len(qubit_ids), len(axes)):
            axes[j].set_visible(False)

        fig.tight_layout()
        self.create_folder_if_not_exists(save_path)
        fig.savefig(save_path + 't1.png', transparent=False, dpi=self.final_figure_quality)

        print('Plot saved to: ', save_path)
    def plot_IBM_qze_compare(self,amps,gains, rounds,save_path):

        q = self.qubit
        gains_q = np.asarray(gains[q])

        amps_q = np.asarray(amps[q])
        rounds_q = np.asarray(rounds[q])
        fig, ax = plt.subplots(figsize=(6, 4))
        from itertools import cycle
        colour_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        for r_id in np.unique(rounds_q):
            mask = rounds_q == r_id
            ax.plot(
                gains_q[mask],
                1.0 / amps_q[mask],
                linestyle="-",
                label=f"Round {r_id}",
                color=next(colour_cycle),
                linewidth=0.8,

            )

        ax.set_title(f"Qubit {self.qubit}")
        ax.set_xlabel("Pulse gain (a.u.)")
        ax.set_ylabel("1 / T1 signal amplitude (a.u.)")
        ax.legend(
            loc="center left",  # anchor on the left‐centre of the legend box
            bbox_to_anchor=(1.02, 0.5),  # shift legend just outside the axes
            frameon=False,  # remove legend border
            fontsize="small"  # or an explicit int, e.g. 8
        )
        fig.tight_layout()

        self.create_folder_if_not_exists(save_path)
        fig.savefig(save_path + f"gamma_q{self.qubit}_slice{self.t1_slice}.png",
                    transparent=False,
                    dpi=self.final_figure_quality)

        print("Plot saved to:", save_path)

    def plot_IBM_qze_normal_compare(self,amps,gains,rounds, save_path):

        q = self.qubit
        gains_q = np.asarray(gains[q])
        amps_q = np.asarray(amps[q])
        rounds_q = np.asarray(rounds[q])
        fig, ax = plt.subplots(figsize=(6, 4))
        from itertools import cycle
        colour_cycle = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
        for r_id in np.unique(rounds_q):
            mask = rounds_q == r_id
            ax.plot(
                gains_q[mask],
                amps_q[mask],
                linestyle="-",
                label=f"Round {r_id}",
                color=next(colour_cycle),
                linewidth=0.8,
            )

        ax.set_title(f"Qubit {self.qubit}")
        ax.set_xlabel("Pulse gain (a.u.)")
        ax.set_ylabel("T1 signal amplitude (a.u.)")
        ax.legend(
            loc="center left",  # anchor on the left‐centre of the legend box
            bbox_to_anchor=(1.02, 0.5),  # shift legend just outside the axes
            frameon=False,  # remove legend border
            fontsize="small"  # or an explicit int, e.g. 8
        )
        fig.tight_layout()

        self.create_folder_if_not_exists(save_path)
        fig.savefig(save_path + f"t1_q{self.qubit}_slice{self.t1_slice}.png",
                    transparent=False,
                    dpi=self.final_figure_quality)

        print("Plot saved to:", save_path)
    def exponential(self, x, a, b, c, d):
        return a * np.exp(- (x - b) / c) + d

    def t1_fit(self, signal, delay_times):

        # Initial guess for parameters
        q1_a_guess = np.max(signal) - np.min(signal)  # Initial guess for amplitude (a)
        q1_b_guess = 0  # Initial guess for time shift (b)
        q1_c_guess = (delay_times[-1] - delay_times[0]) / 5  # Initial guess for decay constant (T1)
        q1_d_guess = np.min(signal)  # Initial guess for baseline (d)

        # Form the guess array
        q1_guess = [q1_a_guess, q1_b_guess, q1_c_guess, q1_d_guess]

        # Define bounds to constrain T1 (c) to be positive, but allow amplitude (a) to be negative
        lower_bounds = [-np.inf, -np.inf, 0, -np.inf]  # Amplitude (a) can be negative/positive, but T1 (c) > 0
        upper_bounds = [np.inf, np.inf, np.inf, np.inf]  # No upper bound on parameters

        # Perform the fit using the 'trf' method with bounds
        q1_popt, q1_pcov = curve_fit(self.exponential, delay_times, signal,
                                     p0=q1_guess, bounds=(lower_bounds, upper_bounds),
                                     method='trf', maxfev=10000)

        # Generate the fitted exponential curve
        q1_fit_exponential = self.exponential(delay_times, *q1_popt)

        # Extract T1 and its error
        T1_est = q1_popt[2]  # Decay constant T1
        T1_err = np.sqrt(q1_pcov[2][2]) if q1_pcov[2][2] >= 0 else float('inf')  # Ensure error is valid

        return q1_fit_exponential, T1_err, T1_est, signal

    def plot_t1_fit_vs_gain(self, amps, gains, rounds, delay_times, save_path):
        """
        For each round:
          - Fit T1 vs delay for each gain slice (x=gain in the old plot).
          - Save a simple plot of data + fit: x=delay_times, y=qubit population.
          - Plot T1 vs gain (2D line plot) and save.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict

        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        if not delay_q or not gains_q or not amps_q or not rounds_q:
            print(f"No data for qubit {q}. Skipping.")
            return

        # Helper: robustly fetch a round label at (i, j)
        def get_round_label(i, j):
            if i < len(rounds_q):
                row = rounds_q[i]
                if isinstance(row, (list, tuple)):
                    if len(row) == 0:
                        return "0"
                    if j < len(row):
                        return str(row[j])
                    return str(row[-1])
                return str(row)
            return "0"

        # Flatten: (round_id, gain, delay, amp)
        all_points = []
        for i in range(len(delay_q)):
            try:
                delay_val = float(delay_q[i])
            except Exception:
                continue
            row_g = gains_q[i] if i < len(gains_q) else []
            row_a = amps_q[i] if i < len(amps_q) else []
            n = min(len(row_g), len(row_a))
            for j in range(n):
                try:
                    g = float(row_g[j])
                    a = float(row_a[j])
                except Exception:
                    continue
                r_id = get_round_label(i, j)
                all_points.append((r_id, g, delay_val, a))

        if not all_points:
            print(f"No numeric points for qubit {q}. Skipping.")
            return

        unique_rounds = sorted({r for (r, _, __, ___) in all_points})

        # Ensure save folder exists
        self.create_folder_if_not_exists(save_path)

        for r_id in unique_rounds:
            # Group by gain for this round
            by_gain = defaultdict(list)  # gain -> list of (delay, amp)
            for (r, g, d, a) in all_points:
                if r == r_id:
                    by_gain[g].append((d, a))

            # For each gain, fit T1 vs delay
            gains_sorted = sorted(by_gain.keys())
            t1_values = []
            t1_gains = []

            for g in gains_sorted:
                pairs = by_gain[g]
                if len(pairs) < 3:
                    continue  # keep it simple: need a few points to fit
                # Sort by delay
                pairs.sort(key=lambda x: x[0])
                dlys = np.array([p[0] for p in pairs], dtype=float)
                sigs = np.array([p[1] for p in pairs], dtype=float)

                try:
                    fit_curve, t1_err, t1_est, _ = self.t1_fit(sigs, dlys)  # uses your existing defs
                except Exception:
                    continue

                # Save data + fit plot (delay on x, population on y)
                fig, ax = plt.subplots(figsize=(5.0, 3.6))
                ax.plot(dlys, sigs, 'o', label='data', ms=3)
                ax.plot(dlys, fit_curve, '-', label='fit')
                ax.set_xlabel("Delay time")
                ax.set_ylabel("Qubit population")
                ax.set_title(f"Qubit {self.qubit + 1} - Round {r_id} - Gain {g} - t1 {t1_est}")
                ax.legend(frameon=False)
                fig.tight_layout()
                os.makedirs(save_path + f't1_fits_q{self.qubit+1}', exist_ok=True)
                outfile_fit = (save_path + f't1_fits_q{self.qubit+1}/'
                               f"t1_fit_q{self.qubit}_slice{self.t1_slice}_round{r_id}_gain{g}.png")
                fig.savefig(outfile_fit, dpi=self.final_figure_quality)
                plt.close(fig)

                if t1_est > 200:
                    continue
                else:
                    t1_gains.append(g)
                    t1_values.append(t1_est)

            # Plot T1 vs gain (simple 2D line plot)
            if t1_gains:
                # sort by gain for a clean line
                order = np.argsort(np.array(t1_gains, dtype=float))
                g_plot = np.array(t1_gains, dtype=float)[order]
                t1_plot = np.array(t1_values, dtype=float)[order]

                fig, ax = plt.subplots(figsize=(5.0, 3.6))
                ax.plot(g_plot, t1_plot, '-o', ms=4)
                ax.set_xlabel("Gain")
                ax.set_ylabel("T1 time")
                ax.set_title(f"Qubit {self.qubit + 1} — Round {r_id}: T1 vs Gain")
                fig.tight_layout()
                os.makedirs(save_path + 't1_vs_gain', exist_ok=True)
                outfile_summary = (save_path + 't1_vs_gain/'
                                   f"t1_vs_gain_q{self.qubit}_slice{self.t1_slice}_round{r_id}.png")
                fig.savefig(outfile_summary, dpi=self.final_figure_quality)
                plt.close(fig)

    def plot_all_t1_heatmaps(self, amps, gains, rounds, delay_times, save_path):
        """
        For each unique round id, make a heatmap where:
          x-axis: pulse gain
          y-axis: delay time
          color:  signal amplitude (amps)
        Works with ragged rows and mismatched lengths by gridding per-round data.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict

        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        if not delay_q or not gains_q or not amps_q or not rounds_q:
            print(f"No data for qubit {q}. Skipping.")
            return

        # Helper: robustly fetch a round label at (i, j)
        def get_round_label(i, j):
            if i < len(rounds_q):
                row = rounds_q[i]
                if isinstance(row, (list, tuple)):
                    if len(row) == 0:
                        return "0"
                    if j < len(row):
                        return str(row[j])
                    # if rounds row shorter than data row, repeat last label
                    return str(row[-1])
                # if someone supplied a scalar label per row
                return str(row)
            return "0"

        # Flatten points: (round_id, gain, delay, amp)
        all_points = []
        for i in range(len(delay_q)):
            delay_val = float(delay_q[i])
            row_g = gains_q[i] if i < len(gains_q) else []
            row_a = amps_q[i] if i < len(amps_q) else []
            n = min(len(row_g), len(row_a))
            for j in range(n):
                try:
                    g = float(row_g[j])
                    a = float(row_a[j])
                except Exception:
                    continue
                r_id = get_round_label(i, j)
                all_points.append((r_id, g, delay_val, a))

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
                # fabricate symmetric edges around the single center
                d = 1.0
                return np.array([centers[0] - d / 2, centers[0] + d / 2])
            mids = (centers[:-1] + centers[1:]) / 2.0
            first = centers[0] - (centers[1] - centers[0]) / 2.0
            last = centers[-1] + (centers[-1] - centers[-2]) / 2.0
            return np.concatenate([[first], mids, [last]])

        for r_id in unique_rounds:
            # Collect this round's points
            pts = [(g, d, a) for (r, g, d, a) in all_points if r == r_id]
            if not pts:
                continue

            gains_r = sorted({g for (g, _, _) in pts})
            delays_r = sorted({d for (_, d, _) in pts})

            # Map (delay_idx, gain_idx) -> list of amplitudes (in case of duplicates)
            bucket = defaultdict(list)
            gi_map = {g: i for i, g in enumerate(gains_r)}
            di_map = {d: i for i, d in enumerate(delays_r)}
            for g, d, a in pts:
                bucket[(di_map[d], gi_map[g])].append(a)

            # Build C grid (Ny x Nx) filled with NaN, average duplicates
            Ny, Nx = len(delays_r), len(gains_r)
            C = np.full((Ny, Nx), np.nan, dtype=float)
            for (iy, ix), vals in bucket.items():
                C[iy, ix] = float(np.nanmean(vals))

            # Build bin edges (no NaNs, strictly increasing)
            x_edges = centers_to_edges(gains_r)
            y_edges = centers_to_edges(delays_r)

            # Plot
            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            mesh = ax.pcolormesh(x_edges, y_edges, C, shading='flat')
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label("Qubit Population")

            ax.set_title(f"Qubit {self.qubit+1} — Round {r_id}")
            ax.set_xlabel("Pulse gain (a.u.)")
            ax.set_ylabel("Delay time")

            # Put nice ticks at the actual centers (optional; comment out if crowded)
            import matplotlib.ticker as mticker
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7, prune=None))

            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            ax.set_yticks(delays_r)

            fig.tight_layout()
            outfile = (save_path +
                       f"t1_heatmap_q{self.qubit}_slice{self.t1_slice}_round{r_id}.png")
            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved heatmap for round {r_id} to: {outfile}")

    def plot_best_ssf_only(
            self,
            amps,
            gains,
            rounds,
            delay_times,
            save_path,
            # --- required for SSF ranking & plotting ---
            ss_class_instance=None,
            ss_cfg=None,
            Ig_calibration=None,
            Ie_calibration=None,
            Qg_calibration=None,
            Qe_calibration=None,
    ):
        """
        Compute SSF per dataset (for this qubit), select the dataset with the highest SSF,
        generate ONLY the SSD/SSF plot for that dataset, and return a summary dict.

        Returns:
            dict | None:
                {
                  'index': <int>,                 # dataset index
                  'round': <str>,                 # round id
                  'fidelity': <float>,            # estimated SSF in [0,1]
                  'gain_repr': <float>,           # representative gain (median if array)
                  'delay_repr': <float>,          # representative delay (median if array)
                  'plot_saved_to': <str | None>,  # folder where hist_ssf saved plots (if any)
                  'Ig': np.ndarray, 'Qg': np.ndarray,
                  'Ie': np.ndarray, 'Qe': np.ndarray,
                }
                or None if inputs insufficient.
        """
        import os
        import numpy as np

        # ---------- helpers ----------
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

        def estimate_ssf(Ig, Qg, Ie, Qe):
            """
            Lightweight SSF estimate via 1D LDA projection with midpoint threshold.
            SSF = 0.5*(Pg_correct + Pe_correct).
            """
            G = np.c_[Ig, Qg]
            E = np.c_[Ie, Qe]
            if min(G.shape[0], E.shape[0]) < 2:
                return np.nan

            mu_g = G.mean(axis=0)
            mu_e = E.mean(axis=0)
            Sg = np.cov(G, rowvar=False)
            Se = np.cov(E, rowvar=False)
            Sw = Sg + Se + 1e-9 * np.eye(2)
            try:
                w = np.linalg.solve(Sw, (mu_e - mu_g))
            except np.linalg.LinAlgError:
                return np.nan

            z_g = G @ w
            z_e = E @ w
            th = 0.5 * (np.mean(z_g) + np.mean(z_e))
            Pg = np.mean(z_g <= th) if z_g.size else np.nan
            Pe = np.mean(z_e > th) if z_e.size else np.nan
            if np.isnan(Pg) or np.isnan(Pe):
                return np.nan
            return 0.5 * (Pg + Pe)

        # ---------- extract this qubit's lists ----------
        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        n = min(len(amps_q), len(gains_q), len(rounds_q), len(delay_q))
        if n == 0 or not (len(amps_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
            print(f"No usable data for qubit {q} (missing lists or length mismatch).")
            return None

        # ---------- require SSD inputs for SSF ranking ----------
        have_ssd_inputs = (
                ss_class_instance is not None and
                ss_cfg is not None and
                Ig_calibration is not None and
                Ie_calibration is not None and
                Qg_calibration is not None and
                Qe_calibration is not None
        )
        if not have_ssd_inputs:
            print("Need ss_class_instance, ss_cfg, and Ig/Ie/Qg/Qe calibration arrays to rank by SSF.")
            return None

        Ig_q = Ig_calibration.get(q, [])
        Ie_q = Ie_calibration.get(q, [])
        Qg_q = Qg_calibration.get(q, [])
        Qe_q = Qe_calibration.get(q, [])
        n_ss = min(len(Ig_q), len(Ie_q), len(Qg_q), len(Qe_q), n)
        if n_ss == 0:
            print("Empty calibration arrays for this qubit; cannot rank by SSF.")
            return None

        # ---------- compute SSF per dataset and pick best ----------
        ssf = np.full(n_ss, np.nan, dtype=float)
        for i in range(n_ss):
            try:
                Ig_i = np.asarray(Ig_q[i]).ravel()
                Qg_i = np.asarray(Qg_q[i]).ravel()
                Ie_i = np.asarray(Ie_q[i]).ravel()
                Qe_i = np.asarray(Qe_q[i]).ravel()
                if min(Ig_i.size, Qg_i.size, Ie_i.size, Qe_i.size) == 0:
                    continue
                ssf[i] = estimate_ssf(Ig_i, Qg_i, Ie_i, Qe_i)
            except Exception:
                continue

        if not np.isfinite(ssf).any():
            print("Failed to compute SSF for all datasets.")
            return None

        best_i = int(np.nanargmax(ssf))
        best_round = str(rounds_q[best_i])
        best_gain_repr = first_scalar_or_nan(gains_q[best_i])
        best_delay_repr = first_scalar_or_nan(delay_q[best_i])

        # ---------- make ONLY the SSD/SSF plot for the best dataset ----------
        plot_dir = None
        analysis_root = os.path.join(save_path, "analysis")
        try:
            self.create_folder_if_not_exists(analysis_root)

            # temporarily override context so the SSD routine saves into our analysis folder
            _old_outer = getattr(ss_class_instance, "outerFolder", None)
            _old_qidx = getattr(ss_class_instance, "QubitIndex", None)
            _old_rnum = getattr(ss_class_instance, "round_num", None)
            _old_name = getattr(ss_class_instance, "expt_name", None)

            ss_class_instance.outerFolder = analysis_root
            ss_class_instance.QubitIndex = getattr(self, "qubit", 0)
            ss_class_instance.round_num = best_round
            ss_class_instance.expt_name = f"t1_ssd_BEST_gain{best_gain_repr:.4g}_delay{best_delay_repr:.4g}_SSF{ssf[best_i]:.4f}"

            I_g = np.asarray(Ig_q[best_i]).ravel()
            Q_g = np.asarray(Qg_q[best_i]).ravel()
            I_e = np.asarray(Ie_q[best_i]).ravel()
            Q_e = np.asarray(Qe_q[best_i]).ravel()

            # Call user-provided SSD routine (it handles plotting/saving)
            ss_class_instance.hist_ssf(
                data=[I_g, Q_g, I_e, Q_e],
                cfg=ss_cfg,
                plot=True
            )
            plot_dir = ss_class_instance.outerFolder  # where it saved

        except Exception as e:
            print(f"[SSD] Could not create SSD plot for best dataset: {e}")
        finally:
            # restore prior context
            try:
                ss_class_instance.outerFolder = _old_outer
                ss_class_instance.QubitIndex = _old_qidx
                ss_class_instance.round_num = _old_rnum
                ss_class_instance.expt_name = _old_name
            except Exception:
                pass

        # ---------- return summary ----------
        return {
            'index': best_i,
            'round': best_round,
            'fidelity': float(ssf[best_i]),
            'gain_repr': float(best_gain_repr),
            'delay_repr': float(best_delay_repr),
            'plot_saved_to': plot_dir,
            'Ig': np.asarray(Ig_q[best_i]).ravel(),
            'Qg': np.asarray(Qg_q[best_i]).ravel(),
            'Ie': np.asarray(Ie_q[best_i]).ravel(),
            'Qe': np.asarray(Qe_q[best_i]).ravel(),
        }

    def plot_worse_ssf_only(
            self,
            amps,
            gains,
            rounds,
            delay_times,
            save_path,
            # --- required for SSF ranking & plotting ---
            ss_class_instance=None,
            ss_cfg=None,
            Ig_calibration=None,
            Ie_calibration=None,
            Qg_calibration=None,
            Qe_calibration=None,
    ):
        """
        Compute SSF per dataset (for this qubit), select the dataset with the highest SSF,
        generate ONLY the SSD/SSF plot for that dataset, and return a summary dict.

        Returns:
            dict | None:
                {
                  'index': <int>,                 # dataset index
                  'round': <str>,                 # round id
                  'fidelity': <float>,            # estimated SSF in [0,1]
                  'gain_repr': <float>,           # representative gain (median if array)
                  'delay_repr': <float>,          # representative delay (median if array)
                  'plot_saved_to': <str | None>,  # folder where hist_ssf saved plots (if any)
                  'Ig': np.ndarray, 'Qg': np.ndarray,
                  'Ie': np.ndarray, 'Qe': np.ndarray,
                }
                or None if inputs insufficient.
        """
        import os
        import numpy as np

        # ---------- helpers ----------
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

        def estimate_ssf(Ig, Qg, Ie, Qe):
            """
            Lightweight SSF estimate via 1D LDA projection with midpoint threshold.
            SSF = 0.5*(Pg_correct + Pe_correct).
            """
            G = np.c_[Ig, Qg]
            E = np.c_[Ie, Qe]
            if min(G.shape[0], E.shape[0]) < 2:
                return np.nan

            mu_g = G.mean(axis=0)
            mu_e = E.mean(axis=0)
            Sg = np.cov(G, rowvar=False)
            Se = np.cov(E, rowvar=False)
            Sw = Sg + Se + 1e-9 * np.eye(2)
            try:
                w = np.linalg.solve(Sw, (mu_e - mu_g))
            except np.linalg.LinAlgError:
                return np.nan

            z_g = G @ w
            z_e = E @ w
            th = 0.5 * (np.mean(z_g) + np.mean(z_e))
            Pg = np.mean(z_g <= th) if z_g.size else np.nan
            Pe = np.mean(z_e > th) if z_e.size else np.nan
            if np.isnan(Pg) or np.isnan(Pe):
                return np.nan
            return 0.5 * (Pg + Pe)

        # ---------- extract this qubit's lists ----------
        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        n = min(len(amps_q), len(gains_q), len(rounds_q), len(delay_q))
        if n == 0 or not (len(amps_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
            print(f"No usable data for qubit {q} (missing lists or length mismatch).")
            return None

        # ---------- require SSD inputs for SSF ranking ----------
        have_ssd_inputs = (
                ss_class_instance is not None and
                ss_cfg is not None and
                Ig_calibration is not None and
                Ie_calibration is not None and
                Qg_calibration is not None and
                Qe_calibration is not None
        )
        if not have_ssd_inputs:
            print("Need ss_class_instance, ss_cfg, and Ig/Ie/Qg/Qe calibration arrays to rank by SSF.")
            return None

        Ig_q = Ig_calibration.get(q, [])
        Ie_q = Ie_calibration.get(q, [])
        Qg_q = Qg_calibration.get(q, [])
        Qe_q = Qe_calibration.get(q, [])
        n_ss = min(len(Ig_q), len(Ie_q), len(Qg_q), len(Qe_q), n)
        if n_ss == 0:
            print("Empty calibration arrays for this qubit; cannot rank by SSF.")
            return None

        # ---------- compute SSF per dataset and pick best ----------
        ssf = np.full(n_ss, np.nan, dtype=float)
        for i in range(n_ss):
            try:
                Ig_i = np.asarray(Ig_q[i]).ravel()
                Qg_i = np.asarray(Qg_q[i]).ravel()
                Ie_i = np.asarray(Ie_q[i]).ravel()
                Qe_i = np.asarray(Qe_q[i]).ravel()
                if min(Ig_i.size, Qg_i.size, Ie_i.size, Qe_i.size) == 0:
                    continue
                ssf[i] = estimate_ssf(Ig_i, Qg_i, Ie_i, Qe_i)
            except Exception:
                continue

        if not np.isfinite(ssf).any():
            print("Failed to compute SSF for all datasets.")
            return None

        best_i = int(np.nanargmin(ssf))
        best_round = str(rounds_q[best_i])
        best_gain_repr = first_scalar_or_nan(gains_q[best_i])
        best_delay_repr = first_scalar_or_nan(delay_q[best_i])

        # ---------- make ONLY the SSD/SSF plot for the best dataset ----------
        plot_dir = None
        analysis_root = os.path.join(save_path, "analysis")
        try:
            self.create_folder_if_not_exists(analysis_root)

            # temporarily override context so the SSD routine saves into our analysis folder
            _old_outer = getattr(ss_class_instance, "outerFolder", None)
            _old_qidx = getattr(ss_class_instance, "QubitIndex", None)
            _old_rnum = getattr(ss_class_instance, "round_num", None)
            _old_name = getattr(ss_class_instance, "expt_name", None)

            ss_class_instance.outerFolder = analysis_root
            ss_class_instance.QubitIndex = getattr(self, "qubit", 0)
            ss_class_instance.round_num = best_round
            ss_class_instance.expt_name = f"t1_ssd_BEST_gain{best_gain_repr:.4g}_delay{best_delay_repr:.4g}_SSF{ssf[best_i]:.4f}"

            I_g = np.asarray(Ig_q[best_i]).ravel()
            Q_g = np.asarray(Qg_q[best_i]).ravel()
            I_e = np.asarray(Ie_q[best_i]).ravel()
            Q_e = np.asarray(Qe_q[best_i]).ravel()

            # Call user-provided SSD routine (it handles plotting/saving)
            ss_class_instance.hist_ssf(
                data=[I_g, Q_g, I_e, Q_e],
                cfg=ss_cfg,
                plot=True
            )
            plot_dir = ss_class_instance.outerFolder  # where it saved

        except Exception as e:
            print(f"[SSD] Could not create SSD plot for best dataset: {e}")
        finally:
            # restore prior context
            try:
                ss_class_instance.outerFolder = _old_outer
                ss_class_instance.QubitIndex = _old_qidx
                ss_class_instance.round_num = _old_rnum
                ss_class_instance.expt_name = _old_name
            except Exception:
                pass

        # ---------- return summary ----------
        return {
            'index': best_i,
            'round': best_round,
            'fidelity': float(ssf[best_i]),
            'gain_repr': float(best_gain_repr),
            'delay_repr': float(best_delay_repr),
            'plot_saved_to': plot_dir,
            'Ig': np.asarray(Ig_q[best_i]).ravel(),
            'Qg': np.asarray(Qg_q[best_i]).ravel(),
            'Ie': np.asarray(Ie_q[best_i]).ravel(),
            'Qe': np.asarray(Qe_q[best_i]).ravel(),
        }


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

    def plot_t1_ssf_only(
            self,
            amps,
            gains,
            rounds,
            delay_times,
            save_path,
            # --- SSF inputs (required to actually plot) ---
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
        SSF-only version of plot_all_t1_heatmaps_with_singular_ssf_plotting.
        For each dataset i of this qubit, generates ONE SSF plot via
        ss_class_instance.hist_ssf_with_annotations(...) and saves results
        under <save_path>/analysis/.

        Inputs (per qubit q) match the original function:
          - amps[q], gains[q], rounds[q], delay_times[q]
          - I_experiment[q], Q_experiment[q] (optional single-shot (I,Q))
          - Ig/Ie/Qg/Qe_calibration[q][gain_index][dataset_index]: arrays for g/e calibration

        Heatmaps are NOT produced here.
        """
        import os
        import numpy as np

        # -------- helpers (kept from your original) --------
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

        def pick_single_shot(x, k=0):
            """
            Return ONE scalar from x.
            - Picks the k-th finite element (default: first, k=0).
            - If scalar, returns it.
            - If empty/all-NaN or k out of range, returns np.nan.
            """
            a = as_1d_array(x)
            if a is None:
                return np.nan
            finite_idx = np.flatnonzero(np.isfinite(a))
            if finite_idx.size == 0:
                return np.nan
            return float(a[finite_idx[0]])

        # ---------------------------------------------------

        # --- new tiny helpers to gracefully read flat/per-gain/per-dataset structures ---
        def nested_get(container, g_idx, d_idx, default=np.nan):
            """
            Try container[g_idx][d_idx] (per-gain-per-dataset),
            then container[g_idx] (per-gain),
            then container[d_idx] (flat per-dataset),
            then the first finite scalar in container,
            else default.
            """
            try:
                # per-gain-per-dataset
                if isinstance(container, (list, tuple)) and g_idx < len(container):
                    maybe = container[g_idx]
                    if isinstance(maybe, (list, tuple)) and d_idx is not None and d_idx < len(maybe):
                        return maybe[d_idx]
                    # per-gain scalar/list
                    return maybe
            except Exception:
                pass
            try:
                # flat per-dataset
                if isinstance(container, (list, tuple)) and d_idx is not None and d_idx < len(container):
                    return container[d_idx]
            except Exception:
                pass
            # fallback to first finite scalar
            val = first_scalar_or_nan(container)
            return val if np.isfinite(val) else default

        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        # optional experiment (I,Q)
        Iexp_q = (I_experiment or {}).get(q, [])
        Qexp_q = (Q_experiment or {}).get(q, [])

        n = min(len(amps_q), len(gains_q), len(rounds_q), len(delay_q)) if all(
            isinstance(x, list) for x in (amps_q, gains_q, rounds_q, delay_q)
        ) else max(len(gains_q), 1)
        if n == 0:
            print(f"No usable data for qubit {q} (missing lists or length mismatch). Skipping.")
            return

        # --------- SSF required inputs check ---------
        do_ssf = (
                ss_class_instance is not None and
                ss_cfg is not None and
                Ig_calibration is not None and
                Ie_calibration is not None and
                Qg_calibration is not None and
                Qe_calibration is not None
        )
        if not do_ssf:
            print("SSF plotting requires ss_class_instance, ss_cfg, and all four calibration dicts. Aborting.")
            return

        Ig_q = Ig_calibration.get(q, [])
        Ie_q = Ie_calibration.get(q, [])
        Qg_q = Qg_calibration.get(q, [])
        Qe_q = Qe_calibration.get(q, [])

        # Outer dimension = number of gains
        num_gains = min(len(Ig_q), len(Ie_q), len(Qg_q), len(Qe_q))
        if num_gains == 0:
            print("SSF requested but no calibration arrays found; aborting SSF.")
            return

        # ensure save root exists (re-using your helper)
        self.create_folder_if_not_exists(save_path)
        analysis_root = os.path.join(save_path, "analysis")
        self.create_folder_if_not_exists(analysis_root)

        # --- temporarily set context on the SSF class for saving like before ---
        _old_outer = getattr(ss_class_instance, "outerFolder", None)
        _old_qidx = getattr(ss_class_instance, "QubitIndex", None)
        _old_rnum = getattr(ss_class_instance, "round_num", None)
        _old_name = getattr(ss_class_instance, "expt_name", None)

        try:
            ss_class_instance.outerFolder = analysis_root
            ss_class_instance.QubitIndex = getattr(self, "qubit", 0)

            for g_idx in range(num_gains):
                # datasets under this gain (inner lists)
                try:
                    n_datasets = min(
                        len(Ig_q[g_idx]),
                        len(Ie_q[g_idx]),
                        len(Qg_q[g_idx]),
                        len(Qe_q[g_idx]),
                    )
                except Exception:
                    print(f"[SSF] Gain index {g_idx}: malformed calibration lists; skipping gain.")
                    continue
                if n_datasets == 0:
                    print(f"[SSF] Gain index {g_idx}: empty calibration; skipping gain.")
                    continue

                # derive a gain label (works if gains_q is scalar, list, per-gain list, etc.)
                g_lbl = first_scalar_or_nan(nested_get(gains_q, g_idx, None, default=np.nan))

                for d_idx in range(n_datasets):
                    # labels: rounds & delays can be flat/per-gain/per-dataset
                    r_raw = nested_get(rounds_q, g_idx, d_idx, default=np.nan)
                    d_lbl = first_scalar_or_nan(nested_get(delay_q, g_idx, d_idx, default=np.nan))
                    r_id = str(int(r_raw)) if np.isfinite(first_scalar_or_nan(r_raw)) else f"{g_idx}_{d_idx}"

                    ss_class_instance.round_num = r_id
                    gain_tag = f"gain_{fmt_p(g_lbl)}_"
                    delay_tag = f"delay_{fmt_p(d_lbl)}"
                    ss_class_instance.expt_name = f"{gain_tag}{delay_tag}_t1_ssf"

                    # read calibration vectors for this (gain, dataset)
                    try:
                        I_g = np.asarray(Ig_q[g_idx][d_idx]).ravel()
                        Q_g = np.asarray(Qg_q[g_idx][d_idx]).ravel()
                        I_e = np.asarray(Ie_q[g_idx][d_idx]).ravel()
                        Q_e = np.asarray(Qe_q[g_idx][d_idx]).ravel()
                        if min(I_g.size, Q_g.size, I_e.size, Q_e.size) == 0:
                            print(f"[SSF] Skipping gain {g_idx} dataset {d_idx}: empty calibration vectors.")
                            continue
                    except Exception as e:
                        print(f"[SSF] Failed to read calibration for gain {g_idx} dataset {d_idx}: {e}")
                        continue

                    # optional single-shot (I,Q) marker (supports flat/per-gain/per-dataset)
                    I_single = pick_single_shot(nested_get(Iexp_q, g_idx, d_idx, default=np.nan))
                    Q_single = pick_single_shot(nested_get(Qexp_q, g_idx, d_idx, default=np.nan))
                    kwargs_meas = {}
                    if np.isfinite(I_single) and np.isfinite(Q_single):
                        kwargs_meas = {"I_meas": I_single, "Q_meas": Q_single}

                    # Call your SSF routine; it handles plotting/saving internally
                    try:
                        ss_class_instance.hist_ssf_with_annotations_tukey(
                            data=[I_g, Q_g, I_e, Q_e],
                            cfg=ss_cfg,
                            plot=True,
                            path_ext='_t1',  # keep same suffix as before for continuity
                            **kwargs_meas
                        )
                        print(f"[SSF] Saved SSF plot for gain {g_idx}, dataset {d_idx} (round {r_id}).")
                    except Exception as e:
                        print(f"[SSF] Failed on gain {g_idx}, dataset {d_idx} (round {r_id}): {e}")

        finally:
            # restore prior context
            if _old_outer is not None:
                ss_class_instance.outerFolder = _old_outer
            if _old_qidx is not None:
                ss_class_instance.QubitIndex = _old_qidx
            if _old_rnum is not None:
                ss_class_instance.round_num = _old_rnum
            if _old_name is not None:
                ss_class_instance.expt_name = _old_name

    def plot_all_t1_heatmaps_with_singular_ssf_vs_time_plotting(
            self,
            amps,
            gains,
            rounds,
            delay_times,
            save_path,
            max_ylabels=6,
            # --- NEW: optional SSD inputs ---
            ss_class_instance=None,
            ss_cfg=None,
            Ig_calibration=None,
            Ie_calibration=None,
            Qg_calibration=None,
            Qe_calibration=None,
            # --- NEW: user-settable threshold for Q channel (applied to both Qe and Qg) ---
            q_threshold=None,
            # --- optional: control Welch parameters for state-PSD ---
            welch_nperseg=None,
            welch_noverlap=None,
    ):
        """
        NEW FORMAT ONLY (augmented to also emit SSD plots per T1 curve)

        NEW BEHAVIOR:
          - If `q_threshold` is provided, we (a) draw it on the Qe/Qg vs time plots as a horizontal line,
            and (b) compute PSD from the *thresholded state* (0/1) that flips whenever the trace crosses
            the threshold, instead of from the raw Q traces.
          - If `q_threshold` is None, we default to the median of the concatenated Qe/Qg samples for that
            dataset to provide a reasonable split.

        Welch PSD of the thresholded state is returned/plotted with units "state²/Hz" (dimensionless²/Hz).
        Saved figure now labels the bottom panel as "Welch PSD (thresholded state)".

        Everything else remains unchanged (heatmaps, CSV of raw Q vs time, etc.).
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict

        # --- helpers (unchanged) ---
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

        # --- extract this qubit's lists (unchanged) ---
        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        n = min(len(amps_q), len(gains_q), len(rounds_q), len(delay_q))
        if n == 0 or not (len(amps_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
            print(f"No usable data for qubit {q} (missing lists or length mismatch). Skipping.")
            return

        # --- OPTIONAL: SSD inputs (unchanged gate) ---
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

        # --- collect points for heatmaps (unchanged) ---
        all_points = []
        import numpy as np
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

            mask = np.isfinite(a_samples) & np.isfinite(g_arr) & np.isfinite(d_arr)
            if not np.any(mask):
                continue

            for g, d, a in zip(g_arr[mask], d_arr[mask], a_samples[mask]):
                all_points.append((r_id, float(g), float(d), float(a)))

        if not all_points:
            print(f"No numeric points for qubit {q}. Skipping.")
            return

        unique_rounds = sorted({r for (r, _, __, ___) in all_points})

        # ensure save root exists
        self.create_folder_if_not_exists(save_path)

        # ---------- global z scale for heatmaps (unchanged) ----------
        all_amps = np.array([a for (_, _, _, a) in all_points], dtype=float)
        global_vmin = float(np.nanmin(all_amps))
        global_vmax = float(np.nanmax(all_amps))

        # ---------- SSD block: add thresholding + state PSD ----------
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
                    d_lbl = first_scalar_or_nan(delay_q[i])

                    ss_class_instance.round_num = r_id
                    gain_tag = f"gain_{fmt_p(g_lbl)}_"
                    delay_tag = f"delay_{fmt_p(d_lbl)}"
                    ss_class_instance.expt_name = f"{gain_tag}{delay_tag}_t1_ssd"

                    round_folder = os.path.join(
                        analysis_root, "ss_plots",
                        f"Q{ss_class_instance.QubitIndex + 1}",
                        f"Round_{r_id}"
                    )
                    self.create_folder_if_not_exists(round_folder)

                    try:
                        I_g = np.asarray(Ig_q[i]).ravel()
                        Q_g = np.asarray(Qg_q[i]).ravel()
                        I_e = np.asarray(Ie_q[i]).ravel()
                        Q_e = np.asarray(Qe_q[i]).ravel()
                        if min(I_g.size, Q_g.size, I_e.size, Q_e.size) == 0:
                            print(f"[SSD] Skipping dataset {i}: empty calibration vectors.")
                            continue

                        # user SSD routine (unchanged)
                        ss_class_instance.hist_ssf(
                            data=[I_g, Q_g, I_e, Q_e],
                            cfg=ss_cfg,
                            plot=True,
                        )

                        # --- align, build timebase (unchanged) ---
                        N = max(min(Q_e.size, Q_g.size), 1)
                        Qe_trim = Q_e[:N].astype(float)
                        Qg_trim = Q_g[:N].astype(float)

                        dt = 30.0 / float(N)
                        t = np.arange(N, dtype=float) * dt
                        fs = 1.0 / dt

                        # --- save raw Q CSV (unchanged columns) ---
                        csv_path = os.path.join(
                            round_folder,
                            f"Q{ss_class_instance.QubitIndex + 1}_Round_{r_id}_{gain_tag}{delay_tag}_ssf_Q_vs_time.csv"
                        )
                        try:
                            import csv
                            with open(csv_path, "w", newline="") as f:
                                w = csv.writer(f)
                                w.writerow(["time_s", "Qe", "Qg"])
                                for k in range(N):
                                    w.writerow([f"{t[k]:.9g}", f"{Qe_trim[k]:.9g}", f"{Qg_trim[k]:.9g}"])
                        except Exception as e_csv:
                            print(f"[SSD] Failed writing CSV ({csv_path}): {e_csv}")

                        # --------- NEW: thresholding to create state traces ----------
                        # default threshold if not provided: median of pooled Qe and Qg
                        if q_threshold is None:
                            thr = float(np.nanmedian(np.concatenate([Qe_trim, Qg_trim])))
                        else:
                            thr = float(q_threshold)

                        # binary states: 1 if above threshold, 0 otherwise
                        Se = (Qe_trim > thr).astype(float)
                        Sg = (Qg_trim > thr).astype(float)

                        # Optionally also save the thresholded states for offline use
                        csv_state_path = os.path.join(
                            round_folder,
                            f"Q{ss_class_instance.QubitIndex + 1}_Round_{r_id}_{gain_tag}{delay_tag}_state_vs_time.csv"
                        )
                        try:
                            import csv
                            with open(csv_state_path, "w", newline="") as f:
                                w = csv.writer(f)
                                w.writerow(["time_s", "Se_state", "Sg_state", "threshold_q"])
                                for k in range(N):
                                    w.writerow([f"{t[k]:.9g}", int(Se[k]), int(Sg[k]), f"{thr:.9g}"])
                        except Exception as e_csv2:
                            print(f"[SSD] Failed writing state CSV ({csv_state_path}): {e_csv2}")

                        # --------- Welch PSD of the thresholded states ----------
                        def _welch_psd(x, dt, nperseg=None, noverlap=None):
                            x = np.asarray(x, dtype=float)
                            Nloc = x.size
                            if Nloc < 4 or not np.all(np.isfinite(x)):
                                return np.array([0.0]), np.array([np.nan])
                            fs_loc = 1.0 / dt
                            try:
                                from scipy import signal
                            except Exception:
                                # fallback simple FFT PSD
                                X = np.fft.rfft(x - np.nanmean(x), n=Nloc)
                                f = np.fft.rfftfreq(Nloc, d=dt)
                                Pxx = (dt / Nloc) * (np.abs(X) ** 2)
                                if Nloc % 2 == 0:
                                    if Pxx.size > 2: Pxx[1:-1] *= 2.0
                                else:
                                    if Pxx.size > 1: Pxx[1:] *= 2.0
                                return f, Pxx
                            if nperseg is None:
                                nperseg = min(256, Nloc)
                            if noverlap is None:
                                noverlap = nperseg // 2
                            nperseg = max(4, min(nperseg, Nloc))
                            noverlap = max(0, min(noverlap, nperseg - 1))
                            f, Pxx = signal.welch(
                                x - np.mean(x),  # remove DC to focus on switching content
                                fs=fs_loc,
                                window="hann",
                                nperseg=nperseg,
                                noverlap=noverlap,
                                detrend="constant",
                                return_onesided=True,
                                scaling="density",
                                average="mean",
                            )
                            return f, Pxx

                        f_e, P_e = _welch_psd(Se, dt, welch_nperseg, welch_noverlap)
                        f_g, P_g = _welch_psd(Sg, dt, welch_nperseg, welch_noverlap)

                        # --------- 3×1 figure: Qe, Qg (with threshold), state-PSD ----------
                        fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.6), gridspec_kw={"hspace": 0.35})
                        ax_e, ax_g, ax_psd = axes

                        # Top: e population vs time + threshold line
                        ax_e.scatter(t, Qe_trim, s=8)  # small markers to reduce overdraw
                        ax_e.axhline(thr, linestyle="--", linewidth=1.0, alpha=0.9, label=f"threshold = {thr:.4g}")
                        ax_e.set_ylabel("Qe (a.u.)")
                        ax_e.set_title("e population (Q channel)")
                        ax_e.legend(loc="best")

                        # Middle: g population vs time + threshold line
                        ax_g.scatter(t, Qg_trim, s=8)
                        ax_g.axhline(thr, linestyle="--", linewidth=1.0, alpha=0.9, label=f"threshold = {thr:.4g}")
                        ax_g.set_ylabel("Qg (a.u.)")
                        ax_g.set_title("g population (Q channel)")
                        ax_g.set_xlabel("Time (s)")
                        ax_g.legend(loc="best")

                        # Bottom: PSD of thresholded states
                        ax_psd.semilogy(f_e, P_e, label="Se PSD (thresholded)")
                        ax_psd.semilogy(f_g, P_g, label="Sg PSD (thresholded)")
                        ax_psd.set_xlim(0.0, fs / 2.0)
                        ax_psd.set_xlabel("Frequency (Hz)")
                        ax_psd.set_ylabel("PSD (state²/Hz)")
                        ax_psd.set_title("Welch PSD (thresholded state)")
                        ax_psd.legend(loc="upper right")

                        fig.suptitle(
                            f"Qubit {ss_class_instance.QubitIndex + 1} — Round {r_id}\n"
                            f"{gain_tag}{delay_tag}".rstrip("_")
                        )
                        fig.tight_layout(rect=[0, 0, 1, 0.93])

                        png_path = os.path.join(
                            round_folder,
                            f"Q{ss_class_instance.QubitIndex + 1}_Round_{r_id}_{gain_tag}{delay_tag}_ssf_Q_vs_time_and_state_psd.png"
                        )
                        fig.savefig(png_path, dpi=getattr(self, "final_figure_quality", 150))
                        plt.close(fig)
                        print(f"[SSD] Saved Q-vs-time (with threshold) + state-PSD plot to: {png_path}")

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

        # ---------- per-round heatmaps (unchanged) ----------
        for r_id in unique_rounds:
            pts = [(g, d, a) for (r, g, d, a) in all_points if r == r_id]
            if not pts:
                continue

            gains_r = sorted({g for (g, _, _) in pts})
            delays_r = sorted({d for (_, d, _) in pts})

            bucket = defaultdict(list)
            gi_map = {g: i for i, g in enumerate(gains_r)}
            di_map = {d: i for i, d in enumerate(delays_r)}
            for g, d, a in pts:
                bucket[(di_map[d], gi_map[g])].append(a)

            Ny, Nx = len(delays_r), len(gains_r)
            C = np.full((Ny, Nx), np.nan, dtype=float)
            for (iy, ix), vals in bucket.items():
                C[iy, ix] = float(np.nanmean(vals))

            x_edges = centers_to_edges(gains_r)
            y_edges = centers_to_edges(delays_r)

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            mesh = ax.pcolormesh(
                x_edges, y_edges, C, shading='flat',
                vmin=global_vmin, vmax=global_vmax
            )
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label("Qubit Population")

            ax.set_title(f"Qubit {self.qubit + 1} — Round {r_id}")
            ax.set_xlabel("Pulse gain (a.u.)")
            ax.set_ylabel("Delay time")

            import matplotlib.ticker as mticker
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7, prune=None))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            if Ny > 0:
                if Ny <= max_ylabels:
                    yticks_idx = list(range(Ny))
                else:
                    yticks_idx = np.linspace(0, Ny - 1, num=max_ylabels, dtype=int).tolist()
                    yticks_idx = sorted(set(yticks_idx))
                yticks_vals = [delays_r[i] for i in yticks_idx]
                ax.set_yticks(yticks_vals)
                ax.set_yticklabels([f"{v:.0f}" for v in yticks_vals])

            fig.tight_layout()
            outfile = (save_path + f"t1_heatmap_q{self.qubit}_round{r_id}.png")
            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved heatmap for round {r_id} to: {outfile}")

    def fit_and_save_t1_slices_new_format(
            self,
            amps,
            gains,
            rounds,
            delay_times,
            save_path,
            min_points_for_fit=4,
            figure_size=(6.0, 4.0),
            # NEW:
            n_bar=None,  # list/np.array OR dict[str]->(list or {"lorentzian"/"gaussian"})
    ):
        """
        For each round and each gain, fit an exponential T1 curve vs delay and save
        a slice plot with the fit overlaid. Only figures are saved (no CSV/TXT).
        Additionally, for each round, save a single plot of T1 vs gain (or vs n̄ if provided) with error bars.

        If n_bar is provided, the summary plot x-axis uses n̄ instead of gain:
          - n_bar can be a single list/array aligned to the round's sorted gains,
          - or a dict mapping round_id -> list, or -> {"gains":..., "lorentzian":..., "gaussian":...}
            (prefers 'lorentzian' if present, else 'gaussian').
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        from scipy.optimize import curve_fit
        import os

        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        n = min(len(amps_q), len(gains_q), len(rounds_q), len(delay_q))
        if n == 0 or not (len(amps_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
            print(f"No usable data for qubit {q} (missing lists or length mismatch). Skipping.")
            return

        # ---------- Flatten all points into (round_id, gain, delay, amp) ----------
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

            mask = np.isfinite(a_samples) & np.isfinite(g_arr) & np.isfinite(d_arr)
            if not np.any(mask):
                continue

            for g_val, d_val, a_val in zip(g_arr[mask], d_arr[mask], a_samples[mask]):
                all_points.append((r_id, float(g_val), float(d_val), float(a_val)))

        if not all_points:
            print(f"No numeric points for qubit {q}. Skipping.")
            return

        # Unique rounds present
        unique_rounds = sorted({r for (r, _, __, ___) in all_points})

        # Ensure base save folder exists
        self.create_folder_if_not_exists(save_path)

        # Helper: average amplitudes per (gain, delay) for one round
        def avg_grid_for_round(r_id):
            pts = [(g, d, a) for (r, g, d, a) in all_points if r == r_id]
            if not pts:
                return {}, {}

            # bucket by (gain, delay)
            by_key = defaultdict(list)
            for g, d, a in pts:
                by_key[(g, d)].append(a)

            # averaged amplitude for each key
            avg = {k: float(np.nanmean(v)) for k, v in by_key.items()}

            # build: gains -> {delay: amp}
            per_gain = defaultdict(dict)
            for (g, d), val in avg.items():
                per_gain[g][d] = val

            # also capture all unique delays seen per round
            return per_gain, {d for (_, d) in by_key.keys()}

        # Pull the n̄ vector for a given round (if provided), aligned to sorted_gains
        def get_nbar_for_round(r_id_str, sorted_gains):
            if n_bar is None:
                return None

            # direct list/array: assume aligned to sorted gains for this round
            if isinstance(n_bar, (list, tuple, np.ndarray)):
                nb = np.asarray(n_bar, float).ravel()
                return nb if nb.size == len(sorted_gains) else None

            # dict-like
            if isinstance(n_bar, dict):
                key = r_id_str if r_id_str in n_bar else str(r_id_str)
                entry = n_bar.get(key, None)
                if entry is None:
                    return None
                if isinstance(entry, dict):
                    candidate = entry.get("lorentzian") or entry.get("gaussian") or entry.get("nbar") or entry.get(
                        "values")
                    if candidate is None:
                        return None
                    nb = np.asarray(candidate, float).ravel()
                    return nb if nb.size == len(sorted_gains) else None
                if isinstance(entry, (list, tuple, np.ndarray)):
                    nb = np.asarray(entry, float).ravel()
                    return nb if nb.size == len(sorted_gains) else None

            return None

        # Fit a single slice (y vs t) with robust defaults
        def fit_slice(t, y):
            t = np.asarray(t, dtype=float).ravel()
            y = np.asarray(y, dtype=float).ravel()

            order = np.argsort(t)
            t = t[order]
            y = y[order]

            # initial guesses
            a_guess = np.nanmax(y) - np.nanmin(y)
            b_guess = float(t[0]) if np.isfinite(t[0]) else 0.0
            c_guess = (t[-1] - t[0]) / 5.0 if t.size >= 2 else max(abs(t[0]), 1.0)
            d_guess = np.nanmin(y)

            p0 = [a_guess, b_guess, max(c_guess, 1e-9), d_guess]
            bounds = ([-np.inf, -np.inf, 0.0, -np.inf], [np.inf, np.inf, np.inf, np.inf])

            try:
                popt, pcov = curve_fit(self.exponential, t, y, p0=p0, bounds=bounds,
                                       method='trf', maxfev=10000)
                yfit = self.exponential(t, *popt)
                T1 = popt[2]
                T1_err = float(np.sqrt(pcov[2][2])) if pcov.shape == (4, 4) and pcov[2][2] >= 0 else float('inf')
                success = True
            except Exception:
                popt, yfit, T1, T1_err, success = [np.nan] * 4, np.full_like(t, np.nan), np.nan, np.nan, False

            return t, y, yfit, popt, T1, T1_err, success

        # ---------- iterate rounds/gains; fit & save ----------
        for r_id in unique_rounds:
            per_gain, _ = avg_grid_for_round(r_id)
            if not per_gain:
                continue

            # Single folder per round (all gains go here)
            round_dir = os.path.join(save_path, f"t1_slices_q{q}_round_{r_id}")
            self.create_folder_if_not_exists(round_dir)

            # stable gain ordering
            sorted_gains = sorted(per_gain.keys())

            # n̄ for this round (if provided)
            nbar_vec = get_nbar_for_round(str(r_id), sorted_gains)

            # collect T1 vs (x) for this round
            t1_points = []  # (gain, T1, T1_err, fit_ok)

            for g_val in sorted_gains:
                # slice for this gain: (t -> y)
                delays = sorted(per_gain[g_val].keys())
                yvals = [per_gain[g_val][d] for d in delays]

                # need enough points to fit
                finite_mask = [np.isfinite(d) and np.isfinite(y) for d, y in zip(delays, yvals)]
                delays = [d for d, m in zip(delays, finite_mask) if m]
                yvals = [y for y, m in zip(yvals, finite_mask) if m]

                if len(delays) < min_points_for_fit:
                    print(f"[q{q} round {r_id} gain {g_val}] Not enough points ({len(delays)}) for fit. Skipping.")
                    continue

                t, y, yfit, popt, T1, T1_err, ok = fit_slice(delays, yvals)

                # save figure with overlay (fit a different color)
                fig, ax = plt.subplots(figsize=figure_size)
                ax.scatter(t, y, s=18, label="data")  # default color (C0)
                ax.plot(t, yfit, linewidth=2.0, label="fit", color='C1')  # different color (C1)

                # title suffix with n̄ if available
                suffix = ""
                if nbar_vec is not None:
                    try:
                        idx = sorted_gains.index(g_val)
                        suffix = f" — n̄ {float(nbar_vec[idx]):g}"
                    except Exception:
                        suffix = ""

                ax.set_title(f"Qubit {q + 1} — Round {r_id} — Gain {g_val}{suffix}")
                ax.set_xlabel("Delay time")
                ax.set_ylabel("Signal (arb.)")
                ax.grid(True, alpha=0.25)

                # annotate T1
                if ok and np.isfinite(T1):
                    ax.text(0.02, 0.98,
                            f"T1 = {T1:.3g} ± {T1_err:.2g}",
                            transform=ax.transAxes, va='top', ha='left')

                ax.legend()
                fig.tight_layout()

                safe_gain = str(g_val).replace('.', 'p').replace('-', 'm')
                f_png = os.path.join(round_dir, f"t1_slice_gain{safe_gain}_q{q}_round_{r_id}.png")
                fig.savefig(f_png, dpi=self.final_figure_quality)
                plt.close(fig)

                t1_points.append((float(g_val), float(T1), float(T1_err), bool(ok)))

                print(f"Saved T1 slice & fit for round {r_id}, gain {g_val} to: {round_dir}")

            # ---- Save a single T1 vs gain (or vs n̄) plot with error bars for this round ----
            if t1_points:
                ok_pts = [(g, t1, e) for (g, t1, e, ok) in t1_points if ok and np.isfinite(t1)]
                if ok_pts:
                    gs, t1s, errs = list(zip(*sorted(ok_pts)))  # sort by gain

                    # T1, err are in µs
                    gs = np.asarray(gs, float)
                    t1_us = np.asarray(t1s, float)
                    err_us = np.asarray(errs, float)

                    # keep only positive, finite T1
                    m = np.isfinite(t1_us) & (t1_us > 0) & np.isfinite(err_us)
                    gs, t1_us, err_us = gs[m], t1_us[m], err_us[m]

                    Gamma = 1000.0 / t1_us  # [1/ms]
                    Gamma_err = 1000.0 * err_us / (t1_us ** 2)  # [1/ms]

                    if nbar_vec is not None and len(nbar_vec) == len(sorted_gains):
                        # map gains -> nbar
                        gain_to_idx = {g: i for i, g in enumerate(sorted_gains)}
                        xs_list, Gamma_list, Gamma_err_list = [], [], []
                        for g_val, T1_val, T1_err_val in zip(gs, t1_us, err_us):
                            idx = gain_to_idx.get(g_val)
                            if idx is None:
                                continue
                            xs_list.append(float(nbar_vec[idx]))
                            # use already-computed Gamma & Gamma_err for this same ordering
                            g_mask = (gs == g_val)
                            Gamma_list.append(float(Gamma[g_mask][0]))
                            Gamma_err_list.append(float(Gamma_err[g_mask][0]))

                        xs = np.asarray(xs_list, float)
                        ys = np.asarray(Gamma_list, float)
                        es = np.asarray(Gamma_err_list, float)

                        # sort by n̄
                        order = np.argsort(xs)
                        xs, ys, es = xs[order], ys[order], es[order]

                        # ---- Single linear Γ vs n̄ plot ----
                        fig2, ax_lin = plt.subplots(
                            figsize=(figure_size[0], figure_size[1])
                        )

                        ax_lin.errorbar(xs, ys, yerr=es, fmt='o', capsize=3,
                                        label=r"$\Gamma = 1/T_1$")
                        ax_lin.set_title(f"Qubit {q + 1} — Round {r_id} — Γ vs n̄")
                        ax_lin.set_xlabel(r"$\bar{n}$")
                        ax_lin.set_ylabel(r"$\Gamma$ (1/ms)")
                        ax_lin.grid(True, alpha=0.25)
                        ax_lin.legend()

                        fig2.tight_layout()
                        # Save to common gamma_vs_nbar folder instead of round-specific folder
                        gamma_folder = os.path.join(save_path, f"gamma_vs_nbar_q{q}")
                        self.create_folder_if_not_exists(gamma_folder)
                        f_vs = os.path.join(gamma_folder, f"Gamma_vs_nbar_round{r_id}.png")
                        fig2.savefig(f_vs, dpi=self.final_figure_quality)
                        plt.close(fig2)

                        print(f"Saved Γ vs n̄ (linear x) for round {r_id} to: {gamma_folder}")


                    else:
                        fig2, ax2 = plt.subplots(figsize=figure_size)
                        ax2.errorbar(gs, Gamma, yerr=Gamma_err, fmt='o', capsize=3, label="Γ = 1/T1")
                        ax2.set_title(f"Qubit {q + 1} — Round {r_id} — Γ vs Gain")
                        ax2.set_xlabel("Gain")
                        ax2.set_ylabel(r"$\Gamma$ (1/ms)")
                        ax2.grid(True, alpha=0.25)
                        ax2.legend()
                        fig2.tight_layout()

                        f_vs = os.path.join(round_dir, f"Gamma_vs_gain_round{r_id}.png")
                        fig2.savefig(f_vs, dpi=self.final_figure_quality)
                        plt.close(fig2)
                        print(f"Saved Γ vs Gain (with error bars) for round {r_id} to: {round_dir}")
                else:
                    print(f"[q{q} round {r_id}] No successful T1 fits to plot.")

    def plot_all_t1_heatmaps_new_format(
            self,
            amps,
            gains,
            rounds,
            delay_times,
            save_path,
            max_ylabels=6,
            n_bar=None,
            use_linear_x=True,
    ):
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        import matplotlib.ticker as mticker
        from scipy.optimize import least_squares

        # -------------------- T1 fit helper --------------------
        def exp_decay(t, A, T1, c):
            return A * np.exp(-t / T1) + c

        def fit_t1_slice(t_us, y):
            """
            Fit y(t) = A*exp(-t/T1) + c.
            t_us in microseconds. Returns T1_us (float) or np.nan.
            """
            t_us = np.asarray(t_us, float)
            y = np.asarray(y, float)
            m = np.isfinite(t_us) & np.isfinite(y)
            if np.sum(m) < 6:
                return np.nan

            t = t_us[m]
            yy = y[m]

            # sort by t
            o = np.argsort(t)
            t = t[o]
            yy = yy[o]

            # shift time so t[0]=0 for numerical stability
            t0 = t[0]
            t = t - t0

            tmax = float(np.max(t))
            if not np.isfinite(tmax) or tmax <= 0:
                return np.nan

            # robust baseline guess from last 20% of points
            n = len(t)
            k = max(1, int(0.2 * n))
            c0 = float(np.nanmedian(yy[-k:]))

            # amplitude guess from early points
            A0 = float(np.nanmedian(yy[:k]) - c0)

            # crude T1 guess: 1/3 of total span (works okay for most traces)
            T10 = max(tmax / 3.0, 0.05)  # in us

            p0 = np.array([A0, T10, c0], float)

            # bounds: T1 positive; A and c fairly free
            dt_min = float(np.min(np.diff(t))) if len(t) >= 2 else (tmax / max(1, len(t) - 1))
            dt_min = dt_min if np.isfinite(dt_min) and dt_min > 0 else 1e-3

            lb = np.array([-np.inf, 0.2 * dt_min, -np.inf], float)
            ub = np.array([np.inf, 10.0 * tmax, np.inf], float)

            def resid(p):
                A, T1, c = p
                return exp_decay(t, A, T1, c) - yy

            try:
                res = least_squares(
                    resid, p0, bounds=(lb, ub),
                    loss="soft_l1",
                    f_scale=(np.nanstd(yy) if np.isfinite(np.nanstd(yy)) and np.nanstd(yy) > 0 else 1.0),
                    max_nfev=3000
                )
                if not res.success:
                    return np.nan
                return float(res.x[1])  # T1 in us
            except Exception:
                return np.nan

        # -------------------- original code below --------------------
        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        n = min(len(amps_q), len(gains_q), len(rounds_q), len(delay_q))
        if n == 0 or not (len(amps_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
            print(f"No usable data for qubit {q} (missing lists or length mismatch). Skipping.")
            return {}

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

            mask = np.isfinite(a_samples) & np.isfinite(g_arr) & np.isfinite(d_arr)
            if not np.any(mask):
                continue

            for g, d, a in zip(g_arr[mask], d_arr[mask], a_samples[mask]):
                all_points.append((r_id, float(g), float(d), float(a)))

        if not all_points:
            print(f"No numeric points for qubit {q}. Skipping.")
            return {}

        unique_rounds = sorted({r for (r, _, __, ___) in all_points})

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

        all_amps = np.array([a for (_, _, _, a) in all_points], dtype=float)
        global_vmin = float(np.nanmin(all_amps))
        global_vmax = float(np.nanmax(all_amps))

        def get_nbar_for_round(r_id_str, gains_sorted):
            if n_bar is None:
                return None
            if isinstance(n_bar, (list, tuple, np.ndarray)):
                nb = np.asarray(n_bar, float).ravel()
                return nb if nb.size == len(gains_sorted) else None
            if isinstance(n_bar, dict):
                key = r_id_str if r_id_str in n_bar else str(r_id_str)
                entry = n_bar.get(key, None)
                if entry is None:
                    return None
                if isinstance(entry, dict):
                    candidate = (
                            entry.get("lorentzian")
                            or entry.get("gaussian")
                            or entry.get("nbar")
                            or entry.get("values")
                    )
                    if candidate is None:
                        return None
                    nb = np.asarray(candidate, float).ravel()
                    return nb if nb.size == len(gains_sorted) else None
                if isinstance(entry, (list, tuple, np.ndarray)):
                    nb = np.asarray(entry, float).ravel()
                    return nb if nb.size == len(gains_sorted) else None
            return None

        # NEW: return object collecting T1 vs gain per round
        t1_by_round = {}

        for r_id in unique_rounds:
            pts = [(g, d, a) for (r, g, d, a) in all_points if r == r_id]
            if not pts:
                continue

            gains_r = sorted({g for (g, _, _) in pts})
            delays_r = sorted({d for (_, d, _) in pts})

            bucket = defaultdict(list)
            gi_map = {g: i for i, g in enumerate(gains_r)}
            di_map = {d: i for i, d in enumerate(delays_r)}
            for g, d, a in pts:
                bucket[(di_map[d], gi_map[g])].append(a)

            Ny, Nx = len(delays_r), len(gains_r)
            C = np.full((Ny, Nx), np.nan, dtype=float)
            for (iy, ix), vals in bucket.items():
                C[iy, ix] = float(np.nanmean(vals))

            # ---------- NEW: extract T1 per gain ----------
            delays_us = np.asarray(delays_r, float)  # assumed already in us (matches your axis label)
            t1_us_list = []
            for ix in range(Nx):
                y_col = C[:, ix]
                t1_us_list.append(fit_t1_slice(delays_us, y_col))
            t1_us_arr = np.asarray(t1_us_list, float)

            # Store for return (aligned with gains_r)
            t1_by_round[str(r_id)] = {
                "gains": np.asarray(gains_r, float),
                "T1_us": t1_us_arr,
                "T1_s": t1_us_arr * 1e-6,
            }

            # ---------- plotting (unchanged) ----------
            nbar_vec = get_nbar_for_round(str(r_id), gains_r)
            if nbar_vec is not None:
                x_vals = np.asarray(nbar_vec, float)
                x_label = "n̄"
            else:
                x_vals = np.asarray(gains_r, float)
                x_label = "Pulse gain (a.u.)"

            order = np.argsort(x_vals)
            x_vals_sorted = x_vals[order]
            C_sorted = C[:, order]

            # --- X axis spacing control ---
            if use_linear_x:
                x_centers = x_vals_sorted.astype(float)
            else:
                x_centers = np.arange(len(x_vals_sorted), dtype=float)

            x_edges = centers_to_edges(x_centers)
            y_edges = centers_to_edges(delays_r)

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            mesh = ax.pcolormesh(
                x_edges,
                y_edges,
                C_sorted,
                shading='flat',
                vmin=global_vmin,
                vmax=global_vmax
            )
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label("Qubit Population")

            ax.set_title(f"Qubit {self.qubit + 1} — Round {r_id}")
            ax.set_xlabel(x_label)
            ax.set_ylabel(r"Delay time ($\mu$s)")

            if use_linear_x:
                ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7, prune=None))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            else:
                max_xticks = 10
                if Nx <= max_xticks:
                    xticks_idx = list(range(Nx))
                else:
                    xticks_idx = np.linspace(0, Nx - 1, num=max_xticks, dtype=int).tolist()
                    xticks_idx = sorted(set(xticks_idx))

                xtick_positions = [x_centers[i] for i in xticks_idx]
                xtick_values = [x_vals_sorted[i] for i in xticks_idx]
                ax.set_xticks(xtick_positions)
                ax.set_xticklabels([f"{v:.3f}" for v in xtick_values], rotation=45, ha='right')

            if Ny > 0:
                if Ny <= max_ylabels:
                    yticks_idx = list(range(Ny))
                else:
                    yticks_idx = np.linspace(0, Ny - 1, num=max_ylabels, dtype=int).tolist()
                    yticks_idx = sorted(set(yticks_idx))
                yticks_vals = [delays_r[i] for i in yticks_idx]
                ax.set_yticks(yticks_vals)
                ax.set_yticklabels([f"{v:.0f}" for v in yticks_vals])

            fig.tight_layout()
            if n_bar is not None:
                outfile = (save_path + f"t1_heatmap_q{self.qubit}_round{r_id}_nbar.png")
            else:
                outfile = (save_path + f"t1_heatmap_q{self.qubit}_round{r_id}.png")

            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved heatmap for round {r_id} to: {outfile}")

        return t1_by_round

    def plot_all_t1_heatmaps_single_calibration(self, Is,Qs,Ig_calibration1, \
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
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])
        I_q = Is.get(q, [])
        Q_q = Qs.get(q, [])
        g = np.mean(Ig_calibration1 + 1j * Qg_calibration1)
        e = np.mean(Ie_calibration1 + 1j * Qe_calibration1)
        denom = np.abs(e - g) ** 2
        if denom <= 0 or not np.isfinite(denom):
            raise ValueError("Best calibration is degenerate (e ≈ g); cannot normalize.")

        # Build amps_q for ALL datasets, using the SAME e/g
        amps_q = []
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
            ax.set_ylabel("Delay time")

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
                ax.set_yticklabels([f"{v:.0f}" for v in yticks_vals])
            # -------------------------------------------------------------------

            fig.tight_layout()
            outfile = (save_path + f"t1_heatmap_q{self.qubit}_round{r_id}.png")
            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved heatmap for round {r_id} to: {outfile}")

    def plot_all_t1_heatmaps_single_calibration_IQ(self, Is, Qs, Ig_calibration1,
                                                   Ie_calibration1, Qe_calibration1, Qg_calibration1, gains, rounds,
                                                   delay_times, save_path, max_ylabels=6):
        """
        NEW FORMAT ONLY — now plots 4 heatmaps per round (I, Q, |IQ|, Calibrated),
        and ALSO saves per-gain T1 curves (delay vs calibrated population) to 't1slices/'.

        Changes vs previous:
          - Adds subplots for I, Q, and raw amplitude sqrt(I^2 + Q^2) alongside calibrated amplitude.
          - Keeps fixed color scales (per-metric) across rounds.
          - Y-axis shows at most `max_ylabels` delay_time tick labels (evenly spaced).
          - NEW: For each round and for each unique gain, plot a T1 curve (delay on x, qubit population on y)
                 and save to: {save_path}/t1slices/t1curve_q{q}_round{round}_gain{gain}.png

        Data model (per qubit q):
          - Is[q][i], Qs[q][i] : 1D arrays of samples for dataset i
          - gains[q][i]        : scalar or per-sample array
          - rounds[q][i]       : round label for dataset i (str/int)
          - delay_times[q][i]  : scalar or per-sample array
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        import matplotlib.ticker as mticker
        import os

        q = self.qubit
        gains_q = gains.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])
        I_q = Is.get(q, [])

        Q_q = Qs.get(q, [])

        # --- Calibration (shared for all datasets) ---
        g = np.mean(Ig_calibration1 + 1j * Qg_calibration1)
        e = np.mean(Ie_calibration1 + 1j * Qe_calibration1)
        denom = np.abs(e - g) ** 2
        if denom <= 0 or not np.isfinite(denom):
            raise ValueError("Best calibration is degenerate (e ≈ g); cannot normalize.")

        # Ensure save folders exist
        self.create_folder_if_not_exists(save_path)
        t1slice_dir = os.path.join(save_path, "t1slices")
        self.create_folder_if_not_exists(t1slice_dir)

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
        n_data = min(len(I_q), len(Q_q), len(gains_q), len(rounds_q), len(delay_q))
        if n_data == 0:
            print(f"No usable data for qubit {q} (missing lists). Skipping.")
            return

        # all_points_<metric>: list of tuples (round_id, gain, delay, value)
        all_points_I = []
        all_points_Q = []
        all_points_abs = []
        all_points_cal = []
        # Calibration fit on projected coordinate x
        z_cal_g = Ig_calibration1 + 1j * Qg_calibration1
        z_cal_e = Ie_calibration1 + 1j * Qe_calibration1
        g = z_cal_g.mean();
        e = z_cal_e.mean()
        u = (e - g) / np.abs(e - g)

        xg = np.real((z_cal_g - g) * np.conj(u))
        xe = np.real((z_cal_e - g) * np.conj(u))

        mu_g, sig_g = np.mean(xg), np.std(xg, ddof=1)
        mu_e, sig_e = np.mean(xe), np.std(xe, ddof=1)
        pi_e = 0.5  # or use relative counts

        def p_e_from_x(x):
            # Gaussian LLR -> P(e|x)
            from numpy import exp
            Ng = exp(-0.5 * ((x - mu_g) / sig_g) ** 2) / (sig_g + 1e-12)
            Ne = exp(-0.5 * ((x - mu_e) / sig_e) ** 2) / (sig_e + 1e-12)
            return (pi_e * Ne) / (pi_e * Ne + (1 - pi_e) * Ng + 1e-300)

        for i in range(n_data):
            I = np.asarray(I_q[i], dtype=float).ravel()
            Q = np.asarray(Q_q[i], dtype=float).ravel()
            if I.size == 0 or Q.size == 0 or I.size != Q.size:
                continue

            z = I + 1j * Q
            x = np.real((z - g) * np.conj(u))
            amp_cal = p_e_from_x(x)
            #amp_cal = np.real(((z - g) * np.conj(e - g)) / np.abs(e - g)**2)#np.abs((z - g) * (e - g) / denom)  # calibrated (population-like)
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

            # delay can be scalar or per-sample
            d_i = delay_q[i]
            if isinstance(d_i, (list, tuple, np.ndarray)):
                d_arr = np.asarray(d_i, dtype=float).ravel()
                if d_arr.size not in (1, I.size):
                    continue
                if d_arr.size == 1:
                    d_arr = np.full(I.shape, float(d_arr[0]), dtype=float)
            else:
                try:
                    d_arr = np.full(I.shape, float(d_i), dtype=float)
                except Exception:
                    continue

            r_id = str(rounds_q[i])

            # keep only finite entries
            mask = (np.isfinite(I) & np.isfinite(Q) &
                    np.isfinite(amp_raw) & np.isfinite(amp_cal) &
                    np.isfinite(g_arr) & np.isfinite(d_arr))
            if not np.any(mask):
                continue

            for g_val, d_val, iv, qv, av, cv in zip(g_arr[mask], d_arr[mask], I[mask], Q[mask], amp_raw[mask],
                                                    amp_cal[mask]):
                g_val = float(g_val);
                d_val = float(d_val)
                all_points_I.append((r_id, g_val, d_val, float(iv)))
                all_points_Q.append((r_id, g_val, d_val, float(qv)))
                all_points_abs.append((r_id, g_val, d_val, float(av)))
                all_points_cal.append((r_id, g_val, d_val, float(cv)))

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
            pts = [(g, d, a) for (r, g, d, a) in points if r == r_id]
            if not pts:
                return None
            gains_r = sorted({g for (g, _, _) in pts})
            delays_r = sorted({d for (_, d, _) in pts})
            gi_map = {g: i for i, g in enumerate(gains_r)}
            di_map = {d: i for i, d in enumerate(delays_r)}
            bucket = defaultdict(list)
            for g, d, a in pts:
                bucket[(di_map[d], gi_map[g])].append(a)
            Ny, Nx = len(delays_r), len(gains_r)
            C = np.full((Ny, Nx), np.nan, dtype=float)
            for (iy, ix), vals in bucket.items():
                C[iy, ix] = float(np.nanmean(vals))
            x_edges = centers_to_edges(gains_r)
            y_edges = centers_to_edges(delays_r)
            return (x_edges, y_edges, C, gains_r, delays_r)

        for r_id in unique_rounds:
            gi = grid_for_round(r_id, all_points_I)
            gq = grid_for_round(r_id, all_points_Q)
            ga = grid_for_round(r_id, all_points_abs)
            gc = grid_for_round(r_id, all_points_cal)
            if not all([gi, gq, ga, gc]):
                print(f"Round {r_id}: incomplete metric grids; skipping.")
                continue

            (xI, yI, CI, gains_Ir, delays_Ir) = gi
            (xQ, yQ, CQ, gains_Qr, delays_Qr) = gq
            (xA, yA, CA, gains_Ar, delays_Ar) = ga
            (xC, yC, CC, gains_Cr, delays_Cr) = gc

            # ---------- Heatmaps (existing behavior) ----------
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
                ax.set_ylabel("Delay time")
                ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # Limit number of Y tick labels (per axis)
            def set_y_ticks(ax, delays_centers):
                Ny = len(delays_centers)
                if Ny == 0:
                    return
                if Ny <= max_ylabels:
                    idx = list(range(Ny))
                else:
                    idx = sorted(set(np.linspace(0, Ny - 1, num=max_ylabels, dtype=int).tolist()))
                vals = [delays_centers[i] for i in idx]
                ax.set_yticks(vals)
                ax.set_yticklabels([f"{v:.0f}" for v in vals])

            set_y_ticks(axI, delays_Ir)
            set_y_ticks(axQ, delays_Qr)
            set_y_ticks(axA, delays_Ar)
            set_y_ticks(axC, delays_Cr)

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

            outfile = (save_path + f"t1_heatmap_quad_q{self.qubit}_round{r_id}.png")
            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved 4-panel heatmaps for round {r_id} to: {outfile}")

            # ---------- NEW: Per-gain T1 curves (delay vs calibrated population) ----------
            # CC is shape (Ny_delays, Nx_gains); delays_Cr length Ny_delays; gains_Cr length Nx_gains
            if CC.size == 0 or len(delays_Cr) == 0 or len(gains_Cr) == 0:
                print(f"Round {r_id}: no calibrated grid for T1 slices; skipping.")
                continue

            delays_arr = np.asarray(delays_Cr, dtype=float)

            for ix, gain_val in enumerate(gains_Cr):
                y_curve = CC[:, ix]  # population vs delay
                # Mask NaNs and sort by delay just in case
                mask = np.isfinite(delays_arr) & np.isfinite(y_curve)
                if not np.any(mask):
                    continue
                x_plot = delays_arr[mask]
                y_plot = y_curve[mask]
                order = np.argsort(x_plot)
                x_plot = x_plot[order]
                y_plot = y_plot[order]

                # Single-axes plot (no specific colors/styles)
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.plot(x_plot, y_plot, marker='o', linestyle='-')
                ax2.set_xlabel("Delay time")
                ax2.set_ylabel("Qubit Population")
                ax2.set_title(f"Qubit {self.qubit + 1} — Round {r_id} — Gain {gain_val:g}")

                # Optional: show a light grid without specifying colors
                ax2.grid(True, which='both', linestyle='--', linewidth=0.6, alpha=0.5)

                fig2.tight_layout()

                slice_name = os.path.join(
                    t1slice_dir,
                    f"t1curve_q{self.qubit}_round{r_id}_gain{gain_val:g}.png"
                )
                fig2.savefig(slice_name, transparent=False, dpi=self.final_figure_quality)
                plt.close(fig2)
                print(f"Saved T1 slice: {slice_name}")

    def plot_t1_vs_delay_per_gain(self, amps, gains, rounds, delay_times, save_path):
        """
        For each unique gain:
          - x-axis: delay time
          - y-axis: signal amplitude
          - one line per round (legend)
        Saves each figure to a subfolder 't1_vs_delay_by_gain' under save_path.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict

        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        if not delay_q or not gains_q or not amps_q or not rounds_q:
            print(f"No data for qubit {q}. Skipping.")
            return

        # Helper: robustly fetch a round label at (i, j)
        def get_round_label(i, j):
            if i < len(rounds_q):
                row = rounds_q[i]
                if isinstance(row, (list, tuple)):
                    if len(row) == 0:
                        return "0"
                    if j < len(row):
                        return str(row[j])
                    return str(row[-1])
                return str(row)
            return "0"

        # Flatten to points: (round_id, gain, delay, amp)
        points = []
        for i in range(len(delay_q)):
            dval = float(delay_q[i])
            row_g = gains_q[i] if i < len(gains_q) else []
            row_a = amps_q[i] if i < len(amps_q) else []
            n = min(len(row_g), len(row_a))
            for j in range(n):
                try:
                    g = float(row_g[j])
                    a = float(row_a[j])
                except Exception:
                    continue
                r_id = get_round_label(i, j)
                points.append((r_id, g, dval, a))

        if not points:
            print(f"No numeric points for qubit {q}. Skipping.")
            return

        # Unique gains and rounds present
        unique_gains = sorted({g for (_, g, __, ___) in points})
        unique_rounds = sorted({r for (r, _, __, ___) in points})

        # Prepare subfolder
        subfolder = os.path.join(save_path, "t1_vs_delay_by_gain")
        self.create_folder_if_not_exists(subfolder)

        # For each gain, gather data per round and plot
        for g_sel in unique_gains:
            # bucket[(round_id, delay)] -> list of amplitudes (average duplicates)
            bucket = defaultdict(list)
            for (r, g, d, a) in points:
                if g == g_sel:  # exact match to this gain value as it appears in data
                    bucket[(r, d)].append(a)

            # If nothing matched (shouldn't happen), skip
            if not bucket:
                continue

            fig, ax = plt.subplots(figsize=(6.5, 4.5))

            # Build a line for each round
            plotted_any = False
            for r_id in unique_rounds:
                # Collect (delay, mean_amp) pairs for this round
                delays = []
                amps_mean = []
                for (r, d), vals in bucket.items():
                    if r == r_id:
                        delays.append(d)
                        amps_mean.append(float(np.nanmean(vals)))
                if not delays:
                    continue
                # Sort by delay
                order = np.argsort(delays)
                x = np.asarray(delays)[order]
                y = np.asarray(amps_mean)[order]

                # Some datasets may have NaNs—mask them out for plotting
                mask = np.isfinite(x) & np.isfinite(y)
                if np.any(mask):
                    ax.plot(x[mask], y[mask], marker='o', linewidth=1.2, markersize=3, label=f"Round {r_id}")
                    plotted_any = True

            if not plotted_any:
                plt.close(fig)
                continue

            ax.set_title(f"Qubit {self.qubit} — Gain {g_sel:g}")
            ax.set_xlabel("Delay time")
            ax.set_ylabel("Qubit Population")
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize="small"
            )
            fig.tight_layout()

            outfile = os.path.join(subfolder, f"t1_line_q{self.qubit}_slice{self.t1_slice}_gain{g_sel:g}.png")
            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved T1 vs delay plot for gain {g_sel:g} to: {outfile}")
    def plot_t1_vs_delay_per_gain_vs_base_t1(self, amps, gains, rounds, delay_times,t1_delay_times,t1_amps, save_path):
        """
        For each unique gain:
          - x-axis: delay time
          - y-axis: signal amplitude
          - one line per round (legend)
        Saves each figure to a subfolder 't1_vs_delay_by_gain' under save_path.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict

        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        if not delay_q or not gains_q or not amps_q or not rounds_q:
            print(f"No data for qubit {q}. Skipping.")
            return

        # Helper: robustly fetch a round label at (i, j)
        def get_round_label(i, j):
            if i < len(rounds_q):
                row = rounds_q[i]
                if isinstance(row, (list, tuple)):
                    if len(row) == 0:
                        return "0"
                    if j < len(row):
                        return str(row[j])
                    return str(row[-1])
                return str(row)
            return "0"

        # Flatten to points: (round_id, gain, delay, amp)
        points = []
        for i in range(len(delay_q)):
            dval = float(delay_q[i])
            row_g = gains_q[i] if i < len(gains_q) else []
            row_a = amps_q[i] if i < len(amps_q) else []
            n = min(len(row_g), len(row_a))
            for j in range(n):
                try:
                    g = float(row_g[j])
                    a = float(row_a[j])
                except Exception:
                    continue
                r_id = get_round_label(i, j)
                points.append((r_id, g, dval, a))

        if not points:
            print(f"No numeric points for qubit {q}. Skipping.")
            return

        # Unique gains and rounds present
        unique_gains = sorted({g for (_, g, __, ___) in points})
        unique_rounds = sorted({r for (r, _, __, ___) in points})

        # Prepare subfolder
        subfolder = os.path.join(save_path, "t1_vs_delay_by_gain_with_base_t1")
        self.create_folder_if_not_exists(subfolder)

        # For each gain, gather data per round and plot
        for g_sel in unique_gains:
            # bucket[(round_id, delay)] -> list of amplitudes (average duplicates)
            bucket = defaultdict(list)
            for (r, g, d, a) in points:
                if g == g_sel:  # exact match to this gain value as it appears in data
                    bucket[(r, d)].append(a)

            # If nothing matched (shouldn't happen), skip
            if not bucket:
                continue

            fig, ax = plt.subplots(figsize=(6.5, 4.5))

            # Build a line for each round
            plotted_any = False
            for r_id in unique_rounds:
                # Collect (delay, mean_amp) pairs for this round
                delays = []
                amps_mean = []
                for (r, d), vals in bucket.items():
                    if r == r_id:
                        delays.append(d)
                        amps_mean.append(float(np.nanmean(vals)))
                if not delays:
                    continue
                # Sort by delay
                order = np.argsort(delays)
                x = np.asarray(delays)[order]
                y = np.asarray(amps_mean)[order]

                # Some datasets may have NaNs—mask them out for plotting
                mask = np.isfinite(x) & np.isfinite(y)
                if np.any(mask):
                    ax.scatter(x[mask], y[mask],  label=f"Round {r_id}")
                    plotted_any = True

            if not plotted_any:
                plt.close(fig)
                continue
            ax.scatter(t1_delay_times, t1_amps[q][0], label=f"Regular T1 no Zeno")
            ax.set_title(f"Qubit {self.qubit+1} — Gain {g_sel:g}")
            ax.set_xlabel("Delay time")
            ax.set_ylabel("Qubit Population")
            ax.legend(
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),
                frameon=False,
                fontsize="small"
            )
            fig.tight_layout()

            outfile = os.path.join(subfolder, f"t1_line_q{self.qubit}_slice{self.t1_slice}_gain{g_sel:g}.png")
            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved T1 vs delay plot for gain {g_sel:g} to: {outfile}")

    def plot_without_errs(self, date_times, t1_vals, show_legends):
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

        #----------------To Plot a specific timeframe------------------
        from datetime import datetime
        year = 2025
        month = 1
        day1 = 22  # Start date
        day2 = 23  # End date
        hour_start = 0  # Start hour
        hour_end = 23  # End hour
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 59)
        #-----------------------------------------------------------------

        font = 14
        titles = [f"Qubit {i+1}" for i in range(self.number_of_qubits)]
        colors = ['orange','blue','purple','green','brown','pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.title('T1 Values vs Time',fontsize = font)
        axes = axes.flatten()

        from datetime import datetime
        for i, ax in enumerate(axes):

            if i >= self.number_of_qubits: # If we have fewer qubits than subplots, stop plotting and hide the rest
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize = font)

            x = date_times[i]
            y = t1_vals[i]

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
            ax.set_xlim(start_time, end_time)

            sorted_x = np.asarray(sorted(x))
            num_points = 5
            indices = np.linspace(0, len(sorted_x) - 1, num_points, dtype=int)

            ax.xaxis.set_major_locator(mdates.AutoDateLocator())  # Automatically choose good tick locations
            # ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))  # Format as month-day
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))  # Show day and time
            ax.tick_params(axis='x', rotation=45)  # Rotate ticks for better readability

            # Disable scientific notation and format y-ticks
            ax.ticklabel_format(style="plain", axis="y")
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.0f}"))  #decimal places


            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time', fontsize=font-2)
            ax.set_ylabel('T1 (us)', fontsize=font-2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'T1_vals.pdf', transparent=True, dpi=self.final_figure_quality)
        print('Plot saved to: ', analysis_folder)
        plt.close()

    def plot_with_errs(self, date_times, t1_vals, t1_fit_err, show_legends,exp_extension=''):
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
        day1 = 22  # Start date
        day2 = 23  # End date
        hour_start = 0  # Start hour
        hour_end = 23  # End hour
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 59)
        # -----------------------------------------------------------------

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        ext = exp_extension.replace('_', '')
        plt.suptitle(f'T1 Values vs Time {ext}', fontsize=font)
        axes = axes.flatten()

        import matplotlib.dates as mdates
        from matplotlib.ticker import StrMethodFormatter

        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            x = date_times[i]
            y = t1_vals[i]
            err = t1_fit_err[i]

            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            combined = list(zip(datetime_objects, y, err))
            combined.sort(key=lambda tup: tup[0])
            if len(combined) == 0:
                ax.set_visible(False)
                continue
            sorted_x, sorted_y, sorted_err = zip(*combined)
            sorted_x = np.array(sorted_x)

            #ax.set_xlim(start_time, end_time)

            ax.errorbar(
                sorted_x, sorted_y, yerr=sorted_err,
                fmt='none',
                ecolor=colors[i],
                elinewidth=1,
                capsize=0
            )

            ax.scatter(
                sorted_x, sorted_y,
                s=10,
                color=colors[i],
                alpha=0.5
            )

            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            ax.tick_params(axis='x', rotation=45)

            ax.ticklabel_format(style="plain", axis="y")

            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time', fontsize=font - 2)
            ax.set_ylabel('T1 (us)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + f'T1_vals{exp_extension}.pdf', transparent=True, dpi=self.final_figure_quality)
        print('Plot saved to:', analysis_folder)
        plt.close()

    def plot_with_errs_single_plot(self, date_times, t1_vals, t1_fit_err, show_legends):
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
        day1 = 22
        day2 = 23
        hour_start = 0
        hour_end = 23
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 59)
        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('T1 Values vs Time', fontsize=font)
        import matplotlib.dates as mdates
        from matplotlib.ticker import StrMethodFormatter
        for i in range(self.number_of_qubits):
            x = date_times[i]
            y = t1_vals[i]
            err = t1_fit_err[i]
            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]
            combined = list(zip(datetime_objects, y, err))
            combined.sort(key=lambda tup: tup[0])
            if len(combined) == 0:
                continue
            sorted_x, sorted_y, sorted_err = zip(*combined)
            sorted_x = np.array(sorted_x)
            ax.errorbar(sorted_x, sorted_y, yerr=sorted_err, fmt='none', ecolor=colors[i], elinewidth=1, capsize=0,
                        label=titles[i] if show_legends else None)
            ax.scatter(sorted_x, sorted_y, s=10, color=colors[i], alpha=0.5)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
        ax.tick_params(axis='x', rotation=45)
        ax.ticklabel_format(style="plain", axis="y")
        if show_legends:
            ax.legend(edgecolor='black')
        ax.set_xlabel('Time', fontsize=font - 2)
        ax.set_ylabel('T1 (us)', fontsize=font - 2)
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.tight_layout()
        plt.savefig(analysis_folder + 'T1_vals_single_plot.pdf', transparent=True, dpi=self.final_figure_quality)
        print('Plot saved to:', analysis_folder)
        plt.close()

    def plot_allan_deviation(self, date_times, vals, show_legends, label="T1"):

        analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/allan_stats/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), sharex=False, sharey=False)
        fig.suptitle(f'Overlapping Allan Deviation of {label} Fluctuations', fontsize=font)
        axes = axes.flatten()

        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]

        # -----------------------------------------------------------------------
        # 2) For each qubit, sort data by timestamp, compute Oadev, and plot
        # -----------------------------------------------------------------------
        for i, ax in enumerate(axes):
            # Hide extra subplots if you have fewer than 6 qubits
            if i >= self.number_of_qubits:
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            # Extract this qubit's data
            datetime_strings = date_times[i]  # list of "YYYY-MM-DD HH:MM:SS"
            data = vals[i]

            # Convert to datetime objects
            dt_objs = [datetime.strptime(s, "%Y-%m-%d %H:%M:%S") for s in datetime_strings]

            # Sort (ascending) by time
            combined = list(zip(dt_objs, data))
            combined.sort(key=lambda x: x[0])  # sort by datetime
            sorted_times, sorted_vals = zip(*combined)

            # Convert times -> seconds since first measurement
            t0 = sorted_times[0]
            time_sec = np.array([(t - t0).total_seconds() for t in sorted_times])
            vals_array = np.array(sorted_vals, dtype=float)

            # If you only have a single point, skip
            if len(time_sec) <= 1:
                ax.text(0.5, 0.5, "Not enough points", ha='center', va='center', transform=ax.transAxes)
                continue

            # Approx. average sample rate for Oadev
            avg_dt = np.mean(np.diff(time_sec))
            if avg_dt <= 0:
                avg_dt = 1.0
            rate = 1.0 / avg_dt

            # Compute overlapping Allan deviation
            # Use 'freq' data_type since label is not a phase measure.
            # We'll auto-select tau points with taus='decade' or you could supply np.logspace(...).
            taus_out, ad, ade, ns = allantools.oadev(
                vals_array,
                rate=rate,
                data_type='freq',
                taus='decade'
            )

            # Plot on log axes to mimic a standard Allan plot
            ax.set_xscale('log')
            ax.set_yscale('log')

            ax.plot(taus_out, ad, marker='o', color=colors[i], label=f"Qubit {i + 1}")

            # Optional: plot error bars
            ax.errorbar(taus_out, ad, yerr=ade, fmt='o', color=colors[i])

            if show_legends:
                ax.legend(loc='best', edgecolor='black')

            ax.set_xlabel(r"$\tau$ (s)", fontsize=font - 2)
            ax.set_ylabel(rf"$\sigma_{{{label}}}(\tau)$ (µs)", fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + f'{label}_allan_deviation.pdf', transparent=True, dpi=self.final_figure_quality)
        plt.close(fig)
