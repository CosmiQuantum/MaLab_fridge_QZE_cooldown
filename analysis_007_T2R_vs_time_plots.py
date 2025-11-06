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
from scipy.stats import norm
from scipy.optimize import curve_fit

class T2rVsTime:
    def __init__(self, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name,fridge='', exp_name = 'ge', qubit=0):
        self.save_figs = save_figs
        self.exp_name = exp_name
        self.qubit=qubit
        self.fridge = fridge
        self.fit_saved = fit_saved
        self.signal = signal
        self.figure_quality = figure_quality
        self.run_name = run_name
        self.number_of_qubits = number_of_qubits
        self.final_figure_quality = final_figure_quality
        self.top_folder_dates = top_folder_dates

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
        # Check if the data is a byte string; decode if necessary.
        if isinstance(data, bytes):
            data_str = data.decode()
        elif isinstance(data, str):
            data_str = data
        else:
            raise ValueError("Unsupported data type. Data should be bytes or string.")

        # Remove extra whitespace and non-numeric characters.
        cleaned_data = ''.join(c for c in data_str if c.isdigit() or c in ['-', '.', ' ', 'e'])

        # Split into individual numbers, removing empty strings.
        numbers = [float(x) for x in cleaned_data.split() if x]
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
    def run_t2_sweep(self, exp_extension='', scaling=False,return_calibration_data=False):
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
                outerFolder_expt = outerFolder + f"/Data_h5/T2{exp_extension}_zeno/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/T2_ge_zeno/"
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
                load_data = H5_class_instance.load_from_h5(data_type=f'T2{exp_extension}_zeno', save_r=int(save_round), scaling=scaling)
                # H5_class_instance.print_h5_contents(h5_file)
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data[f'T2{exp_extension}_zeno']:
                    for dataset in range(len(load_data[f'T2{exp_extension}_zeno'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'T2{exp_extension}_zeno'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T1 = load_data['T1'][q_key].get('T1', [])[0][dataset]
                        # errors = load_data['T1'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data[f'T2{exp_extension}_zeno'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue
                        delays = self.process_h5_data(
                            load_data[f'T2{exp_extension}_zeno'][q_key].get('Delay Times', [])[0][dataset].decode())
                        # try:
                        #     I = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('I', [])[0][dataset].decode())
                        #     Q = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('Q', [])[0][dataset].decode())
                        #     if scaling:
                        #
                        #         Ie = self.process_h5_data(
                        #             load_data[f'T1{exp_extension}'][q_key].get('ss_I_e', [])[0][dataset].decode())
                        #
                        #         Ig = self.process_h5_data(
                        #             load_data[f'T1{exp_extension}'][q_key].get('ss_I_g', [])[0][dataset].decode())
                        #         Qe = self.process_h5_data(
                        #             load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                        #         Qg = self.process_h5_data(
                        #             load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_Q_g', [])[0][dataset].decode())
                        # except:
                        I = self.process_h5_data(load_data[f'T2{exp_extension}_zeno'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data[f'T2{exp_extension}_zeno'][q_key].get('Q', [])[0][dataset].decode())

                        if scaling:
                            Ie = self.process_h5_data(load_data[f'T2{exp_extension}_zeno'][q_key].get('ss_I_e', [])[0][dataset].decode())
                            Ig = self.process_h5_data(load_data[f'T2{exp_extension}_zeno'][q_key].get('ss_I_g', [])[0][dataset].decode())
                            Qe = self.process_h5_data(load_data[f'T2{exp_extension}_zeno'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                            Qg = self.process_h5_data(load_data[f'T2{exp_extension}_zeno'][q_key].get('ss_Q_g', [])[0][dataset].decode())
                        round_num = load_data[f'T2{exp_extension}_zeno'][q_key].get('Round Num', [])[0][dataset]
                        try:
                            batch_num = load_data[f'T2{exp_extension}_zeno'][q_key].get('Batch Num', [])[0][dataset]
                            syst_config = load_data[f'T2{exp_extension}_zeno'][q_key].get('Syst Config', [])[0][dataset].decode()
                            exp_config = load_data[f'T2{exp_extension}_zeno'][q_key].get('Exp Config', [])[0][dataset].decode()
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
            return Is,Qs,amps, gains, rounds_completed, delay_times, Ig_calibration, Ie_calibration, Qe_calibration, Qg_calibration,steps
        else:
            return Is, Qs, amps, gains, rounds_completed, delay_times
    def robust_center(self,z, c=100.5, iters=100, eps=1e-12):
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
    def run_t2_sweep_new(self, exp_extension='', scaling=False, return_calibration_data=False):
        import datetime
        import glob, os, re
        import numpy as np

        # ----------Load/get data------------------------
        Is = {i: [] for i in range(self.number_of_qubits)}
        Qs = {i: [] for i in range(self.number_of_qubits)}
        Ig_calibration = {i: [] for i in range(self.number_of_qubits)}
        Ie_calibration = {i: [] for i in range(self.number_of_qubits)}
        Qg_calibration = {i: [] for i in range(self.number_of_qubits)}
        Qe_calibration = {i: [] for i in range(self.number_of_qubits)}
        amps = {i: [] for i in range(self.number_of_qubits)}  # averaged amplitude list per dataset
        gains = {i: [] for i in range(self.number_of_qubits)}
        rounds_completed = {i: [] for i in range(self.number_of_qubits)}
        reps = []
        steps = 0
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        delay_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}

        # NEW: element-wise averages of I and Q across sublists (per dataset)
        I_avgs = {i: [] for i in range(self.number_of_qubits)}
        Q_avgs = {i: [] for i in range(self.number_of_qubits)}

        def _is_list_of_lists(x):
            return isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple))

        def _avg_over_sublists(list_of_lists):
            """Element-wise average over a list of equal-length sublists."""
            arr = np.array(list_of_lists, dtype=float)  # (n_sublists, n_points)
            return np.mean(arr, axis=0)  # (n_points,)

        # print(self.top_folder_dates)
        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                outerFolder = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date + "/study_data"
                outerFolder_save_plots = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date + "_plots/"
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # ------------------------------------------------Load/Plot/Save T2----------------------------------------------
            if '_' in exp_extension:
                outerFolder_expt = outerFolder + f"/Data_h5/T2{exp_extension}_zeno/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/T2_ge_zeno/"
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
                load_data = H5_class_instance.load_from_h5(data_type=f'T2{exp_extension}_zeno',
                                                           save_r=int(save_round), scaling=scaling)
                # H5_class_instance.print_h5_contents(h5_file)
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data[f'T2{exp_extension}_zeno']:
                    for dataset in range(len(load_data[f'T2{exp_extension}_zeno'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'T2{exp_extension}_zeno'][q_key].get('Dates', [])[0][dataset]):
                            continue

                        date = datetime.datetime.fromtimestamp(
                            load_data[f'T2{exp_extension}_zeno'][q_key].get('Dates', [])[0][dataset])

                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        delays = self.process_h5_data(
                            load_data[f'T2{exp_extension}_zeno'][q_key].get('Delay Times', [])[0][dataset].decode())

                        I = self.process_string_of_nested_lists(
                            load_data[f'T2{exp_extension}_zeno'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_string_of_nested_lists(
                            load_data[f'T2{exp_extension}_zeno'][q_key].get('Q', [])[0][dataset].decode())

                        if scaling:
                            Ie = self.process_string_of_nested_lists(
                                load_data[f'T2{exp_extension}_zeno'][q_key].get('ss_I_e', [])[0][dataset].decode())
                            Ig = self.process_string_of_nested_lists(
                                load_data[f'T2{exp_extension}_zeno'][q_key].get('ss_I_g', [])[0][dataset].decode())
                            Qe = self.process_string_of_nested_lists(
                                load_data[f'T2{exp_extension}_zeno'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                            Qg = self.process_string_of_nested_lists(
                                load_data[f'T2{exp_extension}_zeno'][q_key].get('ss_Q_g', [])[0][dataset].decode())

                        round_num = load_data[f'T2{exp_extension}_zeno'][q_key].get('Round Num', [])[0][dataset]
                        try:
                            batch_num = load_data[f'T2{exp_extension}_zeno'][q_key].get('Batch Num', [])[0][dataset]
                            syst_config = load_data[f'T2{exp_extension}_zeno'][q_key].get('Syst Config', [])[0][
                                dataset].decode()
                            exp_config = load_data[f'T2{exp_extension}_zeno'][q_key].get('Exp Config', [])[0][
                                dataset].decode()
                        except:
                            exp_config = None

                        if len(I) > 0:
                            # keep raw nested I/Q for compatibility
                            Is[q_key].append(I)
                            Qs[q_key].append(Q)

                            # parse meta when available
                            if exp_config is not None:
                                try:
                                    steps = round(float(exp_config.split("Readout_Optimization': ")[-1]
                                                        .split("steps': ")[-1].split(',')[0]), 6)
                                except Exception:
                                    pass
                            try:
                                gain = round(float(syst_config.split("res_gain_qze': ")[-1].split(',')[0]), 6)
                                gains[q_key].append(gain)
                            except Exception:
                                pass

                            rounds_completed[q_key].append(round_we_are_on)

                            # ---- NEW core logic: list-of-lists handling + per-index calibration ----
                            if scaling:
                                I_nested = I if _is_list_of_lists(I) else [I]
                                Q_nested = Q if _is_list_of_lists(Q) else [Q]
                                Ie_nested = Ie if _is_list_of_lists(Ie) else [Ie]
                                Ig_nested = Ig if _is_list_of_lists(Ig) else [Ig]
                                Qe_nested = Qe if _is_list_of_lists(Qe) else [Qe]
                                Qg_nested = Qg if _is_list_of_lists(Qg) else [Qg]

                                calibrated_sublists = []
                                I_sublists, Q_sublists = [], []

                                for sub_I, sub_Q, sub_Ie, sub_Ig, sub_Qe, sub_Qg in zip(
                                        I_nested, Q_nested, Ie_nested, Ig_nested, Qe_nested, Qg_nested
                                ):
                                    sub_I = np.asarray(sub_I, dtype=float)
                                    sub_Q = np.asarray(sub_Q, dtype=float)
                                    sub_Ie = np.asarray(sub_Ie, dtype=float)
                                    sub_Qe = np.asarray(sub_Qe, dtype=float)
                                    sub_Ig = np.asarray(sub_Ig, dtype=float)
                                    sub_Qg = np.asarray(sub_Qg, dtype=float)

                                    e = self.robust_center(sub_Ie + 1j * sub_Qe)
                                    g = self.robust_center(sub_Ig + 1j * sub_Qg)

                                    pop_norm = np.abs(((sub_I + 1j * sub_Q) - g) * (e - g) / (np.abs(e - g) ** 2))
                                    calibrated_sublists.append(pop_norm.tolist())
                                    I_sublists.append(sub_I.tolist())
                                    Q_sublists.append(sub_Q.tolist())

                                amp_avg = _avg_over_sublists(calibrated_sublists)
                                I_avg = _avg_over_sublists(I_sublists)
                                Q_avg = _avg_over_sublists(Q_sublists)

                                amps[q_key].append(amp_avg.tolist())
                                I_avgs[q_key].append(I_avg.tolist())
                                Q_avgs[q_key].append(Q_avg.tolist())

                                # keep the exact calibration data we used (nested)
                                Ig_calibration[q_key].append(Ig_nested)
                                Ie_calibration[q_key].append(Ie_nested)
                                Qg_calibration[q_key].append(Qg_nested)
                                Qe_calibration[q_key].append(Qe_nested)

                            else:
                                I_nested = I if _is_list_of_lists(I) else [I]
                                Q_nested = Q if _is_list_of_lists(Q) else [Q]

                                amp_sublists = []
                                I_sublists, Q_sublists = [], []
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
            # original order preserved; new I/Q averages appended for convenience
            return (I_avgs, Q_avgs, amps, gains, rounds_completed, delay_times,
                    Ig_calibration, Ie_calibration, Qe_calibration, Qg_calibration, steps)
        else:
            # original order preserved; new I/Q averages appended
            return I_avgs, Q_avgs, amps, gains, rounds_completed, delay_times

    def run_t2_sweep_no_zeno(self, exp_extension='', scaling=False,return_calibration_data=False):
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
                outerFolder_expt = outerFolder + f"/Data_h5/T2/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/T2/"
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
                load_data = H5_class_instance.load_from_h5(data_type=f'T2', save_r=int(save_round), scaling=scaling)
                # H5_class_instance.print_h5_contents(h5_file)
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data[f'T2']:
                    for dataset in range(len(load_data[f'T2'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'T2'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T1 = load_data['T1'][q_key].get('T1', [])[0][dataset]
                        # errors = load_data['T1'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data[f'T2'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue
                        delays = self.process_h5_data(
                            load_data[f'T2'][q_key].get('Delay Times', [])[0][dataset].decode())
                        # try:
                        #     I = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('I', [])[0][dataset].decode())
                        #     Q = self.process_h5_data(load_data[f'T1{exp_extension}'][q_key].get('Q', [])[0][dataset].decode())
                        #     if scaling:
                        #
                        #         Ie = self.process_h5_data(
                        #             load_data[f'T1{exp_extension}'][q_key].get('ss_I_e', [])[0][dataset].decode())
                        #
                        #         Ig = self.process_h5_data(
                        #             load_data[f'T1{exp_extension}'][q_key].get('ss_I_g', [])[0][dataset].decode())
                        #         Qe = self.process_h5_data(
                        #             load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                        #         Qg = self.process_h5_data(
                        #             load_data[f'T1{exp_extension}_zeno'][q_key].get('ss_Q_g', [])[0][dataset].decode())
                        # except:
                        I = self.process_h5_data(load_data[f'T2'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data[f'T2'][q_key].get('Q', [])[0][dataset].decode())

                        if scaling:
                            Ie = self.process_h5_data(load_data[f'T2'][q_key].get('ss_I_e', [])[0][dataset].decode())
                            Ig = self.process_h5_data(load_data[f'T2'][q_key].get('ss_I_g', [])[0][dataset].decode())
                            Qe = self.process_h5_data(load_data[f'T2'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                            Qg = self.process_h5_data(load_data[f'T2'][q_key].get('ss_Q_g', [])[0][dataset].decode())
                        round_num = load_data[f'T2'][q_key].get('Round Num', [])[0][dataset]
                        try:
                            batch_num = load_data[f'T2'][q_key].get('Batch Num', [])[0][dataset]
                            syst_config = load_data[f'T2'][q_key].get('Syst Config', [])[0][dataset].decode()
                            exp_config = load_data[f'T2'][q_key].get('Exp Config', [])[0][dataset].decode()
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

                            gain = 0

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
            return Is,Qs,amps, gains, rounds_completed, delay_times, Ig_calibration, Ie_calibration, Qe_calibration, Qg_calibration,steps
        else:
            return Is, Qs, amps, gains, rounds_completed, delay_times
    def run(self,return_errs=False):
        import datetime
        # ----------Load/get data------------------------
        t2_vals = {i: [] for i in range(self.number_of_qubits)}
        t2_errs = {i: [] for i in range(self.number_of_qubits)}
        rounds = []
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}

        for folder_date in self.top_folder_dates:
            outerFolder = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date+ "/study_data" + "/"
            outerFolder_save_plots = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date+ "/study_data" + "_plots/"

            # -------------------------------------------------------Load/Plot/Save T2------------------------------------------
            outerFolder_expt = outerFolder + "/Data_h5/T2_ge/"
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

            for h5_file in h5_files:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)

                # sometimes you get '1(1)' when redownloading the h5 files for some reason
                load_data = H5_class_instance.load_from_h5(data_type='T2', save_r=int(save_round.split('(')[0]))

                for q_key in load_data['T2']:
                    for dataset in range(len(load_data['T2'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data['T2'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T2 = load_data['T2'][q_key].get('T2', [])[0][dataset]
                        # errors = load_data['T2'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data['T2'][q_key].get('Dates', [])[0][dataset])
                        I = self.process_h5_data(load_data['T2'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data['T2'][q_key].get('Q', [])[0][dataset].decode())
                        delay_times = self.process_h5_data(load_data['T2'][q_key].get('Delay Times', [])[0][dataset].decode())
                        # fit = load_data['T2'][q_key].get('Fit', [])[0][dataset]
                        round_num = load_data['T2'][q_key].get('Round Num', [])[0][dataset]
                        batch_num = load_data['T2'][q_key].get('Batch Num', [])[0][dataset]
                        try:
                            exp_config = load_data['T2'][q_key].get('Exp Config', [])[0][dataset].decode()
                            safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                            exp_config = eval(exp_config, safe_globals)
                        except:
                            exp_config =None

                        if len(I) > 0:
                            T2_class_instance = T2RMeasurement(q_key,self.number_of_qubits, outerFolder_save_plots, round_num, self.signal,
                                                               self.save_figs, fit_data=True)
                            try:
                                fitted, t2r_est, t2r_err, plot_sig = T2_class_instance.t2_fit(delay_times, I, Q)
                            except:
                                continue
                            #T2_cfg = exp_config['Ramsey_ge']
                            if t2r_est < 0:
                                print("The value is negative, continuing...")
                                continue
                            if t2r_est > 300:
                                print("The value is above 300 us, this is a bad fit, continuing...")
                                continue
                            if t2r_err >= 0.8 * t2r_est:
                                print(
                                    f"Skipping T2R = {t2r_est:.3f} µs because its error {t2r_err:.3f} µs is >= 80% of its value.")
                                continue
                            t2_vals[q_key].extend([t2r_est])
                            t2_errs[q_key].extend([t2r_err])
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                            del T2_class_instance

                del H5_class_instance
        if return_errs:
            return date_times, t2_vals, t2_errs
        else:
            return date_times, t2_vals

    def plot_all_t2_heatmaps_new_format(self, amps, gains, rounds, delay_times, save_path, max_ylabels=6):
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
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

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
            outfile = (save_path + f"t2_heatmap_q{self.qubit}_round{r_id}.png")
            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved heatmap for round {r_id} to: {outfile}")

    def plot_all_t2_heatmaps_with_singular_ssf_plotting(
            self,
            amps,
            gains,
            rounds,
            delay_times,
            save_path,
            max_ylabels=6,
            # --- optional SSD inputs (same contract as T1 version) ---
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

        T2 heatmaps with:
          - fixed global color scale across rounds (by global min/max amplitude)
          - Y-axis showing at most `max_ylabels` delay tick labels (evenly spaced)
          - OPTIONAL: emit one SSF plot per dataset using provided calibrations,
            with a single (I,Q) experiment point overlaid if available.

        Data model (per qubit q), identical to your T1 function:
          - amps[q]         : list of lists/arrays; amps[q][i] are amplitude samples for dataset i
          - gains[q]        : list; gains[q][i] is scalar or array aligned with amps[q][i]
          - rounds[q]       : list; rounds[q][i] is the round label for dataset i
          - delay_times[q]  : list; delay_times[q][i] scalar or array aligned with amps[q][i]
          - I_experiment[q] : list; I_experiment[q][i] is scalar or array of I-shots for dataset i
          - Q_experiment[q] : list; Q_experiment[q][i] is scalar or array of Q-shots for dataset i

        If SSD is enabled (provide ss_class_instance, ss_cfg and all four calibration dicts),
        then for EACH dataset i we will call:
            ss_class_instance.hist_ssf_with_annotations(
                data=[I_g, Q_g, I_e, Q_e],
                cfg=ss_cfg,
                plot=True,
                I_meas=I_single,   # <- from I_experiment[q][i], if finite
                Q_meas=Q_single    # <- from Q_experiment[q][i], if finite
            )
            ss_class_instance.hist_ssf_with_annotations_new_method(...)
        and direct its output folder to: <save_path>/analysis/ (class handles naming);
        we also stamp qubit/round/gain/delay into the class context and name with *_t2_ssd.
        """
        import os
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        import matplotlib.ticker as mticker

        # --- helpers (mirrors your T1 version) ---
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

        # --- extract this qubit's lists ---
        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        # experiment I/Q per dataset (optional)
        Iexp_q = (I_experiment or {}).get(q, [])
        Qexp_q = (Q_experiment or {}).get(q, [])

        # presence & length checks
        n = min(len(amps_q), len(gains_q), len(rounds_q), len(delay_q))
        if n == 0 or not (len(amps_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
            print(f"No usable data for qubit {q} (missing lists or length mismatch). Skipping.")
            return

        # --- OPTIONAL: pull calibration lists for SSD if provided ---
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

        # --- collect points for heatmaps: (round_id, gain, delay, amp) ---
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
            return

        unique_rounds = sorted({r for (r, _, __, ___) in all_points})

        # ensure save root exists
        self.create_folder_if_not_exists(save_path)

        # ---------- global z scale ----------
        all_amps = np.array([a for (_, _, _, a) in all_points], dtype=float)
        global_vmin = float(np.nanmin(all_amps))
        global_vmax = float(np.nanmax(all_amps))

        # ---------- emit SSD plots per dataset (once per T2 curve) ----------
        if do_ssd:
            analysis_root = os.path.join(save_path, "analysis")
            self.create_folder_if_not_exists(analysis_root)

            # temporarily adjust ss context so its own saver drops files here
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
                    ss_class_instance.expt_name = f"{gain_tag}{delay_tag}_t2_ssd"

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

                    # pick ONE single-shot (I,Q) for this dataset from experiment dicts
                    I_single = np.nan
                    Q_single = np.nan
                    if i < len(Iexp_q):
                        I_single = pick_single_shot(Iexp_q[i])
                    if i < len(Qexp_q):
                        Q_single = pick_single_shot(Qexp_q[i])

                    kwargs_meas = {}
                    if np.isfinite(I_single) and np.isfinite(Q_single):
                        kwargs_meas = {"I_meas": I_single, "Q_meas": Q_single}

                    # call your SSD routines (they handle plotting/saving internally)
                    try:
                        ss_class_instance.hist_ssf_with_annotations(
                            data=[I_g, Q_g, I_e, Q_e],
                            cfg=ss_cfg,
                            plot=True, path_ext='_t2r',
                            **kwargs_meas
                        )
                    except Exception as e:
                        print(f"[SSD] Failed on dataset {i} (round {r_id}): {e}")

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

        # ---------- per-round heatmaps (fixed z across rounds) ----------
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
            outfile = (save_path + f"t2_heatmap_q{self.qubit}_round{r_id}.png")
            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved heatmap for round {r_id} to: {outfile}")

    def plot_all_t2_heatmaps_single_calibration_IQ(self, Is, Qs, Ig_calibration1,
                                                Ie_calibration1, Qe_calibration1, Qg_calibration1, gains, rounds,
                                                delay_times, save_path, max_ylabels=6):
        """
        NEW FORMAT ONLY — now plots 4 heatmaps per round (I, Q, |IQ|, Calibrated)

        Changes vs previous:
          - Adds subplots for I, Q, and raw amplitude sqrt(I^2 + Q^2) alongside calibrated amplitude.
          - Keeps fixed color scales (per-metric) across rounds.
          - Y-axis shows at most `max_ylabels` delay_time tick labels (evenly spaced).

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

        q = self.qubit
        gains_q = gains.get(q, [])
        I_q = Is.get(q, [])
        Q_q = Qs.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        # --- Calibration (shared for all datasets) ---
        g = np.mean(Ig_calibration1 + 1j * Qg_calibration1)
        e = np.mean(Ie_calibration1 + 1j * Qe_calibration1)
        denom = np.abs(e - g) ** 2
        if denom <= 0 or not np.isfinite(denom):
            raise ValueError("Best calibration is degenerate (e ≈ g); cannot normalize.")

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

        # Ensure save folder exists
        self.create_folder_if_not_exists(save_path)

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
            # Build grids for each metric
            gi = grid_for_round(r_id, all_points_I)
            gq = grid_for_round(r_id, all_points_Q)
            ga = grid_for_round(r_id, all_points_abs)
            gc = grid_for_round(r_id, all_points_cal)
            if not all([gi, gq, ga, gc]):
                # If any metric is missing for this round, skip plotting it to avoid mismatched axes.
                print(f"Round {r_id}: incomplete metric grids; skipping.")
                continue

            # Unpack (they may have different bins; that's OK for pcolormesh)
            (xI, yI, CI, gains_I, delays_I) = gi
            (xQ, yQ, CQ, gains_Qr, delays_Qr) = gq
            (xA, yA, CA, gains_Ar, delays_Ar) = ga
            (xC, yC, CC, gains_Cr, delays_Cr) = gc

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

            # Axis labels
            for ax in (axI, axQ, axA, axC):
                ax.set_xlabel("Pulse gain (a.u.)")
                ax.set_ylabel("Delay time")
                ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

            # Limit number of Y tick labels (per-axes, using that axis' centers)
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

            set_y_ticks(axI, delays_I)
            set_y_ticks(axQ, delays_Qr)
            set_y_ticks(axA, delays_Ar)
            set_y_ticks(axC, delays_Cr)

            # Colorbars (one per subplot to preserve independent vmin/vmax)
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

            outfile = (save_path + f"t2_heatmap_quad_q{self.qubit}_round{r_id}.png")
            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved 4-panel heatmaps for round {r_id} to: {outfile}")

    def plot_all_t2_heatmaps_single_calibration(self, Is, Qs, Ig_calibration1, \
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
            outfile = (save_path + f"t2_heatmap_q{self.qubit}_round{r_id}.png")
            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved heatmap for round {r_id} to: {outfile}")
    def plot_all_t2_rounds_IQAmp(
            self,
            I_data,
            Q_data,
            gains,
            rounds,
            delay_times,
            save_path,
            max_ylabels=6,
            use_abs=False,  # False -> raw values (requested). True -> magnitude for I and Q only.
            center_zero=False
    ):
        """
        Three stacked heatmaps: top=I, middle=Q, bottom=amplitude=sqrt(I^2+Q^2).
        I and Q respect `use_abs`; amplitude is always computed from raw I and Q.
        Each panel uses its own color scale (unless center_zero=True).
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict

        q = self.qubit
        gains_q = gains.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])
        I_q = I_data.get(q, [])
        Q_q = Q_data.get(q, [])

        # Ensure all lists align
        n = min(len(I_q), len(Q_q), len(gains_q), len(rounds_q), len(delay_q))
        if n == 0 or not (len(I_q) == len(Q_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
            print(f"No usable data for qubit {q} (missing lists or length mismatch). Skipping.")
            return

        def _as_scalar(x):
            """Return float scalar or np.nan."""
            try:
                if isinstance(x, (list, tuple, np.ndarray)):
                    arr = np.asarray(x, dtype=float).ravel()
                    return float(np.nanmean(arr)) if arr.size > 0 else np.nan
                return float(x)
            except Exception:
                return np.nan

        def collect_points(signal_list, *, apply_abs=False):
            """Return: points=[(round_id, delay, value)], gains_seen, unique_rounds, unique_delays"""
            points = []
            gains_seen_local = []
            for i in range(n):
                r_id = str(rounds_q[i])

                s_samples = np.asarray(signal_list[i], dtype=float).ravel()
                if s_samples.size == 0:
                    continue
                s_vals = np.abs(s_samples) if apply_abs else s_samples

                # gain (collect for info only)
                g_scalar = _as_scalar(gains_q[i])
                if np.isfinite(g_scalar):
                    gains_seen_local.append(g_scalar)

                # delay alignment
                d_i = delay_q[i]
                if isinstance(d_i, (list, tuple, np.ndarray)):
                    d_arr = np.asarray(d_i, dtype=float).ravel()
                    if d_arr.size not in (1, s_samples.size):
                        if d_arr.size == 1:
                            d_arr = np.full_like(s_samples, float(d_arr[0]))
                        else:
                            continue
                    if d_arr.size == 1:
                        d_arr = np.full_like(s_samples, float(d_arr[0]))
                else:
                    d_scalar = _as_scalar(d_i)
                    if not np.isfinite(d_scalar):
                        continue
                    d_arr = np.full_like(s_samples, d_scalar)

                mask = np.isfinite(s_vals) & np.isfinite(d_arr)
                if not np.any(mask):
                    continue

                for d, s in zip(d_arr[mask], s_vals[mask]):
                    points.append((r_id, float(d), float(s)))

            if not points:
                return [], [], [], []

            # Order rounds in first-seen order, delays ascending
            seen_rounds = {}
            for (r, _, __) in points:
                if r not in seen_rounds:
                    seen_rounds[r] = None
            unique_rounds = list(seen_rounds.keys())
            unique_delays = sorted({d for (_, d, __) in points})
            return points, gains_seen_local, unique_rounds, unique_delays

        def collect_amp_points(I_list, Q_list):
            """Compute amplitude from RAW I and Q (no abs) and return points like collect_points."""
            points = []
            gains_seen_local = []
            for i in range(n):
                r_id = str(rounds_q[i])

                I_s = np.asarray(I_list[i], dtype=float).ravel()
                Q_s = np.asarray(Q_list[i], dtype=float).ravel()
                if I_s.size == 0 or Q_s.size == 0:
                    continue

                # length reconcile: require equal or broadcastable delays; if unequal sample lengths, skip this entry
                if I_s.size != Q_s.size:
                    # try simple cases: one is scalar -> broadcast
                    if I_s.size == 1:
                        I_s = np.full_like(Q_s, I_s[0], dtype=float)
                    elif Q_s.size == 1:
                        Q_s = np.full_like(I_s, Q_s[0], dtype=float)
                    else:
                        # cannot align per-sample I and Q -> skip this i
                        continue

                amp = np.sqrt(I_s ** 2 + Q_s ** 2)

                # gain (info only)
                g_scalar = _as_scalar(gains_q[i])
                if np.isfinite(g_scalar):
                    gains_seen_local.append(g_scalar)

                # delay alignment
                d_i = delay_q[i]
                if isinstance(d_i, (list, tuple, np.ndarray)):
                    d_arr = np.asarray(d_i, dtype=float).ravel()
                    if d_arr.size not in (1, amp.size):
                        if d_arr.size == 1:
                            d_arr = np.full_like(amp, float(d_arr[0]))
                        else:
                            continue
                    if d_arr.size == 1:
                        d_arr = np.full_like(amp, float(d_arr[0]))
                else:
                    d_scalar = _as_scalar(d_i)
                    if not np.isfinite(d_scalar):
                        continue
                    d_arr = np.full_like(amp, d_scalar)

                mask = np.isfinite(amp) & np.isfinite(d_arr)
                if not np.any(mask):
                    continue

                for d, a in zip(d_arr[mask], amp[mask]):
                    points.append((r_id, float(d), float(a)))

            if not points:
                return [], [], [], []

            seen_rounds = {}
            for (r, _, __) in points:
                if r not in seen_rounds:
                    seen_rounds[r] = None
            unique_rounds = list(seen_rounds.keys())
            unique_delays = sorted({d for (_, d, __) in points})
            return points, gains_seen_local, unique_rounds, unique_delays

        # Collect I, Q, and Amplitude
        I_points, I_gains, I_rounds, I_delays = collect_points(I_q, apply_abs=use_abs)
        Q_points, Q_gains, Q_rounds, Q_delays = collect_points(Q_q, apply_abs=use_abs)
        A_points, A_gains, A_rounds, A_delays = collect_amp_points(I_q, Q_q)

        if not I_points and not Q_points and not A_points:
            print(f"No numeric I, Q, or amplitude points for qubit {q}. Skipping.")
            return

        # Union axes so all panels share ticks
        unique_rounds = list(dict.fromkeys((I_rounds or []) + (Q_rounds or []) + (A_rounds or [])))  # preserve order
        unique_delays = sorted(set((I_delays or []) + (Q_delays or []) + (A_delays or [])))
        if not unique_rounds or not unique_delays:
            print(f"Insufficient axis values for qubit {q}. Skipping.")
            return

        def build_matrix(points, unique_rounds, unique_delays):
            ri_map = {r: i for i, r in enumerate(unique_rounds)}
            di_map = {d: i for i, d in enumerate(unique_delays)}
            bucket = defaultdict(list)
            for (r, d, val) in points:
                if r in ri_map and d in di_map:
                    bucket[(di_map[d], ri_map[r])].append(val)

            Ny, Nx = len(unique_delays), len(unique_rounds)
            C = np.full((Ny, Nx), np.nan, dtype=float)
            for (iy, ix), vals in bucket.items():
                C[iy, ix] = float(np.nanmean(vals))
            return C

        C_I = build_matrix(I_points, unique_rounds, unique_delays) if I_points else None
        C_Q = build_matrix(Q_points, unique_rounds, unique_delays) if Q_points else None
        C_A = build_matrix(A_points, unique_rounds, unique_delays) if A_points else None

        def centers_to_edges(centers):
            centers = np.asarray(sorted(np.unique(centers)), dtype=float)
            if centers.size == 1:
                d = 1.0
                return np.array([centers[0] - d / 2, centers[0] + d / 2])
            mids = (centers[:-1] + centers[1:]) / 2.0
            first = centers[0] - (centers[1] - centers[0]) / 2.0
            last = centers[-1] + (centers[-1] - centers[-2]) / 2.0
            return np.concatenate([[first], mids, [last]])

        Nx = len(unique_rounds)
        Ny = len(unique_delays)
        x_edges = np.arange(-0.5, Nx + 0.5, 1.0)
        y_edges = centers_to_edges(unique_delays)

        # Per-panel color limits
        def panel_limits(C, force_nonneg=False):
            if C is None or not np.isfinite(C).any():
                return 0.0, 1.0
            vmin = float(np.nanmin(C))
            vmax = float(np.nanmax(C))
            if center_zero and not force_nonneg:
                m = max(abs(vmin), abs(vmax))
                return -m, m
            return (max(0.0, vmin), vmax) if force_nonneg else (vmin, vmax)

        vmin_I, vmax_I = panel_limits(C_I)
        vmin_Q, vmax_Q = panel_limits(C_Q)
        # amplitude is always nonnegative
        vmin_A, vmax_A = panel_limits(C_A, force_nonneg=True)

        self.create_folder_if_not_exists(save_path)

        # --- Figure with three stacked panels ---
        fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(6.5, 11.0), sharex=True)

        panels = [
            (axes[0], C_I, "I ", vmin_I, vmax_I, "Signal value"),
            (axes[1], C_Q, "Q ", vmin_Q, vmax_Q, "Signal value"),
            (axes[2], C_A, "Amplitude (√(I²+Q²)) ", vmin_A, vmax_A, "Amplitude"),
        ]

        for ax, C, label, vmin, vmax, cbar_label in panels:
            ax.set_aspect('auto')
            if C is None:
                ax.text(0.5, 0.5, f"No {label.split()[0]} data", ha='center', va='center')
                continue
            mesh = ax.pcolormesh(x_edges, y_edges, C, shading='flat', vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label(cbar_label)
            ax.set_ylabel("Delay time")
            ax.set_title(f"Qubit {self.qubit + 1} — {label}")

            # Limit number of y tick labels
            if Ny > 0:
                if Ny <= max_ylabels:
                    yticks_idx = list(range(Ny))
                else:
                    yticks_idx = np.linspace(0, Ny - 1, num=max_ylabels, dtype=int).tolist()
                    yticks_idx = sorted(set(yticks_idx))
                yticks_vals = [unique_delays[i] for i in yticks_idx]
                ax.set_yticks(yticks_vals)
                ax.set_yticklabels([f"{v:.0f}" for v in yticks_vals])

        # x ticks on the bottom only
        axes[-1].set_xlabel("Round")
        max_xtick_labels = 10
        step = max(1, int(np.ceil(Nx / max_xtick_labels)))
        xticks = np.arange(0, Nx, step)
        axes[-1].set_xticks(xticks)
        axes[-1].set_xticklabels([unique_rounds[i] for i in xticks], rotation=45, ha='right')

        # FYI if multiple gains were used
        gains_unique = sorted({round(g, 12) for g in (I_gains + Q_gains + A_gains) if np.isfinite(g)})
        if len(gains_unique) > 1:
            print(f"Note: found multiple gains {gains_unique}, but plotting collapsed over gain (x = rounds).")

        fig.tight_layout()
        outfile = (save_path + f"t2_heatmap_IQAMP_raw_q{self.qubit}_by_round.png")
        fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
        plt.close(fig)
        print(f"Saved I/Q/Amplitude heatmaps to: {outfile}")
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
                #print(Ig_i)
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

    def plot_all_t2_rounds_IQ(
            self,
            I_data,
            Q_data,
            gains,
            rounds,
            delay_times,
            save_path,
            max_ylabels=6,
            use_abs=False,  # False -> raw values (requested). True -> magnitude.
            center_zero=False  # True -> vmin/vmax symmetric around 0 per panel.
    ):
        """
        Two stacked heatmaps: top=I, bottom=Q.
        Plots raw I and Q by default (no abs). Each panel uses its own color scale.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict

        q = self.qubit
        gains_q = gains.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])
        I_q = I_data.get(q, [])
        Q_q = Q_data.get(q, [])

        # Ensure all lists align
        n = min(len(I_q), len(Q_q), len(gains_q), len(rounds_q), len(delay_q))
        if n == 0 or not (len(I_q) == len(Q_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
            print(f"No usable data for qubit {q} (missing lists or length mismatch). Skipping.")
            return

        def collect_points(signal_list):
            """Return: points=[(round_id, delay, value)], gains_seen, unique_rounds, unique_delays"""
            points = []
            gains_seen_local = []

            for i in range(n):
                r_id = str(rounds_q[i])

                s_samples = np.asarray(signal_list[i], dtype=float).ravel()
                if s_samples.size == 0:
                    continue
                s_vals = np.abs(s_samples) if use_abs else s_samples

                # gain (collect for info only)
                g_i = gains_q[i]
                if isinstance(g_i, (list, tuple, np.ndarray)):
                    g_arr = np.asarray(g_i, dtype=float).ravel()
                    g_scalar = float(np.nanmean(g_arr)) if g_arr.size > 0 else np.nan
                else:
                    try:
                        g_scalar = float(g_i)
                    except Exception:
                        g_scalar = np.nan
                if np.isfinite(g_scalar):
                    gains_seen_local.append(g_scalar)

                # delay alignment
                d_i = delay_q[i]
                if isinstance(d_i, (list, tuple, np.ndarray)):
                    d_arr = np.asarray(d_i, dtype=float).ravel()
                    if d_arr.size not in (1, s_samples.size):
                        if d_arr.size == 1:
                            d_arr = np.full_like(s_samples, float(d_arr[0]))
                        else:
                            continue
                    if d_arr.size == 1:
                        d_arr = np.full_like(s_samples, float(d_arr[0]))
                else:
                    try:
                        d_scalar = float(d_i)
                    except Exception:
                        continue
                    d_arr = np.full_like(s_samples, d_scalar)

                mask = np.isfinite(s_vals) & np.isfinite(d_arr)
                if not np.any(mask):
                    continue

                for d, s in zip(d_arr[mask], s_vals[mask]):
                    points.append((r_id, float(d), float(s)))

            if not points:
                return [], [], [], []

            # Order rounds in first-seen order, delays ascending
            seen_rounds = {}
            for (r, _, __) in points:
                if r not in seen_rounds:
                    seen_rounds[r] = None
            unique_rounds = list(seen_rounds.keys())
            unique_delays = sorted({d for (_, d, __) in points})
            return points, gains_seen_local, unique_rounds, unique_delays

        # Collect I and Q separately
        I_points, I_gains, I_rounds, I_delays = collect_points(I_q)
        Q_points, Q_gains, Q_rounds, Q_delays = collect_points(Q_q)

        if not I_points and not Q_points:
            print(f"No numeric I or Q points for qubit {q}. Skipping.")
            return

        # Union axes so both panels share ticks
        unique_rounds = list(dict.fromkeys((I_rounds or []) + (Q_rounds or [])))  # preserve order
        unique_delays = sorted(set((I_delays or []) + (Q_delays or [])))
        if not unique_rounds or not unique_delays:
            print(f"Insufficient axis values for qubit {q}. Skipping.")
            return

        def build_matrix(points, unique_rounds, unique_delays):
            ri_map = {r: i for i, r in enumerate(unique_rounds)}
            di_map = {d: i for i, d in enumerate(unique_delays)}
            bucket = defaultdict(list)
            for (r, d, val) in points:
                if r in ri_map and d in di_map:
                    bucket[(di_map[d], ri_map[r])].append(val)

            Ny, Nx = len(unique_delays), len(unique_rounds)
            C = np.full((Ny, Nx), np.nan, dtype=float)
            for (iy, ix), vals in bucket.items():
                C[iy, ix] = float(np.nanmean(vals))
            return C

        C_I = build_matrix(I_points, unique_rounds, unique_delays) if I_points else None
        C_Q = build_matrix(Q_points, unique_rounds, unique_delays) if Q_points else None

        def centers_to_edges(centers):
            centers = np.asarray(sorted(np.unique(centers)), dtype=float)
            if centers.size == 1:
                d = 1.0
                return np.array([centers[0] - d / 2, centers[0] + d / 2])
            mids = (centers[:-1] + centers[1:]) / 2.0
            first = centers[0] - (centers[1] - centers[0]) / 2.0
            last = centers[-1] + (centers[-1] - centers[-2]) / 2.0
            return np.concatenate([[first], mids, [last]])

        Nx = len(unique_rounds)
        Ny = len(unique_delays)
        x_edges = np.arange(-0.5, Nx + 0.5, 1.0)
        y_edges = centers_to_edges(unique_delays)

        # Per-panel color limits
        def panel_limits(C):
            if C is None or not np.isfinite(C).any():
                return 0.0, 1.0
            vmin = float(np.nanmin(C))
            vmax = float(np.nanmax(C))
            if center_zero:
                m = max(abs(vmin), abs(vmax))
                return -m, m
            return vmin, vmax

        vmin_I, vmax_I = panel_limits(C_I)
        vmin_Q, vmax_Q = panel_limits(C_Q)

        self.create_folder_if_not_exists(save_path)

        # --- Figure with two stacked panels ---
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(6.5, 8.0), sharex=True)

        for ax, C, label, vmin, vmax in [
            (axes[0], C_I, "I ", vmin_I, vmax_I),
            (axes[1], C_Q, "Q ", vmin_Q, vmax_Q),
        ]:
            ax.set_aspect('auto')
            if C is None:
                ax.text(0.5, 0.5, f"No {label.split()[0]} data", ha='center', va='center')
                continue
            mesh = ax.pcolormesh(x_edges, y_edges, C, shading='flat', vmin=vmin, vmax=vmax)
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label("Signal value")
            ax.set_ylabel("Delay time")
            ax.set_title(f"Qubit {self.qubit + 1} — {label}")

            # Limit number of y tick labels
            if Ny > 0:
                if Ny <= max_ylabels:
                    yticks_idx = list(range(Ny))
                else:
                    yticks_idx = np.linspace(0, Ny - 1, num=max_ylabels, dtype=int).tolist()
                    yticks_idx = sorted(set(yticks_idx))
                yticks_vals = [unique_delays[i] for i in yticks_idx]
                ax.set_yticks(yticks_vals)
                ax.set_yticklabels([f"{v:.0f}" for v in yticks_vals])

        # x ticks on the bottom only
        axes[-1].set_xlabel("Round")
        max_xtick_labels = 10
        step = max(1, int(np.ceil(Nx / max_xtick_labels)))
        xticks = np.arange(0, Nx, step)
        axes[-1].set_xticks(xticks)
        axes[-1].set_xticklabels([unique_rounds[i] for i in xticks], rotation=45, ha='right')

        # FYI if multiple gains were used
        gains_unique = sorted({round(g, 12) for g in (I_gains + Q_gains) if np.isfinite(g)})
        if len(gains_unique) > 1:
            print(f"Note: found multiple gains {gains_unique}, but plotting collapsed over gain (x = rounds).")

        fig.tight_layout()
        outfile = (save_path + f"t2_heatmap_IQ_raw_q{self.qubit}_by_round.png")
        fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
        plt.close(fig)
        print(f"Saved I/Q heatmaps to: {outfile}")

    def plot_all_t2_rounds_IQ_single_calibration(
            self,
            Is, Qs,
            Ig_calibration1, Qg_calibration1,
            Ie_calibration1, Qe_calibration1,
            gains, rounds, delay_times,
            save_path, max_ylabels=6):
        """
        Like plot_all_t2_rounds(), but:
          - inputs I/Q instead of amplitudes
          - applies one calibration (Ig/Qg, Ie/Qe) across ALL datasets
          - collapses over gain when aggregating, same as the original function

        Data model (per qubit q):
          Is[q][i], Qs[q][i] : 1D arrays of I and Q samples for dataset i (same length)
          gains[q][i]        : scalar or per-sample array (used only for logging; heatmap collapses over gain)
          rounds[q][i]       : round label for dataset i (str/int)
          delay_times[q][i]  : scalar or per-sample array
        """
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from collections import defaultdict

        q = self.qubit
        gains_q = gains.get(q, [])
        I_q = Is.get(q, [])
        Q_q = Qs.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        # --- Shared calibration for all datasets (like your 2nd function) ---
        g = np.mean(np.asarray(Ig_calibration1, dtype=float) + 1j * np.asarray(Qg_calibration1, dtype=float))
        e = np.mean(np.asarray(Ie_calibration1, dtype=float) + 1j * np.asarray(Qe_calibration1, dtype=float))
        denom = np.abs(e - g) ** 2
        if not np.isfinite(denom) or denom <= 0:
            raise ValueError("Calibration is degenerate (e ≈ g); cannot normalize.")

        def centers_to_edges(centers):
            centers = np.asarray(sorted(np.unique(centers)), dtype=float)
            if centers.size == 1:
                d = 1.0
                return np.array([centers[0] - d / 2, centers[0] + d / 2])
            mids = (centers[:-1] + centers[1:]) / 2.0
            first = centers[0] - (centers[1] - centers[0]) / 2.0
            last = centers[-1] + (centers[-2] - centers[-1]) / -2.0  # same as + (centers[-1]-centers[-2])/2
            return np.concatenate([[first], mids, [last]])

        n = min(len(I_q), len(Q_q), len(gains_q), len(rounds_q), len(delay_q))
        if n == 0:
            print(f"No usable data for qubit {q} (missing lists). Skipping.")
            return

        # Collect per-sample calibrated amplitudes, bucketed by (round, delay), collapsing over gain
        all_points = []  # (round_id, delay, calibrated_amp)
        gains_seen = []

        for i in range(n):
            I = np.asarray(I_q[i], dtype=float).ravel()
            Q = np.asarray(Q_q[i], dtype=float).ravel()
            if I.size == 0 or Q.size == 0 or I.size != Q.size:
                continue

            z = I + 1j * Q
            amp_cal = np.abs((z - g) * (e - g) / denom)  # population-like calibrated amplitude

            # Gain can be scalar or per-sample; we only track for info (heatmap collapses over gain)
            g_i = gains_q[i] if i < len(gains_q) else np.nan
            if isinstance(g_i, (list, tuple, np.ndarray)):
                g_arr = np.asarray(g_i, dtype=float).ravel()
                if g_arr.size == 0:
                    g_scalar = np.nan
                else:
                    g_scalar = float(np.nanmean(g_arr))
            else:
                try:
                    g_scalar = float(g_i)
                except Exception:
                    g_scalar = np.nan
            if np.isfinite(g_scalar):
                gains_seen.append(g_scalar)

            # Delay can be scalar or per-sample; broadcast to sample length
            d_i = delay_q[i] if i < len(delay_q) else np.nan
            if isinstance(d_i, (list, tuple, np.ndarray)):
                d_arr = np.asarray(d_i, dtype=float).ravel()
                if d_arr.size not in (1, I.size):
                    # If exactly one delay provided, broadcast; otherwise skip mismatch
                    if d_arr.size == 1:
                        d_arr = np.full(I.shape, float(d_arr[0]))
                    else:
                        continue
                if d_arr.size == 1:
                    d_arr = np.full(I.shape, float(d_arr[0]))
            else:
                try:
                    d_scalar = float(d_i)
                except Exception:
                    continue
                d_arr = np.full(I.shape, d_scalar)

            r_id = str(rounds_q[i])

            mask = np.isfinite(I) & np.isfinite(Q) & np.isfinite(amp_cal) & np.isfinite(d_arr)
            if not np.any(mask):
                continue

            for d_val, a_val in zip(d_arr[mask], amp_cal[mask]):
                all_points.append((r_id, float(d_val), float(a_val)))

        if not all_points:
            print(f"No numeric points for qubit {q}. Skipping.")
            return

        # Warn if multiple gains but we collapse over them
        gains_unique = sorted({round(gv, 12) for gv in gains_seen if np.isfinite(gv)})
        if len(gains_unique) > 1:
            print(f"Note: found multiple gains {gains_unique}, but plotting collapsed over gain (x = rounds).")

        # Build grid: rows = unique delays, cols = unique rounds
        # Keep round order as first-seen (like original), delays sorted numeric
        seen_rounds = {}
        for (r, _, __) in all_points:
            if r not in seen_rounds:
                seen_rounds[r] = None
        unique_rounds = list(seen_rounds.keys())
        unique_delays = sorted({d for (_, d, __) in all_points})

        di_map = {d: i for i, d in enumerate(unique_delays)}
        ri_map = {r: i for i, r in enumerate(unique_rounds)}
        bucket = defaultdict(list)
        for (r, d, a) in all_points:
            bucket[(di_map[d], ri_map[r])].append(a)

        Ny, Nx = len(unique_delays), len(unique_rounds)
        C = np.full((Ny, Nx), np.nan, dtype=float)
        for (iy, ix), vals in bucket.items():
            C[iy, ix] = float(np.nanmean(vals))

        all_vals = np.array([a for (_, _, a) in all_points], dtype=float)
        global_vmin = float(np.nanmin(all_vals))
        global_vmax = float(np.nanmax(all_vals))

        # Edges for pcolormesh
        x_edges = np.arange(-0.5, Nx + 0.5, 1.0)
        y_edges = centers_to_edges(unique_delays)

        self.create_folder_if_not_exists(save_path)

        # --- Plot (fixed figure size, like original) ---
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.set_aspect('auto')

        mesh = ax.pcolormesh(x_edges, y_edges, C, shading='flat', vmin=global_vmin, vmax=global_vmax)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label("Qubit Population (calibrated)")

        ax.set_title(f"Qubit {self.qubit + 1} — T2 Ramsey (IQ-calibrated), collapsed over gain")
        ax.set_xlabel("Round")
        ax.set_ylabel("Delay time")

        # X ticks: fewer labels
        max_xtick_labels = 10
        step = max(1, int(np.ceil(Nx / max_xtick_labels)))
        xticks = np.arange(0, Nx, step)
        ax.set_xticks(xticks)
        ax.set_xticklabels([unique_rounds[i] for i in xticks], rotation=45, ha='right')

        # Y ticks: at most max_ylabels, evenly spaced
        if Ny > 0:
            if Ny <= max_ylabels:
                yticks_idx = list(range(Ny))
            else:
                yticks_idx = sorted(set(np.linspace(0, Ny - 1, num=max_ylabels, dtype=int).tolist()))
            yticks_vals = [unique_delays[i] for i in yticks_idx]
            ax.set_yticks(yticks_vals)
            ax.set_yticklabels([f"{v:.0f}" for v in yticks_vals])

        fig.tight_layout()
        outfile = (save_path + f"t2_heatmap_q{self.qubit}_by_round_single_calibration.png")
        fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
        plt.close(fig)
        print(f"Saved calibrated heatmap to: {outfile}")

    def plot_all_t2_rounds(self, amps, gains, rounds, delay_times, save_path, max_ylabels=6):
        import numpy as np
        import matplotlib.pyplot as plt
        import matplotlib.ticker as mticker
        from collections import defaultdict

        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])
        delay_q = delay_times.get(q, [])

        n = min(len(amps_q), len(gains_q), len(rounds_q), len(delay_q))
        if n == 0 or not (len(amps_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
            print(f"No usable data for qubit {q} (missing lists or length mismatch). Skipping.")
            return

        all_points = []
        gains_seen = []
        for i in range(n):
            r_id = str(rounds_q[i])
            a_samples = np.asarray(amps_q[i], dtype=float).ravel()
            if a_samples.size == 0:
                continue

            g_i = gains_q[i]
            if isinstance(g_i, (list, tuple, np.ndarray)):
                g_arr = np.asarray(g_i, dtype=float).ravel()
                g_scalar = float(np.nanmean(g_arr)) if g_arr.size > 0 else np.nan
            else:
                try:
                    g_scalar = float(g_i)
                except Exception:
                    g_scalar = np.nan
            if np.isfinite(g_scalar):
                gains_seen.append(g_scalar)

            d_i = delay_q[i]
            if isinstance(d_i, (list, tuple, np.ndarray)):
                d_arr = np.asarray(d_i, dtype=float).ravel()
                if d_arr.size not in (1, a_samples.size):
                    if d_arr.size == 1:
                        d_arr = np.full_like(a_samples, float(d_arr[0]))
                    else:
                        continue
                if d_arr.size == 1:
                    d_arr = np.full_like(a_samples, float(d_arr[0]))
            else:
                try:
                    d_scalar = float(d_i)
                except Exception:
                    continue
                d_arr = np.full_like(a_samples, d_scalar)

            mask = np.isfinite(a_samples) & np.isfinite(d_arr)
            if not np.any(mask):
                continue

            for d, a in zip(d_arr[mask], a_samples[mask]):
                all_points.append((r_id, float(d), float(a)))

        if not all_points:
            print(f"No numeric points for qubit {q}. Skipping.")
            return

        gains_unique = sorted({round(g, 12) for g in gains_seen if np.isfinite(g)})
        if len(gains_unique) > 1:
            print(f"Note: found multiple gains {gains_unique}, but plotting collapsed over gain (x = rounds).")

        seen = {}
        for (r, _, __) in all_points:
            if r not in seen:
                seen[r] = None
        unique_rounds = list(seen.keys())
        unique_delays = sorted({d for (_, d, __) in all_points})

        di_map = {d: i for i, d in enumerate(unique_delays)}
        ri_map = {r: i for i, r in enumerate(unique_rounds)}
        bucket = defaultdict(list)
        for (r, d, a) in all_points:
            bucket[(di_map[d], ri_map[r])].append(a)

        Ny, Nx = len(unique_delays), len(unique_rounds)
        C = np.full((Ny, Nx), np.nan, dtype=float)
        for (iy, ix), vals in bucket.items():
            C[iy, ix] = float(np.nanmean(vals))

        all_amps = np.array([a for (_, _, a) in all_points], dtype=float)
        global_vmin = float(np.nanmin(all_amps))
        global_vmax = float(np.nanmax(all_amps))

        def centers_to_edges(centers):
            centers = np.asarray(sorted(np.unique(centers)), dtype=float)
            if centers.size == 1:
                d = 1.0
                return np.array([centers[0] - d / 2, centers[0] + d / 2])
            mids = (centers[:-1] + centers[1:]) / 2.0
            first = centers[0] - (centers[1] - centers[0]) / 2.0
            last = centers[-1] + (centers[-1] - centers[-2]) / 2.0
            return np.concatenate([[first], mids, [last]])

        x_edges = np.arange(-0.5, Nx + 0.5, 1.0)
        y_edges = centers_to_edges(unique_delays)

        self.create_folder_if_not_exists(save_path)

        # --- Fixed size like before ---
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        # Keep automatic aspect so it doesn't stretch with many rounds
        ax.set_aspect('auto')

        mesh = ax.pcolormesh(x_edges, y_edges, C, shading='flat', vmin=global_vmin, vmax=global_vmax)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label("Qubit Population")

        ax.set_title(f"Qubit {self.qubit + 1} — T2 Ramsey heatmap taken repeatedly for zeno gain 0.01")
        ax.set_xlabel("Round")
        ax.set_ylabel("Delay time")

        max_xtick_labels = 10  # change if you want more/less
        step = max(1, int(np.ceil(Nx / max_xtick_labels)))
        xticks = np.arange(0, Nx, step)

        ax.set_xticks(xticks)
        ax.set_xticklabels([unique_rounds[i] for i in xticks], rotation=45, ha='right')
        # Fewer x-axis tick labels (match labels to ticks)
        max_xtick_labels = 10  # tune if needed
        Nx = len(unique_rounds)
        step = max(1, int(np.ceil(Nx / max_xtick_labels)))
        xticks = np.arange(0, Nx, step)

        ax.set_xticks(xticks)
        ax.set_xticklabels([unique_rounds[i] for i in xticks], rotation=45, ha='right')

        if Ny > 0:
            if Ny <= max_ylabels:
                yticks_idx = list(range(Ny))
            else:
                yticks_idx = np.linspace(0, Ny - 1, num=max_ylabels, dtype=int).tolist()
                yticks_idx = sorted(set(yticks_idx))
            yticks_vals = [unique_delays[i] for i in yticks_idx]
            ax.set_yticks(yticks_vals)
            ax.set_yticklabels([f"{v:.0f}" for v in yticks_vals])

        fig.tight_layout()
        outfile = (save_path + f"t2_heatmap_q{self.qubit}_by_round.png")
        fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
        plt.close(fig)
        print(f"Saved heatmap to: {outfile}")


    def plot_all_t2_rounds_adapted_for_gain(self, amps, gains, rounds, delay_times, save_path, max_ylabels=6):
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict

        q = self.qubit
        gains_q = gains.get(q, [])
        amps_q = amps.get(q, [])
        rounds_q = rounds.get(q, [])  # not used for axes, but kept for parity/checks
        delay_q = delay_times.get(q, [])

        n = min(len(amps_q), len(gains_q), len(rounds_q), len(delay_q))
        if n == 0 or not (len(amps_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
            print(f"No usable data for qubit {q} (missing lists or length mismatch). Skipping.")
            return

        all_points = []  # (gain, delay, amp)

        for i in range(n):
            # amps for this "round"
            a_samples = np.asarray(amps_q[i], dtype=float).ravel()
            if a_samples.size == 0:
                continue

            # align gains element-wise with amps (broadcast if scalar/size==1)
            g_i = gains_q[i]
            if isinstance(g_i, (list, tuple, np.ndarray)):
                g_arr = np.asarray(g_i, dtype=float).ravel()
                if g_arr.size == 1:
                    g_arr = np.full_like(a_samples, float(g_arr[0]))
                elif g_arr.size != a_samples.size:
                    # cannot align gains to amps for this segment; skip it
                    continue
            else:
                try:
                    g_scalar = float(g_i)
                except Exception:
                    continue
                g_arr = np.full_like(a_samples, g_scalar)

            # align delays element-wise with amps (broadcast if scalar/size==1)
            d_i = delay_q[i]
            if isinstance(d_i, (list, tuple, np.ndarray)):
                d_arr = np.asarray(d_i, dtype=float).ravel()
                if d_arr.size == 1:
                    d_arr = np.full_like(a_samples, float(d_arr[0]))
                elif d_arr.size != a_samples.size:
                    continue
            else:
                try:
                    d_scalar = float(d_i)
                except Exception:
                    continue
                d_arr = np.full_like(a_samples, d_scalar)

            # keep only finite triplets
            mask = np.isfinite(a_samples) & np.isfinite(d_arr) & np.isfinite(g_arr)
            if not np.any(mask):
                continue

            # collect (gain, delay, amp)
            for g, d, a in zip(g_arr[mask], d_arr[mask], a_samples[mask]):
                all_points.append((float(g), float(d), float(a)))

        if not all_points:
            print(f"No numeric points for qubit {q}. Skipping.")
            return

        # unique axes values
        unique_gains = sorted({g for (g, _, __) in all_points})
        unique_delays = sorted({d for (_, d, __) in all_points})

        gi_map = {g: i for i, g in enumerate(unique_gains)}
        di_map = {d: i for i, d in enumerate(unique_delays)}

        # bin amplitudes by (delay_idx, gain_idx)
        bucket = defaultdict(list)
        for (g, d, a) in all_points:
            bucket[(di_map[d], gi_map[g])].append(a)

        Ny, Nx = len(unique_delays), len(unique_gains)
        C = np.full((Ny, Nx), np.nan, dtype=float)
        for (iy, ix), vals in bucket.items():
            C[iy, ix] = float(np.nanmean(vals))

        # color scale from all amplitudes
        all_amps = np.array([a for (_, _, a) in all_points], dtype=float)
        global_vmin = float(np.nanmin(all_amps))
        global_vmax = float(np.nanmax(all_amps))

        # helper: centers -> bin edges
        def centers_to_edges(centers):
            centers = np.asarray(sorted(np.unique(centers)), dtype=float)
            if centers.size == 1:
                d = 1.0
                return np.array([centers[0] - d / 2, centers[0] + d / 2])
            mids = (centers[:-1] + centers[1:]) / 2.0
            first = centers[0] - (centers[1] - centers[0]) / 2.0
            last = centers[-1] + (centers[-1] - centers[-2]) / 2.0
            return np.concatenate([[first], mids, [last]])

        # edges for pcolormesh
        x_edges = centers_to_edges(unique_gains)  # gains on x
        y_edges = centers_to_edges(unique_delays)  # delays on y

        # ensure output folder exists
        self.create_folder_if_not_exists(save_path)

        # plot
        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        ax.set_aspect('auto')

        mesh = ax.pcolormesh(x_edges, y_edges, C, shading='flat', vmin=global_vmin, vmax=global_vmax)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label("Qubit Population")

        ax.set_title(f"Qubit {self.qubit + 1} — T2 Ramsey heatmap by gain")
        ax.set_xlabel("Gain")
        ax.set_ylabel("Delay time")

        # thin x tick labels to ~10
        max_xtick_labels = 10
        if Nx > 0:
            step = max(1, int(np.ceil(Nx / max_xtick_labels)))
            xt_idx = np.arange(0, Nx, step)
            ax.set_xticks([unique_gains[i] for i in xt_idx])
            ax.set_xticklabels([f"{unique_gains[i]:g}" for i in xt_idx], rotation=45, ha='right')

        # y ticks (same thinning logic)
        if Ny > 0:
            if Ny <= max_ylabels:
                yticks_idx = list(range(Ny))
            else:
                yticks_idx = np.linspace(0, Ny - 1, num=max_ylabels, dtype=int).tolist()
                yticks_idx = sorted(set(yticks_idx))
            yticks_vals = [unique_delays[i] for i in yticks_idx]
            ax.set_yticks(yticks_vals)
            ax.set_yticklabels([f"{v:.0f}" for v in yticks_vals])

        fig.tight_layout()
        outfile = (save_path + f"t2_heatmap_q{self.qubit}_by_gain.png")
        fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
        plt.close(fig)
        print(f"Saved heatmap to: {outfile}")

    def plot_without_errs(self, date_times, t2_vals, show_legends):
        # ---------------------------------plot-----------------------------------------------------
        analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.title('T2R Values vs Time', fontsize=font)
        axes = axes.flatten()
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        from datetime import datetime
        for i, ax in enumerate(axes):

            ax.set_title(titles[i], fontsize=font)

            x = date_times[i]
            y = t2_vals[i]

            # Convert strings to datetime objects.
            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            # Combine datetime objects and y values into a list of tuples and sort by datetime.
            combined = list(zip(datetime_objects, y))
            combined.sort(reverse=True, key=lambda x: x[0])

            # Unpack them back into separate lists, in order from latest to most recent.
            sorted_x, sorted_y = zip(*combined)
            ax.scatter(sorted_x, sorted_y, color=colors[i])
            # print(len(sorted_y))
            # print(len(sorted_x))

            sorted_x = np.asarray(sorted(x))

            num_points = 5
            indices = np.linspace(0, len(sorted_x) - 1, num_points, dtype=int)

            # Set new x-ticks using the datetime objects at the selected indices
            ax.set_xticks(sorted_x[indices])
            ax.set_xticklabels([dt for dt in sorted_x[indices]], rotation=45)

            ax.scatter(x, y, color=colors[i])
            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time', fontsize=font - 2)
            ax.set_ylabel('T2R (us)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'T2R_vals.pdf', transparent=True, dpi=self.final_figure_quality)

        # plt.close()

    def plot_with_errs(self, date_times, t2_vals, t2_fit_err, show_legends):
        # ---------------------------------plot-----------------------------------------------------
        analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.suptitle('T2R Values vs Time', fontsize=font)
        axes = axes.flatten()

        from datetime import datetime
        import matplotlib.dates as mdates

        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            x = date_times[i]
            y = t2_vals[i]
            err = t2_fit_err[i]

            datetime_objects = [datetime.strptime(date_string, "%Y-%m-%d %H:%M:%S") for date_string in x]

            combined = list(zip(datetime_objects, y, err))
            combined.sort(key=lambda tup: tup[0])
            sorted_x, sorted_y, sorted_err = zip(*combined)
            sorted_x = np.array(sorted_x)

            ax.errorbar(
                sorted_x, sorted_y, yerr=sorted_err,
                fmt='none',  #no marker
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

            #ax.xaxis.set_major_locator(mdates.AutoDateLocator())
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            ax.tick_params(axis='x', rotation=45)

            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time', fontsize=font - 2)
            ax.set_ylabel('T2R (us)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'T2R_vals.pdf', transparent=True, dpi=self.final_figure_quality)
        plt.close()

    def plot_with_errs_single_plot(self, date_times, t2_vals, t2_fit_err, show_legends):
        analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
        self.create_folder_if_not_exists(analysis_folder)
        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('T2R Values vs Time', fontsize=font)
        from datetime import datetime
        import matplotlib.dates as mdates
        for i in range(self.number_of_qubits):
            x = date_times[i]
            y = t2_vals[i]
            err = t2_fit_err[i]
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
        if show_legends:
            ax.legend(edgecolor='black')
        ax.set_xlabel('Time', fontsize=font - 2)
        ax.set_ylabel('T2R (us)', fontsize=font - 2)
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.tight_layout()
        plt.savefig(analysis_folder + 'T2R_vals_single_plot.pdf', transparent=True, dpi=self.final_figure_quality)
        plt.close()
