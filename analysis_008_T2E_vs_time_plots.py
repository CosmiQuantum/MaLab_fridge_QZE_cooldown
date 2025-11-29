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

class T2eVsTime:
    def __init__(self, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name, fridge, exp_name='ge', qubit=0):
        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.signal = signal
        self.figure_quality = figure_quality
        self.run_name = run_name
        self.number_of_qubits = number_of_qubits
        self.final_figure_quality = final_figure_quality
        self.top_folder_dates = top_folder_dates
        self.fridge = fridge
        self.exp_name=exp_name
        self.qubit=qubit

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

    def exponential(x, a, b, c, d):
        return a * np.exp(-(x - b) / c) + d

    def optimal_bins(data):
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
    def run_t2_sweep_new(self, exp_extension='', scaling=False, return_calibration_data=False, weighted_mean=True):
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
            outerFolder_expt = outerFolder + f"/Data_h5/T2E{exp_extension}_zeno/"
            
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
                load_data = H5_class_instance.load_from_h5(data_type=f'T2E{exp_extension}_zeno', save_r=int(save_round), scaling=scaling)

                for q_key in load_data[f'T2E{exp_extension}_zeno']:
                    for dataset in range(len(load_data[f'T2E{exp_extension}_zeno'][q_key].get('Dates', [])[0])):
                        delays = self.process_h5_data(
                            load_data[f'T2E{exp_extension}_zeno'][q_key].get('Delay Times', [])[0][dataset].decode())

                        I = self.process_string_of_nested_lists(
                            load_data[f'T2E{exp_extension}_zeno'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_string_of_nested_lists(
                            load_data[f'T2E{exp_extension}_zeno'][q_key].get('Q', [])[0][dataset].decode())
                        
                        gains_swept = self.process_h5_data(
                            load_data[f'T2E{exp_extension}_zeno'][q_key].get('Gains', [])[0][dataset].decode())

                        if scaling:
                            Ie = self.process_string_of_nested_lists(
                                load_data[f'T2E{exp_extension}_zeno'][q_key].get('ss_I_e', [])[0][dataset].decode())
                            Ig = self.process_string_of_nested_lists(
                                load_data[f'T2E{exp_extension}_zeno'][q_key].get('ss_I_g', [])[0][dataset].decode())
                            Qe = self.process_string_of_nested_lists(
                                load_data[f'T2E{exp_extension}_zeno'][q_key].get('ss_Q_e', [])[0][dataset].decode())
                            Qg = self.process_string_of_nested_lists(
                                load_data[f'T2E{exp_extension}_zeno'][q_key].get('ss_Q_g', [])[0][dataset].decode())

                        if len(I) > 0:
                            round_current = int(folder_date.split('round')[-1])
                            rounds_completed[q_key].append(round_current)
                            
                            # Assume I and Q are list-of-lists (one per gain/repetition), single calibration per round
                            if scaling:
                                # Single calibration for all gains/repetitions
                                Ie_cal = np.asarray(Ie[0], dtype=float)
                                Ig_cal = np.asarray(Ig[0], dtype=float)
                                Qe_cal = np.asarray(Qe[0], dtype=float)
                                Qg_cal = np.asarray(Qg[0], dtype=float)
                                
                                e = np.mean(Ie_cal + 1j * Qe_cal)
                                g = np.mean(Ig_cal + 1j * Qg_cal)
                                
                                # Apply calibration to each repetition's data
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
                                    # Average over repetitions (single gain or no sweep)
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

    def run(self,return_errs=False):
        import datetime

        # ----------Load/get data------------------------
        t2e_vals = {i: [] for i in range(self.number_of_qubits)}
        t2e_errs = {i: [] for i in range(self.number_of_qubits)}
        rounds = []
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}

        for folder_date in self.top_folder_dates:
            if self.fridge.upper() == 'QUIET':
                outerFolder = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date+ "/study_data" + "/"
                outerFolder_save_plots = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date+ "/study_data" + "_plots/"
            elif self.fridge.upper() == 'NEXUS':
                outerFolder = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "/"
                outerFolder_save_plots = f"/home/nexusadmin/qick/NEXUS_sandbox/Data/{self.run_name}/" + folder_date + "_plots/"
            else:
                raise ValueError("fridge must be either 'QUIET' or 'NEXUS'")

            # -------------------------------------------------------Load/Plot/Save T2E------------------------------------------
            outerFolder_expt = outerFolder + "/Data_h5/T2E_ge/"
            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

            for h5_file in h5_files:
                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type='T2E', save_r=int(save_round))

                # Define specific days to exclude
                exclude_dates = {
                    datetime.date(2025, 1, 26),  # power outage
                    datetime.date(2025, 1, 29),  # HEMT Issues
                    datetime.date(2025, 1, 30),  # HEMT Issues
                    datetime.date(2025, 1, 31)  # Optimization Issues and non RR work in progress
                }

                for q_key in load_data['T2E']:
                    for dataset in range(len(load_data['T2E'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data['T2E'][q_key].get('Dates', [])[0][dataset]):
                            continue
                        # T2 = load_data['T2E'][q_key].get('T2', [])[0][dataset]
                        # errors = load_data['T2E'][q_key].get('Errors', [])[0][dataset]
                        date = datetime.datetime.fromtimestamp(load_data['T2E'][q_key].get('Dates', [])[0][dataset])

                        # Skip processing if the date (as a date object) is in the excluded set
                        if date.date() in exclude_dates:
                            print(f"Skipping data for {date} (excluded date)")
                            continue

                        I = self.process_h5_data(load_data['T2E'][q_key].get('I', [])[0][dataset].decode())
                        Q = self.process_h5_data(load_data['T2E'][q_key].get('Q', [])[0][dataset].decode())
                        delay_times = self.process_h5_data(load_data['T2E'][q_key].get('Delay Times', [])[0][dataset].decode())
                        # fit = load_data['T2E'][q_key].get('Fit', [])[0][dataset]
                        round_num = load_data['T2E'][q_key].get('Round Num', [])[0][dataset]
                        batch_num = load_data['T2E'][q_key].get('Batch Num', [])[0][dataset]
                        try:
                            syst_config = load_data['T2E'][q_key].get('Syst Config', [])[0][dataset].decode()
                            exp_config = load_data['T2E'][q_key].get('Exp Config', [])[0][dataset].decode()
                            safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                            exp_config = eval(exp_config, safe_globals)
                        except:
                            exp_config =None

                        if len(I) > 0:
                            T2E_class_instance = T2EMeasurement(q_key, self.number_of_qubits, outerFolder_save_plots, round_num, self.signal, self.save_figs,
                                                               fit_data=True)
                            try:
                                fitted, t2e_est, t2e_err, plot_sig = T2E_class_instance.t2_fit(delay_times, I, Q)
                            except Exception as e:
                                print(f"good fit not found, error: {e}")
                                continue
                            #T2E_cfg = exp_config['SpinEcho_ge']
                            if t2e_est < 0:
                                print("The value is negative, continuing...")
                                continue
                            if t2e_est > 300:
                                print("The value is above 300 us, this is a bad fit, continuing...")
                                continue
                            if t2e_err >= 0.4 * t2e_est:
                                print(
                                    f"Skipping T2R = {t2e_est:.3f} µs because its error {t2e_err:.3f} µs is >= 80% of its value.")
                                continue
                            t2e_vals[q_key].extend([t2e_est])
                            t2e_errs[q_key].extend([t2e_err])
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                            del T2E_class_instance
                del H5_class_instance
        if return_errs:
            return date_times, t2e_vals, t2e_errs
        else:
            return date_times, t2e_vals
    def plot_all_t2_heatmaps_new_format(
            self,
            amps,
            gains,
            rounds,
            delay_times,
            save_path,
            max_ylabels=6,
            n_bar=None,
            save_individual_plots=True
    ):
        """
        Heatmaps + per-gain T2E fits + T2E vs nbar plotting.
        """
        import numpy as np
        import matplotlib.pyplot as plt
        from collections import defaultdict
        import matplotlib.ticker as mticker
        from scipy.optimize import curve_fit

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
            a_samples = np.asarray(amps_q[i], dtype=float).ravel()
            if a_samples.size == 0: continue

            g_i = gains_q[i]
            g_arr = np.asarray(g_i, dtype=float).ravel() if isinstance(g_i, (list, tuple, np.ndarray)) else None
            if g_arr is None or g_arr.size == 1:
                try: g_scalar = float(g_i)
                except: continue
                g_arr = np.full(a_samples.shape, g_scalar, dtype=float)
            elif g_arr.size != a_samples.size: continue

            d_i = delay_q[i]
            d_arr = np.asarray(d_i, dtype=float).ravel() if isinstance(d_i, (list, tuple, np.ndarray)) else None
            if d_arr is None or d_arr.size == 1:
                try: d_scalar = float(d_i)
                except: continue
                d_arr = np.full(a_samples.shape, d_scalar, dtype=float)
            elif d_arr.size != a_samples.size: continue

            mask = np.isfinite(a_samples) & np.isfinite(g_arr) & np.isfinite(d_arr)
            if not np.any(mask): continue

            for g, d, a in zip(g_arr[mask], d_arr[mask], a_samples[mask]):
                all_points.append((r_id, float(g), float(d), float(a)))

        if not all_points:
            print(f"No numeric points for qubit {q}. Skipping.")
            return

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
            if n_bar is None: return None
            if isinstance(n_bar, (list, tuple, np.ndarray)):
                nb = np.asarray(n_bar, float).ravel()
                return nb if nb.size == len(gains_sorted) else None
            if isinstance(n_bar, dict):
                key = r_id_str if r_id_str in n_bar else str(r_id_str)
                entry = n_bar.get(key, None)
                if entry is None: return None
                if isinstance(entry, dict):
                    candidate = entry.get("lorentzian") or entry.get("gaussian") or entry.get("nbar") or entry.get("values")
                    if candidate is None: return None
                    nb = np.asarray(candidate, float).ravel()
                    return nb if nb.size == len(gains_sorted) else None
                if isinstance(entry, (list, tuple, np.ndarray)):
                    nb = np.asarray(entry, float).ravel()
                    return nb if nb.size == len(gains_sorted) else None
            return None

        # Exponential decay function
        def exp_decay(t, A, T2, C):
            return A * np.exp(-t / T2) + C

        for r_id in unique_rounds:
            pts = [(g, d, a) for (r, g, d, a) in all_points if r == r_id]
            if not pts: continue

            gains_r = sorted({g for (g, _, _) in pts})
            delays_r = sorted({d for (_, d, _) in pts})

            bucket = defaultdict(list)
            gi_map = {g: i for i, g in enumerate(gains_r)}
            di_map = {d: i for i, d in enumerate(delays_r)}
            for g, d, a in pts:
                bucket[(di_map[d], gi_map[g])].append(a)

            Ny, Nx = len(delays_r), len(gains_r)
            C = np.full((Ny, Nx), np.nan, dtype=float)
            
            # For T2E vs nbar plotting
            t2e_vals = []
            t2e_errs = []
            gains_for_plot = []
            
            # Fit each gain slice
            if save_individual_plots:
                round_dir = os.path.join(save_path, f"t2e_slices_q{q}_round_{r_id}")
                self.create_folder_if_not_exists(round_dir)

            for ix, g_val in enumerate(gains_r):
                # Collect data for this gain
                d_vals = []
                a_vals = []
                for iy, d_val in enumerate(delays_r):
                    vals = bucket.get((iy, ix), [])
                    if vals:
                        avg_a = float(np.nanmean(vals))
                        C[iy, ix] = avg_a
                        d_vals.append(d_val)
                        a_vals.append(avg_a)
                
                # Fit
                if len(d_vals) >= 4:
                    try:
                        p0 = [np.max(a_vals)-np.min(a_vals), d_vals[-1]/2.0, np.min(a_vals)]
                        popt, pcov = curve_fit(exp_decay, d_vals, a_vals, p0=p0, maxfev=2000)
                        perr = np.sqrt(np.diag(pcov))
                        t2e = popt[1]
                        t2e_err = perr[1]
                        
                        if 0 < t2e < 500 and t2e_err < t2e:
                            t2e_vals.append(t2e)
                            t2e_errs.append(t2e_err)
                            gains_for_plot.append(g_val)
                            
                            # Save individual plot
                            if save_individual_plots:
                                fig_slice, ax_slice = plt.subplots(figsize=(6, 4))
                                ax_slice.plot(d_vals, a_vals, 'o', label='Data')
                                t_fit = np.linspace(min(d_vals), max(d_vals), 100)
                                ax_slice.plot(t_fit, exp_decay(t_fit, *popt), '-', label=f'Fit T2={t2e:.2f}us')
                                ax_slice.set_title(f"Q{q+1} R{r_id} Gain {g_val}")
                                ax_slice.set_xlabel("Delay (us)")
                                ax_slice.set_ylabel("Population")
                                ax_slice.legend()
                                safe_gain = str(g_val).replace('.', 'p').replace('-', 'm')
                                fig_slice.savefig(os.path.join(round_dir, f"t2e_slice_gain{safe_gain}.png"), dpi=100)
                                plt.close(fig_slice)
                    except:
                        pass

            # Heatmap Plotting
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
            x_edges = centers_to_edges(x_vals_sorted)
            y_edges = centers_to_edges(delays_r)

            fig, ax = plt.subplots(figsize=(6.5, 4.5))
            mesh = ax.pcolormesh(x_edges, y_edges, C_sorted, shading='flat', vmin=global_vmin, vmax=global_vmax)
            cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
            cbar.set_label("Qubit Population")
            ax.set_title(f"Qubit {self.qubit + 1} — Round {r_id} (T2E)")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Delay time")
            ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7, prune=None))
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            if Ny > 0:
                yt = list(range(Ny)) if Ny <= max_ylabels else sorted(set(np.linspace(0, Ny - 1, max_ylabels, dtype=int)))
                ax.set_yticks([delays_r[i] for i in yt])
                ax.set_yticklabels([f"{delays_r[i]:.0f}" for i in yt])

            fig.tight_layout()
            outfile = (save_path + f"t2e_heatmap_q{self.qubit}_round{r_id}{'_nbar' if n_bar is not None else ''}.png")
            fig.savefig(outfile, transparent=False, dpi=self.final_figure_quality)
            plt.close(fig)
            print(f"Saved T2E heatmap for round {r_id} to: {outfile}")
            
            # T2E vs nbar/gain Plotting
            if len(t2e_vals) > 1:
                fig_vs, ax_vs = plt.subplots(figsize=(6, 4))
                
                # Determine x-axis for this plot
                if nbar_vec is not None:
                    # Map gains to nbar
                    gain_to_nbar = {g: n for g, n in zip(gains_r, nbar_vec)}
                    x_plot = [gain_to_nbar[g] for g in gains_for_plot]
                    xlabel_vs = r"$\bar{n}$"
                else:
                    x_plot = gains_for_plot
                    xlabel_vs = "Gain"
                
                ax_vs.errorbar(x_plot, t2e_vals, yerr=t2e_errs, fmt='o', capsize=3)
                ax_vs.set_title(f"Qubit {self.qubit + 1} — Round {r_id} — T2E vs {xlabel_vs}")
                ax_vs.set_xlabel(xlabel_vs)
                ax_vs.set_ylabel(r"$T_{2E}$ ($\mu$s)")
                ax_vs.grid(True, alpha=0.25)
                fig_vs.tight_layout()
                
                # Save to common folder
                vs_folder = os.path.join(save_path, f"t2e_vs_nbar_q{q}")
                self.create_folder_if_not_exists(vs_folder)
                fig_vs.savefig(os.path.join(vs_folder, f"T2E_vs_nbar_round{r_id}.png"), dpi=self.final_figure_quality)
                plt.close(fig_vs)
                print(f"Saved T2E vs {xlabel_vs} for round {r_id}")

    def plot_without_errs(self, date_times, t2e_vals, show_legends):
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
        plt.title('T2E Values vs Time', fontsize=font)
        axes = axes.flatten()

        from datetime import datetime
        for i, ax in enumerate(axes):

            if i >= self.number_of_qubits:  # If we have fewer qubits than subplots, stop plotting and hide the rest
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            x = date_times[i]
            y = t2e_vals[i]

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
            ax.yaxis.set_major_formatter(StrMethodFormatter("{x:.2f}"))  # 2 decimal places

            if show_legends:
                ax.legend(edgecolor='black')
            ax.set_xlabel('Time', fontsize=font - 2)
            ax.set_ylabel('T2E (us)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'T2E_vals.png', transparent=False, dpi=self.final_figure_quality)
        print('Plot saved at: ', analysis_folder)
        plt.close()

    def plot_with_errs(self, date_times, t2e_vals, t2e_fit_err, show_legends):
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
        day1 = 24
        day2 = 25
        hour_start = 0
        hour_end = 12
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 0)
        # -----------------------------------------------------------------

        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        plt.suptitle('T2E Values vs Time', fontsize=font)
        axes = axes.flatten()

        import matplotlib.dates as mdates
        from matplotlib.ticker import StrMethodFormatter

        for i, ax in enumerate(axes):
            if i >= self.number_of_qubits:
                ax.set_visible(False)
                continue

            ax.set_title(titles[i], fontsize=font)

            x = date_times[i]
            y = t2e_vals[i]
            err = t2e_fit_err[i]

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
            ax.set_ylabel('T2E (us)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(analysis_folder + 'T2E_vals.png', transparent=False, dpi=self.final_figure_quality)
        print('Plot saved at: ', analysis_folder)
        plt.close()

    def plot_with_errs_single_plot(self, date_times, t2e_vals, t2e_fit_err, show_legends):
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
        day1 = 24
        day2 = 25
        hour_start = 0
        hour_end = 12
        start_time = datetime(year, month, day1, hour_start, 0)
        end_time = datetime(year, month, day2, hour_end, 0)
        font = 14
        titles = [f"Qubit {i + 1}" for i in range(self.number_of_qubits)]
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(12, 8))
        fig.suptitle('T2E Values vs Time', fontsize=font)
        import matplotlib.dates as mdates
        from matplotlib.ticker import StrMethodFormatter
        for i in range(self.number_of_qubits):
            x = date_times[i]
            y = t2e_vals[i]
            err = t2e_fit_err[i]
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
        ax.set_ylabel('T2E (us)', fontsize=font - 2)
        ax.tick_params(axis='both', which='major', labelsize=8)
        plt.tight_layout()
        plt.savefig(analysis_folder + 'T2E_vals_single_plot.png', transparent=False, dpi=self.final_figure_quality)
        print('Plot saved at: ', analysis_folder)
        plt.close()

