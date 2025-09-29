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
    def run_t2_sweep(self, exp_extension='', scaling=False):
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
                            safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                            exp_config = eval(exp_config, safe_globals)
                        except:
                            exp_config =None

                        if len(I) > 0:
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
                            else:
                                amp=np.hypot(I, Q)
                            amps[q_key].append(amp.tolist())
                            delay_times[q_key].append(delays)
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                del H5_class_instance
        return Is,Qs,amps, gains, rounds_completed, delay_times
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
            ax.set_xticks(gains_r)
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
