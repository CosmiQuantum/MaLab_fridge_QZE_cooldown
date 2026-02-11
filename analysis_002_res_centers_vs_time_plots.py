import numpy as np
import os
import sys
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
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.optimize import curve_fit

class ResonatorFreqVsTime:
    def __init__(self, figure_quality, final_figure_quality, number_of_qubits, top_folder_dates, save_figs, fit_saved,
                 signal, run_name):
        self.figure_quality = figure_quality
        self.number_of_qubits = number_of_qubits
        self.save_figs = save_figs
        self.fit_saved = fit_saved
        self.signal = signal
        self.run_name = run_name
        self.top_folder_dates = top_folder_dates
        self.final_figure_quality = final_figure_quality


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

    def run(self,exp_extension=''):
        import datetime
        # ----------Load/get data------------------------
        resonator_centers = {i: [] for i in range(self.number_of_qubits)}
        rounds = []
        reps = []
        file_names = []
        date_times = {i: [] for i in range(self.number_of_qubits)}
        mean_values = {}

        for folder_date in self.top_folder_dates:
            outerFolder = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date + "/study_data"
            outerFolder_save_plots = f"M:/_Data/20250822 - Olivia/{self.run_name}/" + folder_date + "_plots/"

            # ------------------------------------------Load/Plot/Save Res Spec------------------------------------

            if '_' in exp_extension:
                outerFolder_expt = outerFolder + f"/Data_h5/Res{exp_extension}/"
            else:
                outerFolder_expt = outerFolder + "/Data_h5/Res_ge/"

            h5_files = glob.glob(os.path.join(outerFolder_expt, "*.h5"))

            for h5_file in h5_files:

                save_round = h5_file.split('Num_per_batch')[-1].split('.')[0]
                H5_class_instance = Data_H5(h5_file)
                #H5_class_instance.print_h5_contents(h5_file)
                load_data = H5_class_instance.load_from_h5(data_type=f'Res{exp_extension}', save_r=int(save_round))


                # just look at this resonator data, should have batch_num of arrays in each one
                # right now the data writes the same thing batch_num of times, so it will do the same 5 datasets 5 times, until you fix this just grab the first one (All 5)
                for q_key in load_data[f'Res{exp_extension}']:
                    # print("all batch_num datasets------------------------", load_data['Res'][q_key].get('Amps', [])[0])
                    # print("one dataset------------------------",load_data['Res'][q_key].get('Amps', [])[0][0].decode())
                    # go through each dataset in the batch and plot
                    for dataset in range(len(load_data[f'Res{exp_extension}'][q_key].get('Dates', [])[0])):
                        if 'nan' in str(load_data[f'Res{exp_extension}'][q_key].get('Dates', [])[0][dataset]):
                            continue

                        date = datetime.datetime.fromtimestamp(
                            load_data[f'Res{exp_extension}'][q_key].get('Dates', [])[0][dataset])  # single date per dataset

                        freq_pts = self.process_h5_data(load_data[f'Res{exp_extension}'][q_key].get('freq_pts', [])[0][0].decode())

                        freq_center = self.process_h5_data( str(load_data[f'Res{exp_extension}'][q_key].get('freq_center', [])[0][0]))
                        freqs_found = self.string_to_float_list(load_data[f'Res{exp_extension}'][q_key].get('Found Freqs', [])[0][
                                                               dataset].decode())  # comes in as a list of floats in string format, need to convert
                        amps = self.process_string_of_nested_lists(
                            load_data[f'Res{exp_extension}'][q_key].get('Amps', [])[0][0].decode())  # list of lists
                        round_num = load_data[f'Res{exp_extension}'][q_key].get('Round Num', [])[0][dataset]  # already a float
                        batch_num = load_data[f'Res{exp_extension}'][q_key].get('Batch Num', [])[0][dataset]

                        try:
                            exp_config = load_data[f'Res{exp_extension}'][q_key].get('Exp Config', [])[0][dataset].decode()
                            safe_globals = {"np": np, "array": np.array, "__builtins__": {}}
                            exp_config = eval(exp_config, safe_globals)
                        except:
                            exp_config = None



                        if len(freq_pts) > 0:
                            res_class_instance = ResonanceSpectroscopy(q_key, self.number_of_qubits, outerFolder_save_plots, round_num, self.save_figs)
                            #res_spec_cfg = exp_config['res_spec']
                            res_freqs = res_class_instance.get_results(freq_pts, freq_center, amps)

                            resonator_centers[q_key].extend([res_freqs[q_key]])
                            date_times[q_key].extend([date.strftime("%Y-%m-%d %H:%M:%S")])

                            del res_class_instance

                del H5_class_instance
        return date_times, resonator_centers

    def plot(self, date_times, resonator_centers, show_legends, exp_extension=''):
        # ---------------------------------plot-----------------------------------------------------
        analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_folder)
        analysis_folder = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/features_vs_time/"
        self.create_folder_if_not_exists(analysis_folder)

        font = 14
        colors = ['orange', 'blue', 'purple', 'green', 'brown', 'pink']

        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        ext = exp_extension.split('_')[0]
        fig.suptitle(f'Resonator centers vs Time {ext}', fontsize=font)

        titles = [f"Res {i + 1}" for i in range(self.number_of_qubits)]

        from datetime import datetime

        for i, ax in enumerate(axes):
            # If you only have self.number_of_qubits qubits, blank the extra subplots.
            if i >= self.number_of_qubits:
                ax.axis("off")
                continue

            ax.set_title(titles[i], fontsize=font)

            # ---- Skip if missing / empty for this i ----
            if (
                    i >= len(date_times)
                    or i >= len(resonator_centers)
                    or not date_times[i]  # covers None, [], "", etc.
                    or not resonator_centers[i]
            ):
                ax.axis("off")  # "put nothing"
                continue

            x = date_times[i]
            y = resonator_centers[i]

            # Convert strings to datetime objects.
            datetime_objects = [datetime.strptime(ds, "%Y-%m-%d %H:%M:%S") for ds in x]

            # Sort by datetime (latest first)
            combined = list(zip(datetime_objects, y))
            combined.sort(key=lambda t: t[0], reverse=True)

            sorted_x, sorted_y = zip(*combined)  # tuples of datetimes and y's

            ax.scatter(sorted_x, sorted_y, color=colors[i])

            # xticks: pick up to 5 evenly-spaced points
            num_points = min(5, len(sorted_x))
            indices = np.linspace(0, len(sorted_x) - 1, num_points, dtype=int)

            ax.set_xticks([sorted_x[idx] for idx in indices])
            ax.set_xticklabels([sorted_x[idx] for idx in indices], rotation=45)

            if show_legends:
                ax.legend(edgecolor='black')

            ax.set_xlabel('Time (Days)', fontsize=font - 2)
            ax.set_ylabel('Resonator Center (MHz)', fontsize=font - 2)
            ax.tick_params(axis='both', which='major', labelsize=8)

        plt.tight_layout()
        plt.savefig(
            analysis_folder + f'Res_Centers{exp_extension}.pdf',
            transparent=True,
            dpi=self.final_figure_quality
        )

        #plt.show()
    def plot_both_transitions(self, date_times_ge,
        resonator_centers_ge,
        date_times_fe=None,
        resonator_centers_fe=None,
        show_legends=True,
        exp_extension=""
    ):
        """
        Plot resonator centers vs time for both GE and FE transitions on the same subplots.
        - date_times_ge / resonator_centers_ge: lists (len = number_of_qubits) of equal-length sequences
          of "%Y-%m-%d %H:%M:%S" strings and float values (MHz) for the GE transition.
        - date_times_fe / resonator_centers_fe: same structure for the FE transition (optional).
        """

        # --------------------------------- setup + folders ---------------------------------
        analysis_root = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_root)
        analysis_folder = f"{analysis_root}features_vs_time/"
        self.create_folder_if_not_exists(analysis_folder)

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter
        import matplotlib.dates as mdates
        from datetime import datetime
        import matplotlib.dates as mdates

        def _sorted_xy(times_str_list, y_vals):
            if not times_str_list or not y_vals:
                return np.array([]), np.array([])

            dt = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in times_str_list]
            pairs = sorted(zip(dt, y_vals), key=lambda t: t[0])
            xs, ys = zip(*pairs)

            # Convert datetime -> matplotlib "date numbers"
            xs_num = mdates.date2num(xs)
            return np.asarray(xs_num), np.asarray(ys)

        font = 14
        ge_color = "tab:blue"
        fe_color = "tab:orange"

        # Figure: same 2x3 grid (assumes up to 6 resonators)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        # Nice overall title
        ext = exp_extension.split("_")[0]
        fig.suptitle(f"Resonator centers vs Time {ext} (GE & FE)", fontsize=font)

        # Date tick formatter
        locator = mdates.AutoDateLocator()
        formatter = mdates.ConciseDateFormatter(locator)

        def _sorted_xy(times_str_list, y_vals):
            """Return datetime-sorted (x, y). Accepts empty/None gracefully."""
            if not times_str_list or not y_vals or len(times_str_list) == 0 or len(y_vals) == 0:
                return np.array([]), np.array([])
            dt = [datetime.strptime(ts, "%Y-%m-%d %H:%M:%S") for ts in times_str_list]
            pairs = sorted(zip(dt, y_vals), key=lambda t: t[0])  # oldest -> newest
            xs, ys = zip(*pairs) if pairs else ([], [])
            return np.asarray(xs), np.asarray(ys)

        for i, ax in enumerate(axes[: self.number_of_qubits]):
            ax.set_title(f"Res {i + 1}", fontsize=font)

            # ----------------------------- GE data -----------------------------
            x_ge, y_ge = _sorted_xy(date_times_ge[i], resonator_centers_ge[i])
            if x_ge.size:
                ax.scatter(x_ge, y_ge, s=14, label="GE", color=ge_color)

            # ----------------------------- FE data (optional) -----------------------------
            if date_times_fe is not None and resonator_centers_fe is not None:
                x_fe, y_fe = _sorted_xy(date_times_fe[i], resonator_centers_fe[i])
                if x_fe.size:
                    ax.scatter(x_fe, y_fe, s=20, label="FE", color=fe_color)

            # Axes formatting
            # Make locator/formatter per-axis (recommended)
            locator = mdates.AutoDateLocator()
            formatter = mdates.ConciseDateFormatter(locator)

            x_ge, y_ge = _sorted_xy(date_times_ge[i], resonator_centers_ge[i])
            if x_ge.size:
                ax.scatter(x_ge, y_ge, s=14, label="GE", color=ge_color)

            ax.xaxis.set_major_locator(locator)
            ax.xaxis.set_major_formatter(formatter)

            ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.tick_params(axis="both", which="major", labelsize=8)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")

            ax.set_xlabel("Time", fontsize=font - 2)
            ax.set_ylabel("Resonator Center (MHz)", fontsize=font - 2)


            if show_legends:
                ax.legend(edgecolor="black", fontsize=8)

        # If there are fewer than 6 qubits, hide unused subplots
        for j in range(self.number_of_qubits, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig(
            analysis_folder + f"Res_Centers_GE_FE{exp_extension}.pdf",
            transparent=True,
            dpi=self.final_figure_quality,
        )
        plt.close(fig)

    def plot_both_transitions_hist(
        self,
        date_times_ge,
        resonator_centers_ge,
        date_times_fe=None,
        resonator_centers_fe=None,
        show_legends=True,
        exp_extension="",
        bins=30                       # "fd" => Freedman–Diaconis; or pass an int for fixed bins
    ):
        """
        For each resonator (subplot), histogram the resonator-center values for GE (and FE if provided).
        Titles include mean (μ) and sample standard deviation (σ) in MHz.

        Parameters
        ----------
        date_times_ge : list[list[str]]
            Unused here; preserved to mirror plot_both_transitions signature.
        resonator_centers_ge : list[list[float]]
            Per-resonator GE center values (MHz).
        date_times_fe : list[list[str]] or None
            Unused here; preserved to mirror plot_both_transitions signature.
        resonator_centers_fe : list[list[float]] or None
            Per-resonator FE center values (MHz) (optional).
        show_legends : bool
            Whether to show the legend.
        exp_extension : str
            Suffix added to saved filename.
        bins : "fd" or int or sequence
            Binning mode for histograms (default "fd"). You can pass an int for fixed bin count.
        """

        # --------------------------------- setup + folders ---------------------------------
        analysis_root = f"M:/_Data/20250822 - Olivia/{self.run_name}/benchmark_analysis_plots/"
        self.create_folder_if_not_exists(analysis_root)
        analysis_folder = f"{analysis_root}histograms/"
        self.create_folder_if_not_exists(analysis_folder)

        import numpy as np
        import matplotlib.pyplot as plt
        from matplotlib.ticker import FormatStrFormatter

        font = 14
        ge_color = "tab:blue"
        fe_color = "tab:orange"

        # Figure: same 2x3 grid (assumes up to 6 resonators)
        fig, axes = plt.subplots(2, 3, figsize=(12, 8))
        axes = axes.flatten()

        ext = exp_extension.split("_")[0]
        fig.suptitle(f"Resonator Centers Histograms {ext} (GE & FE)", fontsize=font)

        def _clean_vals(vs):
            """Return finite float numpy array; empty if None/empty."""
            if vs is None:
                return np.array([])
            arr = np.asarray(vs, dtype=float)
            if arr.size == 0:
                return arr
            return arr[np.isfinite(arr)]

        def _stats(arr):
            """Return (mu, sigma, n). Use sample std (ddof=1) when n>1."""
            n = int(arr.size)
            if n == 0:
                return np.nan, np.nan, 0
            mu = float(np.mean(arr))
            sigma = float(np.std(arr, ddof=1)) if n > 1 else 0.0
            return mu, sigma, n

        # loop resonators
        for i, ax in enumerate(axes[: self.number_of_qubits]):
            ge_vals = _clean_vals(resonator_centers_ge[i]) if i < len(resonator_centers_ge) else np.array([])
            fe_vals = _clean_vals(resonator_centers_fe[i]) if (resonator_centers_fe is not None and i < len(resonator_centers_fe)) else np.array([])

            # Shared bin edges per resonator for fair comparison
            combined = np.concatenate([ge_vals, fe_vals]) if fe_vals.size else ge_vals
            if combined.size >= 2:
                try:
                    # numpy's Freedman–Diaconis
                    bin_edges = np.histogram_bin_edges(combined, bins=bins)
                except Exception:
                    # fallback if numpy can't compute edges (e.g., all values equal)
                    bin_edges = np.histogram_bin_edges(combined, bins='auto')
            elif combined.size == 1:
                # make a small window around the single value
                v = combined[0]
                span = max(abs(v) * 0.01, 0.1)  # 1% or 0.1 MHz minimum span
                bin_edges = np.linspace(v - span, v + span, 5)
            else:
                bin_edges = np.linspace(0, 1, 5)  # dummy; plot will be empty

            # Plot histograms
            plotted_any = False
            if ge_vals.size:
                ax.hist(
                    ge_vals,
                    bins=bin_edges,
                    alpha=0.45,
                    label="GE",
                    color=ge_color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                mu_ge, sig_ge, n_ge = _stats(ge_vals)
                ax.axvline(mu_ge, color=ge_color, linestyle="--", linewidth=1.2)
                plotted_any = True
            else:
                mu_ge = sig_ge = n_ge = np.nan

            if fe_vals.size:
                ax.hist(
                    fe_vals,
                    bins=bin_edges,
                    alpha=0.45,
                    label="FE",
                    color=fe_color,
                    edgecolor="black",
                    linewidth=0.5,
                )
                mu_fe, sig_fe, n_fe = _stats(fe_vals)
                ax.axvline(mu_fe, color=fe_color, linestyle="--", linewidth=1.2)
                plotted_any = True
            else:
                mu_fe = sig_fe = n_fe = np.nan

            # Title with stats
            title_parts = [f"Res {i + 1}"]
            if not np.isnan(mu_ge):
                title_parts.append(f"GE μ={mu_ge:.1f} σ={sig_ge:.1f} \n (n={int(n_ge)})")
            if not np.isnan(mu_fe):
                title_parts.append(f"FE μ={mu_fe:.1f} σ={sig_fe:.1f} \n (n={int(n_fe)})")
            ax.set_title(" | ".join(title_parts), fontsize=10)

            # Axes formatting
            ax.set_xlabel("Resonator Center (MHz)", fontsize=font - 2)
            ax.set_ylabel("Count", fontsize=font - 2)
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
            ax.tick_params(axis="both", which="major", labelsize=8)


            if show_legends and plotted_any:
                ax.legend(edgecolor="black", fontsize=8)

            # If nothing to plot, soften the panel
            if not plotted_any:
                ax.set_facecolor("#f5f5f5")
                ax.text(0.5, 0.5, "No data", ha="center", va="center", transform=ax.transAxes, fontsize=10)
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        # Hide unused subplots
        for j in range(self.number_of_qubits, len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout(rect=(0, 0, 1, 0.96))
        plt.savefig(
            analysis_folder + f"Res_Centers_Hist_GE_FE{exp_extension}.pdf",
            transparent=True,
            dpi=self.final_figure_quality,
        )
        plt.close(fig)
