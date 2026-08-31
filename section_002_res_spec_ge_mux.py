import matplotlib.pyplot as plt
from qick.asm_v2 import AveragerProgramV2
from tqdm import tqdm
from build_state import *
from expt_config import *
import copy
import datetime
import logging
import os


class SingleToneSpectroscopyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_chs = cfg['ro_ch']
        res_ch = cfg['res_ch']
        print(cfg['res_length'],cfg['res_freq_ge'],cfg['res_gain_ge'],cfg['ro_phase'])
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        self.declare_readout(ch=cfg['ro_ch'], length=cfg['res_length'])

        self.add_readoutconfig(ch=ro_chs, name="myro",
                               freq=cfg['res_freq_ge'],
                               gen_ch=res_ch,
                               outsel='product')
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_chs,
                       style="const",
                       length=cfg["res_length"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

    def _body(self, cfg):
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'], ddr4=True)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)


class ResonanceSpectroscopy:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, round_num, save_figs, experiment = None,
                 verbose = False, logger = None, qick_verbose=True, unmasking_resgain = False):
        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder
        self.expt_name = "res_spec"
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.round_num = round_num
        self.save_figs = save_figs
        self.experiment = experiment

        self.exp_cfg = expt_cfg[self.expt_name]
        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]

        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        if experiment is not None:
            self.q_config = all_qubit_state(experiment, self.number_of_qubits)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} Res Spec configuration: {self.config}')
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} Res Spec configuration: ', self.config)

    def run(self):
        fpts = self.exp_cfg["start"] + self.exp_cfg["step_size"] * np.arange(self.exp_cfg["steps"])
        fcenter = self.config['res_freq_ge']

        amps = []
        for index,f in enumerate(tqdm(fpts)):
            self.config["res_freq_ge"] = fcenter + f
            prog = SingleToneSpectroscopyProgram(self.experiment.soccfg, reps=self.exp_cfg["reps"], final_delay=self.config['relax_delay'], cfg=self.config)
            iq_list = prog.acquire(self.experiment.soc, rounds=self.exp_cfg["rounds"], progress=self.qick_verbose)
            amp = np.abs(iq_list[0][0][0] + 1j * iq_list[0][0][1])
            amps.append(amp)
        amps = np.array(amps)
        res_freqs = self.plot_results(fpts, fcenter, amps) #return freqs from plotting loop so we can use to update experiment

        return res_freqs, fpts, fcenter, amps, self.config

    def plot_results(self, fpts, fcenter, amps, reloaded_config = None, fig_quality = 100):
        res_freqs = []
        plt.figure(figsize=(12, 8))
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
        })


        plt.subplot(2, 3, 1)
        # plt.plot(fpts + fcenter[i], amps[i], '-', linewidth=1.5)
        plt.plot([f + fcenter[0] for f in fpts], amps[0], '-', linewidth=1.5)
        extremum = os.environ.get("PUCQ4_RES_EXTREMUM", "dip").strip().lower()
        if extremum not in {"dip", "peak"}:
            raise ValueError("PUCQ4_RES_EXTREMUM must be 'dip' or 'peak'")
        index = np.argmax(amps) if extremum == "peak" else np.argmin(amps)
        freq_r = fpts[index] + fcenter[0]

        print(freq_r)

        res_freqs.append(freq_r)

        plt.axvline(freq_r, linestyle='--', color='orange', linewidth=1.5)
        plt.title(f"Resonator {self.QubitIndex + 1} {freq_r:.3f} MHz", pad=10)

        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude (a.u.)")

        plt.ylim(plt.ylim()[0] - 0.05 * (plt.ylim()[1] - plt.ylim()[0]), plt.ylim()[1])

        if self.experiment is not None:
            plt.suptitle(f"G resonator spectroscopy {self.config['reps']}*{self.config['rounds']} avgs", fontsize=24, y=0.95)
        else:
            plt.suptitle(f"G resonator spectroscopy {reloaded_config ['reps']}*{reloaded_config ['rounds']} avgs",
                         fontsize=24, y=0.95)
        plt.tight_layout(pad=2.0)

        if self.save_figs:
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name + "_ge_plots")
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name)
            plt.savefig(file_name + ".png", dpi=fig_quality)
            plt.savefig(file_name + ".pdf", dpi=fig_quality)
        plt.close()

        res_freqs = [round(x, 5) for x in res_freqs]
        return res_freqs

    def plot_results(self, fpts, fcenter, amps, reloaded_config=None, fig_quality=100):
        res_freqs = []
        plt.figure(figsize=(12, 8))
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
        })

        plt.subplot(2, 3, 1)
        # plt.plot(fpts + fcenter[i], amps[i], '-', linewidth=1.5)
        plt.plot([f + fcenter for f in fpts], amps, '-', linewidth=1.5)
        extremum = os.environ.get("PUCQ4_RES_EXTREMUM", "dip").strip().lower()
        if extremum not in {"dip", "peak"}:
            raise ValueError("PUCQ4_RES_EXTREMUM must be 'dip' or 'peak'")
        index = np.argmax(amps) if extremum == "peak" else np.argmin(amps)
        freq_r = fpts[index] + fcenter

        print(freq_r)

        res_freqs.append(freq_r)

        plt.axvline(freq_r, linestyle='--', color='orange', linewidth=1.5)
        plt.title(f"Resonator {self.QubitIndex + 1} {freq_r:.3f} MHz", pad=10)

        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude (a.u.)")

        plt.ylim(plt.ylim()[0] - 0.05 * (plt.ylim()[1] - plt.ylim()[0]), plt.ylim()[1])

        if self.experiment is not None:
            plt.suptitle(f"G resonator spectroscopy {self.config['reps']}*{self.config['rounds']} avgs", fontsize=24,
                         y=0.95)
        else:
            plt.suptitle(f"G resonator spectroscopy {reloaded_config['reps']}*{reloaded_config['rounds']} avgs",
                         fontsize=24, y=0.95)
        plt.tight_layout(pad=2.0)

        if self.save_figs:
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name + "_ge_plots")
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt,
                                     f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name)
            plt.savefig(file_name + ".png", dpi=fig_quality)
            plt.savefig(file_name + ".pdf", dpi=fig_quality)
        plt.close()

        res_freqs = [round(x, 5) for x in res_freqs]
        return res_freqs

    def plot_results_reloaded(self, fpts, fcenter, amps, reloaded_config = None, fig_quality = 100):
        res_freqs = []
        plt.figure(figsize=(12, 8))
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
        })


        plt.subplot(2, 3, 1)
        # plt.plot(fpts + fcenter[i], amps[i], '-', linewidth=1.5)
        plt.plot([f + fcenter[0] for f in fpts], amps[0], '-', linewidth=1.5)
        freq_r = fpts[np.argmin(amps)] + fcenter[0]

        print(freq_r)

        res_freqs.append(freq_r)

        plt.axvline(freq_r, linestyle='--', color='orange', linewidth=1.5)
        plt.title(f"Resonator {self.QubitIndex + 1} {freq_r:.3f} MHz", pad=10)

        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude (a.u.)")

        plt.ylim(plt.ylim()[0] - 0.05 * (plt.ylim()[1] - plt.ylim()[0]), plt.ylim()[1])

        if self.experiment is not None:
            plt.suptitle(f"G resonator spectroscopy {self.config['reps']}*{self.config['rounds']} avgs", fontsize=24, y=0.95)
        else:
            plt.suptitle(f"G resonator spectroscopy {reloaded_config ['reps']}*{reloaded_config ['rounds']} avgs",
                         fontsize=24, y=0.95)
        plt.tight_layout(pad=2.0)

        if self.save_figs:
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name + "_ge_plots")
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name)
            plt.savefig(file_name + ".png", dpi=fig_quality)
            plt.savefig(file_name + ".pdf", dpi=fig_quality)
        plt.close()

        res_freqs = [round(x, 5) for x in res_freqs]
        return res_freqs

    def plot_results_overlay(
            self,
            ge_fpts, ge_fcenter, ge_amps, ge_reloaded_config,
            ef_fpts, ef_fcenter, ef_amps, ef_reloaded_config,
            fig_quality=100,
            *,
            label_ge="Qubit in g",
            label_ef="Qubit in e",
            ge_date=None,
            ef_date=None,
            fh_freq_pts=None,
            fh_freq_center=None,
            fh_amps=None,
            fh_cfg=None,
            label_fh="Qubit in f",
            fh_date=None,
    ):
        from scipy.optimize import curve_fit

        plt.rcParams.update({
            "font.size": 16,
            "axes.titlesize": 16,
            "axes.labelsize": 18,
            "xtick.labelsize": 16,
            "ytick.labelsize": 16,
            "legend.fontsize": 16,
        })

        # ---- Lorentzian model (dip): offset - A / (1 + ((f - f0) / (kappa/2))^2) ----
        def lorentzian_dip(f, f0, kappa, A, offset):
            return offset - A / (1.0 + ((f - f0) / (kappa / 2.0)) ** 2)

        def fit_lorentzian(freqs, amps_1d):
            """Fit a Lorentzian dip. Returns (f0, kappa, A, offset, fit_curve) or Nones."""
            freqs = np.asarray(freqs, dtype=float)
            amps_1d = np.asarray(amps_1d, dtype=float)
            try:
                offset0 = np.max(amps_1d)
                idx_min = np.argmin(amps_1d)
                f0_guess = freqs[idx_min]
                A_guess = offset0 - amps_1d[idx_min]
                # guess kappa as FWHM: width where amp drops to half-depth
                half_depth = offset0 - A_guess / 2.0
                below = np.where(amps_1d <= half_depth)[0]
                if len(below) >= 2:
                    kappa_guess = abs(freqs[below[-1]] - freqs[below[0]])
                else:
                    kappa_guess = (freqs[-1] - freqs[0]) / 10.0
                kappa_guess = max(kappa_guess, (freqs[1] - freqs[0]) * 2)  # floor

                p0 = [f0_guess, kappa_guess, A_guess, offset0]
                bounds_lo = [freqs[0], 0, 0, -np.inf]
                bounds_hi = [freqs[-1], freqs[-1] - freqs[0], np.inf, np.inf]

                popt, _ = curve_fit(
                    lorentzian_dip, freqs, amps_1d,
                    p0=p0, bounds=(bounds_lo, bounds_hi), maxfev=10000,
                )
                f0_fit, kappa_fit, A_fit, offset_fit = popt
                fit_curve = lorentzian_dip(freqs, *popt)
                return f0_fit, kappa_fit, A_fit, offset_fit, fit_curve
            except Exception as e:
                print(f"Lorentzian fit failed: {e}")
                return None, None, None, None, None

        fig, ax = plt.subplots(figsize=(14, 10))

        # ---- build x-axes ----
        ge_x = np.array([f + ge_fcenter[0] for f in ge_fpts])
        ef_x = np.array([f + ef_fcenter[0] for f in ef_fpts])

        # ---- plot data ----
        ax.plot(ge_x, ge_amps[0], "-", linewidth=2, label=label_ge, alpha=0.7)
        ax.plot(ef_x, ef_amps[0], "-", linewidth=2, label=label_ef, alpha=0.7)

        fh_x = None
        if fh_freq_pts is not None and fh_freq_center is not None and fh_amps is not None:
            fh_x = np.array([f + fh_freq_center[0] for f in fh_freq_pts])
            ax.plot(fh_x, fh_amps[0], "-", linewidth=2, label=label_fh, alpha=0.7)

        # ---- fit Lorentzians ----
        ge_f0, ge_kappa, _, _, ge_fit = fit_lorentzian(ge_x, ge_amps[0])
        ef_f0, ef_kappa, _, _, ef_fit = fit_lorentzian(ef_x, ef_amps[0])
        fh_f0, fh_kappa, fh_fit = None, None, None
        if fh_x is not None:
            fh_f0, fh_kappa, _, _, fh_fit = fit_lorentzian(fh_x, fh_amps[0])

        # ---- plot fits ----
        if ge_fit is not None:
            ax.plot(ge_x, ge_fit, "--", linewidth=2, color="tab:blue", label=f"{label_ge} fit")
        if ef_fit is not None:
            ax.plot(ef_x, ef_fit, "--", linewidth=2, color="tab:orange", label=f"{label_ef} fit")
        if fh_fit is not None:
            ax.plot(fh_x, fh_fit, "--", linewidth=2, color="tab:green", label=f"{label_fh} fit")

        # ---- resonance markers (from fit if available, else argmin) ----
        ge_freq_r = ge_f0 if ge_f0 is not None else None
        ef_freq_r = ef_f0 if ef_f0 is not None else None
        fh_freq_r = fh_f0 if fh_f0 is not None else None
        if ge_freq_r is None:
            try:
                ge_freq_r = ge_fpts[np.argmin(ge_amps[0])] + ge_fcenter[0]
            except Exception:
                pass
        if ef_freq_r is None:
            try:
                ef_freq_r = ef_fpts[np.argmin(ef_amps[0])] + ef_fcenter[0]
            except Exception:
                pass
        if fh_freq_r is None and fh_x is not None:
            try:
                fh_freq_r = fh_freq_pts[np.argmin(fh_amps[0])] + fh_freq_center[0]
            except Exception:
                pass

        if ge_freq_r is not None:
            ax.axvline(ge_freq_r, linestyle="--", linewidth=1.5, color="tab:blue", alpha=0.5)
        if ef_freq_r is not None:
            ax.axvline(ef_freq_r, linestyle="--", linewidth=1.5, color="tab:orange", alpha=0.5)
        if fh_freq_r is not None:
            ax.axvline(fh_freq_r, linestyle="--", linewidth=1.5, color="tab:green", alpha=0.5)

        # ---- compute chi shifts (MHz → kHz) ----
        chi_ge_ef = (ef_freq_r - ge_freq_r) * 1e3 if (ge_freq_r and ef_freq_r) else None
        chi_ef_fh = (fh_freq_r - ef_freq_r) * 1e3 if (ef_freq_r and fh_freq_r) else None
        chi_ge_fh = (fh_freq_r - ge_freq_r) * 1e3 if (ge_freq_r and fh_freq_r) else None

        # ---- build title ----
        title_lines = [f"Resonator {self.QubitIndex + 1}"]

        # Line 2: resonance frequencies
        freq_parts = []
        if ge_freq_r is not None:
            freq_parts.append(f"g: {ge_freq_r:.4f} MHz")
        if ef_freq_r is not None:
            freq_parts.append(f"e: {ef_freq_r:.4f} MHz")
        if fh_freq_r is not None:
            freq_parts.append(f"f: {fh_freq_r:.4f} MHz")
        if freq_parts:
            title_lines.append("  |  ".join(freq_parts))

        # Line 3: kappa values (in kHz)
        kappa_parts = []
        if ge_kappa is not None:
            kappa_parts.append(f"κ_g: {ge_kappa * 1e3:.1f} kHz")
        if ef_kappa is not None:
            kappa_parts.append(f"κ_e: {ef_kappa * 1e3:.1f} kHz")
        if fh_kappa is not None:
            kappa_parts.append(f"κ_f: {fh_kappa * 1e3:.1f} kHz")
        if kappa_parts:
            title_lines.append("  |  ".join(kappa_parts))

        # Line 4: chi shifts
        chi_parts = []
        if chi_ge_ef is not None:
            chi_parts.append(f"χ(g→e): {chi_ge_ef:+.1f} kHz")
        if chi_ef_fh is not None:
            chi_parts.append(f"χ(e→f): {chi_ef_fh:+.1f} kHz")
        if chi_ge_fh is not None:
            chi_parts.append(f"χ(g→f): {chi_ge_fh:+.1f} kHz")
        if chi_parts:
            title_lines.append("  |  ".join(chi_parts))

        ax.set_title("\n".join(title_lines), pad=14, linespacing=1.6, fontsize=14)

        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Amplitude (a.u.)")
        ax.legend()

        yl = ax.get_ylim()
        ax.set_ylim(yl[0] - 0.05 * (yl[1] - yl[0]), yl[1])

        fig.suptitle("Resonator Spectroscopy Overlay (g vs e vs f)", fontsize=22, y=1.01)

        # footer timestamp
        date_parts = []
        if ge_date:
            date_parts.append(f"ge={ge_date}")
        if ef_date:
            date_parts.append(f"ef={ef_date}")
        if fh_date:
            date_parts.append(f"fh={fh_date}")
        if date_parts:
            fig.text(0.01, 0.01, "paired: " + "  ->  ".join(date_parts), fontsize=11)

        plt.tight_layout(pad=2.5)

        if self.save_figs:
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name + "_ge_ef_fh_plots")
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(
                outerFolder_expt,
                f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{now}_{self.expt_name}_ge_ef_fh_overlay"
            )
            plt.savefig(file_name + ".png", dpi=fig_quality, bbox_inches="tight")
            plt.savefig(file_name + ".pdf", dpi=fig_quality, bbox_inches="tight")

        plt.close()

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def get_results(self, fpts, fcenter, amps):
        res_freqs = []

        for i in range(self.number_of_qubits):
            freq_r = fpts[np.argmin(amps[0])] + fcenter[0]
            res_freqs.append(freq_r)

        res_freqs = [round(x, 7) for x in res_freqs]
        return res_freqs

class PostProcessResonanceSpectroscopy:
    def __init__(self, QubitIndex,  outerFolder, round_num, save_figs, experiment = None):
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.expt_name = "res_spec"
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.round_num = round_num
        self.save_figs = save_figs
        self.experiment = experiment




