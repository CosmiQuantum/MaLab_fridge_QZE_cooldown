from scipy.optimize import curve_fit
from build_task import *
from build_state import *
from expt_config import *
from system_config import *
import matplotlib.pyplot as plt
import numpy as np
import logging

# ═══════════════════════════════════════════════════════════════════════════════
#  QICK Programs — 2D sweeps (qubit_freq inner × res_freq outer)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  The 2D sweep pattern follows PulseProbeSpectroscopyProgram in
#  section_004_qubit_spec_ge_zeno.py:
#     • Inner loop  →  added FIRST  →  qubit pulse frequency
#     • Outer loop  →  added SECOND →  stark tone (resonator drive) frequency
#
#  Output from acquire():
#     iq_list[0][0].T  →  shape [2, n_outer * n_inner]
#     I = iq_list[0][0].T[0].reshape(n_outer, n_inner)   # [res_freq, qubit_freq]
#     Q = iq_list[0][0].T[1].reshape(n_outer, n_inner)
# ═══════════════════════════════════════════════════════════════════════════════

class CKPProgram_g(AveragerProgramV2):
    """CKP qubit spectroscopy — |g⟩ preparation, 2D sweep."""

    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        # ── resonator / readout ──
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        self.declare_readout(ch=cfg['ro_ch'], length=cfg['res_length'])

        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'],
                               gen_ch=res_ch,
                               outsel='product')
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        # Stark tone — frequency swept via outer loop
        self.add_pulse(ch=res_ch, name="stark_tone",
                       style="const",
                       length=cfg['ckp_length'],
                       freq=QickSweep1D("res_freq_loop",
                                        cfg['res_freq_start'],
                                        cfg['res_freq_stop']),
                       phase=cfg['ro_phase'],
                       gain=cfg['ckp_gain']
                       )

        # Normal readout pulse (fixed)
        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        # ── qubit ──
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])

        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'],
                       length=cfg['sigma'] * 4, even_length=False)
        self.add_gauss(ch=qubit_ch, name="ramp_ckz", sigma=cfg['sigma_ckz'],
                       length=cfg['sigma_ckz'] * 4, even_length=False)

        # Qubit probe pulse — frequency swept via inner loop
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp_ckz",
                       freq=QickSweep1D("qubit_pulse_loop",
                                        cfg['qubit_freq_ge'] + cfg["start_freq"],
                                        cfg['qubit_freq_ge'] + cfg["end_freq"]),
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )

        # Inner loop added FIRST, outer loop added SECOND
        self.add_loop("qubit_pulse_loop", cfg["qubit_pulse_steps"])   # inner
        self.add_loop("res_freq_loop", cfg["res_freq_steps"])          # outer

    def _body(self, cfg):
        self.pulse(ch=self.cfg['res_ch'], name="stark_tone", t=0)          # play stark tone
        self.pulse(ch=cfg['qubit_ch'], name="qubit_pulse",
                   t=cfg['qubit_pulse_delay'])                              # play qubit pulse with delay
        self.delay_auto(t=0)
        self.delay_auto(t=cfg['readout_pulse_delay'])                       # wait for ring-down
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


class CKPProgram_e(AveragerProgramV2):
    """CKP qubit spectroscopy — |e⟩ preparation (π pulse first), 2D sweep."""

    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        # ── resonator / readout ──
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        self.declare_readout(ch=cfg['ro_ch'], length=cfg['res_length'])

        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'],
                               gen_ch=res_ch,
                               outsel='product')
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)

        # Stark tone — frequency swept via outer loop
        self.add_pulse(ch=res_ch, name="stark_tone",
                       style="const",
                       length=cfg['ckp_length'],
                       freq=QickSweep1D("res_freq_loop",
                                        cfg['res_freq_start'],
                                        cfg['res_freq_stop']),
                       phase=cfg['ro_phase'],
                       gain=cfg['ckp_gain']
                       )

        # Normal readout pulse (fixed)
        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        # ── qubit ──
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])

        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'],
                       length=cfg['sigma'] * 4, even_length=False)
        self.add_gauss(ch=qubit_ch, name="ramp_ckz", sigma=cfg['sigma_ckz'],
                       length=cfg['sigma_ckz'] * 4, even_length=False)

        # Qubit probe pulse — frequency swept via inner loop
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp_ckz",
                       freq=QickSweep1D("qubit_pulse_loop",
                                        cfg['qubit_freq_ge'] + cfg["start_freq"],
                                        cfg['qubit_freq_ge'] + cfg["end_freq"]),
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )

        # Pi pulse for |e⟩ preparation (fixed freq/gain)
        self.add_pulse(ch=qubit_ch, name="pi_pulse",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )

        # Inner loop added FIRST, outer loop added SECOND
        self.add_loop("qubit_pulse_loop", cfg["qubit_pulse_steps"])   # inner
        self.add_loop("res_freq_loop", cfg["res_freq_steps"])          # outer

    def _body(self, cfg):
        self.pulse(ch=cfg['qubit_ch'], name="pi_pulse")                     # put qubit in |e⟩
        self.delay_auto(t=0)
        self.pulse(ch=self.cfg['res_ch'], name="stark_tone", t=0)          # play stark tone
        self.pulse(ch=cfg['qubit_ch'], name="qubit_pulse",
                   t=cfg['qubit_pulse_delay'])                              # play qubit pulse with delay
        self.delay_auto(t=0)
        self.delay_auto(t=cfg['readout_pulse_delay'])                       # wait for ring-down
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


# ═══════════════════════════════════════════════════════════════════════════════
#  CKP Measurement class
# ═══════════════════════════════════════════════════════════════════════════════

class CKPMeasurement:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, round_num,
                 signal, save_figs, res_freq_ckp, res_phase_ckp,
                 experiment=None, fit_data=None, verbose=False, logger=None,
                 qick_verbose=True):
        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.fit_data = fit_data
        self.expt_name = "ckp_nbar_calibration"
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num
        self.signal = signal
        self.number_of_qubits = number_of_qubits
        self.save_figs = save_figs
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger(
            "custom_logger_for_rr_only")

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment,
                                            self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name,
                                                self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            self.config['res_freq_ckp'] = res_freq_ckp
            self.config['res_phase_ckp'] = res_phase_ckp
            stark_mask = np.arange(0, self.number_of_qubits + 1)
            stark_mask = np.delete(stark_mask, QubitIndex)
            if self.verbose:
                print(f'Q {self.QubitIndex + 1} Round {self.round_num} '
                      f'CKP configuration: ', self.config)
            self.logger.info(
                f'Q {self.QubitIndex + 1} Round {self.round_num} '
                f'CKP configuration:{self.config}')

    # ─────────────────────────────────────────────────────────────────────
    #  run()  —  2D QICK sweep with per-gain SSF calibration
    # ─────────────────────────────────────────────────────────────────────
    def run(self):
        now = datetime.datetime.now()

        from section_005_single_shot_ge import (SingleShotProgram_g,
                                                 SingleShotProgram_e)

        # ── SSF config (used for per-gain calibrations) ──
        q_config = all_qubit_state(self.experiment, self.number_of_qubits)
        ss_exp_cfg = add_qubit_experiment(expt_cfg, 'Readout_Optimization',
                                          self.QubitIndex)
        ss_config = {**q_config[self.Qubit], **ss_exp_cfg}

        # ── Sweep arrays ──
        gain_sweep = np.linspace(self.config["start_gain"],
                                 self.config["end_gain"],
                                 num=self.config["gain_steps"])
        res_freq_sweep = np.linspace(self.config["res_freq_start"],
                                     self.config["res_freq_stop"],
                                     num=self.config["res_freq_steps"])

        # ── CKP config overrides (same as before) ──
        self.config["sigma_ckz"] = 0.12
        self.config["qubit_gain_ge"] = 0.5787
        self.config["ckp_length"] = (self.config["qubit_pulse_delay"]
                                     + self.config["sigma_ckz"] * 4)

        n_res = self.config["res_freq_steps"]
        n_qf = self.config["qubit_pulse_steps"]

        # ── Per-gain containers ──
        I_g_nested = []     # [gain_idx] → shape [n_res, n_qf]
        Q_g_nested = []
        I_e_nested = []
        Q_e_nested = []

        # Per-gain SSF calibration data
        ss_I_g_per_gain = []
        ss_Q_g_per_gain = []
        ss_I_e_per_gain = []
        ss_Q_e_per_gain = []

        qu_freq_sweep = None   # captured once from the first QICK program

        for gi, g in enumerate(gain_sweep):
            self.config['ckp_gain'] = float(np.round(g, 3))
            print(f'\n── CKP gain {gi+1}/{len(gain_sweep)}: '
                  f'{self.config["ckp_gain"]:.3f} ──')

            # ──────────────────────────────────────────────────────────
            #  Per-gain SSF calibration (separate g and e)
            # ──────────────────────────────────────────────────────────
            print('  SSF calibration for |g⟩ …')
            ssp_g = SingleShotProgram_g(
                self.experiment.soccfg, reps=1,
                final_delay=ss_config['relax_delay'], cfg=ss_config)
            iq_list_g_cal = ssp_g.acquire(self.experiment.soc,
                                          rounds=1, progress=True)
            ss_I_g = iq_list_g_cal[0][0].T[0]
            ss_Q_g = iq_list_g_cal[0][0].T[1]

            print('  SSF calibration for |e⟩ …')
            ssp_e = SingleShotProgram_e(
                self.experiment.soccfg, reps=1,
                final_delay=ss_config['relax_delay'], cfg=ss_config)
            iq_list_e_cal = ssp_e.acquire(self.experiment.soc,
                                          rounds=1, progress=True)
            ss_I_e = iq_list_e_cal[0][0].T[0]
            ss_Q_e = iq_list_e_cal[0][0].T[1]

            ss_I_g_per_gain.append(ss_I_g)
            ss_Q_g_per_gain.append(ss_Q_g)
            ss_I_e_per_gain.append(ss_I_e)
            ss_Q_e_per_gain.append(ss_Q_e)

            # ──────────────────────────────────────────────────────────
            #  2D CKP acquisition: |g⟩ panel
            # ──────────────────────────────────────────────────────────
            print('  CKP |g⟩ 2D sweep …')
            ckp_g = CKPProgram_g(
                self.experiment.soccfg,
                reps=self.config['reps'],
                final_delay=self.config['relax_delay'],
                cfg=self.config
            )

            iq_list_g = ckp_g.acquire(
                self.experiment.soc,
                rounds=self.config['rounds'],
                progress=self.qick_verbose
            )
            iqg = iq_list_g[0][0].T
            i_g_2d = iqg[0].reshape(n_res, n_qf)   # [res_freq, qubit_freq]
            q_g_2d = iqg[1].reshape(n_res, n_qf)

            # ──────────────────────────────────────────────────────────
            #  2D CKP acquisition: |e⟩ panel
            # ──────────────────────────────────────────────────────────
            print('  CKP |e⟩ 2D sweep …')
            ckp_e = CKPProgram_e(
                self.experiment.soccfg,
                reps=self.config['reps'],
                final_delay=self.config['relax_delay'],
                cfg=self.config
            )

            iq_list_e = ckp_e.acquire(
                self.experiment.soc,
                rounds=self.config['rounds'],
                progress=self.qick_verbose
            )
            iqe = iq_list_e[0][0].T
            i_e_2d = iqe[0].reshape(n_res, n_qf)
            q_e_2d = iqe[1].reshape(n_res, n_qf)

            # ── Store ──
            I_g_nested.append(i_g_2d)
            Q_g_nested.append(q_g_2d)
            I_e_nested.append(i_e_2d)
            Q_e_nested.append(q_e_2d)

            # Capture qubit frequency sweep array once
            if qu_freq_sweep is None:
                qu_freq_sweep = ckp_g.get_pulse_param(
                    "qubit_pulse", "freq", as_array=True)

        # ── Plotting (same diagnostics as before) ──
        if self.save_figs:
            ro_gain = self.config['res_gain_ge']
            if isinstance(ro_gain, (list, tuple, np.ndarray)):
                ro_gain = ro_gain[self.QubitIndex]
            ro_gain = float(ro_gain)
            gain_index = int(np.argmin(np.abs(gain_sweep - ro_gain)))

            # Use the SSF from the closest-to-readout gain for the fig2 plot
            ref_ss_I_g = ss_I_g_per_gain[gain_index]
            ref_ss_Q_g = ss_Q_g_per_gain[gain_index]
            ref_ss_I_e = ss_I_e_per_gain[gain_index]
            ref_ss_Q_e = ss_Q_e_per_gain[gain_index]

            self.plot_ckp_debug_curves(
                I_g_nested, Q_g_nested, I_e_nested, Q_e_nested,
                qu_freq_sweep, gain_sweep, res_freq_sweep,
                gain_index=gain_index,
                save_path=os.path.join(self.outerFolder,
                                       f"Q{self.QubitIndex + 1}"
                                       f"_ckp_debug_curves.png"),
                show=False
            )

            self.plot_ckp_fig2_style(
                I_g_nested, Q_g_nested, I_e_nested, Q_e_nested,
                qu_freq_sweep, gain_sweep, res_freq_sweep,
                ss_I_g=ref_ss_I_g, ss_Q_g=ref_ss_Q_g,
                ss_I_e=ref_ss_I_e, ss_Q_e=ref_ss_Q_e,
                gain_index=gain_index,
                save_path=os.path.join(
                    self.outerFolder,
                    f"Q{self.QubitIndex + 1}_ckp_fig2_style.png"),
                show=False,
                target_gain=ro_gain
            )

        return (I_g_nested, Q_g_nested, I_e_nested, Q_e_nested,
                qu_freq_sweep, gain_sweep, res_freq_sweep, self.config,
                ss_I_g_per_gain, ss_Q_g_per_gain,
                ss_I_e_per_gain, ss_Q_e_per_gain)

    # ─────────────────────────────────────────────────────────────────────
    #  Helper methods (unchanged from original)
    # ─────────────────────────────────────────────────────────────────────

    def iq_to_pe(self, I, Q, g_ref, e_ref, clip=True):
        """
        Convert IQ data to calibrated excited-state probability using
        projection onto the g→e axis in IQ space.
        """
        Z = np.asarray(I) + 1j * np.asarray(Q)
        denom = np.abs(e_ref - g_ref) ** 2

        if denom == 0:
            pe = np.zeros_like(np.asarray(I), dtype=float)
        else:
            pe = np.real(((Z - g_ref) * np.conj(e_ref - g_ref)) / denom)

        if clip:
            pe = np.clip(pe, 0.0, 1.0)

        return pe

    def extract_peak_centers(self, flip_map, qu_freq_sweep):
        """
        For each resonator-frequency row, find the qubit frequency where
        flip probability is largest. Uses a small quadratic interpolation
        around the max when possible.
        """
        centers = []
        x = np.asarray(qu_freq_sweep, dtype=float)

        for row in np.asarray(flip_map):
            idx = int(np.argmax(row))

            if 0 < idx < len(x) - 1:
                xs = x[idx - 1: idx + 2]
                ys = row[idx - 1: idx + 2]

                try:
                    a, b, c = np.polyfit(xs, ys, 2)
                    if abs(a) > 1e-15:
                        xv = -b / (2 * a)
                        if xs[0] <= xv <= xs[-1]:
                            centers.append(xv)
                        else:
                            centers.append(x[idx])
                    else:
                        centers.append(x[idx])
                except Exception:
                    centers.append(x[idx])
            else:
                centers.append(x[idx])

        return np.array(centers)

    def lorentzian_dip(self, x, x0, depth, width, offset):
        return offset - depth / (1.0 + ((x - x0) / width) ** 2)

    def fit_branch_curve(self, x, y):
        """
        Smooth fit for extracted branch centers vs resonator drive frequency.
        """
        try:
            p0 = [x[np.argmin(y)], np.max(y) - np.min(y), 0.002, np.max(y)]
            popt, _ = curve_fit(self.lorentzian_dip, x, y, p0=p0,
                                maxfev=10000)
            return self.lorentzian_dip(x, *popt)
        except Exception:
            return None

    def get_closest_gain_index(self, gain_sweep):
        """
        Find the CKP sweep gain closest to the normal readout gain
        for this qubit.
        """
        ro_gain = self.config['res_gain_ge']

        # handle either scalar or per-qubit list/array
        if isinstance(ro_gain, (list, tuple, np.ndarray)):
            ro_gain = ro_gain[self.QubitIndex]

        ro_gain = float(ro_gain)
        gain_sweep = np.array(gain_sweep, dtype=float)

        gain_index = int(np.argmin(np.abs(gain_sweep - ro_gain)))
        return gain_index, ro_gain

    def plot_ckp_slice(self, I_g_nested, Q_g_nested, I_e_nested, Q_e_nested,
                       qu_freq_sweep, gain_sweep, res_freq_sweep,
                       gain_index=0, save_path=None, show=False,
                       target_gain=None):
        """
        Plot one CKP slice at a fixed resonator gain.
        x-axis: qubit spectroscopy frequency
        y-axis: resonator CKP drive frequency
        color: magnitude sqrt(I^2 + Q^2)
        """

        import os
        import numpy as np
        import matplotlib.pyplot as plt

        # shape: [res_freq_idx][qubit_freq_idx]
        Ig = np.array(I_g_nested[gain_index])
        Qg = np.array(Q_g_nested[gain_index])
        Ie = np.array(I_e_nested[gain_index])
        Qe = np.array(Q_e_nested[gain_index])

        mag_g = np.sqrt(Ig ** 2 + Qg ** 2)
        mag_e = np.sqrt(Ie ** 2 + Qe ** 2)
        mag_diff = mag_e - mag_g

        fig, axes = plt.subplots(1, 3, figsize=(16, 4),
                                 constrained_layout=True)

        extent = [
            qu_freq_sweep[0], qu_freq_sweep[-1],
            res_freq_sweep[0], res_freq_sweep[-1]
        ]

        im0 = axes[0].imshow(
            mag_g,
            aspect='auto',
            origin='lower',
            extent=extent
        )
        axes[0].set_title(f'|g⟩ prep, gain={gain_sweep[gain_index]:.3f}')
        axes[0].set_xlabel('Qubit probe frequency')
        axes[0].set_ylabel('Resonator CKP frequency')
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(
            mag_e,
            aspect='auto',
            origin='lower',
            extent=extent
        )
        axes[1].set_title(f'|e⟩ prep, gain={gain_sweep[gain_index]:.3f}')
        axes[1].set_xlabel('Qubit probe frequency')
        axes[1].set_ylabel('Resonator CKP frequency')
        fig.colorbar(im1, ax=axes[1])

        im2 = axes[2].imshow(
            mag_diff,
            aspect='auto',
            origin='lower',
            extent=extent
        )
        axes[2].set_title('|e|-|g| contrast')
        axes[2].set_xlabel('Qubit probe frequency')
        axes[2].set_ylabel('Resonator CKP frequency')
        fig.colorbar(im2, ax=axes[2])

        selected_gain = gain_sweep[gain_index]

        if target_gain is None:
            gain_label = f"{selected_gain:.3f}"
        else:
            gain_label = (f"{selected_gain:.3f} "
                          f"(closest to ro gain {target_gain:.3f})")

        axes[0].set_title(f'|g⟩ prep, gain={gain_label}')
        axes[1].set_title(f'|e⟩ prep, gain={gain_label}')
        axes[2].set_title('|e|-|g| contrast')

        if save_path is not None:
            folder = os.path.dirname(save_path)
            if folder:
                self.create_folder_if_not_exists(folder)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_ckp_fig2_style(self, I_g_nested, Q_g_nested,
                            I_e_nested, Q_e_nested,
                            qu_freq_sweep, gain_sweep, res_freq_sweep,
                            ss_I_g, ss_Q_g, ss_I_e, ss_Q_e,
                            gain_index=0, save_path=None, show=False,
                            target_gain=None):

        import os
        import numpy as np
        import matplotlib.pyplot as plt

        # fixed-gain slice: shape [res_freq_idx, qubit_freq_idx]
        Ig = np.array(I_g_nested[gain_index], dtype=float)
        Qg = np.array(Q_g_nested[gain_index], dtype=float)
        Ie = np.array(I_e_nested[gain_index], dtype=float)
        Qe = np.array(Q_e_nested[gain_index], dtype=float)

        # single-shot calibration points
        g_ref = np.mean(np.asarray(ss_I_g) + 1j * np.asarray(ss_Q_g))
        e_ref = np.mean(np.asarray(ss_I_e) + 1j * np.asarray(ss_Q_e))

        # calibrated excited-state probability
        pe_gprep = self.iq_to_pe(Ig, Qg, g_ref, e_ref, clip=True)
        pe_eprep = self.iq_to_pe(Ie, Qe, g_ref, e_ref, clip=True)

        # convert to flip probability
        # start in |0⟩ → flip means ending in |1⟩
        flip_g = pe_gprep

        # start in |1⟩ → flip means ending in |0⟩
        flip_e = 1.0 - pe_eprep
        flip_e = np.clip(flip_e, 0.0, 1.0)

        # extract branch centers row-by-row
        g_centers = self.extract_peak_centers(flip_g, qu_freq_sweep)
        e_centers = self.extract_peak_centers(flip_e, qu_freq_sweep)

        # smooth curves
        g_fit = self.fit_branch_curve(res_freq_sweep, g_centers)
        e_fit = self.fit_branch_curve(res_freq_sweep, e_centers)

        fig = plt.figure(figsize=(7.0, 8.6))
        gs = fig.add_gridspec(
            4, 1,
            height_ratios=[0.16, 1.0, 1.0, 0.9],
            hspace=0.12
        )

        cax = fig.add_subplot(gs[0])
        ax0 = fig.add_subplot(gs[1])
        ax1 = fig.add_subplot(gs[2], sharex=ax0)
        ax2 = fig.add_subplot(gs[3], sharex=ax0)

        extent = [
            res_freq_sweep[0], res_freq_sweep[-1],
            qu_freq_sweep[0], qu_freq_sweep[-1]
        ]

        # top panel: |0⟩

        ax0.plot(res_freq_sweep, g_centers, 'o', ms=4, mfc='white',
                 mec='white')
        ax0.text(0.16, 0.18, r'$|0\rangle$', transform=ax0.transAxes,
                 fontsize=18, color='blue')

        X, Y = np.meshgrid(res_freq_sweep, qu_freq_sweep)

        im0 = ax0.pcolormesh(
            X, Y, flip_g.T,
            shading='nearest',
            vmin=0.0,
            vmax=1.0,
            cmap='viridis_r'
        )

        im1 = ax1.pcolormesh(
            X, Y, flip_e.T,
            shading='nearest',
            vmin=0.0,
            vmax=1.0,
            cmap='viridis_r'
        )

        ax1.plot(res_freq_sweep, e_centers, 'o', ms=4, mfc='white',
                 mec='white')
        ax1.text(0.16, 0.18, r'$|1\rangle$', transform=ax1.transAxes,
                 fontsize=18, color='red')

        # bottom panel: extracted branches
        ax2.plot(res_freq_sweep, g_centers, 'o-', lw=2, ms=4, color='blue',
                 label=r'$|0\rangle$')
        ax2.plot(res_freq_sweep, e_centers, 'o-', lw=2, ms=4, color='red',
                 label=r'$|1\rangle$')
        ax2.legend(loc='lower left', fontsize=12)

        # labels
        ax1.set_ylabel("Qubit pulse frequency")
        ax2.set_ylabel("Qubit pulse frequency")
        ax2.set_xlabel("Resonator drive frequency")

        plt.setp(ax0.get_xticklabels(), visible=False)
        plt.setp(ax1.get_xticklabels(), visible=False)

        # shared colorbar on top
        cbar = fig.colorbar(im0, cax=cax, orientation='horizontal')
        cax.invert_xaxis()
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        cbar.set_label("Flip probability")

        selected_gain = gain_sweep[gain_index]
        if target_gain is None:
            fig.suptitle(f"CKP slice at gain = {selected_gain:.3f}",
                         y=0.995)
        else:
            fig.suptitle(
                f"CKP slice at gain = {selected_gain:.3f} "
                f"(closest to ro gain {target_gain:.3f})",
                y=0.995
            )

        if save_path is not None:
            folder = os.path.dirname(save_path)
            if folder:
                self.create_folder_if_not_exists(folder)
            fig.savefig(save_path, dpi=200, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

    def plot_ckp_debug_curves(self, I_g_nested, Q_g_nested,
                              I_e_nested, Q_e_nested,
                              qu_freq_sweep, gain_sweep, res_freq_sweep,
                              gain_index=0, res_indices=None,
                              save_path=None, show=False):
        import os
        import numpy as np
        import matplotlib.pyplot as plt

        if res_indices is None:
            n = len(res_freq_sweep)
            res_indices = sorted(set([0, n // 2, n - 1]))

        fig, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
        axes = np.array(axes)

        for ridx in res_indices:
            Ig = np.array(I_g_nested[gain_index][ridx], dtype=float)
            Qg = np.array(Q_g_nested[gain_index][ridx], dtype=float)
            Ie = np.array(I_e_nested[gain_index][ridx], dtype=float)
            Qe = np.array(Q_e_nested[gain_index][ridx], dtype=float)

            Mg = np.sqrt(Ig ** 2 + Qg ** 2)
            Me = np.sqrt(Ie ** 2 + Qe ** 2)

            label = f"f_res={res_freq_sweep[ridx]:.4f}"

            axes[0, 0].plot(qu_freq_sweep, Ig, label=label)
            axes[0, 1].plot(qu_freq_sweep, Qg, label=label)
            axes[0, 2].plot(qu_freq_sweep, Mg, label=label)

            axes[1, 0].plot(qu_freq_sweep, Ie, label=label)
            axes[1, 1].plot(qu_freq_sweep, Qe, label=label)
            axes[1, 2].plot(qu_freq_sweep, Me, label=label)

        axes[0, 0].set_title("|0⟩ prep: I")
        axes[0, 1].set_title("|0⟩ prep: Q")
        axes[0, 2].set_title("|0⟩ prep: |IQ|")
        axes[1, 0].set_title("|1⟩ prep: I")
        axes[1, 1].set_title("|1⟩ prep: Q")
        axes[1, 2].set_title("|1⟩ prep: |IQ|")

        for ax in axes.ravel():
            ax.set_xlabel("Qubit probe frequency")
            ax.legend(fontsize=8)

        plt.tight_layout()

        if save_path is not None:
            folder = os.path.dirname(save_path)
            if folder:
                self.create_folder_if_not_exists(folder)
            plt.savefig(save_path, dpi=200, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

    def set_res_gain_ge(self, QUBIT_INDEX, num_qubits=6):
        """Sets the gain for the selected qubit to 1, others to 0."""
        res_gain_ge = [0] * num_qubits  # Initialize all gains to 0
        if 0 <= QUBIT_INDEX < num_qubits:
            res_gain_ge[QUBIT_INDEX] = 1
        return res_gain_ge

    def exponential(self, x, a, b, c, d):
        return a * np.exp(-(x - b) / c) + d

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)