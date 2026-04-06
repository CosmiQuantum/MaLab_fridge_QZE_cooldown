
import datetime
import numpy as np
import logging
np.set_printoptions(threshold=1000000000000000)
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import math
import h5py
# Assuming these are defined elsewhere and importable
from build_task import *
from build_state import *
from expt_config import *
from system_config import QICK_experiment
import copy
import os

# Both g and e during the same experiment.
class SingleShotProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_chs = cfg['ro_chs']
        gen_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']
        self.declare_gen(ch=gen_ch, nqz=cfg['nqz_res'])
        self.declare_readout(ch=cfg['ro_ch'], length=cfg['res_length'])

        self.add_readoutconfig(ch=ro_chs, name="myro",
                               freq=cfg['res_freq_ge'],
                               gen_ch=gen_ch,
                               outsel='product')
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.declare_readout(ch=cfg['ro_ch'], length=cfg['res_length'])
        self.add_pulse(ch=gen_ch, name="res_pulse", ro_ch=ro_chs,
                       style="const",
                       length=cfg["res_length"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])

        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 5, even_length=True)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_chs[0],
                       style="arb",
                       envelope="ramp",
                       freq=cfg['f_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_gain'],
                       )

        #         self.add_loop("shotloop", cfg["steps"]) # number of total shots
        self.add_loop("gainloop", cfg["expts"])  # Pulse / no Pulse loop

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play pulse
        self.delay_auto(0.01)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # play probe pulse
        self.trigger(ros=cfg['ro_chs'], pins=[0], t=cfg['trig_time'])


# Separate g and e per each experiment defined.
class SingleShotProgram_g(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_chs = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

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

        self.add_loop("shotloop", cfg["steps"])  # number of total shots

    def _body(self, cfg):
        self.delay_auto(0.01)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # play probe pulse
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])
        # relax delay ...


class SingleShotProgram_e(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        self.declare_readout(ch=cfg['ro_ch'], length=cfg['res_length'])

        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'],
                               gen_ch=res_ch,
                               outsel='product')
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])

        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)

        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )
        print('cfg[pi_amp]',cfg['pi_amp'])
        print('cfg[qubit_freq_ge]', cfg['qubit_freq_ge'])
        print('cfg[res_freq_ge]', cfg['res_freq_ge'])
        print('cfg[res_gain_ge]', cfg['res_gain_ge'])
        self.add_loop("shotloop", cfg["steps"])  # number of total shots

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play pulse
        self.delay_auto(0.0)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)  # play probe pulse
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class SingleShot:
    def __init__(self, QubitIndex, number_of_qubits,  outerFolder, round_num, save_figs=False, experiment = None,
                 verbose = False, logger = None, qick_verbose=True, unmasking_resgain = False, long_readout=False):
        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        if long_readout:
            self.expt_name = "Readout_Optimization_300k"
        else:
            self.expt_name = "Readout_Optimization"
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.round_num = round_num
        self.save_figs = save_figs
        self.experiment = experiment
        self.number_of_qubits = number_of_qubits
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")
        self.exp_cfg = expt_cfg[self.expt_name]

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} Single Shot configuration: ', self.config)
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} Single Shot configuration: {self.config}')

        self.q1_t1 = []
        self.q1_t1_err = []
        self.dates = []


    def fidelity_test(self, soccfg, soc):
        # Run the single shot programs (g and e)
        ssp_g = SingleShotProgram_g(soccfg,  reps=1, final_delay=self.config['relax_delay'],
                                    cfg=self.config)
        iq_list_g = ssp_g.acquire(soc, rounds=1, progress=False)

        ssp_e = SingleShotProgram_e(soccfg,  reps=1, final_delay=self.config['relax_delay'],
                                    cfg=self.config)
        iq_list_e = ssp_e.acquire(soc, rounds=1, progress=False)

        # Use the fidelity calculation from SingleShotGE
        fidelity, _, _, _,_ = self.hist_ssf(
            data=[iq_list_g[0][0].T[0], iq_list_g[0][0].T[1],
                  iq_list_e[0][0].T[0], iq_list_e[0][0].T[1]],
            cfg=self.config, plot=False)

        return fidelity

    def run(self,return_thres=False):
        ssp_g = SingleShotProgram_g(self.experiment.soccfg, reps=1, final_delay=self.config['relax_delay'], cfg=self.config)
        iq_list_g = ssp_g.acquire(self.experiment.soc, rounds=1, progress=True)

        ssp_e = SingleShotProgram_e(self.experiment.soccfg, reps=1, final_delay=self.config['relax_delay'], cfg=self.config)
        iq_list_e = ssp_e.acquire(self.experiment.soc, rounds=1, progress=True)

        if return_thres:
            raw_g = ssp_g.get_raw()
            raw_e = ssp_e.get_raw()
            #thresh = int((np.mean(raw_g[0][:, :, 0]) + np.mean(raw_e[0][:, :, 0])) / 2)
            #fid, angle = self.plot_results(iq_list_g, iq_list_e, self.QubitIndex)
            fid, angle, thresh = self.plot_results(iq_list_g, iq_list_e, self.QubitIndex, return_thres=return_thres)

            return fid, angle, iq_list_g, iq_list_e, self.config, thresh
        else:
            fid, angle = self.plot_results(iq_list_g, iq_list_e, self.QubitIndex)
            return fid, angle, iq_list_g, iq_list_e, self.config

    def plot_results(self, iq_list_g, iq_list_e, QubitIndex,  fig_quality=100, return_thres=False):
        I_g = iq_list_g[0][0].T[0]
        Q_g = iq_list_g[0][0].T[1]
        I_e = iq_list_e[0][0].T[0]
        Q_e = iq_list_e[0][0].T[1]
        # I_g = iq_list_g[self.QubitIndex][:, :, 0, 0][0]
        # Q_g = iq_list_g[self.QubitIndex][:, :, 0, 1][0]
        # I_e = iq_list_e[self.QubitIndex][:, :, 0, 0][0]
        # Q_e = iq_list_e[self.QubitIndex][:, :, 0, 1][0]


        fid, threshold, angle, ig_new, ie_new = self.hist_ssf(data=[I_g, Q_g, I_e, Q_e], cfg=self.config, plot=self.save_figs,  fig_quality=fig_quality)
        if self.verbose: print('Optimal fidelity after rotation = %.3f' % fid)
        if self.verbose: print('Optimal angle after rotation = %f' % angle)
        self.logger.info('Optimal fidelity after rotation = %.3f' % fid)
        self.logger.info('Optimal angle after rotation = %f' % angle)
        if return_thres:
            return fid, angle, threshold
        else:
            return fid, angle

    def hist_ssf(self, data=None, cfg=None, plot=True,  fig_quality = 100):

        ig = data[0]
        qg = data[1]
        ie = data[2]
        qe = data[3]

        numbins = round(math.sqrt(float(cfg["steps"])))

        xg, yg = np.median(ig), np.median(qg)
        xe, ye = np.median(ie), np.median(qe)

        if plot == True:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
            fig.tight_layout()

            axs[0].scatter(ig, qg, label='g', color='b', marker='*', alpha=0.2)
            axs[0].scatter(ie, qe, label='e', color='r', marker='*', alpha=0.2)
            axs[0].scatter(xg, yg, color='k', marker='o')
            axs[0].scatter(xe, ye, color='k', marker='o')
            axs[0].set_xlabel('I (a.u.)')
            axs[0].set_ylabel('Q (a.u.)')
            axs[0].legend(loc='upper right')
            axs[0].set_title('Unrotated')
            axs[0].axis('equal')
        """Compute the rotation angle"""
        theta = -np.arctan2((ye - yg), (xe - xg))
        """Rotate the IQ data"""
        ig_new = ig * np.cos(theta) - qg * np.sin(theta)
        qg_new = ig * np.sin(theta) + qg * np.cos(theta)
        ie_new = ie * np.cos(theta) - qe * np.sin(theta)
        qe_new = ie * np.sin(theta) + qe * np.cos(theta)

        """New means of each blob"""
        xg, yg = np.median(ig_new), np.median(qg_new)
        xe, ye = np.median(ie_new), np.median(qe_new)

        # print(xg, xe)
        #xlims = [xg - ran, xg + ran]
        xlims = [np.min(ig_new), np.max(ie_new)]

        if plot == True:
            axs[1].scatter(ig_new, qg_new, label='g', color='b', marker='*', alpha=0.2)
            axs[1].scatter(ie_new, qe_new, label='e', color='r', marker='*', alpha=0.2)
            axs[1].scatter(xg, yg, color='k', marker='o')
            axs[1].scatter(xe, ye, color='k', marker='o')
            axs[1].set_xlabel('I (a.u.)')
            axs[1].legend(loc='lower right')
            axs[1].set_title(f'Rotated Theta:{round(theta, 5)}')
            axs[1].axis('equal')

            """X and Y ranges for histogram"""
            ng, binsg, pg = axs[2].hist(ig_new, bins=numbins, range=xlims, color='b', label='g', alpha=0.2)
            ne, binse, pe = axs[2].hist(ie_new, bins=numbins, range=xlims, color='r', label='e', alpha=0.2)

            axs[2].set_xlabel('I(a.u.)')
        else:
            ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
            ne, binse = np.histogram(ie_new, bins=numbins, range=xlims)

        """Compute the fidelity using overlap of the histograms"""
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
        tind = contrast.argmax()
        threshold = binsg[tind]
        fid = contrast[tind]
        #axs[2].set_title(f"Fidelity = {fid * 100:.2f}%")


        if plot == True:
            self.create_folder_if_not_exists(self.outerFolder)
            outerFolder_expt = os.path.join(self.outerFolder, "ss_repeat_meas_ge")
            self.create_folder_if_not_exists(outerFolder_expt)
            outerFolder_expt = os.path.join(outerFolder_expt, "Q" + str(self.QubitIndex + 1))
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt,
                                     f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + self.expt_name+ f"{formatted_datetime}_"  + f"_q{self.QubitIndex + 1}.png")

            axs[2].set_title(f"Fidelity = {fid * 100:.2f}%")
            fig.savefig(file_name,  dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)

        return fid, threshold, theta, ig_new, ie_new

    def hist_ssf_with_annotations_new_method(self, data=None, cfg=None, plot=True, fig_quality=100, I_meas=None, Q_meas=None):
        """
        Visualizes the calibration:
            z = I + 1j*Q
            e = mean(Ie + 1j*Qe)
            g = mean(Ig + 1j*Qg)
            pop_lin = Re(((z - g) * conj(e - g)) / |e - g|^2)
            amp = pop_lin (usually clipped to [0,1] for display)

        Plots g/e clouds, centroids (means), g->e axis, g->z vector,
        the projection point z_proj on the g->e line, and the perpendicular
        from z to z_proj.
        """
        import os, math, datetime
        import numpy as np
        import matplotlib.pyplot as plt

        ig = np.asarray(data[0])
        qg = np.asarray(data[1])
        ie = np.asarray(data[2])
        qe = np.asarray(data[3])

        numbins = round(math.sqrt(float(cfg["steps"])))

        # --- Centroids (MEANS, as in your formula) ---
        g_c = np.mean(ig + 1j * qg)
        e_c = np.mean(ie + 1j * qe)

        # If no measurement provided, pick a sample from e cloud
        if (I_meas is None) or (Q_meas is None):
            z_ix = len(ie) // 2
            I_meas = float(ie[z_ix])
            Q_meas = float(qe[z_ix])

        # --- Complex forms ---
        z_c = complex(I_meas, Q_meas)
        eg = e_c - g_c
        zg = z_c - g_c

        # --- Your calibration mapping (linear projection along g->e) ---
        pop_lin = float(np.real(zg * np.conj(eg)) / (np.abs(eg) ** 2 + 1e-12))
        amp = float(np.clip(pop_lin, 0.0, 1.0))  # clip for display

        # --- Projection point on the line g->e ---
        z_proj = g_c + pop_lin * eg
        perp_vec = z_c - z_proj  # purely perpendicular component to the g->e axis

        if plot:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(17, 5))
            fig.tight_layout()

            # -------- Panel 0: Unrotated with full geometry --------
            axs[0].scatter(ig, qg, label='g shots', color='b', marker='*', alpha=0.2)
            axs[0].scatter(ie, qe, label='e shots', color='r', marker='*', alpha=0.2)

            axs[0].scatter([g_c.real], [g_c.imag], color='k', marker='X', s=100, label='g mean')
            axs[0].scatter([e_c.real], [e_c.imag], color='k', marker='X', s=100, label='e mean')

            # g->e axis (dashed)
            axs[0].plot([g_c.real, e_c.real], [g_c.imag, e_c.imag], linestyle='--', linewidth=2, label='g → e')

            # g->z arrow
            axs[0].arrow(g_c.real, g_c.imag, (z_c - g_c).real, (z_c - g_c).imag,
                         length_includes_head=True, head_width=0.04, alpha=0.9)
            axs[0].scatter([z_c.real], [z_c.imag], s=80, label='measured z')
            axs[0].annotate("z", xy=(z_c.real, z_c.imag), xytext=(z_c.real + 0.05, z_c.imag + 0.05))

            # projection point and perpendicular
            axs[0].scatter([z_proj.real], [z_proj.imag], s=70, marker='D', label='z_proj (projection)')
            axs[0].plot([z_c.real, z_proj.real], [z_c.imag, z_proj.imag], linestyle=':', linewidth=2,
                        label='⊥ from z to axis')

            # Info box
            axs[0].text(0.02, 0.02,
                        "Calibration:\n"
                        "pop_lin = Re(((z−g)·conj(e−g)) / |e−g|²)\n"
                        f"pop_lin = {pop_lin:.3f}\n"
                        f"amp (clipped) = {amp:.3f}",
                        transform=axs[0].transAxes)
            axs[0].set_xlabel('I (a.u.)')
            axs[0].set_ylabel('Q (a.u.)')
            axs[0].set_title('Unrotated (projection geometry)')
            axs[0].legend(loc='upper right')
            axs[0].axis('equal')

            # -------- Rotation so g->e is horizontal (for intuition) --------
            theta = -np.angle(eg)  # same as -arctan2(Im(eg), Re(eg))
            rot = np.exp(1j * theta)

            ig_r = (ig + 1j * qg) * rot
            ie_r = (ie + 1j * qe) * rot
            g_r = g_c * rot
            e_r = e_c * rot
            z_r = z_c * rot
            zproj_r = z_proj * rot

            # Panel 1: Rotated with same geometry
            axs[1].scatter(ig_r.real, ig_r.imag, label='g shots', color='b', marker='*', alpha=0.2)
            axs[1].scatter(ie_r.real, ie_r.imag, label='e shots', color='r', marker='*', alpha=0.2)
            axs[1].scatter([g_r.real], [g_r.imag], color='k', marker='X', s=100, label='g mean (rot)')
            axs[1].scatter([e_r.real], [e_r.imag], color='k', marker='X', s=100, label='e mean (rot)')

            # g->e axis ~ horizontal
            axs[1].plot([g_r.real, e_r.real], [g_r.imag, e_r.imag], linestyle='--', linewidth=2, label='g → e')

            # g->z vector, z point
            axs[1].arrow(g_r.real, g_r.imag, (z_r - g_r).real, (z_r - g_r).imag,
                         length_includes_head=True, head_width=0.04, alpha=0.9)
            axs[1].scatter([z_r.real], [z_r.imag], s=80, label='z (rot)')
            axs[1].annotate("z", xy=(z_r.real, z_r.imag), xytext=(z_r.real + 0.05, z_r.imag + 0.05))

            # projection point and perpendicular
            axs[1].scatter([zproj_r.real], [zproj_r.imag], s=70, marker='D', label='z_proj (rot)')
            axs[1].plot([z_r.real, zproj_r.real], [z_r.imag, zproj_r.imag], linestyle=':', linewidth=2,
                        label='⊥ to axis')

            # In the rotated frame, pop_lin equals the fraction along x between g and e.
            # Show this explicitly:
            eg_len = np.abs(e_r - g_r)
            x_frac = float((zproj_r.real - g_r.real) / (eg_len + 1e-12))
            axs[1].text(0.02, 0.02,
                        f"Rotated view:\n"
                        f"(z_proj.x − g.x) / |e−g| = {x_frac:.3f}\n"
                        f"pop_lin = {pop_lin:.3f}",
                        transform=axs[1].transAxes)

            axs[1].set_xlabel('I (rot)')
            axs[1].set_title(f'Rotated (θ = {np.degrees(theta):.2f}°)')
            axs[1].legend(loc='lower right')
            axs[1].axis('equal')

            # -------- Panel 2: Histogram of rotated I (as before) --------
            # Use real parts of rotated clouds for 1D discrimination view
            ig_new = ig_r.real
            ie_new = ie_r.real
            xlims = [min(np.min(ig_new), np.min(ie_new)), max(np.max(ig_new), np.max(ie_new))]

            ng, binsg, _ = axs[2].hist(ig_new, bins=numbins, range=xlims, color='b', label='g', alpha=0.2)
            ne, binse, _ = axs[2].hist(ie_new, bins=numbins, range=xlims, color='r', label='e', alpha=0.2)

            # Fidelity via histogram overlap (unchanged)
            contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
            tind = contrast.argmax()
            threshold = binsg[tind]
            fid = contrast[tind]

            #axs[2].axvline(threshold, linestyle='--', linewidth=1.5)
            axs[2].set_title(f"Fidelity = {fid * 100:.2f}%")
            axs[2].set_xlabel('I (rot, a.u.)')
            axs[2].legend(loc='upper right')

            # ---- Save like before ----
            self.create_folder_if_not_exists(self.outerFolder)
            out_dir = os.path.join(self.outerFolder, "ss_repeat_meas_ge")
            self.create_folder_if_not_exists(out_dir)
            out_dir = os.path.join(out_dir, "Q" + str(self.QubitIndex + 1))
            self.create_folder_if_not_exists(out_dir)
            out_dir = os.path.join(out_dir, "new_method_samples_at_end")
            self.create_folder_if_not_exists(out_dir)
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(
                out_dir,
                f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{self.expt_name}{now}_q{self.QubitIndex + 1}.png"
            )
            plt.tight_layout()
            fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)

            return fid, threshold, float(theta), ig_new, ie_new
        else:
            # non-plot path still computes fid/threshold on rotated-I
            theta = -np.angle(eg)
            rot = np.exp(1j * theta)
            ig_r = (ig + 1j * qg) * rot
            ie_r = (ie + 1j * qe) * rot
            ig_new = ig_r.real
            ie_new = ie_r.real
            xlims = [min(np.min(ig_new), np.min(ie_new)), max(np.max(ig_new), np.max(ie_new))]
            ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
            ne, binse = np.histogram(ie_new, bins=numbins, range=xlims)
            contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
            tind = contrast.argmax()
            threshold = binsg[tind]
            fid = contrast[tind]
            return fid, threshold, float(theta), ig_new, ie_new


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

    def hist_ssf_with_annotations(self, data=None, cfg=None, plot=True, fig_quality=100, I_meas=None, Q_meas=None,
                                  path_ext='', ):
        """
        Plots g/e calibration clouds, shows their *means*, rotates the IQ plane,
        and overlays vectors g->e and g->z for a single experiment point (I_meas, Q_meas).

        Adds: dashed vertical lines on the rotated-I histogram at the g/e mean I-locations.

        Returns:
            fid, threshold, theta, ig_new, ie_new
        """
        import os, math, datetime
        import numpy as np
        import matplotlib.pyplot as plt

        ig = data[0]
        qg = data[1]
        ie = data[2]
        qe = data[3]


        n_total = len(ig) + len(ie)  # use total sample count
        numbins = max(1, int(math.ceil(math.log2(n_total) + 1)))

        # # Use means for the centroids (requested)
        # gx_mean, gy_mean = float(np.mean(ig)), float(np.mean(qg))
        # ex_mean, ey_mean = float(np.mean(ie)), float(np.mean(qe))
        #
        # # (Optional) medians if you still want to compare visually
        # gx_med, gy_med = float(np.median(ig)), float(np.median(qg))
        # ex_med, ey_med = float(np.median(ie)), float(np.median(qe))

        # Use robust centers for the centroids
        gc = self.robust_center(ig + 1j * qg)  # complex
        ec = self.robust_center(ie + 1j * qe)  # complex
        gx_mean, gy_mean = float(np.real(gc)), float(np.imag(gc))
        ex_mean, ey_mean = float(np.real(ec)), float(np.imag(ec))

        # (Optional) medians if you still want to compare visually
        gx_med, gy_med = float(np.median(ig)), float(np.median(qg))
        ex_med, ey_med = float(np.median(ie)), float(np.median(qe))

        if plot:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4))
            fig.tight_layout()

            # ------- Unrotated -------
            axs[0].scatter(ig, qg, label='g shots', color='b', marker='*', alpha=0.2)
            axs[0].scatter(ie, qe, label='e shots', color='r', marker='*', alpha=0.2)

            axs[0].scatter([gx_mean], [gy_mean], color='k', marker='X', s=80, label='g mean')
            axs[0].scatter([ex_mean], [ey_mean], color='k', marker='X', s=80, label='e mean')

            # tiny dots for medians (optional)
            axs[0].scatter([gx_med], [gy_med], color='k', marker='o', s=20, alpha=0.6, label='g median')
            axs[0].scatter([ex_med], [ey_med], color='k', marker='o', s=20, alpha=0.6, label='e median')

            axs[0].plot([gx_mean, ex_mean], [gy_mean, ey_mean], linestyle='--', linewidth=2, label='g → e')

            # If user supplied a measurement, plot it (unrotated)
            if (I_meas is not None) and (Q_meas is not None):
                axs[0].scatter([I_meas], [Q_meas], s=80, label='meas z (raw)')
                axs[0].annotate("z", xy=(I_meas, Q_meas), xytext=(I_meas + 0.05, Q_meas + 0.05))

            axs[0].set_xlabel('I (a.u.)')
            axs[0].set_ylabel('Q (a.u.)')
            axs[0].legend(loc='upper right')
            axs[0].set_title('Unrotated')
            axs[0].axis('equal')

        # ------- Rotation so g->e is horizontal (use means) -------
        theta = -np.arctan2((ey_mean - gy_mean), (ex_mean - gx_mean))
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        # Rotate calibration clouds
        ig_new = ig * cos_t - qg * sin_t
        qg_new = ig * sin_t + qg * cos_t
        ie_new = ie * cos_t - qe * sin_t
        qe_new = ie * sin_t + qe * cos_t

        # Rotate centroids
        gx_mean_r = gx_mean * cos_t - gy_mean * sin_t
        gy_mean_r = gx_mean * sin_t + gy_mean * cos_t
        ex_mean_r = ex_mean * cos_t - ey_mean * sin_t
        ey_mean_r = ex_mean * sin_t + ey_mean * cos_t

        # ----- Rotate the single experiment point if provided -----
        if (I_meas is not None) and (Q_meas is not None):
            zI_r = I_meas * cos_t - Q_meas * sin_t
            zQ_r = I_meas * sin_t + Q_meas * cos_t
        else:
            # fallback: pick a sample from e cloud (rotated)
            z_ix = len(ie_new) // 2
            zI_r, zQ_r = float(ie_new[z_ix]), float(qe_new[z_ix])

        # X range for hist
        xlims = [min(np.min(ig_new), np.min(ie_new)), max(np.max(ig_new), np.max(ie_new))]

        if plot:
            # ------- Rotated scatter & vectors -------
            axs[1].scatter(ig_new, qg_new, label='g shots', color='b', marker='*', alpha=0.2)
            axs[1].scatter(ie_new, qe_new, label='e shots', color='r', marker='*', alpha=0.2)
            axs[1].scatter([gx_mean_r], [gy_mean_r], color='k', marker='X', s=80, label='g mean (rot)')
            axs[1].scatter([ex_mean_r], [ey_mean_r], color='k', marker='X', s=80, label='e mean (rot)')

            # g->e vector
            axs[1].plot([gx_mean_r, ex_mean_r], [gy_mean_r, ey_mean_r], linestyle='--', linewidth=2, label='g → e')

            # g->z vector and z point
            axs[1].arrow(gx_mean_r, gy_mean_r, (zI_r - gx_mean_r), (zQ_r - gy_mean_r),
                         length_includes_head=True, head_width=0.04, alpha=0.9)
            axs[1].scatter([zI_r], [zQ_r], s=80, label='meas z (rot)')
            axs[1].annotate("z", xy=(zI_r, zQ_r), xytext=(zI_r + 0.05, zQ_r + 0.05))

            # Numbers block
            zg = complex(zI_r - gx_mean_r, zQ_r - gy_mean_r)
            eg = complex(ex_mean_r - gx_mean_r, ey_mean_r - gy_mean_r)
            pop_norm = np.abs(zg) / (np.abs(eg) + 1e-12)
            pop_proj = (((zI_r - gx_mean_r) * (ex_mean_r - gx_mean_r) + (zQ_r - gy_mean_r) * (ey_mean_r - gy_mean_r))
                        / ((ex_mean_r - gx_mean_r) ** 2 + (ey_mean_r - gy_mean_r) ** 2 + 1e-12))
            axs[1].text(0.02, 0.02,
                        f"|z−g| = {np.abs(zg):.3f}\n|e−g| = {np.abs(eg):.3f}\n"
                        f"|z−g|/|e−g| = {pop_norm:.3f}\nproj = {np.clip(pop_proj, 0, 1):.3f}",
                        transform=axs[1].transAxes)

            axs[1].set_xlabel('I (a.u.)')
            axs[1].legend(loc='lower right')
            axs[1].set_title(f'Rotated  (θ = {round(theta, 5)})')
            axs[1].axis('equal')

            # ------- Histograms in rotated I -------
            ng, binsg, _ = axs[2].hist(ig_new, bins=numbins, range=xlims, color='b', label='g', alpha=0.2)
            ne, binse, _ = axs[2].hist(ie_new, bins=numbins, range=xlims, color='r', label='e', alpha=0.2)

            # NEW: dashed vertical lines at the mean I-positions for g and e (in the rotated frame)
            axs[2].axvline(gx_mean_r, linestyle='--', linewidth=2, color='b', label='g mean (I)')
            axs[2].axvline(ex_mean_r, linestyle='--', linewidth=2, color='r', label='e mean (I)')
            # Optional: annotate the values
            axs[2].annotate(f"g μI={gx_mean_r:.3f}", xy=(gx_mean_r, 0), xytext=(5, 10),
                            textcoords='offset points', rotation=90, va='bottom', ha='left')
            axs[2].annotate(f"e μI={ex_mean_r:.3f}", xy=(ex_mean_r, 0), xytext=(5, 10),
                            textcoords='offset points', rotation=90, va='bottom', ha='left')

            axs[2].set_xlabel('I (a.u.)')
        else:
            ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
            ne, binse = np.histogram(ie_new, bins=numbins, range=xlims)

        # ------- Fidelity via histogram overlap (unchanged) -------
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
        tind = contrast.argmax()
        threshold = binsg[tind]
        fid = contrast[tind]

        if plot:
            self.create_folder_if_not_exists(self.outerFolder)
            out_dir = os.path.join(self.outerFolder, f"ss_repeat_meas_ge{path_ext}")
            self.create_folder_if_not_exists(out_dir)
            out_dir = os.path.join(out_dir, "Q" + str(self.QubitIndex + 1))
            self.create_folder_if_not_exists(out_dir)
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(
                out_dir,
                f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{self.expt_name}{now}_q{self.QubitIndex + 1}.png"
            )
            axs[2].set_title(f"Fidelity = {fid * 100:.2f}%")
            plt.tight_layout()
            fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)

        return fid, threshold, theta, ig_new, ie_new

    def robust_center_debug(self,z, c=5.5, iters=100, eps=1e-12, keep_path_steps=5):
        """
        Run robust_center but return useful intermediates for visualization.
        Returns:
            {
              'mu0': complex median start,
              's0': float initial MAD-based scale (from distances to mu0),
              'u0': np.array of standardized distances in first step,
              'w0': np.array of Tukey weights from the first step,
              'cutoff_radius': c*s0,
              'path': [complex mu after each iter] (first keep_path_steps+1 entries incl. mu0)
              'final_mu': complex final location
            }
        """
        I = np.real(z)
        Q = np.imag(z)

        mu_I, mu_Q = np.median(I), np.median(Q)  # start
        path = [mu_I + 1j * mu_Q]

        # first pass: distances from median
        d0 = np.hypot(I - mu_I, Q - mu_Q)
        s0 = 1.4826 * np.median(np.abs(d0 - np.median(d0))) + eps
        u0 = d0 / (c * s0 + eps)
        w0 = (1 - u0 ** 2) ** 2
        w0[u0 >= 1] = 0.0
        cutoff_radius = c * s0

        # run full IRLS, store a short path
        for k in range(iters):
            d = np.hypot(I - mu_I, Q - mu_Q)
            s = 1.4826 * np.median(np.abs(d - np.median(d))) + eps
            u = d / (c * s + eps)
            w = (1 - u ** 2) ** 2
            w[u >= 1] = 0.0
            if np.all(w == 0):
                w = np.ones_like(d)

            mu_I = np.sum(w * I) / (np.sum(w) + eps)
            mu_Q = np.sum(w * Q) / (np.sum(w) + eps)

            if k < keep_path_steps:
                path.append(mu_I + 1j * mu_Q)

        return {
            'mu0': path[0],
            's0': s0,
            'u0': u0,
            'w0': w0,
            'cutoff_radius': cutoff_radius,
            'path': path,
            'final_mu': mu_I + 1j * mu_Q
        }

    def hist_ssf_with_annotations_tukey(
            self, data=None, cfg=None, plot=True, fig_quality=100,
            I_meas=None, Q_meas=None, path_ext='',
            show_weight_labels=False, max_weight_labels=10,
            show_path_steps=True, path_steps=5, tukey_c=5.5):
        """
        Same outputs, but with robust_center step-by-step annotations:
          - plot starting medians (mu0)
          - draw Tukey cutoff circle (radius = c*s from the median) where weights drop to 0
          - color points by first-iteration weight w0
          - print a few weight values near selected points
          - optionally show the first few iteration updates of mu ("path")

        Returns:
            fid, threshold, theta, ig_new, ie_new
        """
        import os, math, datetime
        import numpy as np
        import matplotlib.pyplot as plt

        ig, qg, ie, qe = data[0], data[1], data[2], data[3]
        zg = ig + 1j * qg
        ze = ie + 1j * qe

        # --- robust centers (final) for centroids ---
        gc = self.robust_center(zg, c=tukey_c)  # final robust center g
        ec = self.robust_center(ze, c=tukey_c)  # final robust center e
        gx_mean, gy_mean = float(np.real(gc)), float(np.imag(gc))
        ex_mean, ey_mean = float(np.real(ec)), float(np.imag(ec))

        # --- debug traces for step-by-step visualization (short path) ---
        dbg_g = self.robust_center_debug(zg, c=tukey_c, keep_path_steps=path_steps)
        dbg_e = self.robust_center_debug(ze, c=tukey_c, keep_path_steps=path_steps)

        gx_med0, gy_med0 = float(np.real(dbg_g['mu0'])), float(np.imag(dbg_g['mu0']))
        ex_med0, ey_med0 = float(np.real(dbg_e['mu0'])), float(np.imag(dbg_e['mu0']))

        # also keep plain medians if desired
        gx_med, gy_med = float(np.median(ig)), float(np.median(qg))
        ex_med, ey_med = float(np.median(ie)), float(np.median(qe))

        n_total = len(ig) + len(ie)
        numbins = max(1, int(math.ceil(math.log2(n_total) + 1)))

        if plot:
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(18, 5))
            fig.tight_layout()

            # ------- Unrotated with robust-center debugging -------
            # Color points by first-iteration Tukey weight (w0) to show influence
            # Normalize colors separately for g/e for clarity.
            g_colors = dbg_g['w0']
            e_colors = dbg_e['w0']

            sc_g = axs[0].scatter(ig, qg, c=g_colors, cmap='viridis', marker='o', alpha=0.7, label='g shots')
            sc_e = axs[0].scatter(ie, qe, c=e_colors, cmap='plasma', marker='o', alpha=0.7, label='e shots')

            from matplotlib.colors import Normalize

            from matplotlib.colors import Normalize

            # Bigger canvas + extra bottom margin for colorbars
            fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(22, 7))
            fig.subplots_adjust(bottom=0.20, wspace=0.28)  # <- room for two cbars below axs[0]

            # ---------- Unrotated with two horizontal colorbars UNDER axs[0] ----------
            norm = Normalize(vmin=0.0, vmax=1.0)

            sc_g = axs[0].scatter(
                ig, qg, c=dbg_g['w0'], cmap='viridis', norm=norm,
                marker='o', alpha=0.70, edgecolors='none', label='g shots'
            )
            sc_e = axs[0].scatter(
                ie, qe, c=dbg_e['w0'], cmap='plasma', norm=norm,
                marker='s', alpha=0.55, edgecolors='none', label='e shots'
            )

            # ---- two small horizontal colorbars centered under axs[0] ----
            axpos = axs[0].get_position()  # [x0, y0, width, height] in figure coords

            cbar_h = 0.025  # height of each colorbar (in fig coords)
            gap_y = 0.04  # vertical gap below axs[0]
            bar_w = axpos.width * 0.30  # each bar = 40% of axes width
            bar_gap = axpos.width * 0.05  # gap between the two bars (10% of axes width)

            # center the pair under axs[0]
            total_w = 2 * bar_w + bar_gap
            x_start = axpos.x0 + (axpos.width - total_w) / 2
            y_cbar = max(0.02, axpos.y0 - gap_y)  # keep above bottom of figure

            cax_g = fig.add_axes([x_start, y_cbar, bar_w, cbar_h])
            cax_e = fig.add_axes([x_start + bar_w + bar_gap, y_cbar, bar_w, cbar_h])

            cbar_g = fig.colorbar(sc_g, cax=cax_g, orientation='horizontal')
            cbar_e = fig.colorbar(sc_e, cax=cax_e, orientation='horizontal')
            cbar_g.set_label('g first-step weight  $w_0$', labelpad=2)
            cbar_e.set_label('e first-step weight  $w_0$', labelpad=2)
            for cb in (cbar_g, cbar_e):
                cb.ax.tick_params(labelsize=9)

            # Starting medians (mu0) = robust_center start
            axs[0].scatter([gx_med0], [gy_med0], color='k', marker='^', s=90, label='g start (median)')
            axs[0].scatter([ex_med0], [ey_med0], color='k', marker='v', s=90, label='e start (median)')

            # Final robust centers
            axs[0].scatter([gx_mean], [gy_mean], color='k', marker='X', s=110, label='g robust center')
            axs[0].scatter([ex_mean], [ey_mean], color='k', marker='X', s=110, label='e robust center')

            # Optional: show short path (a few IRLS updates) so you see how mu moves
            if show_path_steps:
                path_g = np.array(dbg_g['path'])
                path_e = np.array(dbg_e['path'])
                axs[0].plot(np.real(path_g), np.imag(path_g), linestyle='--', linewidth=2, label='g μ path')
                axs[0].plot(np.real(path_e), np.imag(path_e), linestyle='--', linewidth=2, label='e μ path')

            # Tukey cutoff circles: radius = c*s from the starting median (where w goes to zero on 1st step)
            theta_circle = np.linspace(0, 2 * np.pi, 361)
            for (cx, cy, rad) in [
                (gx_med0, gy_med0, float(dbg_g['cutoff_radius'])),
                (ex_med0, ey_med0, float(dbg_e['cutoff_radius']))]:
                axs[0].plot(cx + rad * np.cos(theta_circle), cy + rad * np.sin(theta_circle),
                            linestyle=':', linewidth=2, color='k')

            # Label a few representative weights (near selected distances) so it’s easy to read
            if show_weight_labels and max_weight_labels > 0:
                def annotate_weights(I, Q, w, cx, cy, N):
                    # pick points near quantiles of distance to spread labels
                    d = np.hypot(I - cx, Q - cy)
                    qs = np.linspace(0.05, 0.95, min(N, 10))
                    targets = np.quantile(d, qs)
                    for t in targets:
                        idx = np.argmin(np.abs(d - t))
                        axs[0].annotate(f"w={w[idx]:.2f}",
                                        xy=(I[idx], Q[idx]),
                                        xytext=(5, 5),
                                        textcoords='offset points',
                                        fontsize=9, bbox=dict(boxstyle='round,pad=0.2', fc='white', alpha=0.6))

                annotate_weights(ig, qg, dbg_g['w0'], gx_med0, gy_med0, max_weight_labels)
                annotate_weights(ie, qe, dbg_e['w0'], ex_med0, ey_med0, max_weight_labels)

            # Optional raw measurement
            if (I_meas is not None) and (Q_meas is not None):
                axs[0].scatter([I_meas], [Q_meas], s=90, edgecolor='k', facecolor='none', linewidth=1.5,
                               label='meas z (raw)')
                axs[0].annotate("z", xy=(I_meas, Q_meas), xytext=(I_meas + 0.05, Q_meas + 0.05))

            axs[0].set_xlabel('I (a.u.)')
            axs[0].set_ylabel('Q (a.u.)')
            axs[0].set_title('Unrotated — robust_center internals')
            axs[0].legend(loc='best')
            axs[0].axis('equal')

        # ------- Rotation so g->e is horizontal (use robust centers) -------
        theta = -np.arctan2((ey_mean - gy_mean), (ex_mean - gx_mean))
        cos_t, sin_t = np.cos(theta), np.sin(theta)

        ig_new = ig * cos_t - qg * sin_t
        qg_new = ig * sin_t + qg * cos_t
        ie_new = ie * cos_t - qe * sin_t
        qe_new = ie * sin_t + qe * cos_t

        gx_mean_r = gx_mean * cos_t - gy_mean * sin_t
        gy_mean_r = gx_mean * sin_t + gy_mean * cos_t
        ex_mean_r = ex_mean * cos_t - ey_mean * sin_t
        ey_mean_r = ex_mean * sin_t + ey_mean * cos_t

        if (I_meas is not None) and (Q_meas is not None):
            zI_r = I_meas * cos_t - Q_meas * sin_t
            zQ_r = I_meas * sin_t + Q_meas * cos_t
        else:
            z_ix = len(ie_new) // 2
            zI_r, zQ_r = float(ie_new[z_ix]), float(qe_new[z_ix])

        xlims = [min(np.min(ig_new), np.min(ie_new)), max(np.max(ig_new), np.max(ie_new))]

        if plot:
            # ------- Rotated scatter & vectors -------
            axs[1].scatter(ig_new, qg_new, label='g shots', color='b', marker='*', alpha=0.25)
            axs[1].scatter(ie_new, qe_new, label='e shots', color='r', marker='*', alpha=0.25)
            axs[1].scatter([gx_mean_r], [gy_mean_r], color='k', marker='X', s=90, label='g robust center (rot)')
            axs[1].scatter([ex_mean_r], [ey_mean_r], color='k', marker='X', s=90, label='e robust center (rot)')

            axs[1].plot([gx_mean_r, ex_mean_r], [gy_mean_r, ey_mean_r], linestyle='--', linewidth=2, label='g → e')

            axs[1].arrow(gx_mean_r, gy_mean_r, (zI_r - gx_mean_r), (zQ_r - gy_mean_r),
                         length_includes_head=True, head_width=0.04, alpha=0.9)
            axs[1].scatter([zI_r], [zQ_r], s=80, label='meas z (rot)')
            axs[1].annotate("z", xy=(zI_r, zQ_r), xytext=(zI_r + 0.05, zQ_r + 0.05))

            zg_vec = complex(zI_r - gx_mean_r, zQ_r - gy_mean_r)
            eg_vec = complex(ex_mean_r - gx_mean_r, ey_mean_r - gy_mean_r)
            pop_norm = np.abs(zg_vec) / (np.abs(eg_vec) + 1e-12)
            pop_proj = (((zI_r - gx_mean_r) * (ex_mean_r - gx_mean_r) + (zQ_r - gy_mean_r) * (ey_mean_r - gy_mean_r))
                        / ((ex_mean_r - gx_mean_r) ** 2 + (ey_mean_r - gy_mean_r) ** 2 + 1e-12))
            axs[1].text(0.02, 0.02,
                        f"|z−g| = {np.abs(zg_vec):.3f}\n|e−g| = {np.abs(eg_vec):.3f}\n"
                        f"|z−g|/|e−g| = {pop_norm:.3f}\nproj = {np.clip(pop_proj, 0, 1):.3f}",
                        transform=axs[1].transAxes)

            axs[1].set_xlabel('I (a.u.)')
            axs[1].legend(loc='lower right')
            axs[1].set_title(f'Rotated  (θ = {round(theta, 5)})')
            axs[1].axis('equal')

            # ------- Histograms in rotated I -------
            ng, binsg, _ = axs[2].hist(ig_new, bins=numbins, range=xlims, color='b', label='g', alpha=0.25)
            ne, binse, _ = axs[2].hist(ie_new, bins=numbins, range=xlims, color='r', label='e', alpha=0.25)

            axs[2].axvline(gx_mean_r, linestyle='--', linewidth=2, color='b', label='g robust μI')
            axs[2].axvline(ex_mean_r, linestyle='--', linewidth=2, color='r', label='e robust μI')

            axs[2].set_xlabel('I (a.u.)')
        else:
            ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
            ne, binse = np.histogram(ie_new, bins=numbins, range=xlims)

        # ------- Fidelity via histogram overlap (unchanged) -------
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) / (0.5 * ng.sum() + 0.5 * ne.sum())))
        tind = contrast.argmax()
        threshold = binsg[tind]
        fid = contrast[tind]

        if plot:
            self.create_folder_if_not_exists(self.outerFolder)
            out_dir = os.path.join(self.outerFolder, f"ss_repeat_meas_ge{path_ext}")
            self.create_folder_if_not_exists(out_dir)
            out_dir = os.path.join(out_dir, "Q" + str(self.QubitIndex + 1))
            self.create_folder_if_not_exists(out_dir)
            now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(
                out_dir,
                f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{self.expt_name}{now}_q{self.QubitIndex + 1}.png"
            )
            axs[2].set_title(f"Fidelity = {fid * 100:.2f}%")
            plt.tight_layout()
            fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)

        return fid, threshold, theta, ig_new, ie_new

    def only_hist_ssf(self, data=None, cfg=None, plot=True, fig_quality=100, plot_title="Run 3"):
        import math
        import numpy as np
        import matplotlib.pyplot as plt
        import os
        import datetime

        # Unpack IQ data
        ig = data[0]
        qg = data[1]
        ie = data[2]
        qe = data[3]

        # Determine number of bins for the histogram
        numbins = round(math.sqrt(float(cfg["steps"])))

        # Compute medians (used for rotation angle calculation)
        xg, yg = np.median(ig), np.median(qg)
        xe, ye = np.median(ie), np.median(qe)

        # Compute rotation angle
        theta = -np.arctan2((ye - yg), (xe - xg))

        # Rotate the IQ data
        ig_new = ig * np.cos(theta) - qg * np.sin(theta)
        qg_new = ig * np.sin(theta) + qg * np.cos(theta)
        ie_new = ie * np.cos(theta) - qe * np.sin(theta)
        qe_new = ie * np.sin(theta) + qe * np.cos(theta)

        # New medians after rotation (not used further in plotting)
        xg, yg = np.median(ig_new), np.median(qg_new)
        xe, ye = np.median(ie_new), np.median(qe_new)

        # Define histogram range from the rotated ground state to the excited state
        xlims = [np.min(ig_new), np.max(ie_new)]
        ng, binsg = np.histogram(ig_new, bins=numbins, range=xlims)
        ne, binse = np.histogram(ie_new, bins=numbins, range=xlims)
        # Compute the fidelity using the overlap of the histograms
        contrast = np.abs(((np.cumsum(ng) - np.cumsum(ne)) /
                           (0.5 * ng.sum() + 0.5 * ne.sum())))
        tind = contrast.argmax()
        threshold = binsg[tind]
        fid = contrast[tind]
        if plot:
            # Create figure and axis for the histogram
            fig, ax = plt.subplots(figsize=(8, 6))

            # Plot histogram for ground state and first excited state with updated labels
            ng, binsg, _ = ax.hist(ig_new, bins=numbins, range=xlims, color='b',
                                   label='Ground', alpha=0.5)
            ne, binse, _ = ax.hist(ie_new, bins=numbins, range=xlims, color='r',
                                   label='First Excited State', alpha=0.5)

            # Set axis labels with 12-point font
            ax.set_xlabel('I (a.u.)', fontsize=12)
            ax.set_ylabel('Counts', fontsize=12)
            # Set plot title using the provided parameter
            ax.set_title(plot_title + f'   SSF: {int(fid * 100)}%', fontsize=12)
            ax.legend()

            # Save the figure
            self.create_folder_if_not_exists(self.outerFolder)
            outerFolder_expt = os.path.join(self.outerFolder, "ss_repeat_meas_ge")
            self.create_folder_if_not_exists(outerFolder_expt)
            outerFolder_expt = os.path.join(outerFolder_expt, "Q" + str(self.QubitIndex + 1))
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(outerFolder_expt,
                                     f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}.png")
            fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)





        return fid, threshold, theta, ig_new, ie_new

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)


class GainFrequencySweep:
    def __init__(self,qubit_index, number_of_qubits, list_of_all_qubits, experiment, optimal_lengths=None, output_folder="/default/path/", unmasking_resgain = False):
        self.qubit_index = qubit_index
        self.list_of_all_qubits = list_of_all_qubits
        self.output_folder = output_folder
        self.expt_name = "Readout_Optimization"
        self.Qubit = 'Q' + str(self.qubit_index)
        self.optimal_lengths = optimal_lengths
        self.number_of_qubits = number_of_qubits

        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.unmasking_resgain = unmasking_resgain

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [qubit_index]

        self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
        self.config = {**self.q_config[self.Qubit], **self.exp_cfg}

    def set_res_gain_ge(self, QUBIT_INDEX, set_gain, num_qubits=6):
        """Sets the gain for the selected qubit to 1, others to 0."""
        res_gain_ge = [0] * num_qubits  # Initialize all gains to 0
        if 0 <= QUBIT_INDEX < num_qubits:  # makes sure you are within the range of options
            res_gain_ge[QUBIT_INDEX] = set_gain  # Set the gain for the selected qubit
        return res_gain_ge

    def run_sweep(self, freq_range, gain_range, freq_steps, gain_steps):
        freq_step_size = (freq_range[1] - freq_range[0]) / freq_steps
        gain_step_size = (gain_range[1] - gain_range[0]) / gain_steps
        results = []

        # Use the optimal readout length for the current qubit
        readout_length = self.optimal_lengths[self.qubit_index]
        for freq_step in range(freq_steps):
            freq = freq_range[0] + freq_step * freq_step_size
            #print('Running for res_freq: ', freq, '...')
            fid_results = []
            for gain_step in range(gain_steps):
                #experiment = QICK_experiment(self.output_folder)
                #experiment = QICK_experiment(self.output_folder, DAC_attenuator1=10, DAC_attenuator2=5, ADC_attenuator=10)
                fresh_experiment = copy.deepcopy(self.experiment)
                gain = gain_range[0] + gain_step * gain_step_size


                # Update config with current gain and frequency values
                fresh_experiment.readout_cfg['res_freq_ge']= freq
                fresh_experiment.readout_cfg['res_length'] = readout_length  # Set the optimal readout length for the qubit

                fresh_experiment.readout_cfg['res_gain_ge'] = gain

                # Initialize SingleShotGE instance for fidelity calculation
                round_num = 0
                save_figs = True
                import time

                while True:
                    try:
                        single_shot = SingleShot(
                            self.qubit_index, self.number_of_qubits, self.output_folder,
                            round_num, save_figs, fresh_experiment,
                            unmasking_resgain=self.unmasking_resgain
                        )
                        fidelity = single_shot.fidelity_test(fresh_experiment.soccfg, fresh_experiment.soc)
                        break  #it worked
                    except Exception as e:
                        print(f"[retry] SingleShot failed: {e}. Trying again in 2s…")
                        time.sleep(2)
                fid_results.append(fidelity)
                del fresh_experiment
                del single_shot

            results.append(fid_results)

        return results

