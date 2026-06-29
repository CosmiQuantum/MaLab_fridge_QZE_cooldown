from build_task import *
from build_state import *
from expt_config import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import datetime
import copy
import visdom
import logging
from section_005_single_shot_ge import SingleShotProgram_g, SingleShotProgram_e
class QubitSpectroscopyDualStark:
    def __init__(self, QubitIndex, number_of_qubits,  outerFolder,  round_num, signal, save_figs, experiment = None,
                 live_plot = None, verbose = False, logger = None, qick_verbose=True, increase_reps = False,
                 increase_reps_to = 500, plot_fit=True, zeno_stark=False, zeno_stark_pulse_gain=None,
                 ext_q_spec=False, high_gain_q_spec=False, fit_data=True, unmasking_resgain = False):

        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.plot_fit=plot_fit
        self.zeno_stark = zeno_stark
        self.zeno_stark_pulse_gain = zeno_stark_pulse_gain
        self.ext_q_spec = ext_q_spec
        self.fit_data = fit_data
        self.high_gain_q_spec = high_gain_q_spec
        self.expt_name = "qubit_spec_ge_dual_stark_drive"
        self.signal = signal
        self.save_figs = save_figs
        self.experiment = experiment
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num
        self.number_of_qubits = number_of_qubits
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")
        self.increase_reps = increase_reps
        self.increase_reps_to = increase_reps_to

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]

        if experiment is not None:
            if self.zeno_stark:
                qze_mask = np.arange(0, self.number_of_qubits + 1)
                qze_mask = np.delete(qze_mask, QubitIndex)
                self.exp_cfg['qze_mask'] = qze_mask
                self.experiment.readout_cfg['res_gain_qze'] = [self.experiment.readout_cfg['res_gain_ge'][QubitIndex],
                                                               0, 0, 0, 0, 0, self.zeno_stark_pulse_gain]
                self.experiment.readout_cfg['res_freq_qze'] = self.experiment.readout_cfg['res_freq_ge']
                self.experiment.readout_cfg['res_phase_qze'] = self.experiment.readout_cfg['res_phase']
                if len(self.experiment.readout_cfg['res_freq_qze']) < 7:  # otherise it keeps appending
                    self.experiment.readout_cfg['res_freq_qze'].append(
                        experiment.readout_cfg['res_freq_qze'][self.QubitIndex])
                    self.experiment.readout_cfg['res_phase_qze'].append(
                        experiment.readout_cfg['res_phase_qze'][self.QubitIndex])

            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)

            self.live_plot = live_plot
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            print(self.config)
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} Qubit Spec configuration: ', self.config)
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} Qubit Spec configuration: {self.config}')
    def run_res_stark(self,return_fwhm=False, scaling=False,qze_pulse='const'):

        if self.increase_reps:
            self.config['reps'] = self.increase_reps_to
        if scaling:
            q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            ss_exp_cfg = add_qubit_experiment(expt_cfg, 'Readout_Optimization', self.QubitIndex)
            ss_config = {**q_config[self.Qubit], **ss_exp_cfg}


        # iq_lists= []
        if self.live_plot:
            I, Q, freqs = self.live_plotting(qspec)
        else:
            Is_all=[]
            Qs_all=[]
            ss_I_g_all = []
            ss_Q_g_all = []
            ss_I_e_all = []
            ss_Q_e_all = []
            I_shots_all=[]
            Q_shots_all=[]

            qspec = ResStark(self.experiment.soccfg, reps=self.config['reps'], final_delay=0.5,
                                                  cfg=self.config)

            iq_list = qspec.acquire(self.experiment.soc, rounds=self.exp_cfg["rounds"], progress=self.qick_verbose)
            iq_list = iq_list[0][0].T
            I = iq_list[0] # shape 3, 500 for 3 gains in the sweep and 500 steps (reps already averaged over here)
            Q = iq_list[1]
            Is_all.append(I)
            Qs_all.append(Q)

            raw_0 = qspec.get_raw()  # I,Q data without normalizing to readout window, subtracting readout offset, or rotation/thresholding
            A = np.squeeze(raw_0[0])
            I_shots = A[:, :,
                      0]  # if you have 4 steps and 3 shots/reps this is like [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
            Q_shots = A[:, :, 1]

            I_shots_all.append(I_shots)
            Q_shots_all.append(Q_shots)

            if scaling:
                ssp_g = SingleShotProgram_g(self.experiment.soccfg, reps=1, final_delay=ss_config['relax_delay'],
                                            cfg=ss_config)
                ssp_e = SingleShotProgram_e(self.experiment.soccfg, reps=1, final_delay=ss_config['relax_delay'],
                                            cfg=ss_config)
                iq_list_g = ssp_g.acquire(self.experiment.soc, rounds=1, progress=True)
                iq_list_e = ssp_e.acquire(self.experiment.soc, rounds=1, progress=True)

                ss_I_g = iq_list_g[0][0].T[0]
                ss_Q_g = iq_list_g[0][0].T[1]
                ss_I_e = iq_list_e[0][0].T[0]
                ss_Q_e = iq_list_e[0][0].T[1]

                ss_I_g_all.append(ss_I_g)
                ss_Q_g_all.append(ss_Q_g)
                ss_I_e_all.append(ss_I_e)
                ss_Q_e_all.append(ss_Q_e)

            freqs = qspec.get_pulse_param('qubit_pulse', "freq", as_array=True) #only need to get it once
            gains = qspec.get_pulse_param('qze_pulse', "gain", as_array=True)

            if return_fwhm:
                largest_amp_curve_mean, y_data_fit, fwhm = self.plot_results_interweaved_cal(Is_all, Qs_all, freqs, config=self.config,
                                                                               return_fwhm=return_fwhm,scaling=scaling,Ie = ss_I_e_all,
                                                                               Ig = ss_I_g_all, Qe = ss_Q_e_all, Qg = ss_Q_g_all)
                return Is_all, Qs_all, freqs, y_data_fit, largest_amp_curve_mean, self.config, fwhm, ss_Q_e_all\
                    , ss_Q_g_all, ss_I_e_all, ss_I_g_all, I_shots_all, Q_shots_all
            else:
                largest_amp_curve_mean, y_data_fit = self.plot_results_interweaved_cal(Is_all, Qs_all, freqs,gains, config=self.config,
                                                                         return_fwhm=return_fwhm,scaling=scaling, Ie = ss_I_e_all,
                                                                         Ig = ss_I_g_all, Qe = ss_Q_e_all, Qg = ss_Q_g_all)
                return Is_all, Qs_all, freqs, y_data_fit, largest_amp_curve_mean, self.config, ss_Q_e_all, ss_Q_g_all,ss_I_e_all\
                    , ss_I_g_all, I_shots_all, Q_shots_all, gains
    def run_off_resonant_qstark(self,return_fwhm=False, scaling=False, qze_pulse='const'):

        if self.increase_reps:
            self.config['reps'] = self.increase_reps_to
        if scaling:
            q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            ss_exp_cfg = add_qubit_experiment(expt_cfg, 'Readout_Optimization', self.QubitIndex)
            ss_config = {**q_config[self.Qubit], **ss_exp_cfg}


        # iq_lists= []
        if self.live_plot:
            I, Q, freqs = self.live_plotting(qspec)
        else:
            Is_all=[]
            Qs_all=[]
            ss_I_g_all = []
            ss_Q_g_all = []
            ss_I_e_all = []
            ss_Q_e_all = []
            I_shots_all=[]
            Q_shots_all=[]

            qspec = OffResonantQSpecDrive(self.experiment.soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'],
                                                  cfg=self.config)

            iq_list = qspec.acquire(self.experiment.soc, rounds=self.exp_cfg["rounds"], progress=self.qick_verbose)
            iq_list = iq_list[0][0].T
            I = iq_list[0] # shape 3, 500 for 3 gains in the sweep and 500 steps (reps already averaged over here)
            Q = iq_list[1]
            Is_all.append(I)
            Qs_all.append(Q)

            raw_0 = qspec.get_raw()  # I,Q data without normalizing to readout window, subtracting readout offset, or rotation/thresholding
            A = np.squeeze(raw_0[0])
            I_shots = A[:, :,
                      0]  # if you have 4 steps and 3 shots/reps this is like [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
            Q_shots = A[:, :, 1]

            I_shots_all.append(I_shots)
            Q_shots_all.append(Q_shots)

            if scaling:
                ssp_g = SingleShotProgram_g(self.experiment.soccfg, reps=1, final_delay=ss_config['relax_delay'],
                                            cfg=ss_config)
                ssp_e = SingleShotProgram_e(self.experiment.soccfg, reps=1, final_delay=ss_config['relax_delay'],
                                            cfg=ss_config)
                iq_list_g = ssp_g.acquire(self.experiment.soc, rounds=1, progress=True)
                iq_list_e = ssp_e.acquire(self.experiment.soc, rounds=1, progress=True)

                ss_I_g = iq_list_g[0][0].T[0]
                ss_Q_g = iq_list_g[0][0].T[1]
                ss_I_e = iq_list_e[0][0].T[0]
                ss_Q_e = iq_list_e[0][0].T[1]

                ss_I_g_all.append(ss_I_g)
                ss_Q_g_all.append(ss_Q_g)
                ss_I_e_all.append(ss_I_e)
                ss_Q_e_all.append(ss_Q_e)

            freqs = qspec.get_pulse_param('qubit_pulse', "freq", as_array=True) #only need to get it once
            gains = qspec.get_pulse_param('qubit_stark_pulse', "gain", as_array=True)

            if return_fwhm:
                largest_amp_curve_mean, y_data_fit, fwhm = self.plot_results_interweaved_cal(Is_all, Qs_all, freqs, config=self.config,
                                                                               return_fwhm=return_fwhm,scaling=scaling,Ie = ss_I_e_all,
                                                                               Ig = ss_I_g_all, Qe = ss_Q_e_all, Qg = ss_Q_g_all)
                return Is_all, Qs_all, freqs, y_data_fit, largest_amp_curve_mean, self.config, fwhm, ss_Q_e_all\
                    , ss_Q_g_all, ss_I_e_all, ss_I_g_all, I_shots_all, Q_shots_all
            else:
                largest_amp_curve_mean, y_data_fit = self.plot_results_interweaved_cal(Is_all, Qs_all, freqs,gains, config=self.config,
                                                                         return_fwhm=return_fwhm,scaling=scaling, Ie = ss_I_e_all,
                                                                         Ig = ss_I_g_all, Qe = ss_Q_e_all, Qg = ss_Q_g_all)
                return Is_all, Qs_all, freqs, y_data_fit, largest_amp_curve_mean, self.config, ss_Q_e_all, ss_Q_g_all,ss_I_e_all\
                    , ss_I_g_all, I_shots_all, Q_shots_all, gains

    def run_off_resonant_qstark_tof(self, gains=None, soft_avgs=None):

        # how long the decimated capture window is. Default spans the whole sequence
        # (stark pulse + a bit before for the qubit pulse + readout after). If the decimated
        # buffer overflows on hardware, shorten qubit_stark_pulse_length or set
        # 'tof_readout_length' smaller in expt_config.
        if 'tof_readout_length' not in self.config or self.config['tof_readout_length'] is None:
            self.config['tof_readout_length'] = self.config['qubit_stark_pulse_length'] + 12

        # The decimated ADC buffer is limited (e.g. 16384 samples). A long window will raise
        # "requested readout length ... exceeds buffer size". Clamp the window to what the
        # buffer can hold (queried from soccfg so it's board-agnostic). We still trigger at
        # t=0, so the full off-resonant stark pulse stays in view as long as it fits (~29 us
        # is plenty for the 25 us stark pulse); only the trailing readout pulse may be clipped.
        try:
            ro_ch = self.config['ro_ch']
            ro_cfg = self.experiment.soccfg['readouts'][ro_ch]
            # the buffer-length key has different names across QICK versions
            maxlen_samps = None
            for k in ('maxlen', 'buf_maxlen', 'avg_maxlen', 'max_length'):
                if k in ro_cfg:
                    maxlen_samps = ro_cfg[k]
                    break
            if maxlen_samps is not None:
                max_window_us = float(self.experiment.soccfg.cycles2us(maxlen_samps, ro_ch=ro_ch)) * 0.97
                if self.config['tof_readout_length'] > max_window_us:
                    if self.verbose:
                        print(f"TOF window {self.config['tof_readout_length']} us exceeds decimated "
                              f"buffer (~{max_window_us:.2f} us max); clamping to {max_window_us:.2f} us. "
                              f"This still captures the full {self.config['qubit_stark_pulse_length']} us "
                              f"stark pulse.")
                    self.logger.info(f"Clamping TOF readout window to {max_window_us:.2f} us "
                                     f"(decimated buffer limit).")
                    self.config['tof_readout_length'] = max_window_us
            else:
                # couldn't find the buffer-size key for this QICK version; keep configured value.
                # If acquire_decimated raises a buffer-size error, lower 'tof_readout_length'.
                self.logger.info("Could not determine decimated buffer max from soccfg; "
                                 f"using configured tof_readout_length={self.config['tof_readout_length']} us.")
        except Exception as e:
            self.logger.info(f"TOF window auto-clamp skipped ({e}); "
                             f"using tof_readout_length={self.config['tof_readout_length']} us.")

        # add_qubit_experiment() turns qubit_freq_ge into a QickSweep1D over the 'freqloop'
        # loop (for the normal qspec measurement). The TOF check does NOT sweep frequency
        # (only gain vs time), so pin qubit_freq_ge to a fixed scalar (center of that sweep)
        # to avoid a dangling 'freqloop' reference when the program compiles.
        # (QickSweep1D is a factory, not a type, so check for "not a plain number" instead.)
        if not isinstance(self.config.get('qubit_freq_ge'), (int, float)):
            self.config['qubit_freq_ge'] = float((self.config['start'] + self.config['stop']) / 2)

        # gain sweep axis (one decimated acquire per gain point)
        if gains is None:
            gains = np.linspace(self.config['start_qubit_stark_gain'],
                                self.config['stop_qubit_stark_gain'],
                                self.config['stark_gain_steps'])
        gains = np.asarray(gains, dtype=float)

        if soft_avgs is None:
            soft_avgs = self.config['rounds']

        I_traces = []
        Q_traces = []
        mag_traces = []
        t = None
        for g in gains:
            self.config['qubit_stark_gain_tof'] = float(g)
            prog = OffResonantQSpecDriveTOF_Q(self.experiment.soccfg, reps=1,
                                            final_delay=self.config['relax_delay'],
                                            cfg=self.config)
            # prog = MuxProgram(self.experiment.soccfg, reps=1, final_delay=5, cfg=self.config)

            iq_list = prog.acquire_decimated(self.experiment.soc, rounds=500,
                                             progress=self.qick_verbose)


            if t is None:
                t = prog.get_time_axis(ro_index=0)
            I = iq_list[0][:, 0]
            Q = iq_list[0][:, 1]
            mag = np.abs(iq_list[0].dot([1, 1j]))
            I_traces.append(I)
            Q_traces.append(Q)
            mag_traces.append(mag)
            if self.verbose:
                print(f"TOF stark check Q{self.QubitIndex + 1}: gain={g:.4g}, "
                      f"peak |IQ|={np.max(mag):.1f}")

        I_traces = np.array(I_traces)
        Q_traces = np.array(Q_traces)
        mag_traces = np.array(mag_traces)

        if self.save_figs:
            self.plot_tof_2d(t, gains, I_traces, Q_traces, mag_traces)
        return t, gains, I_traces, Q_traces, mag_traces, self.config

        # prog = MuxProgram(self.experiment.soccfg, reps=1, final_delay=5, cfg=self.config)
        #
        # iq_list = prog.acquire_decimated(self.experiment.soc, rounds=1000,#soft_avgs,
        #                                  progress=self.qick_verbose)
        # t = prog.get_time_axis(ro_index=0)
        # I = iq_list[0][:, 0]
        # Q = iq_list[0][:, 1]
        # print(iq_list[0].shape)
        #
        # mag = np.abs(I + 1j * Q)  # or: np.abs(iq_list[0].dot([1, 1j]))
        #
        # plt.figure(figsize=(10, 5))
        # plt.plot(t, mag, label="magnitude")
        # plt.xlabel("Time (us)")
        # plt.ylabel("|IQ| (a.u.)")
        # plt.legend()
        # plt.show()
        #
        # return t, gains, I, Q, mag, self.config

    def plot_tof_2d(self, t, gains, I_traces, Q_traces, mag_traces, fig_quality=100):
        """2D heatmap (time vs gain, color=|IQ|) plus per-gain magnitude line traces."""
        t = np.asarray(t)
        gains = np.asarray(gains)

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 10))
        plt.rcParams.update({'font.size': 14})

        # --- 2D heatmap: x = time, y = stark gain, color = magnitude ---
        extent = (t[0], t[-1], gains[0], gains[-1])
        im = ax1.imshow(mag_traces, origin='lower', aspect='auto',
                        extent=extent, interpolation='nearest')
        ax1.set_xlabel("Time (us)")
        ax1.set_ylabel("Stark pulse gain")
        cbar = fig.colorbar(im, ax=ax1)
        cbar.set_label("|IQ| (a.u.)")
        ax1.set_title(
            f"Stark pulse TOF Q{self.QubitIndex + 1} "
            f"(all pulses @ res freq {self.config['res_freq_ge']:.2f} MHz, "
            f"stark len {self.config['qubit_stark_pulse_length']} us)",
            fontsize=12)

        # --- line traces: magnitude vs time for each gain ---
        for g, mag in zip(gains, mag_traces):
            ax2.plot(t, mag, label=f"gain={g:.3g}")
        ax2.set_xlabel("Time (us)")
        ax2.set_ylabel("|IQ| (a.u.)")
        ax2.legend(fontsize=9, ncol=2)
        ax2.set_title("Magnitude trace per gain (amplitude should grow with gain)",
                      fontsize=12)

        plt.tight_layout()

        if self.save_figs:
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name + "_tof_plots")
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(
                outerFolder_expt,
                f"R_{self.round_num}_Q_{self.QubitIndex + 1}_"
                f"{formatted_datetime}_{self.expt_name}_tof.png")
            fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
        plt.close(fig)

    def live_plotting(self, qspec):
        I = Q = expt_mags = expt_phases = expt_pop = None
        viz = visdom.Visdom()
        if not viz.check_connection(timeout_seconds=5):
            raise RuntimeError("Visdom server not connected!")
        viz.close(win=None)  # close previous plots
        for ii in range(self.config["rounds"]):
            iq_list = qspec.acquire(self.experiment.soc, rounds=1, progress=self.qick_verbose)
            freqs = qspec.get_pulse_param('qubit_pulse', "freq", as_array=True)
            iq_list = iq_list[0][0].T
            this_I = (iq_list[0])
            this_Q = (iq_list[1])
            #this_I = iq_list[self.QubitIndex][0, :, 0]
            #this_Q = iq_list[self.QubitIndex][0, :, 1]

            if I is None:  # ii == 0
                I, Q = this_I, this_Q
            else:
                I = (I * ii + this_I) / (ii + 1.0)
                Q = (Q * ii + this_Q) / (ii + 1.0)

            viz.line(X=freqs, Y=I, opts=dict(height=400, width=700, title='Qubit Spectroscopy I', showlegend=True, xlabel='expt_pts'),win='QSpec_I')
            viz.line(X=freqs, Y=Q, opts=dict(height=400, width=700, title='Qubit Spectroscopy Q', showlegend=True, xlabel='expt_pts'),win='QSpec_Q')
        return I, Q, freqs

    def plot_results(self, I, Q, freqs, config=None, fig_quality=100, sigma_guess=1, return_fwhm=False, scaling=False,
                     Ie=None, Ig=None, Qe=None, Qg=None):
        if scaling:
            e = np.mean((Ie + 1j * Qe))
            g = np.mean((Ig + 1j * Qg))
            ### Normalization ###
            pop_norm = abs(((I + 1j * Q) - g) * (e - g) / abs(e - g) ** 2)
            ydata = pop_norm

            freqs = np.array(freqs)
            freq_q = freqs[np.argmax(I)]

            mean_y_data,y_data_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, fit_err = self.fit_lorenzian_scaled(
                ydata, freqs,
                freq_q, sigma_guess)

            # Check if the returned values are all None
            if (mean_y_data is None and y_data_fit is None
                    and largest_amp_curve_mean is None and largest_amp_curve_fwhm is None):
                # If so, return None for the values in this definition as well
                return None, None, None

            # If we get here, the fit was successful and we can proceed with plotting
            fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))
            plt.rcParams.update({'font.size': 18})

            # I subplot
            ax1.plot(freqs, ydata, label='Qubit Population', linewidth=2)
            ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=16)
            ax1.legend()

            # Plot the fits
            if self.plot_fit:
                ax1.plot(freqs, y_data_fit, 'r--', label='Lorentzian Fit')
                #ax1.axvline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)

            # Calculate the middle of the plot area
            plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

            if self.plot_fit:
                # Add title, centered on the plot area
                if config is not None:  # then its been passed to this definition, so use that
                    fig.text(plot_middle, 0.98,
                             f"Qubit Spectroscopy Q{self.QubitIndex + 1}, %.2f MHz" % largest_amp_curve_mean +
                             f" FWHM: {round(largest_amp_curve_fwhm, 1)}" +
                             f", {config['reps']}*{config['rounds']} avgs, Zeno pulse gain {round(self.config['res_gain_qze'],3)}",
                             fontsize=14, ha='center', va='top')
                else:
                    fig.text(plot_middle, 0.98,
                             f"Qubit Spectroscopy Q{self.QubitIndex + 1}, %.2f MHz" % largest_amp_curve_mean +
                             f" FWHM: {round(largest_amp_curve_fwhm, 1)}" +
                             f", {self.config['reps']}*{self.config['rounds']} avgs",
                             fontsize=24, ha='center', va='top')
            else:
                # Add title, centered on the plot area
                if config is not None:  # then its been passed to this definition, so use that
                    fig.text(plot_middle, 0.98,
                             f"Qubit Spectroscopy Q{self.QubitIndex + 1}" +
                             f", {config['reps']}*{config['rounds']} avgs",
                             fontsize=24, ha='center', va='top')
                else:
                    fig.text(plot_middle, 0.98,
                             f"Qubit Spectroscopy Q{self.QubitIndex + 1}",
                             fontsize=24, ha='center', va='top')

                    # Adjust spacing
            plt.tight_layout()

            # Adjust the top margin to make room for the title
            plt.subplots_adjust(top=0.93)

            ### Save figure
            if self.save_figs:
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name + "_plots")
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" +
                                         f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)
            if return_fwhm:
                return largest_amp_curve_mean, y_data_fit, largest_amp_curve_fwhm
            else:
                return largest_amp_curve_mean, y_data_fit
        else:
            freqs = np.array(freqs)
            freq_q = freqs[np.argmax(I)]

            mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, fit_err = self.fit_lorenzian(I, Q, freqs,
                                                                                                              freq_q,sigma_guess)

            # Check if the returned values are all None
            if (mean_I is None and mean_Q is None and I_fit is None and Q_fit is None
                    and largest_amp_curve_mean is None and largest_amp_curve_fwhm is None):
                # If so, return None for the values in this definition as well
                return None, None, None

            # If we get here, the fit was successful and we can proceed with plotting
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            plt.rcParams.update({'font.size': 18})

            # I subplot
            ax1.plot(freqs, I, label='I', linewidth=2)
            ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=16)
            ax1.legend()

            # Q subplot
            ax2.plot(freqs, Q, label='Q', linewidth=2)
            ax2.set_xlabel("Qubit Frequency (MHz)", fontsize=20)
            ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
            ax2.tick_params(axis='both', which='major', labelsize=16)
            ax2.legend()
            # Plot the fits
            if self.plot_fit:
                ax1.plot(freqs, I_fit, 'r--', label='Lorentzian Fit')
                #ax1.axvline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)

                ax2.plot(freqs, Q_fit, 'r--', label='Lorentzian Fit')
                #ax2.axvline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)

            # Calculate the middle of the plot area
            plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

            if self.plot_fit:
                # Add title, centered on the plot area
                if config is not None:  # then its been passed to this definition, so use that
                    fig.text(plot_middle, 0.98,
                             f"Qubit Spectroscopy Q{self.QubitIndex + 1}, %.2f MHz" % largest_amp_curve_mean +
                             f" FWHM: {round(largest_amp_curve_fwhm, 1)}" +
                             f", {config['reps']}*{config['rounds']} avgs",
                             fontsize=24, ha='center', va='top')
                else:
                    fig.text(plot_middle, 0.98,
                             f"Qubit Spectroscopy Q{self.QubitIndex + 1}, %.2f MHz" % largest_amp_curve_mean +
                             f" FWHM: {round(largest_amp_curve_fwhm, 1)}" +
                             f", {self.config['reps']}*{self.config['rounds']} avgs",
                             fontsize=24, ha='center', va='top')
            else:
                # Add title, centered on the plot area
                if config is not None:  # then its been passed to this definition, so use that
                    fig.text(plot_middle, 0.98,
                             f"Qubit Spectroscopy Q{self.QubitIndex + 1}" +
                             f", {config['reps']}*{config['rounds']} avgs",
                             fontsize=24, ha='center', va='top')
                else:
                    fig.text(plot_middle, 0.98,
                             f"Qubit Spectroscopy Q{self.QubitIndex + 1}",
                             fontsize=24, ha='center', va='top')


                    # Adjust spacing
            plt.tight_layout()

            # Adjust the top margin to make room for the title
            plt.subplots_adjust(top=0.93)

            ### Save figure
            if self.save_figs:
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name + "_plots")
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" +
                                         f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)
            if return_fwhm:
                return largest_amp_curve_mean, I_fit, Q_fit, largest_amp_curve_fwhm
            else:
                return largest_amp_curve_mean, I_fit, Q_fit

    def plot_results_interweaved_cal(self, I, Q, freqs, gains,
                                     config=None, fig_quality=100, sigma_guess=1,
                                     return_fwhm=False, scaling=False,
                                     Ie=None, Ig=None, Qe=None, Qg=None):

        # --- helpers ---
        def _to_stack(x):
            """Return a list of np arrays; if x is already 2D-like (list of arrays), keep;
               if 1D or 2D np.ndarray, wrap into length-1 list for uniform handling."""
            if x is None:
                return None
            if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple, np.ndarray)):
                return [np.asarray(row) for row in x]
            else:
                return [np.asarray(x)]

        freqs = np.asarray(freqs)
        gains = np.asarray(gains)

        I_stack = _to_stack(I)
        Q_stack = _to_stack(Q)

        if scaling:
            # Expect calibration per trace. Allow single calibration to be broadcast to all traces.
            Ie_stack = _to_stack(Ie)
            Ig_stack = _to_stack(Ig)
            Qe_stack = _to_stack(Qe)
            Qg_stack = _to_stack(Qg)

            n_traces = len(I_stack)
            # Broadcast single calibration to all traces if user passed 1 set
            if len(Ie_stack) == 1 and n_traces > 1:
                Ie_stack = Ie_stack * n_traces
                Ig_stack = Ig_stack * n_traces
                Qe_stack = Qe_stack * n_traces
                Qg_stack = Qg_stack * n_traces

            # Basic checks
            assert len(Q_stack) == n_traces, "I and Q must have the same number of traces"
            assert Ie_stack is not None and len(Ie_stack) == n_traces, "Calibration lists must match number of traces"
            assert len(Ig_stack) == n_traces and len(Qe_stack) == n_traces and len(Qg_stack) == n_traces, \
                "Calibration lists must match number of traces"

            # 1) Per-trace calibration -> population (works for 1D or 2D arrays)
            pop_traces = []
            for k in range(n_traces):
                I_k = np.asarray(I_stack[k])
                Q_k = np.asarray(Q_stack[k])

                e_k = np.mean(Ie_stack[k] + 1j * Qe_stack[k])
                g_k = np.mean(Ig_stack[k] + 1j * Qg_stack[k])
                denom = np.abs(e_k - g_k) ** 2
                # protect against pathological calibration
                if denom == 0:
                    raise ValueError("Calibration |e-g| is zero for trace index {}.".format(k))

                z_k = I_k + 1j * Q_k
                pop_k = np.abs(((z_k - g_k) * (e_k - g_k)) / denom)  # same shape as I_k/Q_k
                pop_traces.append(pop_k)

            # 2) Average calibrated populations across traces
            pop_arr = np.stack(pop_traces, axis=0)  # shape: (n_traces, ...) -> 2D or 3D
            pop_mean = np.mean(pop_arr, axis=0)  # shape: 1D (freq) or 2D (gains, freq)
            # print("freqs: ", freqs)
            # print("gains: ", gains)
            # print("pop_mean: (gains, freq)", pop_mean)
            # 3) For Lorentzian fit, we need a 1D population vs freq.
            #    If we have a gain axis, average over gains.
            if pop_mean.ndim == 1:
                # old 1D case
                ydata_1d = pop_mean
            elif pop_mean.ndim == 2:
                # new 2D case: average over gains -> 1D vs frequency
                # pop_mean shape assumed (n_gains, n_freqs)
                ydata_1d = np.mean(pop_mean, axis=0)
            else:
                raise ValueError("Unexpected population array dimensionality: {}".format(pop_mean.ndim))

            freq_q = freqs[np.argmax(ydata_1d)]

            mean_y_data, y_data_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, fit_err = \
                self.fit_lorenzian_scaled(ydata_1d, freqs, freq_q, sigma_guess)

            if (mean_y_data is None and y_data_fit is None and
                    largest_amp_curve_mean is None and largest_amp_curve_fwhm is None):
                return None, None, None

            # --- plotting ---
            plt.rcParams.update({'font.size': 18})

            # If we have 2D data (gains × freqs), make a heatmap.
            if pop_mean.ndim == 2:
                n_gains, n_freqs = pop_mean.shape

                # Sanity checks
                if len(freqs) != n_freqs:
                    raise ValueError(
                        f"freqs length ({len(freqs)}) does not match population freq dimension ({n_freqs})"
                    )
                if len(gains) != n_gains:
                    raise ValueError(
                        f"gains length ({len(gains)}) does not match population gain dimension ({n_gains})"
                    )

                fig, ax = plt.subplots(1, 1, figsize=(10, 6))

                # imshow expects shape (Ny, Nx) = (len(freqs), len(gains))
                # pop_mean is (n_gains, n_freqs), so transpose to (n_freqs, n_gains)
                im = ax.imshow(
                    pop_mean.T,
                    origin='lower',
                    aspect='auto',
                    extent=(gains[0], gains[-1], freqs[0], freqs[-1]),
                    interpolation = 'nearest'
                )

                ax.set_xlabel("Gain", fontsize=20)
                ax.set_ylabel("Qubit Frequency (MHz)", fontsize=20)
                ax.tick_params(axis='both', which='major', labelsize=16)

                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("Qubit Population (avg)", fontsize=18)

                # # Optionally overlay the fitted peak (averaged over gain)
                # if self.plot_fit:
                #     ax.axhline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)

                # Title text
                plot_middle = (ax.get_position().x0 + ax.get_position().x1) / 2
                if self.plot_fit:
                    if config is not None:
                        fig.text(
                            plot_middle, 0.98,
                            f"Qubit Spectroscopy Q{self.QubitIndex + 1}, %.2f MHz" % largest_amp_curve_mean +
                            f" FWHM: {round(largest_amp_curve_fwhm, 1)}" +
                            f", {config['reps']}*{config['rounds']} avgs",
                            fontsize=16, ha='center', va='top'
                        )
                    else:
                        fig.text(
                            plot_middle, 0.98,
                            f"Qubit Spectroscopy Q{self.QubitIndex + 1}, %.2f MHz" % largest_amp_curve_mean +
                            f" FWHM: {round(largest_amp_curve_fwhm, 1)}" +
                            f", {self.config['reps']}*{self.config['rounds']} avgs",
                            fontsize=16, ha='center', va='top'
                        )
                else:
                    if config is not None:
                        fig.text(
                            plot_middle, 0.98,
                            f"Qubit Spectroscopy Q{self.QubitIndex + 1}" +
                            f", {config['reps']}*{config['rounds']} avgs",
                            fontsize=16, ha='center', va='top'
                        )
                    else:
                        fig.text(
                            plot_middle, 0.98,
                            f"Qubit Spectroscopy Q{self.QubitIndex + 1}",
                            fontsize=16, ha='center', va='top'
                        )

            else:
                # Fallback: old 1D-style plot (no gain axis)
                fig, ax1 = plt.subplots(1, 1, figsize=(10, 5))
                ax1.plot(freqs, ydata_1d, label='', linewidth=2)
                ax1.set_ylabel("Qubit Population (avg)", fontsize=20)
                ax1.set_xlabel("Qubit Frequency (MHz)", fontsize=20)
                ax1.tick_params(axis='both', which='major', labelsize=16)
                ax1.legend()

                if self.plot_fit:
                    ax1.plot(freqs, y_data_fit, 'r--', label='Lorentzian Fit')
                    #ax1.axvline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)

                plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2
                if self.plot_fit:
                    if config is not None:
                        fig.text(
                            plot_middle, 0.98,
                            f"Qubit Spectroscopy Q{self.QubitIndex + 1}, %.2f MHz" % largest_amp_curve_mean +
                            f" FWHM: {round(largest_amp_curve_fwhm, 1)}" +
                            f", {config['reps']}*{config['rounds']} avgs",
                            fontsize=14, ha='center', va='top'
                        )
                    else:
                        fig.text(
                            plot_middle, 0.98,
                            f"Qubit Spectroscopy Q{self.QubitIndex + 1}, %.2f MHz" % largest_amp_curve_mean +
                            f" FWHM: {round(largest_amp_curve_fwhm, 1)}" +
                            f", {self.config['reps']}*{self.config['rounds']} avgs",
                            fontsize=24, ha='center', va='top'
                        )
                else:
                    if config is not None:
                        fig.text(
                            plot_middle, 0.98,
                            f"Qubit Spectroscopy Q{self.QubitIndex + 1}" +
                            f", {config['reps']}*{config['rounds']} avgs",
                            fontsize=24, ha='center', va='top'
                        )
                    else:
                        fig.text(
                            plot_middle, 0.98,
                            f"Qubit Spectroscopy Q{self.QubitIndex + 1}",
                            fontsize=24, ha='center', va='top'
                        )

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)

            if self.save_figs:
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name + "_plots")
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(
                    outerFolder_expt,
                    f"R_{self.round_num}_"
                    + f"Q_{self.QubitIndex + 1}_"
                    + f"{formatted_datetime}_"
                    + self.expt_name
                    + f"_q{self.QubitIndex + 1}.png"
                )
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)

            if return_fwhm:
                return largest_amp_curve_mean, y_data_fit, largest_amp_curve_fwhm
            else:
                return largest_amp_curve_mean, y_data_fit

        else:
            # Non-scaling: handle 1D or 2D I/Q.
            I_arr = np.stack(I_stack, axis=0)  # (n_traces, ...) -> 2D or 3D
            Q_arr = np.stack(Q_stack, axis=0)

            I_mean_all = np.mean(I_arr, axis=0)  # 1D (freq) or 2D (gains, freq)
            Q_mean_all = np.mean(Q_arr, axis=0)

            # For fitting, reduce to 1D vs frequency if necessary
            if I_mean_all.ndim == 1:
                I_mean_1d = I_mean_all
                Q_mean_1d = Q_mean_all
            elif I_mean_all.ndim == 2:
                I_mean_1d = np.mean(I_mean_all, axis=0)  # average over gains
                Q_mean_1d = np.mean(Q_mean_all, axis=0)
            else:
                raise ValueError("Unexpected dimensionality for I/Q in non-scaling mode: {}".format(I_mean_all.ndim))

            freq_q = freqs[np.argmax(I_mean_1d)]

            mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, fit_err = \
                self.fit_lorenzian(I_mean_1d, Q_mean_1d, freqs, freq_q, sigma_guess)

            if (mean_I is None and mean_Q is None and I_fit is None and Q_fit is None
                    and largest_amp_curve_mean is None and largest_amp_curve_fwhm is None):
                return None, None, None

            plt.rcParams.update({'font.size': 18})

            # Keep original 1D plots based on averaged traces.
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

            ax1.plot(freqs, I_mean_1d, label='I (avg)', linewidth=2)
            ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=16)
            ax1.legend()

            ax2.plot(freqs, Q_mean_1d, label='Q (avg)', linewidth=2)
            ax2.set_xlabel("Qubit Frequency (MHz)", fontsize=20)
            ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
            ax2.tick_params(axis='both', which='major', labelsize=16)
            ax2.legend()

            if self.plot_fit:
                ax1.plot(freqs, I_fit, 'r--', label='Lorentzian Fit')
                #ax1.axvline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)
                ax2.plot(freqs, Q_fit, 'r--', label='Lorentzian Fit')
                #ax2.axvline(largest_amp_curve_mean, color='orange', linestyle='--', linewidth=2)

            plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2
            if self.plot_fit:
                if config is not None:
                    fig.text(
                        plot_middle, 0.98,
                        f"Qubit Spectroscopy Q{self.QubitIndex + 1}, %.2f MHz" % largest_amp_curve_mean +
                        f" FWHM: {round(largest_amp_curve_fwhm, 1)}" +
                        f", {config['reps']}*{config['rounds']} avgs",
                        fontsize=24, ha='center', va='top'
                    )
                else:
                    fig.text(
                        plot_middle, 0.98,
                        f"Qubit Spectroscopy Q{self.QubitIndex + 1}, %.2f MHz" % largest_amp_curve_mean +
                        f" FWHM: {round(largest_amp_curve_fwhm, 1)}" +
                        f", {self.config['reps']}*{self.config['rounds']} avgs",
                        fontsize=24, ha='center', va='top'
                    )
            else:
                if config is not None:
                    fig.text(
                        plot_middle, 0.98,
                        f"Qubit Spectroscopy Q{self.QubitIndex + 1}" +
                        f", {config['reps']}*{config['rounds']} avgs",
                        fontsize=24, ha='center', va='top'
                    )
                else:
                    fig.text(
                        plot_middle, 0.98,
                        f"Qubit Spectroscopy Q{self.QubitIndex + 1}",
                        fontsize=24, ha='center', va='top'
                    )

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)

            if self.save_figs:
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name + "_plots")
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(
                    outerFolder_expt,
                    f"R_{self.round_num}_"
                    + f"Q_{self.QubitIndex + 1}_"
                    + f"{formatted_datetime}_"
                    + self.expt_name
                    + f"_q{self.QubitIndex + 1}.png"
                )
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)

            if return_fwhm:
                return largest_amp_curve_mean, I_fit, Q_fit, largest_amp_curve_fwhm
            else:
                return largest_amp_curve_mean, I_fit, Q_fit

    def get_results(self, I, Q, freqs):
        freqs = np.array(freqs)
        freq_q = freqs[np.argmax(I)]

        mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, qspec_fit_err = self.fit_lorenzian(I, Q, freqs, freq_q)

        return largest_amp_curve_mean, I_fit, Q_fit, qspec_fit_err

    def get_results_Two_peaks(self, I, Q, freqs):
        freqs = np.array(freqs)
        freq_q = freqs[np.argmax(I)]

        I_fit, Q_fit, fit_err_I, fit_err_Q, mean_I_1, mean_I_2, mean_Q_1, mean_Q_2, sigma_I_1, sigma_I_2,  sigma_Q_1, sigma_Q_2, height_I_1, height_I_2, height_Q_1, height_Q_2, base_I_1, base_I_2, base_Q_1, base_Q_2 = self.fit_lorenzian_two_peaks(I, Q, freqs)

        return  mean_I_1, mean_I_2, mean_Q_1, mean_Q_2


    def lorentzian(self, f, f0, gamma, A, B):

        return A * gamma ** 2 / ((f - f0) ** 2 + gamma ** 2) + B

    def Two_lorentzians_Q(self, f, f0_1, gamma_1, A_1, B_1, f0_2, gamma_2, A_2, B_2):

        return -1*(A_1 * gamma_1 ** 2 / ((f - f0_1) ** 2 + gamma_1 ** 2) + B_1) - (A_2 * gamma_2 ** 2 / ((f - f0_2) ** 2 + gamma_2 ** 2) + B_2)

    def Two_lorentzians_I(self, f, f0_1, gamma_1, A_1, B_1, f0_2, gamma_2, A_2, B_2):

        return (A_1 * gamma_1 ** 2 / ((f - f0_1) ** 2 + gamma_1 ** 2) + B_1) + (A_2 * gamma_2 ** 2 / ((f - f0_2) ** 2 + gamma_2 ** 2) + B_2)

    def max_offset_difference_with_x(self, x_values, y_values, offset):
        max_average_difference = -1
        corresponding_x = None

        # average all 3 to avoid noise spikes
        for i in range(len(y_values) - 2):
            # group 3 vals
            y_triplet = y_values[i:i + 3]

            # avg differences for these 3 vals
            average_difference = sum(abs(y - offset) for y in y_triplet) / 3

            # see if this is the highest difference yet
            if average_difference > max_average_difference:
                max_average_difference = average_difference
                # x value for the middle y value in the 3 vals
                corresponding_x = x_values[i + 1]

        return corresponding_x, max_average_difference

    def fit_lorenzian(self, I, Q, freqs, freq_q, sigma_guess = 1):
        try:
            # Initial guesses for I and Q
            initial_guess_I = [freq_q, sigma_guess, np.max(I), np.min(I)]
            initial_guess_Q = [freq_q, sigma_guess, np.max(Q), np.min(Q)]

            # First round of fits (to get rough estimates)
            params_I, _ = curve_fit(self.lorentzian, freqs, I, p0=initial_guess_I)
            params_Q, _ = curve_fit(self.lorentzian, freqs, Q, p0=initial_guess_Q)

            # Use these fits to refine guesses
            x_max_diff_I, max_diff_I = self.max_offset_difference_with_x(freqs, I, params_I[3])
            x_max_diff_Q, max_diff_Q = self.max_offset_difference_with_x(freqs, Q, params_Q[3])
            initial_guess_I = [x_max_diff_I, sigma_guess, np.max(I), np.min(I)]
            initial_guess_Q = [x_max_diff_Q, sigma_guess, np.max(Q), np.min(Q)]

            # Second (refined) round of fits, this time capturing the covariance matrices
            params_I, cov_I = curve_fit(self.lorentzian, freqs, I, p0=initial_guess_I)
            params_Q, cov_Q = curve_fit(self.lorentzian, freqs, Q, p0=initial_guess_Q)

            # Create the fitted curves
            I_fit = self.lorentzian(freqs, *params_I)
            Q_fit = self.lorentzian(freqs, *params_Q)

            # Calculate errors from the covariance matrices
            fit_err_I = np.sqrt(np.diag(cov_I))
            fit_err_Q = np.sqrt(np.diag(cov_Q))

            # Extract fitted means and FWHM (assuming params[0] is the mean and params[1] relates to the width)
            mean_I = params_I[0]
            mean_Q = params_Q[0]
            fwhm_I = 2 * params_I[1]
            fwhm_Q = 2 * params_Q[1]

            # Calculate the amplitude differences from the fitted curves
            amp_I_fit = abs(np.max(I_fit) - np.min(I_fit))
            amp_Q_fit = abs(np.max(Q_fit) - np.min(Q_fit))

            # Choose which curve to use based on the input signal indicator
            if 'None' in self.signal or self.signal is None:
                if amp_I_fit > amp_Q_fit:
                    largest_amp_curve_mean = mean_I
                    largest_amp_curve_fwhm = fwhm_I
                    # error on the Q fit's center frequency (first parameter):
                    qspec_fit_err = fit_err_I[0]
                else:
                    largest_amp_curve_mean = mean_Q
                    largest_amp_curve_fwhm = fwhm_Q
                    qspec_fit_err = fit_err_Q[0]
            elif 'I' in self.signal:
                largest_amp_curve_mean = mean_I
                largest_amp_curve_fwhm = fwhm_I
                qspec_fit_err = fit_err_I[0]
            elif 'Q' in self.signal:
                largest_amp_curve_mean = mean_Q
                largest_amp_curve_fwhm = fwhm_Q
                qspec_fit_err = fit_err_Q[0]
            else:
                print('Invalid signal passed, please choose "I", "Q", or "None".')
                return None

        except Exception as e:
            if self.verbose: print("Error during Lorentzian fit:", e)
            self.logger.info(f'Error during Lorentzian fit: {e}')
            # Return all desired results including the error on the Q fit
            mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, qspec_fit_err = None, None, None, None, None,None,None
        return mean_I, mean_Q, I_fit, Q_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, qspec_fit_err

    def fit_lorenzian_scaled(self, y_data, freqs, freq_q, sigma_guess = 1):
        try:
            # Initial guesse
            initial_guess_y_data = [freq_q, sigma_guess, np.max(y_data), np.min(y_data)]


            # First round of fits (to get rough estimates)
            params_y_data, _ = curve_fit(self.lorentzian, freqs, y_data, p0=initial_guess_y_data)

            # Use these fits to refine guesses
            x_max_diff_y_data, max_diff_y_data = self.max_offset_difference_with_x(freqs, y_data, params_y_data[3])
            initial_guess_y_data = [x_max_diff_y_data, sigma_guess, np.max(y_data), np.min(y_data)]


            # Second (refined) round of fits, this time capturing the covariance matrices
            params_y_data, cov_y_data = curve_fit(self.lorentzian, freqs, y_data, p0=initial_guess_y_data)

            # Create the fitted curves
            y_data_fit = self.lorentzian(freqs, *params_y_data)

            # Calculate errors from the covariance matrices
            fit_err_y_data = np.sqrt(np.diag(cov_y_data))

            # Extract fitted means and FWHM (assuming params[0] is the mean and params[1] relates to the width)
            mean_y_data = params_y_data[0]
            fwhm_y_data = 2 * params_y_data[1]

            # Calculate the amplitude differences from the fitted curves
            amp_I_fit = abs(np.max(y_data_fit) - np.min(y_data_fit))

            # Choose which curve to use based on the input signal indicator
            largest_amp_curve_mean = mean_y_data
            largest_amp_curve_fwhm = fwhm_y_data
            # error on the Q fit's center frequency (first parameter):
            qspec_fit_err = fit_err_y_data[0]


        except Exception as e:
            if self.verbose: print("Error during Lorentzian fit:", e)
            self.logger.info(f'Error during Lorentzian fit: {e}')
            # Return all desired results including the error on the Q fit
            mean_y_data, y_data_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, qspec_fit_err = None, None, None, None, None,None,None
        return mean_y_data, y_data_fit, largest_amp_curve_mean, largest_amp_curve_fwhm, qspec_fit_err

    def fit_lorenzian_two_peaks(self, I, Q, freqs ):
        try:
            # Initial guesses for I and Q
            mean_guess_1 = 3450.75
            mean_guess_2 = 3450.86
            sigma_guess_1 = 0.08
            sigma_guess_2 = 0.08
            base1 = 71.5
            base2 = 71.5
            height1 = 4.5
            height2 = 4
            initial_guess_I = [freqs, mean_guess_1, sigma_guess_1, base1, height1, mean_guess_2, sigma_guess_2, base2, height2]
            initial_guess_Q = [freqs, mean_guess_1, sigma_guess_1, base1, height1, mean_guess_2, sigma_guess_2, base2, height2]

            # First round of fits (to get rough estimates)
            # params_I, _ = curve_fit(self.Two_lorentzians_I, freqs, I, p0=initial_guess_I)
            # params_Q, _ = curve_fit(self.Two_lorentzians_Q, freqs, Q, p0=initial_guess_Q)
            #
            # # Use these fits to refine guesses
            # x_max_diff_I, max_diff_I = self.max_offset_difference_with_x(freqs, I, params_I[3])
            # x_max_diff_Q, max_diff_Q = self.max_offset_difference_with_x(freqs, Q, params_Q[3])
            # initial_guess_I = [x_max_diff_I, sigma_guess, np.max(I), np.min(I)]
            # initial_guess_Q = [x_max_diff_Q, sigma_guess, np.max(Q), np.min(Q)]

            # Second (refined) round of fits, this time capturing the covariance matrices
            params_I, cov_I = curve_fit(self.Two_lorentzians_I, freqs, I, p0=initial_guess_I)
            params_Q, cov_Q = curve_fit(self.Two_lorentzians_Q, freqs, Q, p0=initial_guess_Q)

            # Create the fitted curves
            I_fit = self.Two_lorentzians_I(freqs, *params_I)
            Q_fit = self.Two_lorentzians_Q(freqs, *params_Q)

            # Calculate errors from the covariance matrices
            fit_err_I = np.sqrt(np.diag(cov_I))
            fit_err_Q = np.sqrt(np.diag(cov_Q))

            # Extract fitted means and FWHM (assuming params[0] is the mean and params[1] relates to the width)
            mean_I_1 = params_I[0]
            mean_I_2 = params_I[4]
            mean_Q_1 = params_Q[0]
            mean_Q_2 = params_Q[4]
            sigma_I_1= 2*params_I[1]
            sigma_I_2 = 2*params_I[5]
            sigma_Q_1 = 2*params_Q[1]
            sigma_Q_2 = 2*params_Q[5]
            height_I_1 = params_I[3]
            height_I_2 = params_I[7]
            height_Q_1 = params_Q[3]
            height_Q_2 = params_Q[7]
            base_I_1= params_I[2]
            base_I_2 = params_I[6]
            base_Q_1 = params_Q[2]
            base_Q_2 = params_Q[6]
            # fwhm_I_1 = 2 * params_I[1]
            # fwhm_I_2 = 2 * params_I[1]
            # fwhm_Q = 2 * params_Q[1]
            # fwhm_Q = 2 * params_Q[1]

            # Calculate the amplitude differences from the fitted curves
            amp_I_fit = abs(np.max(I_fit) - np.min(I_fit))
            amp_Q_fit = abs(np.max(Q_fit) - np.min(Q_fit))

            # Choose which curve to use based on the input signal indicator
            # if 'None' in self.signal or self.signal is None:
            #     if amp_I_fit > amp_Q_fit:
            #         largest_amp_curve_mean = mean_I
            #         largest_amp_curve_fwhm = fwhm_I
            #         # error on the Q fit's center frequency (first parameter):
            #         qspec_fit_err = fit_err_I[0]
            #     else:
            #         largest_amp_curve_mean = mean_Q
            #         largest_amp_curve_fwhm = fwhm_Q
            #         qspec_fit_err = fit_err_Q[0]
            # elif 'I' in self.signal:
            #     largest_amp_curve_mean = mean_I
            #     largest_amp_curve_fwhm = fwhm_I
            #     qspec_fit_err = fit_err_I[0]
            # elif 'Q' in self.signal:
            #     largest_amp_curve_mean = mean_Q
            #     largest_amp_curve_fwhm = fwhm_Q
            #     qspec_fit_err = fit_err_Q[0]
            # else:
            #     print('Invalid signal passed, please choose "I", "Q", or "None".')
            #     return None

            # Return all desired results including the error on the Q fit
        except Exception as e:
            if self.verbose: print("Error during Lorentzian fit:", e)
            self.logger.info(f'Error during Lorentzian fit: {e}')
        # return None, None, None, None, None, None, None
        return I_fit, Q_fit, fit_err_I, fit_err_Q, mean_I_1, mean_I_2, mean_Q_1, mean_Q_2, sigma_I_1, sigma_I_2,  sigma_Q_1, sigma_Q_2, height_I_1, height_I_2, height_Q_1, height_Q_2, base_I_1, base_I_2, base_Q_1, base_Q_2

        # except Exception as e:
        #     if self.verbose: print("Error during Lorentzian fit:", e)
        #     self.logger.info(f'Error during Lorentzian fit: {e}')
        #     return None, None,None,None,None,None,None

    def create_folder_if_not_exists(self, folder_path):
        import os
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)


class PulseProbeSpectroscopyProgram(AveragerProgramV2):
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

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_pulse(ch=res_ch, name="qze_pulse",  ro_ch=ro_ch,
                       style="const",
                       length=cfg['qubit_length_ge']+cfg["readout_pulse_delay"],#+3us for res ring up time
                       freq=cfg['res_freq_qze'],
                       phase=cfg['res_phase_qze'],
                       gain=QickSweep1D("gain_loop", cfg["gain_start"], cfg["gain_stop"])
                       )

        self.add_loop("freqloop", cfg["steps"])
        self.add_loop("gain_loop", cfg["gain_steps"])  # inn
    def _body(self, cfg):
        self.pulse(ch=cfg['res_ch'], name="qze_pulse", t=0)
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=cfg["readout_pulse_delay"])  # play probe pulse after ring up
        self.delay_auto(t=cfg["readout_pulse_delay"], tag='waiting')  # Wait til qubit pulse is done and resonator rings down before proceeding
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class PulseProbeSpectroscopyProgramSingleGain(AveragerProgramV2):
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

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_pulse(ch=res_ch, name="qze_pulse",  ro_ch=ro_ch,
                       style="const",
                       length=cfg['qubit_length_ge']+3,#+3us for res ring up time
                       freq=cfg['res_freq_qze'],
                       phase=cfg['res_phase_qze'],
                       gain=cfg['res_gain_qze']
                       )

        self.add_loop("freqloop", cfg["steps"])




    def _body(self, cfg):
        self.pulse(ch=cfg['res_ch'], name="qze_pulse", t=0)
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=3)  # play probe pulse after ring up
        self.delay_auto(t=5, tag='waiting')  # Wait til qubit pulse is done and resonator rings down before proceeding
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])
class PulseProbeSpectroscopyProgramFlatTop(AveragerProgramV2):
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
        self.add_loop("freqloop", cfg["steps"])
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_gauss(ch=res_ch, name="qze_flat_top", sigma=0.05,
                       length=0.2, even_length=True)
        self.add_pulse(ch=res_ch, name="qze_pulse",
                       style="flat_top",
                       envelope="qze_flat_top",
                       length=cfg['qubit_length_ge'],
                       freq=cfg['res_freq_qze'],
                       phase=cfg['res_phase_qze'],
                       gain=cfg['res_gain_qze']
                       )


    def _body(self, cfg):
        self.pulse(ch=cfg['res_ch'], name="qze_pulse", t=0)
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse
        self.delay_auto(t=5, tag='waiting')  # Wait til qubit pulse is done and resonator rings down before proceeding
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class PulseProbeSpectroscopyProgram_WithStark(AveragerProgramV2):
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
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['qubit_length_ge']-0.11,#
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_pulse(ch=res_ch, name="proj_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'] - 0.11,#
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )
        self.add_loop("freqloop", cfg["steps"])

    def _body(self, cfg):
        self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse
        self.delay_auto(t=0, tag='waiting')  # Wait til qubit pulse is done before proceeding
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class PulseProbeSpectroscopyProgram_WithStark_WaitForRingUp(AveragerProgramV2):
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
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['qubit_length_ge'] - cfg['qubit_pi_len'],  #
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_pulse(ch=res_ch, name="proj_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'] - cfg['qubit_pi_len'] + cfg['res_ring_up_time'],  #add ring up time, 2us
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )
        self.add_loop("freqloop", cfg["steps"])

    def _body(self, cfg):
        self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=cfg['res_ring_up_time'])  # play probe pulse after res ring up to get saturated resonator stark/zeno tone
        self.delay_auto(t=0.0, tag='wait')  # wait for stark tone to finish
        self.delay(t=cfg['res_ring_up_time']) #wait for ring down
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0) #ring down time, then res readout pulse
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


class ResStark(AveragerProgramV2):
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

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_pulse(ch=res_ch, name="qze_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['qubit_length_ge'] + cfg["readout_pulse_delay"],  # +3us for res ring up time
                       freq=cfg['res_freq_qze'],
                       phase=cfg['res_phase_qze'],
                       gain=QickSweep1D("gain_loop", cfg["gain_start"], cfg["gain_stop"])
                       )

        self.add_loop("freqloop", cfg["steps"])
        self.add_loop("gain_loop", cfg["gain_steps"])  # inn

    def _body(self, cfg):
        self.pulse(ch=cfg['res_ch'], name="qze_pulse", t=0)
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse",
                   t=cfg["readout_pulse_delay"])  # play probe pulse after ring up
        self.delay_auto(t=cfg["readout_pulse_delay"],
                        tag='waiting')  # Wait til qubit pulse is done and resonator rings down before proceeding
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class OffResonantQSpecDrive(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        self.declare_readout(ch=cfg['ro_ch'], length=cfg['res_length'])

        self.add_readoutconfig(ch=ro_ch, name="myro", freq=cfg['res_freq_ge'], gen_ch=res_ch, outsel='product')
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,  style="const", length=cfg["res_length"], freq=cfg['res_freq_ge'], phase=cfg['ro_phase'], gain=cfg['res_gain_ge'])

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch,style="const", length=cfg['qubit_pulse_length'],  freq=cfg['qubit_freq_ge'], phase=cfg['ro_phase'],gain=cfg['qubit_pulse_gain'],)
        self.add_pulse(ch=qubit_ch, name="qubit_stark_pulse", ro_ch=ro_ch, style="const", length=cfg['qubit_stark_pulse_length'], freq=cfg["qubit_stark_freq"], phase=cfg['qubit_phase'],
                       gain=QickSweep1D("stark_gain_loop", cfg["start_qubit_stark_gain"], cfg["stop_qubit_stark_gain"]))
        self.add_loop("freqloop", cfg["steps"])
        self.add_loop("stark_gain_loop", cfg["stark_gain_steps"])
    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_stark_pulse", t=0)
        self.delay_auto(t=0, tag='waiting2')
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)
        self.delay_auto(t=0, tag='waiting')
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class MuxProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        self.declare_readout(ch=cfg['ro_ch'], length=cfg['res_length'])

        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'],
                               gen_ch=res_ch,
                               outsel='product')

        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        # print(cfg["res_length"],cfg['ro_phase'],cfg['res_gain_ge'])
        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        # same gaussian qubit pulse as OffResonantQSpecDrive, but at the resonator frequency
        # self.add_gauss(ch=res_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=res_ch, name="qubit_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['sigma'] * 4,
                       freq=cfg['res_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )

        # off-resonant stark pulse — at the resonator frequency, fixed gain (Python sweeps it)
        self.add_pulse(ch=res_ch, name="qubit_stark_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['qubit_stark_pulse_length'],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_stark_gain_tof'])

    def _body(self, cfg):
        # trigger at t=0 so the decimated capture starts at the beginning of the sequence
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=0, ddr4=True)
        self.pulse(ch=cfg['res_ch'], name="qubit_pulse", t=0)
        self.delay_auto(t=0.1, ros=False, tag='waiting')
        self.pulse(ch=cfg['res_ch'], name="qubit_stark_pulse", t=0)
        self.delay_auto(t=0.1, ros=False, tag='waiting2')
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
class OffResonantQSpecDriveTOF(AveragerProgramV2):

    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        # the qubit-channel pulses are now emitted at the resonator frequency for this loopback
        # timing check, so declare the qubit gen in the resonator's Nyquist zone (nqz_res) so it
        # can actually produce ~res_freq_ge instead of the ~qubit_freq_ge zone.
        # self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_res'])

        # long readout window so the decimated capture spans the whole pulse sequence
        self.declare_readout(ch=ro_ch, length=cfg['tof_readout_length'])

        # downconvert at the resonator frequency: every pulse is played at res_freq_ge so the
        # whole sequence shows up on the readout line / ADC for a pure timing+amplitude check
        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'],
                               gen_ch=res_ch,
                               outsel='product')
        self.send_readoutconfig(ch=ro_ch, name="myro", t=0)

        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        # same gaussian qubit pulse as OffResonantQSpecDrive, but at the resonator frequency
        # self.add_gauss(ch=res_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=res_ch, name="qubit_pulse",ro_ch=ro_ch,
                       style="const",
                       length=cfg['sigma'] * 4,
                       freq=cfg['res_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )

        # off-resonant stark pulse — at the resonator frequency, fixed gain (Python sweeps it)
        self.add_pulse(ch=res_ch, name="qubit_stark_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['qubit_stark_pulse_length'],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_stark_gain_tof'])

    def _body(self, cfg):
        # trigger at t=0 so the decimated capture starts at the beginning of the sequence
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=0, ddr4=True)
        self.pulse(ch=cfg['res_ch'], name="qubit_pulse", t=0)
        self.delay_auto(t=0.1, ros=False, tag='waiting')
        self.pulse(ch=cfg['res_ch'], name="qubit_stark_pulse", t=0)
        self.delay_auto(t=0.1, ros=False, tag='waiting2')
        self.pulse(ch=cfg['res_ch'], name="res_pulse")

class OffResonantQSpecDriveTOF_Q(AveragerProgramV2):

    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']

        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'])
        # the qubit-channel pulses are now emitted at the resonator frequency for this loopback
        # timing check, so declare the qubit gen in the resonator's Nyquist zone (nqz_res) so it
        # can actually produce ~res_freq_ge instead of the ~qubit_freq_ge zone.
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_res'])

        # long readout window so the decimated capture spans the whole pulse sequence
        self.declare_readout(ch=ro_ch, length=cfg['tof_readout_length'])

        # downconvert at the resonator frequency: every pulse is played at res_freq_ge so the
        # whole sequence shows up on the readout line / ADC for a pure timing+amplitude check
        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'],
                               gen_ch=res_ch,
                               outsel='product')
        self.send_readoutconfig(ch=ro_ch, name="myro", t=0)

        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']*2
                       )

        # same gaussian qubit pulse as OffResonantQSpecDrive, but at the resonator frequency
        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)

        self.add_pulse(ch=qubit_ch, name="qubit_pulse",ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp",
                       # length=cfg['sigma'] * 4,
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=1,#cfg['pi_amp'],
                       )

        self.add_pulse(ch=qubit_ch, name="qubit_stark_pulse", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp",
                       # length=cfg['qubit_stark_pulse_length'],
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=1,)#cfg['pi_amp'])#cfg['qubit_stark_gain_tof'])

    def _body(self, cfg):
        # trigger at t=0 so the decimated capture starts at the beginning of the sequence
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=0, ddr4=True)
        self.pulse(ch=cfg['qubit_ch'], name="qubit_pulse", t=0)
        self.delay_auto(t=10, ros=False, tag='waiting')
        self.pulse(ch=cfg['qubit_ch'], name="qubit_stark_pulse", t=0)
        self.delay_auto(t=1, ros=False, tag='waiting2')
        self.pulse(ch=cfg['res_ch'], name="res_pulse")