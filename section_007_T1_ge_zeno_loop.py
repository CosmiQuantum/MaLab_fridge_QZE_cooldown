import numpy as np
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from build_task import *
from build_state import *
from expt_config import *
from system_config import *
import copy
import visdom
import logging
from section_005_single_shot_ge import SingleShotProgram_g, SingleShotProgram_e
class T1ProgramIBMZeno(AveragerProgramV2):
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
        self.add_pulse(ch=res_ch, name="res_pulse",ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        self.add_pulse(ch=res_ch, name="qze_pulse",
                       style="const",
                       length=QickSweep1D("waitloop", cfg['start'], cfg['stop']),
                       freq=cfg['res_freq_qze'],
                       phase=cfg['res_phase_qze'],
                       gain=cfg['res_gain_qze']
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

        self.add_loop("waitloop", cfg["steps"])

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse
        self.delay_auto(tag='wait_pi_pulse')                          # wait for it to be done, now qubit is in e
        self.pulse(ch=cfg['res_ch'], name="qze_pulse", t=0.01)           # play res pulse that has same length as wait_time
        self.delay_auto(tag='wait_qze_pulse')                         # wait for that pulse to finish
        self.delay_auto(t=5, tag='wait_for_ring_down')
        self.pulse(ch=cfg['res_ch'], name="res_pulse")           # play readout pulse after 5 us for ring down
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])
class T1ProgramIBMZenoFlatTop(AveragerProgramV2):
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
        self.add_pulse(ch=res_ch, name="res_pulse",ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        self.add_gauss(ch=res_ch, name="qze_flat_top", sigma=0.05,
                       length=0.2, even_length=True)
        self.add_pulse(ch=res_ch, name="qze_pulse",
                       style="flat_top",
                       envelope="qze_flat_top",
                       length=QickSweep1D("waitloop", cfg['start'], cfg['stop']),
                       freq=cfg['res_freq_qze'],
                       phase=cfg['res_phase_qze'],
                       gain=cfg['res_gain_qze']
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

        self.add_loop("waitloop", cfg["steps"])

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse
        self.delay_auto(tag='wait_pi_pulse')                          # wait for it to be done, now qubit is in e
        self.pulse(ch=cfg['res_ch'], name="qze_pulse", t=0.01)           # play res pulse that has same length as wait_time
        self.delay_auto(tag='wait_qze_pulse')                         # wait for that pulse to finish
        self.delay_auto(t=5, tag='wait_for_ring_down')
        self.pulse(ch=cfg['res_ch'], name="res_pulse")           # play readout pulse after 5 us for ring down
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


class T1Measurement_with_Zeno_loop:
    def __init__(self, QubitIndex, number_of_qubits,  outerFolder, round_num, signal, save_figs, experiment = None,
                 live_plot = None, fit_data = None, increase_qubit_reps = False, qubit_to_increase_reps_for = None,
                 multiply_qubit_reps_by = 0, verbose = False, logger = None, qick_verbose=True, save_shots=False,
                 set_relax_delay=False, relax_delay=1000,unmasking_resgain = False, zeno_pulse_gain=0, slice=20):

        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder
        self.fit_data = fit_data
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.experiment = experiment
        self.expt_name = "T1_IBM_qze_loop"
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num
        self.live_plot = live_plot
        self.signal = signal
        self.save_figs = save_figs
        self.verbose = verbose
        self.save_shots = save_shots
        self.set_relax_delay = set_relax_delay
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")
        self.zeno_pulse_gain = zeno_pulse_gain

        self.exp_cfg["wait_time"] = slice
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")


        self.experiment.readout_cfg['res_gain_qze'] = self.zeno_pulse_gain
        self.experiment.readout_cfg['res_freq_qze'] = self.experiment.readout_cfg['res_freq_ge']
        self.experiment.readout_cfg['res_phase_qze'] = self.experiment.readout_cfg['res_phase']



        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            if increase_qubit_reps:
                    if self.QubitIndex==qubit_to_increase_reps_for:
                        self.logger.info(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        if self.verbose: print(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        self.config["reps"] *= multiply_qubit_reps_by
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} T1 configuration: {self.config}')
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} T1 configuration: {self.config}')
            if self.set_relax_delay:
                self.config['relax_delay'] = relax_delay
                print(f'set t1 relax delay to {relax_delay} us')

    def run(self, thresholding=False, scaling=False,qze_pulse='const'):
        now = datetime.datetime.now()
        if qze_pulse=='flat_top':
            t1 = T1ProgramIBMZenoFlatTop(self.experiment.soccfg, reps=self.config['reps'],
                                  final_delay=self.config['relax_delay'], cfg=self.config)

        else:
            t1 = T1ProgramIBMZeno(self.experiment.soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'], cfg=self.config)

        if scaling:
            q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            ss_exp_cfg = add_qubit_experiment(expt_cfg, 'Readout_Optimization', self.QubitIndex)
            ss_config = {**q_config[self.Qubit], **ss_exp_cfg}

            ssp_g = SingleShotProgram_g(self.experiment.soccfg, reps=1, final_delay=ss_config['relax_delay'],
                                        cfg=ss_config)
            ssp_e = SingleShotProgram_e(self.experiment.soccfg, reps=1, final_delay=ss_config['relax_delay'],
                                        cfg=ss_config)

        if self.live_plot:
            I, Q, delay_times = self.live_plotting(t1, thresholding)
        else:
            if thresholding:
                iq_list = t1.acquire(self.experiment.soc, rounds=self.config['rounds'],
                                           threshold=self.experiment.readout_cfg["threshold"],
                                           angle=self.experiment.readout_cfg["ro_phase"], progress=True)
            else:
                Is_all = []
                Qs_all = []
                ss_I_g_all = []
                ss_Q_g_all = []
                ss_I_e_all = []
                ss_Q_e_all = []
                for round_num in range(self.config["rounds"]):
                    iq_list = t1.acquire(self.experiment.soc, rounds=1, progress=True)
                    iq_list = iq_list[0][0].T
                    I = (iq_list[0])
                    Q = (iq_list[1])
                    Is_all.append(I)
                    Qs_all.append(Q)

                    if scaling:
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

            delay_times = t1.get_pulse_param(pulsename='qze_pulse', parname='length', as_array=True)


            if self.plot_results:
                self.plot_results_interweaved_cal(Is_all, Qs_all, delay_times, now, scaling=scaling, Ie = ss_I_e_all
                                  , Ig = ss_I_g_all, Qe = ss_Q_e_all, Qg = ss_Q_g_all)
            q1_fit_exponential, T1_est, T1_err = None, None, None
            return T1_est, T1_err, Is_all, Qs_all, delay_times, q1_fit_exponential, self.config, ss_Q_e_all\
                , ss_Q_g_all,ss_I_e_all, ss_I_g_all


    def live_plotting(self, t1, thresholding):
        I = Q = expt_mags = expt_phases = expt_pop = None
        viz = visdom.Visdom()
        if not viz.check_connection(timeout_seconds=5):
            raise RuntimeError("Visdom server not connected!")
        for ii in range(self.config["rounds"]):
            #iq_list = t1.acquire(self.experiment.soc, rounds=1, progress=True)
            if thresholding:
                iq_list = t1.acquire(self.experiment.soc, rounds=1,
                                           threshold=self.experiment.readout_cfg["threshold"],
                                           angle=self.experiment.readout_cfg["ro_phase"], progress=True)
            else:
                iq_list = t1.acquire(self.experiment.soc, rounds=1, progress=True)
            delay_times = t1.get_time_param('wait', "t", as_array=True)
            iq_list = iq_list[0][0].T
            this_I = (iq_list[0])
            this_Q = (iq_list[1])

            if I is None:  # ii == 0
                I, Q = this_I, this_Q
            else:
                I = (I * ii + this_I) / (ii + 1.0)
                Q = (Q * ii + this_Q) / (ii + 1.0)

            viz.line(X=delay_times, Y=I, opts=dict(height=400, width=700, title='T1 I', showlegend=True, xlabel='expt_pts'),win='T1_I')
            viz.line(X=delay_times, Y=Q, opts=dict(height=400, width=700, title='T1 Q', showlegend=True, xlabel='expt_pts'),win='T1_Q')
        return I, Q, delay_times

    def exponential(self, x, a, b, c, d):
        return a * np.exp(-(x - b) / c) + d

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def exponential(self, x, a, b, c, d):
        return a * np.exp(- (x - b) / c) + d

    def t1_fit(self, I, Q, delay_times, mag=None):
        if 'I' in self.signal:
            signal = I
            plot_sig = 'I'
        elif 'Q' in self.signal:
            signal = Q
            plot_sig = 'Q'
        elif 'Mag' in self.signal:
            if mag is not None:
                signal=mag
            else:
                signal = np.hypot(I, Q)
            plot_sig = 'Q'
        else:
            if abs(I[-1] - I[0]) > abs(Q[-1] - Q[0]):
                signal = I
                plot_sig = 'I'
            else:
                signal = Q
                plot_sig = 'Q'

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

        return q1_fit_exponential, T1_err, T1_est, plot_sig

    def plot_results_interweaved_cal(self, I, Q, delay_times, now, config=None, fig_quality=100,
                     scaling=False, Ie=None, Ig=None, Qe=None, Qg=None):
        def _as_sets(x):
            """
            Normalize input to a list of 1D numpy arrays.
            - If x is 1D -> returns [np.asarray(x)]  (backwards compatible)
            - If x is list-of-lists/2D -> returns [np.asarray(row) for row in x]
            """
            arr = np.asarray(x, dtype=object)
            if arr.ndim == 1 or (arr.ndim == 2 and arr.dtype != object and arr.shape[0] == 1):
                return [np.asarray(x, dtype=float)]
            # assume iterable of iterables
            return [np.asarray(xx, dtype=float) for xx in x]

        def _calibrated_population(I, Q, Ie, Ig, Qe, Qg):
            """
            Single-dataset calibration: returns population trace for one IQ dataset.
            """
            IQ = np.asarray(I) + 1j * np.asarray(Q)
            e = np.mean(np.asarray(Ie) + 1j * np.asarray(Qe))
            g = np.mean(np.asarray(Ig) + 1j * np.asarray(Qg))
            return np.abs((IQ - g) * (e - g) / (np.abs(e - g) ** 2))

        def calibrate_and_average(I, Q, Ie, Ig, Qe, Qg):
            """
            Vectorized over datasets:
            - I,Q are list-of-lists (or 1D).
            - Ie,Ig,Qe,Qg are list-of-lists (or 1D).
            Returns:
              y_avg: averaged calibrated population (1D array)
              y_each: list of calibrated populations, one per dataset
            """
            I_sets = _as_sets(I)
            Q_sets = _as_sets(Q)
            Ie_sets = _as_sets(Ie)
            Ig_sets = _as_sets(Ig)
            Qe_sets = _as_sets(Qe)
            Qg_sets = _as_sets(Qg)

            # datasets correspond 1:1
            pops = [
                _calibrated_population(Ii, Qi, Iei, Igi, Qei, Qgi)
                for Ii, Qi, Iei, Igi, Qei, Qgi in zip(I_sets, Q_sets, Ie_sets, Ig_sets, Qe_sets, Qg_sets)
            ]
            y_avg = np.mean(np.stack(pops, axis=0), axis=0)
            return y_avg, pops

        def average_IQ(I, Q):
            """
            Averages raw I,Q over datasets (useful when scaling=False).
            """
            I_sets = _as_sets(I)
            Q_sets = _as_sets(Q)
            I_avg = np.mean(np.stack(I_sets, axis=0), axis=0)
            Q_avg = np.mean(np.stack(Q_sets, axis=0), axis=0)
            return I_avg, Q_avg
        if scaling:
            # --- NEW: handle list-of-lists calibration + average calibrated populations ---
            if any(v is None for v in (Ie, Ig, Qe, Qg)):
                raise ValueError("scaling=True requires Ie, Ig, Qe, Qg (each list-of-lists or 1D).")

            ydata, _ = calibrate_and_average(I, Q, Ie, Ig, Qe, Qg)

            fig, (ax1) = plt.subplots(1, 1)
            plt.rcParams.update({'font.size': 18})

            # Center title on the axes area
            plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

            if self.fit_data:
                # Fit using magnitude data (already a population)
                self.signal = 'Mag'
                q1_fit_exponential, T1_err, T1_est, plot_sig = self.t1_fit(None, None, delay_times, mag=ydata)

                ax1.plot(delay_times, q1_fit_exponential, '-', color='red', linewidth=3, label="Fit")

                if config is not None:
                    fig.text(
                        plot_middle, 0.98,
                        f"Q{self.QubitIndex + 1} "
                        f"T1={T1_est:.2f} us, {float(config['reps'])}*{float(config['rounds'])} avgs, "
                        f"Zeno pulse gain {round(self.config['res_gain_qze'], 3)}",
                        fontsize=14, ha='center', va='top'
                    )
                else:
                    fig.text(
                        plot_middle, 0.98,
                        f"T1 Q{self.QubitIndex + 1}, T1 {T1_est:.2f} us, "
                        f"{self.config['reps']}*{self.config['rounds']} avgs, "
                        f"Zeno pulse gain {round(self.config['res_gain_qze'], 3)}",
                        fontsize=14, ha='center', va='top'
                    )
            else:
                if config is not None:
                    fig.text(
                        plot_middle, 0.98,
                        f"T1 Q{self.QubitIndex + 1}, {float(config['reps'])}*{float(config['rounds'])} avgs, "
                        f"Zeno pulse gain {round(self.config['res_gain_qze'], 3)}",
                        fontsize=14, ha='center', va='top'
                    )
                else:
                    fig.text(plot_middle, 0.98, f"T1 Q{self.QubitIndex + 1}", fontsize=24, ha='center', va='top')
                q1_fit_exponential = T1_est = T1_err = None

            # Plot averaged calibrated population
            ax1.plot(delay_times, ydata, label="Gain (a.u.)", linewidth=2)
            ax1.set_ylabel("Qubit Population", fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=16)

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)

            if self.save_figs:
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(
                    outerFolder_expt,
                    f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}.png"
                )
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)

        else:
            # --- OLD behavior but averages across datasets if list-of-lists ---
            I_avg, Q_avg = average_IQ(I, Q)

            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            plt.rcParams.update({'font.size': 18})
            plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

            if self.fit_data:
                q1_fit_exponential, T1_err, T1_est, plot_sig = self.t1_fit(I_avg, Q_avg, delay_times)

                if 'I' in plot_sig:
                    ax1.plot(delay_times, q1_fit_exponential, '-', color='red', linewidth=3, label="Fit")
                else:
                    ax2.plot(delay_times, q1_fit_exponential, '-', color='red', linewidth=3, label="Fit")

                if config is not None:
                    fig.text(
                        plot_middle, 0.98,
                        f"Q{self.QubitIndex + 1} T1={T1_est:.2f} us, {float(config['reps'])}*{float(config['rounds'])} avgs,",
                        fontsize=24, ha='center', va='top'
                    )
                else:
                    fig.text(
                        plot_middle, 0.98,
                        f"T1 Q{self.QubitIndex + 1}, T1 {T1_est:.2f} us, {self.config['reps']}*{self.config['rounds']} avgs,",
                        fontsize=24, ha='center', va='top'
                    )
            else:
                if config is not None:
                    fig.text(
                        plot_middle, 0.98,
                        f"T1 Q{self.QubitIndex + 1}, {float(config['reps'])}*{float(config['rounds'])} avgs,",
                        fontsize=24, ha='center', va='top'
                    )
                else:
                    fig.text(plot_middle, 0.98, f"T1 Q{self.QubitIndex + 1}", fontsize=24, ha='center', va='top')
                q1_fit_exponential = T1_est = T1_err = None

            # I subplot
            ax1.plot(delay_times, I_avg, label="Gain (a.u.)", linewidth=2)
            ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=16)

            # Q subplot
            ax2.plot(delay_times, Q_avg, label="Q", linewidth=2)
            ax2.set_xlabel("Delay time (us)", fontsize=20)
            ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
            ax2.tick_params(axis='both', which='major', labelsize=16)

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)

            if self.save_figs:
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(
                    outerFolder_expt,
                    f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}.png"
                )
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)

    def plot_results(self, I, Q, delay_times, now, config = None, fig_quality =100,scaling=False, Ie=None, Ig=None, Qe=None, Qg=None):

        if scaling:
            e = np.mean((Ie + 1j * Qe))
            g = np.mean((Ig + 1j * Qg))
            ### Normalization ###
            pop_norm = abs(((I + 1j * Q) - g) * (e - g) / abs(e - g) ** 2)
            ydata = pop_norm
            fig, (ax1) = plt.subplots(1, 1)

            plt.rcParams.update({'font.size': 18})

            # Calculate the middle of the plot area
            plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

            if self.fit_data:
                self.signal = 'Mag'
                q1_fit_exponential, T1_err, T1_est, plot_sig = self.t1_fit(I, Q, delay_times, mag=ydata)

                ax1.plot(delay_times, q1_fit_exponential, '-', color='red', linewidth=3, label="Fit")

                # Add title, centered on the plot area
                if config is not None:
                    fig.text(plot_middle, 0.98,
                             f"Q{self.QubitIndex + 1} " + f"T1={T1_est:.2f} us" + f", {float(config['reps'])}*{float(config['rounds'])} avgs, Zeno pulse gain {round(self.config['res_gain_qze'],3)}",
                             fontsize=14, ha='center',
                             va='top')  # , pi gain %.2f" % float(config['pi_amp']) + f", {float(config['sigma']) * 1000} ns sigma
                else:
                    fig.text(plot_middle, 0.98,
                             f"T1 Q{self.QubitIndex + 1}, T1 %.2f us" % T1_est + f", {self.config['reps']}*{self.config['rounds']} avgs, Zeno pulse gain {round(self.config['res_gain_qze'],3)}",
                             fontsize=14, ha='center', va='top')

            else:
                if config is not None:
                    fig.text(plot_middle, 0.98,
                             f"T1 Q{self.QubitIndex + 1}" + f", {float(config['reps'])}*{float(config['rounds'])} avgs, Zeno pulse gain {round(self.config['res_gain_qze'],3)}",
                             fontsize=14, ha='center',
                             va='top')  # , pi gain %.2f" % float(config['pi_amp']) + f", {float(config['sigma']) * 1000} ns sigma"   you can put this back once you save configs properly for when replotting
                else:
                    fig.text(plot_middle, 0.98,
                             f"T1 Q{self.QubitIndex + 1}",
                             fontsize=24, ha='center', va='top')
                q1_fit_exponential = None
                T1_est = None
                T1_err = None

            # I subplot
            ax1.plot(delay_times, ydata, label="Gain (a.u.)", linewidth=2)
            ax1.set_ylabel("Qubit Population", fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=16)
            # ax1.axvline(freq_q, color='orange', linestyle='--', linewidth=2)


            # Adjust spacing
            plt.tight_layout()

            # Adjust the top margin to make room for the title
            plt.subplots_adjust(top=0.93)
            if self.save_figs:
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(outerFolder_expt,
                                         f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')  # , facecolor='white'
            plt.close(fig)
        else:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            plt.rcParams.update({'font.size': 18})

            # Calculate the middle of the plot area
            plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2


            if self.fit_data:
                q1_fit_exponential, T1_err, T1_est, plot_sig = self.t1_fit(I, Q, delay_times)

                if 'I' in plot_sig:
                    ax1.plot(delay_times, q1_fit_exponential, '-', color='red', linewidth=3, label="Fit")
                else:
                    ax2.plot(delay_times, q1_fit_exponential, '-', color='red', linewidth=3, label="Fit")

                # Add title, centered on the plot area
                if config is not None:
                    fig.text(plot_middle, 0.98,
                             f"Q{self.QubitIndex + 1} " + f"T1={T1_est:.2f} us" + f", {float(config['reps'])}*{float(config['rounds'])} avgs,",
                             fontsize=24, ha='center',
                             va='top')  # , pi gain %.2f" % float(config['pi_amp']) + f", {float(config['sigma']) * 1000} ns sigma
                else:
                    fig.text(plot_middle, 0.98,
                             f"T1 Q{self.QubitIndex + 1}, T1 %.2f us" % T1_est + f", {self.config['reps']}*{self.config['rounds']} avgs,",
                             fontsize=24, ha='center', va='top')

            else:
                if config is not None:
                    fig.text(plot_middle, 0.98,
                             f"T1 Q{self.QubitIndex + 1}" + f", {float(config['reps'])}*{float(config['rounds'])} avgs,",
                             fontsize=24, ha='center',
                             va='top')  # , pi gain %.2f" % float(config['pi_amp']) + f", {float(config['sigma']) * 1000} ns sigma"   you can put this back once you save configs properly for when replotting
                else:
                    fig.text(plot_middle, 0.98,
                             f"T1 Q{self.QubitIndex + 1}",
                             fontsize=24, ha='center', va='top')
                q1_fit_exponential = None
                T1_est = None
                T1_err = None

            # I subplot
            ax1.plot(delay_times, I, label="Gain (a.u.)", linewidth=2)
            ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=16)
            # ax1.axvline(freq_q, color='orange', linestyle='--', linewidth=2)

            # Q subplot
            ax2.plot(delay_times, Q, label="Q", linewidth=2)
            ax2.set_xlabel("Delay time (us)", fontsize=20)
            ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
            ax2.tick_params(axis='both', which='major', labelsize=16)
            # ax2.axvline(freq_q, color='orange', linestyle='--', linewidth=2)

            # Adjust spacing
            plt.tight_layout()

            # Adjust the top margin to make room for the title
            plt.subplots_adjust(top=0.93)
            if self.save_figs:
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')  # , facecolor='white'
            plt.close(fig)


