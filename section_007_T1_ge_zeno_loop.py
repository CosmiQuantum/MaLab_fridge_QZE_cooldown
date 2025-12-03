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

        self.add_pulse(ch=res_ch, name="qze_pulse", ro_ch=ro_ch,
                       style="const",
                       length=QickSweep1D("waitloop", cfg['start'], cfg['stop']),
                       freq=cfg['res_freq_qze'],
                       phase=cfg['res_phase_qze'],
                       gain=QickSweep1D("gain_loop", cfg["gain_start"], cfg["gain_stop"])
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])
        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )

        self.add_loop("waitloop", cfg["steps"])
        self.add_loop("gain_loop", cfg["gain_steps"])  # inner loop

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse
        self.delay_auto(tag='wait_pi_pulse')                          # wait for it to be done, now qubit is in e
        self.pulse(ch=cfg['res_ch'], name="qze_pulse", t=0.01)           # play res pulse that has same length as wait_time
        self.delay_auto(tag='wait_qze_pulse')                         # wait for that pulse to finish
        self.delay_auto(t=5, tag='wait_for_ring_down')
        self.pulse(ch=cfg['res_ch'], name="res_pulse")           # play readout pulse after 5 us for ring down
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class T1ProgramIBMZenoSingleGain(AveragerProgramV2):
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

        self.add_pulse(ch=res_ch, name="qze_pulse", ro_ch=ro_ch,
                       style="const",
                       length=QickSweep1D("waitloop", cfg['start'], cfg['stop']),
                       freq=cfg['res_freq_qze'],
                       phase=cfg['res_phase_qze'],
                       gain=cfg['res_gain_qze']
                       )

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])
        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse", ro_ch=ro_ch,
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

    def run(self, thresholding=False, scaling=False, qze_pulse='const'):
        now = datetime.datetime.now()

        if scaling:
            q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            ss_exp_cfg = add_qubit_experiment(expt_cfg, 'Readout_Optimization', self.QubitIndex)
            ss_config = {**q_config[self.Qubit], **ss_exp_cfg}



        if self.live_plot:
            t1 = T1ProgramIBMZeno(self.experiment.soccfg, reps=self.config['reps'],
                                  final_delay=self.config['relax_delay'], cfg=self.config)
            I, Q, delay_times = self.live_plotting(t1, thresholding)
        else:
            if thresholding:
                t1 = T1ProgramIBMZeno(self.experiment.soccfg, reps=self.config['reps'],
                                      final_delay=self.config['relax_delay'], cfg=self.config)

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
                I_shots_all = []
                Q_shots_all = []

                t1 = T1ProgramIBMZeno(self.experiment.soccfg, reps=self.config['reps'],
                                      final_delay=self.config['relax_delay'], cfg=self.config)

                iq_list = t1.acquire(self.experiment.soc, rounds=self.config["rounds"], progress=True)
                iq_list = iq_list[0][0].T
                I = iq_list[0]
                Q = iq_list[1]
                Is_all.append(I)
                Qs_all.append(Q)

                raw_0 = t1.get_raw()  # I,Q data without normalizing to readout window, subtracting readout offset, or rotation/thresholding
                A = np.squeeze(raw_0[0])
                I_shots = A[:, :,
                          0]  # if you have 4 steps and 3 shots/reps this is like [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
                Q_shots = A[:, :, 1]

                I_shots_all.append(I_shots)
                Q_shots_all.append(Q_shots)

                if scaling:
                    ssp_g = SingleShotProgram_g(self.experiment.soccfg, reps=1,
                                                final_delay=ss_config['relax_delay'],
                                                cfg=ss_config)
                    ssp_e = SingleShotProgram_e(self.experiment.soccfg, reps=1,
                                                final_delay=ss_config['relax_delay'],
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

            delay_times = t1.get_pulse_param(pulsename='qze_pulse', parname='length', as_array=True)
            gains = t1.get_pulse_param('qze_pulse', "gain", as_array=True)

            if self.plot_results:
                self.plot_results_interweaved_cal_t1(Is_all, Qs_all, delay_times, gains=gains, scaling=scaling, Ie = ss_I_e_all
                                  , Ig = ss_I_g_all, Qe = ss_Q_e_all, Qg = ss_Q_g_all)
            q1_fit_exponential, T1_est, T1_err = None, None, None
            return T1_est, T1_err, Is_all, Qs_all, delay_times, q1_fit_exponential, self.config, ss_Q_e_all\
                , ss_Q_g_all,ss_I_e_all, ss_I_g_all, I_shots_all, Q_shots_all, gains

    def run_single_gain(self, thresholding=False, scaling=False, qze_pulse='const'):
        now = datetime.datetime.now()

        if scaling:
            q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            ss_exp_cfg = add_qubit_experiment(expt_cfg, 'Readout_Optimization', self.QubitIndex)
            ss_config = {**q_config[self.Qubit], **ss_exp_cfg}

        if self.live_plot:
            t1 = T1ProgramIBMZenoSingleGain(self.experiment.soccfg, reps=self.config['reps'],
                                  final_delay=self.config['relax_delay'], cfg=self.config)
            I, Q, delay_times = self.live_plotting(t1, thresholding)
        else:
            if thresholding:
                t1 = T1ProgramIBMZenoSingleGain(self.experiment.soccfg, reps=self.config['reps'],
                                      final_delay=self.config['relax_delay'], cfg=self.config)

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
                I_shots_all = []
                Q_shots_all = []

                t1 = T1ProgramIBMZenoSingleGain(self.experiment.soccfg, reps=self.config['reps'],
                                      final_delay=self.config['relax_delay'], cfg=self.config)

                iq_list = t1.acquire(self.experiment.soc, rounds=self.config["rounds"], progress=True)
                iq_list = iq_list[0][0].T
                I = iq_list[0]
                Q = iq_list[1]
                Is_all.append(I)
                Qs_all.append(Q)

                raw_0 = t1.get_raw()  # I,Q data without normalizing to readout window, subtracting readout offset, or rotation/thresholding
                A = np.squeeze(raw_0[0])
                I_shots = A[:, :,
                          0]  # if you have 4 steps and 3 shots/reps this is like [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
                Q_shots = A[:, :, 1]

                I_shots_all.append(I_shots)
                Q_shots_all.append(Q_shots)

                if scaling:
                    ssp_g = SingleShotProgram_g(self.experiment.soccfg, reps=1,
                                                final_delay=ss_config['relax_delay'],
                                                cfg=ss_config)
                    ssp_e = SingleShotProgram_e(self.experiment.soccfg, reps=1,
                                                final_delay=ss_config['relax_delay'],
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

            delay_times = t1.get_pulse_param(pulsename='qze_pulse', parname='length', as_array=True)

            if self.plot_results:
                self.plot_results(I, Q, delay_times,now,  scaling=scaling, Ie = ss_I_e
                                  , Ig = ss_I_g, Qe = ss_Q_e, Qg = ss_Q_g)
            q1_fit_exponential, T1_est, T1_err = None, None, None
            return T1_est, T1_err, Is_all, Qs_all, delay_times, q1_fit_exponential, self.config, ss_Q_e_all\
                , ss_Q_g_all,ss_I_e_all, ss_I_g_all, I_shots_all, Q_shots_all


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

    def plot_results_interweaved_cal_t1(self, I, Q, delay_times, gains,
                                        config=None, fig_quality=100,
                                        scaling=False, Ie=None, Ig=None, Qe=None, Qg=None):
        """
        T1 plotting with support for:
          - 1D data (delay_times)
          - 2D data (gains x delay_times), plotted as a heatmap:
                x-axis: gain
                y-axis: delay time
        No fitting is performed.
        """

        # --- helpers (same idea as in spectroscopy function) ---
        def _to_stack(x):
            """
            Return a list of np arrays; if x is already 2D-like (list of arrays), keep;
            if 1D or 2D np.ndarray, wrap into length-1 list for uniform handling.
            """
            if x is None:
                return None
            if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple, np.ndarray)):
                return [np.asarray(row) for row in x]
            else:
                return [np.asarray(x)]

        delay_times = np.asarray(delay_times)
        gains = np.asarray(gains) if gains is not None else None

        I_stack = _to_stack(I)
        Q_stack = _to_stack(Q)

        plt.rcParams.update({'font.size': 18})

        if scaling:
            # --- scaling: use calibration to compute populations; support 1D & 2D ---

            # Normalize calibration inputs similarly
            if any(v is None for v in (Ie, Ig, Qe, Qg)):
                raise ValueError("scaling=True requires Ie, Ig, Qe, Qg (each list/array).")

            Ie_stack = _to_stack(Ie)
            Ig_stack = _to_stack(Ig)
            Qe_stack = _to_stack(Qe)
            Qg_stack = _to_stack(Qg)

            n_traces = len(I_stack)

            # Broadcast single calibration set to all traces if needed
            if len(Ie_stack) == 1 and n_traces > 1:
                Ie_stack = Ie_stack * n_traces
                Ig_stack = Ig_stack * n_traces
                Qe_stack = Qe_stack * n_traces
                Qg_stack = Qg_stack * n_traces

            # Basic checks
            assert len(Q_stack) == n_traces, "I and Q must have the same number of traces"
            assert len(Ie_stack) == n_traces and len(Ig_stack) == n_traces \
                   and len(Qe_stack) == n_traces and len(Qg_stack) == n_traces, \
                "Calibration lists must match number of traces"

            # 1) Per-trace calibration -> population (works for 1D or 2D arrays)
            pop_traces = []
            for k in range(n_traces):
                I_k = np.asarray(I_stack[k])
                Q_k = np.asarray(Q_stack[k])

                e_k = np.mean(Ie_stack[k] + 1j * Qe_stack[k])
                g_k = np.mean(Ig_stack[k] + 1j * Qg_stack[k])
                denom = np.abs(e_k - g_k) ** 2
                if denom == 0:
                    raise ValueError(f"Calibration |e-g| is zero for trace index {k}.")

                z_k = I_k + 1j * Q_k
                pop_k = np.abs(((z_k - g_k) * (e_k - g_k)) / denom)  # same shape as I_k/Q_k
                pop_traces.append(pop_k)

            # 2) Average calibrated populations across traces
            pop_arr = np.stack(pop_traces, axis=0)  # shape: (n_traces, ...) -> 2D or 3D
            pop_mean = np.mean(pop_arr, axis=0)  # shape: 1D (delay) or 2D (gains, delay)

            # --- plotting (no fits) ---
            if pop_mean.ndim == 2:
                # 2D case: (n_gains, n_delays) -> heatmap
                if gains is None:
                    raise ValueError("For 2D data (gains x delay_times), 'gains' must be provided.")

                n_gains, n_delays = pop_mean.shape

                if gains.shape[0] != n_gains:
                    raise ValueError(
                        f"gains length ({gains.shape[0]}) does not match population gain dimension ({n_gains})"
                    )
                if delay_times.shape[0] != n_delays:
                    raise ValueError(
                        f"delay_times length ({delay_times.shape[0]}) does not match population delay dimension ({n_delays})"
                    )

                fig, ax = plt.subplots(1, 1, figsize=(10, 6))

                # imshow expects shape (Ny, Nx) = (len(y), len(x))
                # pop_mean is (n_gains, n_delays), so transpose to (n_delays, n_gains)
                im = ax.imshow(
                    pop_mean.T,
                    origin='lower',
                    aspect='auto',
                    extent=(gains[0], gains[-1], delay_times[0], delay_times[-1])
                )

                ax.set_xlabel("Gain", fontsize=20)
                ax.set_ylabel("Delay Time (us)", fontsize=20)
                ax.tick_params(axis='both', which='major', labelsize=16)

                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("Qubit Population (avg)", fontsize=18)

                # Title
                plot_middle = (ax.get_position().x0 + ax.get_position().x1) / 2
                if config is not None:
                    fig.text(
                        plot_middle, 0.98,
                        f"T1 Q{self.QubitIndex + 1}, "
                        f"{float(config['reps'])}*{float(config['rounds'])} avgs",
                        fontsize=16, ha='center', va='top'
                    )
                else:
                    fig.text(
                        plot_middle, 0.98,
                        f"T1 Q{self.QubitIndex + 1}",
                        fontsize=16, ha='center', va='top'
                    )

            elif pop_mean.ndim == 1:
                # 1D case: simple line plot vs delay_times
                if delay_times.shape[0] != pop_mean.shape[0]:
                    raise ValueError(
                        f"delay_times length ({delay_times.shape[0]}) does not match data length ({pop_mean.shape[0]})"
                    )

                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                ax.plot(delay_times, pop_mean, linewidth=2, label="Qubit Population (avg)")
                ax.set_xlabel("Delay Time (us)", fontsize=20)
                ax.set_ylabel("Qubit Population", fontsize=20)
                ax.tick_params(axis='both', which='major', labelsize=16)
                ax.legend()

                plot_middle = (ax.get_position().x0 + ax.get_position().x1) / 2
                if config is not None:
                    fig.text(
                        plot_middle, 0.98,
                        f"T1 Q{self.QubitIndex + 1}, "
                        f"{float(config['reps'])}*{float(config['rounds'])} avgs",
                        fontsize=16, ha='center', va='top'
                    )
                else:
                    fig.text(
                        plot_middle, 0.98,
                        f"T1 Q{self.QubitIndex + 1}",
                        fontsize=16, ha='center', va='top'
                    )
            else:
                raise ValueError(f"Unexpected population array dimensionality: {pop_mean.ndim}")

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
            # --- non-scaling: average I/Q and plot; support 1D & 2D (heatmap of magnitude for 2D) ---
            I_arr = np.stack(I_stack, axis=0)  # (n_traces, ...) -> 2D or 3D
            Q_arr = np.stack(Q_stack, axis=0)

            I_mean_all = np.mean(I_arr, axis=0)  # 1D (delay) or 2D (gains, delay)
            Q_mean_all = np.mean(Q_arr, axis=0)

            if I_mean_all.ndim == 1:
                # Original 1D behavior: I/Q vs delay_times
                if delay_times.shape[0] != I_mean_all.shape[0]:
                    raise ValueError(
                        f"delay_times length ({delay_times.shape[0]}) does not match data length ({I_mean_all.shape[0]})"
                    )

                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

                plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

                # I subplot
                ax1.plot(delay_times, I_mean_all, label="I (avg)", linewidth=2)
                ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
                ax1.tick_params(axis='both', which='major', labelsize=16)
                ax1.legend()

                # Q subplot
                ax2.plot(delay_times, Q_mean_all, label="Q (avg)", linewidth=2)
                ax2.set_xlabel("Delay Time (us)", fontsize=20)
                ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
                ax2.tick_params(axis='both', which='major', labelsize=16)
                ax2.legend()

                if config is not None:
                    fig.text(
                        plot_middle, 0.98,
                        f"T1 Q{self.QubitIndex + 1}, "
                        f"{float(config['reps'])}*{float(config['rounds'])} avgs",
                        fontsize=16, ha='center', va='top'
                    )
                else:
                    fig.text(
                        plot_middle, 0.98,
                        f"T1 Q{self.QubitIndex + 1}",
                        fontsize=16, ha='center', va='top'
                    )

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

            elif I_mean_all.ndim == 2:
                # 2D case: make a heatmap of the magnitude sqrt(I^2+Q^2)
                if gains is None:
                    raise ValueError("For 2D data (gains x delay_times), 'gains' must be provided.")

                mag = np.sqrt(I_mean_all ** 2 + Q_mean_all ** 2)  # shape (n_gains, n_delays)
                n_gains, n_delays = mag.shape

                if gains.shape[0] != n_gains:
                    raise ValueError(
                        f"gains length ({gains.shape[0]}) does not match gain dimension ({n_gains})"
                    )
                if delay_times.shape[0] != n_delays:
                    raise ValueError(
                        f"delay_times length ({delay_times.shape[0]}) does not match delay dimension ({n_delays})"
                    )

                fig, ax = plt.subplots(1, 1, figsize=(10, 6))

                im = ax.imshow(
                    mag.T,
                    origin='lower',
                    aspect='auto',
                    extent=(gains[0], gains[-1], delay_times[0], delay_times[-1])
                )

                ax.set_xlabel("Gain", fontsize=20)
                ax.set_ylabel("Delay Time (us)", fontsize=20)
                ax.tick_params(axis='both', which='major', labelsize=16)

                cbar = fig.colorbar(im, ax=ax)
                cbar.set_label("Signal Magnitude (avg)", fontsize=18)

                plot_middle = (ax.get_position().x0 + ax.get_position().x1) / 2
                if config is not None:
                    fig.text(
                        plot_middle, 0.98,
                        f"T1 Q{self.QubitIndex + 1}, "
                        f"{float(config['reps'])}*{float(config['rounds'])} avgs",
                        fontsize=16, ha='center', va='top'
                    )
                else:
                    fig.text(
                        plot_middle, 0.98,
                        f"T1 Q{self.QubitIndex + 1}",
                        fontsize=16, ha='center', va='top'
                    )

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
                raise ValueError(f"Unexpected dimensionality for I/Q in non-scaling mode: {I_mean_all.ndim}")

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


