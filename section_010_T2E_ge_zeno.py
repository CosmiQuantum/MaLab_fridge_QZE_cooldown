from prompt_toolkit.key_binding.bindings.named_commands import self_insert
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from build_task import *
from build_state import *
from expt_config import *
from system_config import *
import copy
import visdom
from scipy import optimize
from sklearn import preprocessing
import matplotlib.pyplot as plt
from typing import List, Union
import itertools
import json
import numpy as np
import warnings
from scipy.optimize import OptimizeWarning
import logging
from section_005_single_shot_ge import SingleShotProgram_g, SingleShotProgram_e

class Fit:
    """
    This class takes care of the fitting to the measured data.
    It includes:
        - Fitting to: linear line
                      T1 experiment
                      echo experiment
                      transmission resonator spectroscopy
                      reflection resonator spectroscopy
        - Printing the initial guess and fitting results
        - Plotting the data and the fitting function
        - Saving the data
    """

    # Remove optimize warnings
    warnings.simplefilter("ignore", RuntimeWarning)
    warnings.simplefilter("ignore", OptimizeWarning)

    @staticmethod
    def linear(
        x_data: Union[np.ndarray, List[float]],
        y_data: Union[np.ndarray, List[float]],
        guess=None,
        verbose=False,
        plot=False,
        save=False,
    ) -> dict:
        """
        Create a linear fit of the form

        .. math::
        f(x) = a * x + b

        for unknown parameters :
             a - The slope of the function
             b - The free parameter of the function

         :param x_data: The data on the x-axis
         :param y_data: The data on the y-axis
         :param dict guess: Dictionary containing the initial guess for the fitting parameters (guess=dict(a=20))
         :param verbose: if True prints the initial guess and fitting results
         :param plot: if True plots the data and the fitting function
         :param save: if not False saves the data into a json file
                      The id of the file is save='id'. The name of the json file is `id.json`
         :return: A dictionary of (fit_func, a, b)

        """

        # Normalizing the vectors
        xn = preprocessing.normalize([x_data], return_norm=True)
        yn = preprocessing.normalize([y_data], return_norm=True)
        x = xn[0][0]
        y = yn[0][0]
        x_normal = xn[1][0]
        y_normal = yn[1][0]

        # Finding an initial guess to the slope
        a0 = (y[-1] - y[0]) / (x[-1] - x[0])

        # Finding an initial guess to the free parameter
        b0 = y[0]

        # Check user guess
        if guess is not None:
            for key in guess.keys():
                if key == "a":
                    a0 = float(guess[key]) * x_normal / y_normal
                elif key == "b":
                    b0 = float(guess[key]) / y_normal
                else:
                    raise Exception(
                        f"The key '{key}' specified in 'guess' does not match a fitting parameters for this function."
                    )
        # Print the initial guess if verbose=True
        if verbose:
            print(f"Initial guess:\n" f" a = {a0 * y_normal / x_normal:.3f}, \n" f" b = {b0 * y_normal:.3f}")

        # Fitting function
        def func(x_var, c0, c1):
            return a0 * c0 * x_var + b0 * c1

        def fit_type(x_var, a):
            return func(x_var, a[0], a[1])

        popt, pcov = optimize.curve_fit(func, x, y, p0=[1, 1])
        perr = np.sqrt(np.diag(pcov))

        # Output the fitting function and its parameters
        out = {
            "fit_func": lambda x_var: fit_type(x_var / x_normal, popt) * y_normal,
            "a": [
                popt[0] * a0 * y_normal / x_normal,
                perr[0] * a0 * y_normal / x_normal,
            ],
            "b": [popt[1] * b0 * y_normal, perr[1] * b0 * y_normal],
        }
        # Print the fitting results if verbose=True
        if verbose:
            print(
                f"Fitting results:\n"
                f" a = {out['a'][0]:.3f} +/- {out['a'][1]:.3f}, \n"
                f" b = {out['b'][0]:.3f} +/- {out['b'][1]:.3f}"
            )
        # Plot the data and the fitting function if plot=True
        if plot:
            plt.plot(x_data, fit_type(x, popt) * y_normal)
            plt.plot(
                x_data,
                y_data,
                ".",
                label=f"a  = {out['a'][0]:.1f} +/- {out['a'][1]:.1f} \n b  = {out['b'][0]:.1f} +/- {out['b'][1]:.1f}",
            )
            plt.legend(loc="upper right")
        # Save the data in a json file named 'id.json' if save=id
        if save:
            fit_params = dict(itertools.islice(out.items(), 1, len(out)))
            fit_params["x_data"] = x_data.tolist()
            fit_params["y_data"] = y_data.tolist()
            fit_params["y_fit"] = (func(x, popt[0], popt[1]) * y_normal).tolist()
            json_object = json.dumps(fit_params)
            if save[-5:] == ".json":
                save = save[:-5]
            with open(f"{save}.json", "w") as outfile:
                outfile.write(json_object)

        return out

class T2EProgram(AveragerProgramV2):
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

        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])
        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="qubit_pulse1",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'] ,
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'] / 2,
                       )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse_pi",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse2",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'] + cfg['wait_time']*360*cfg['ramsey_freq'], # current phase + time * 2pi * echo freq
                       gain=cfg['pi_amp'] / 2,
                      )
        self.add_pulse(ch=res_ch, name="qze_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg['wait_time']/2,  # varying in the loop
                       freq=cfg['res_freq_qze'],
                       phase=cfg['res_phase_qze'],
                       gain=QickSweep1D("gain_loop", cfg["gain_start"], cfg["gain_stop"])
                       )

        self.add_loop("waitloop", cfg["steps"])
        self.add_loop("gain_loop", cfg["gain_steps"])  # inner loop

    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse1", t=0)  # put qubit on equator
        self.delay_auto(0.01, tag='wait0') # wait for pulse to finish
        self.pulse(ch=cfg['res_ch'], name="qze_pulse", t=0.01) # do the resonator qze pulse for delay_time/2
        self.delay_auto(0.01, tag='wait1') # wait for pulse to finish
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse_pi", t=0)  # echo around equator
        self.delay_auto(0.01, tag='wait2') # wait for pulse to finish
        self.pulse(ch=cfg['res_ch'], name="qze_pulse", t=0.01) # do the resonator qze pulse for delay_time/2
        self.delay_auto(0.01, tag='wait3') # wait for pulse to finish
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse2", t=0)  # put on the z axis
        self.delay_auto(t=5, tag='wait_for_ring_down')  # wait_time after last pulse
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class T2EMeasurementZeno:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, round_num, signal, save_figs, experiment = None,
                 live_plot = None, fit_data = None, increase_qubit_reps = False, qubit_to_increase_reps_for = None,
                 multiply_qubit_reps_by = 0, verbose = False, logger = None, qick_verbose=True, unmasking_resgain = False):
        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.fit_data = fit_data
        self.expt_name = "SpinEcho_ge_zeno"
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num
        self.signal = signal
        self.save_figs = save_figs
        self.live_plot = live_plot
        self.number_of_qubits = number_of_qubits
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [QubitIndex]

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            if increase_qubit_reps:
                    if self.QubitIndex==qubit_to_increase_reps_for:
                        if self.verbose: print(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        self.logger.info(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        self.config["reps"] *=multiply_qubit_reps_by
                        # self.config['echo_freq'] = 2 * self.config['echo_freq']
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} T2E configuration: ', self.config)
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} T2E configuration: {self.config}')

    def t2_fit(self, x_data, I, Q, verbose = False, guess=None, plot=False,amp=None):
        #fitting code adapted from https://github.com/qua-platform/py-qua-tools/blob/37c741ade5a8f91888419c6fd23fd34e14372b06/qualang_tools/plot/fitting.py

        if abs(I[-1] - I[0]) > abs(Q[-1] - Q[0]):
            y_data = I
            plot_sig = 'I'
        elif amp is not None:
            y_data = amp
            plot_sig = 'Pop'
        else:
            y_data = Q
            plot_sig = 'Q'

        # Normalizing the vectors
        xn = preprocessing.normalize([x_data], return_norm=True)
        yn = preprocessing.normalize([y_data], return_norm=True)
        x = xn[0][0]
        y = yn[0][0]
        x_normal = xn[1][0]
        y_normal = yn[1][0]

        # Compute the FFT for guessing the frequency
        fft = np.fft.fft(y)
        f = np.fft.fftfreq(len(x))
        # Take the positive part only
        fft = fft[1: len(f) // 2]
        f = f[1: len(f) // 2]
        # Remove the DC peak if there is one
        if (np.abs(fft)[1:] - np.abs(fft)[:-1] > 0).any():
            first_read_data_ind = np.where(np.abs(fft)[1:] - np.abs(fft)[:-1] > 0)[0][0]  # away from the DC peak
            fft = fft[first_read_data_ind:]
            f = f[first_read_data_ind:]

        # Finding a guess for the frequency
        out_freq = f[np.argmax(np.abs(fft))]
        guess_freq = out_freq / (x[1] - x[0])

        # The period is 1 / guess_freq --> number of oscillations --> peaks decay to get guess_T2
        period = int(np.ceil(1 / out_freq))
        peaks = (
                np.array([np.std(y[i * period: (i + 1) * period]) for i in range(round(len(y) / period))]) * np.sqrt(
            2) * 2
        )

        # Finding a guess for the decay (slope of log(peaks))
        if len(peaks) > 1:
            guess_T2 = -1 / ((np.log(peaks)[-1] - np.log(peaks)[0]) / (period * (len(peaks) - 1))) * (x[1] - x[0])
        else:
            guess_T2 = 100 / x_normal

        # Finding a guess for the offsets
        initial_offset = np.mean(y[:period])
        final_offset = np.mean(y[-period:])

        # Finding a guess for the phase
        guess_phase = np.angle(fft[np.argmax(np.abs(fft))]) - guess_freq * 2 * np.pi * x[0]

        # Check user guess
        if guess is not None:
            for key in guess.keys():
                if key == "f":
                    guess_freq = float(guess[key]) * x_normal
                elif key == "phase":
                    guess_phase = float(guess[key])
                elif key == "T2":
                    guess_T2 = float(guess[key]) * x_normal
                elif key == "amp":
                    peaks[0] = float(guess[key]) / y_normal
                elif key == "initial_offset":
                    initial_offset = float(guess[key]) / y_normal
                elif key == "final_offset":
                    final_offset = float(guess[key]) / y_normal
                else:
                    raise Exception(
                        f"The key '{key}' specified in 'guess' does not match a fitting parameters for this function."
                    )

        # Print the initial guess if verbose=True
        if verbose:
            print(
                f"Initial guess:\n"
                f" f = {guess_freq / x_normal:.3f}, \n"
                f" phase = {guess_phase:.3f}, \n"
                f" T2 = {guess_T2 * x_normal:.3f}, \n"
                f" amp = {peaks[0] * y_normal:.3f}, \n"
                f" initial offset = {initial_offset * y_normal:.3f}, \n"
                f" final_offset = {final_offset * y_normal:.3f}"
            )

        # Fitting function
        def func(x_var, a0, a1, a2, a3, a4, a5):
            return final_offset * a4 * (1 - np.exp(-x_var / (guess_T2 * a1))) + peaks[0] / 2 * a2 * (
                    np.exp(-x_var / (guess_T2 * a1))
                    * (a5 * initial_offset / peaks[0] * 2 + np.cos(2 * np.pi * a0 * guess_freq * x + a3))
            )

        def fit_type(x_var, a):
            return func(x_var, a[0], a[1], a[2], a[3], a[4], a[5])

        popt, pcov = optimize.curve_fit(
            func,
            x,
            y,
            p0=[1, 1, 1, guess_phase, 1, 1],
        )

        perr = np.sqrt(np.diag(pcov))

        # Output the fitting function and its parameters
        out = {
            "fit_func": lambda x_var: fit_type(x_var / x_normal, popt) * y_normal,
            "f": [popt[0] * guess_freq / x_normal, perr[0] * guess_freq / x_normal],
            "phase": [popt[3] % (2 * np.pi), perr[3] % (2 * np.pi)],
            "T2": [(guess_T2 * popt[1]) * x_normal, perr[1] * guess_T2 * x_normal],
            "amp": [peaks[0] * popt[2] * y_normal, perr[2] * peaks[0] * y_normal],
            "initial_offset": [
                popt[5] * initial_offset * y_normal,
                perr[5] * initial_offset * y_normal,
            ],
            "final_offset": [
                final_offset * popt[4] * y_normal,
                perr[4] * final_offset * y_normal,
            ],
        }
        # Print the fitting results if verbose=True
        if verbose:
            print(
                f"Fitting results:\n"
                f" f = {out['f'][0] * 1000:.3f} +/- {out['f'][1] * 1000:.3f} MHz, \n"
                f" phase = {out['phase'][0]:.3f} +/- {out['phase'][1]:.3f} rad, \n"
                f" T2 = {out['T2'][0]:.2f} +/- {out['T2'][1]:.3f} ns, \n"
                f" amp = {out['amp'][0]:.2f} +/- {out['amp'][1]:.3f} a.u., \n"
                f" initial offset = {out['initial_offset'][0]:.2f} +/- {out['initial_offset'][1]:.3f}, \n"
                f" final_offset = {out['final_offset'][0]:.2f} +/- {out['final_offset'][1]:.3f} a.u."
            )
        # Plot the data and the fitting function if plot=True
        if plot:
            plt.plot(x_data, fit_type(x, popt) * y_normal)
            plt.plot(
                x_data,
                y_data,
                ".",
                label=f"T2  = {out['T2'][0]:.1f} +/- {out['T2'][1]:.1f}ns \n f = {out['f'][0] * 1000:.3f} +/- {out['f'][1] * 1000:.3f} MHz",
            )
            plt.legend(loc="upper right")
        t2e_est = out['T2'][0] #in ns
        t2e_err = out['T2'][1] #in ns
        return fit_type(x, popt) * y_normal, t2e_est, t2e_err, plot_sig

    def run(self, thresholding=False, correction=False, scaling=False):
        now = datetime.datetime.now()
        echo = T2EProgram(self.experiment.soccfg, reps=self.config['reps'],
                                final_delay=self.config['relax_delay'],
                                cfg=self.config)
        if scaling:
            q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            ss_exp_cfg = add_qubit_experiment(expt_cfg, 'Readout_Optimization', self.QubitIndex)
            ss_config = {**q_config[self.Qubit], **ss_exp_cfg}

            ssp_g = SingleShotProgram_g(self.experiment.soccfg, reps=1, final_delay=ss_config['relax_delay'],
                                        cfg=ss_config)
            ssp_e = SingleShotProgram_e(self.experiment.soccfg, reps=1, final_delay=ss_config['relax_delay'],
                                        cfg=ss_config)

        # for live plotting open http://localhost:8097/ on firefox
        if self.live_plot:
            I, Q, delay_times = self.live_plotting(echo, thresholding)
        else:
            if thresholding:
                echo = T2EProgram(self.experiment.soccfg, reps=self.config['reps'],
                                    final_delay=self.config['relax_delay'],
                                    cfg=self.config)
                iq_list = echo.acquire(self.experiment.soc, rounds=self.config['rounds'],
                                         threshold=self.experiment.readout_cfg["threshold"],
                                         angle=self.experiment.readout_cfg["ro_phase"], progress=self.qick_verbose)
            else:
                Is_all = []
                Qs_all = []
                ss_I_g_all = []
                ss_Q_g_all = []
                ss_I_e_all = []
                ss_Q_e_all = []
                I_shots_all = []
                Q_shots_all = []

                echo = T2EProgram(self.experiment.soccfg, reps=self.config['reps'],
                                  final_delay=self.config['relax_delay'],
                                  cfg=self.config)
                iq_list = echo.acquire(self.experiment.soc, rounds=self.config["rounds"], progress=self.qick_verbose)
                iq_list = iq_list[0][0].T
                I = iq_list[0]
                Q = iq_list[1]
                Is_all.append(I)
                Qs_all.append(Q)

                raw_0 = echo.get_raw()  # I,Q data without normalizing to readout window, subtracting readout offset, or rotation/thresholding
                A = np.squeeze(raw_0[0])
                I_shots = A[:, :,
                          0]  # if you have 4 steps and 3 shots/reps this is like [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
                Q_shots = A[:, :, 1]

                I_shots_all.append(I_shots)
                Q_shots_all.append(Q_shots)

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

            delay_times = echo.get_pulse_param(pulsename='qze_pulse', parname='length', as_array=True) * 2
            gains = echo.get_pulse_param('qze_pulse', "gain", as_array=True)
            self.plot_results_interweaved_cal(Is_all, Qs_all, delay_times, gains=gains, scaling=scaling, Ie=ss_I_e_all
                                              , Ig=ss_I_g_all, Qe=ss_Q_e_all, Qg=ss_Q_g_all)
            return None, None, Is_all, Qs_all, delay_times, None, self.config, ss_Q_e_all \
                , ss_Q_g_all, ss_I_e_all, ss_I_g_all, I_shots_all, Q_shots_all,gains


    def plot_results_interweaved_cal(self, I, Q, delay_times, gains,
                                     config=None, fig_quality=100,
                                     scaling=False, Ie=None, Ig=None, Qe=None, Qg=None):
        """
        T2R plotting with support for:
          - 1D data (delay_times)
          - 2D data (gains x delay_times), plotted as a heatmap:
                x-axis: gain
                y-axis: delay time
        No fitting is performed.
        """

        # --- helpers (same style as spectroscopy / T1 version) ---
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
                    extent=(gains[0], gains[-1], delay_times[0], delay_times[-1]),
                    interpolation = 'nearest'
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
                        f"T2E Q{self.QubitIndex + 1}, "
                        f"{float(config['reps'])}*{float(config['rounds'])} avgs",
                        fontsize=16, ha='center', va='top'
                    )
                else:
                    fig.text(
                        plot_middle, 0.98,
                        f"T2E Q{self.QubitIndex + 1}",
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
                        f"T2R Q{self.QubitIndex + 1}, "
                        f"{float(config['reps'])}*{float(config['rounds'])} avgs",
                        fontsize=16, ha='center', va='top'
                    )
                else:
                    fig.text(
                        plot_middle, 0.98,
                        f"T2R Q{self.QubitIndex + 1}",
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
                        f"T2R Q{self.QubitIndex + 1}, "
                        f"{float(config['reps'])}*{float(config['rounds'])} avgs",
                        fontsize=16, ha='center', va='top'
                    )
                else:
                    fig.text(
                        plot_middle, 0.98,
                        f"T2R Q{self.QubitIndex + 1}",
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
                    extent=(gains[0], gains[-1], delay_times[0], delay_times[-1]),
                    interpolation = 'nearest'
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
                        f"T2R Q{self.QubitIndex + 1}, "
                        f"{float(config['reps'])}*{float(config['rounds'])} avgs",
                        fontsize=16, ha='center', va='top'
                    )
                else:
                    fig.text(
                        plot_middle, 0.98,
                        f"T2R Q{self.QubitIndex + 1}",
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

    def live_plotting(self, echo,thresholding):
        I = Q = expt_mags = expt_phases = expt_pop = None
        viz = visdom.Visdom()
        if not viz.check_connection(timeout_seconds=5):
            raise RuntimeError("Visdom server not connected!")
        for ii in range(self.config["rounds"]):
            if thresholding:
                iq_list = echo.acquire(self.experiment.soc, rounds=1,
                                       threshold=self.experiment.readout_cfg["threshold"],
                                       angle=self.experiment.readout_cfg["ro_phase"], progress=self.qick_verbose)
            else:
                iq_list = echo.acquire(self.experiment.soc, rounds=1, progress=self.qick_verbose)

            delay_times1 = echo.get_time_param('wait1', "t", as_array=True)
            delay_times2 = echo.get_time_param('wait2', "t", as_array=True)
            delay_times = delay_times1 + delay_times2
            iq_list = iq_list[0][0].T
            this_I = (iq_list[0])
            this_Q = (iq_list[1])

            if I is None:  # ii == 0
                I, Q = this_I, this_Q
            else:
                I = (I * ii + this_I) / (ii + 1.0)
                Q = (Q * ii + this_Q) / (ii + 1.0)

            viz.line(X=delay_times, Y=I, opts=dict(height=400, width=700, title='T2 Echo I', showlegend=True, xlabel='expt_pts'),win='T2E_I')
            viz.line(X=delay_times, Y=Q, opts=dict(height=400, width=700, title='T2 Echo Q', showlegend=True, xlabel='expt_pts'),win='T2E_Q')
        return I, Q, delay_times

    def set_res_gain_ge(self, QUBIT_INDEX, num_qubits=6):
        """Sets the gain for the selected qubit to 1, others to 0."""
        res_gain_ge = [0] * num_qubits  # Initialize all gains to 0
        if 0 <= QUBIT_INDEX < num_qubits:  # makes sure you are within the range of options
            res_gain_ge[QUBIT_INDEX] = 1  # Set the gain for the selected qubit
        return res_gain_ge

    def exponential(self, x, a, b, c, d):
        return a * np.exp(-(x - b) / c) + d

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)

    def plot_results(self, I, Q, delay_times, now, fit, t2e_est, t2e_err, plot_sig, config = None, fig_quality = 100):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        plt.rcParams.update({'font.size': 18})

        # Calculate the middle of the plot area
        plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

        if self.fit_data:
            if 'I' in plot_sig:
                ax1.plot(delay_times, fit, '-', color='red', linewidth=3, label="Fit")
            if 'Q' in plot_sig:
                ax2.plot(delay_times, fit, '-', color='red', linewidth=3, label="Fit")

            # Add title, centered on the plot area
            if config is not None:
                fig.text(plot_middle, 0.98,
                         f"Q{self.QubitIndex + 1}" + f" T2E={t2e_est:.2f} us" + f", {float(config['reps'])}*{float(config['rounds'])} avgs,",
                         fontsize=24, ha='center', va='top') #, pi gain %.2f" % float(config['pi_amp']) + f", {float(config['sigma']) * 1000} ns sigma
            else:
                fig.text(plot_middle, 0.98,
                         f"T2 Q{self.QubitIndex + 1}, T2E %.2f us" % float(
                             t2e_est) + f", {float(self.config['reps'])}*{float(self.config['rounds'])} avgs,",
                         fontsize=24, ha='center', va='top')

        else:
            # Add title, centered on the plot area
            if config is not None:
                fig.text(plot_middle, 0.98,
                         f"T2 Q{self.QubitIndex + 1}, pi gain %.2f" % float(config[
                             'pi_amp']) + f", {float(config['sigma']) * 1000} ns sigma" + f", {float(config['reps'])}*{float(config['rounds'])} avgs," ,
                         fontsize=24, ha='center', va='top')
            else:
                fig.text(plot_middle, 0.98,
                         f"T2 Q{self.QubitIndex + 1}, pi gain %.2f" % float(self.config[
                                                                                'pi_amp']) + f", {float(self.config['sigma']) * 1000} ns sigma" + f", {float(self.config['reps'])}*{float(self.config['rounds'])} avgs,",
                         fontsize=24, ha='center', va='top')

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


