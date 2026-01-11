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


class Fit:
    """
    This class takes care of the fitting to the measured data.
    It includes:
        - Fitting to: linear line
                      T1 experiment
                      Ramsey experiment
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

class starkT2RProgram(AveragerProgramV2):
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
        self.add_pulse(ch=qubit_ch, name="qubit_pulse1", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'] / 2,
                       )
        self.add_pulse(ch=res_ch, name="qze_pulse", ro_ch=ro_ch,
                       style="const",
                       length=QickSweep1D("waitloop", cfg['start'], cfg['stop']),  # varying in the loop
                       freq=cfg['res_freq_qze'],
                       phase=cfg['res_phase_qze'],
                       gain=cfg['stark_gain']
                       )

        self.add_pulse(ch=qubit_ch, name="qubit_pulse2", ro_ch=ro_ch,
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'] + cfg['wait_time'] * 360 * cfg['ramsey_freq'],
                       # current phase + time * 2pi * ramsey freq
                       gain=cfg['pi_amp'] / 2,
                       )

        self.add_loop("waitloop", cfg["steps"])


    def _body(self, cfg):
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse1", t=0)  # put on equator
        self.pulse(ch=cfg['res_ch'], name="qze_pulse",
                   t=0.01)  # play res pulse that has same length as wait_time and let evolve for wait time
        self.delay_auto(0.01, tag='wait')  # wait_time after last pulse
        self.pulse(ch=self.cfg["qubit_ch"], name="qubit_pulse2", t=0)  # put on z axis
        self.delay_auto(0.01)  # wait_time after last pulse
        self.delay_auto(t=5, tag='wait_for_ring_down')
        self.pulse(ch=cfg['res_ch'], name="res_pulse")  # play res pulse 5 us after everything
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class starkT2RMeasurement:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, round_num, signal, save_figs, experiment = None,
                 fit_data = None, verbose = False, logger = None, qick_verbose=True):
        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.outerFolder = outerFolder
        self.fit_data = fit_data
        self.expt_name = "Ramsey_stark"
        self.Qubit = 'Q' + str(self.QubitIndex)
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.round_num = round_num
        self.signal = signal
        self.number_of_qubits = number_of_qubits
        self.save_figs = save_figs
        self.verbose = verbose
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} T2R configuration: ', self.config)
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} T2R configuration:{self.config}')

    def t2_fit(self, x_data, I, Q, verbose = True, guess=None, plot=False):
        #fitting code adapted from https://github.com/qua-platform/py-qua-tools/blob/37c741ade5a8f91888419c6fd23fd34e14372b06/qualang_tools/plot/fitting.py

        # Calculate amplitude
        y_data = np.hypot(I, Q)
        plot_sig = 'Amplitude'

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

        popt, pcov = optimize.curve_fit(func,x,y,p0=[1, 1, 1, guess_phase, 1, 1])

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
                f" f = {out['f'][0]:.3f} +/- {out['f'][1]:.3f} MHz, \n"
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

        t2r_est = out['T2'][0] #in ns
        t2r_err = out['T2'][1] #in ns
        f_est = out['f'][0] # in MHz
        f_err = out['f'][1] # in MHz

        return fit_type(x, popt) * y_normal, t2r_est, t2r_err, f_est, f_err, plot_sig

    def run(self, thresholding=False):
        now = datetime.datetime.now()

        gain_sweep = np.linspace(self.config["start_gain"], self.config["end_gain"],num=self.config["gain_steps"])

        I = []
        Q = []
        fit = []
        t2r_est = []
        t2r_err = []
        f_err =[]
        f_est =[]
        plot_sig =[]

        for g in gain_sweep:
            self.config['stark_gain'] = np.round(g,3)
            ramsey = starkT2RProgram(self.experiment.soccfg, reps=self.config['reps'], final_delay=self.config['relax_delay'],
                         cfg=self.config)

            if thresholding:
                iq_list = ramsey.acquire(self.experiment.soc, rounds=self.config['rounds'],
                                         threshold=self.experiment.readout_cfg["threshold"],
                                         angle=self.experiment.readout_cfg["ro_phase"], progress=self.qick_verbose)
            else:
                iq_list = ramsey.acquire(self.experiment.soc, rounds=self.config['rounds'], progress=self.qick_verbose)

            iq_list = iq_list[0][0].T
            i0 = (iq_list[0])
            q0 = (iq_list[1])
            I.append(i0)
            Q.append(q0)
            delay_times = ramsey.get_pulse_param(pulsename='qze_pulse', parname='length', as_array=True)

            if self.fit_data:
                fit0, t2r_est0, t2r_err0, f_est0, f_err0, plot_sig0 = self.t2_fit(delay_times, i0, q0)
            else:
                fit0, t2r_est0, t2r_err0, f_est0, f_err0, plot_sig0 = None, None, None, None


            if self.save_figs:
                self.plot_results(i0, q0, delay_times, now, fit0, t2r_est0, t2r_err0, f_est0, f_err0, plot_sig0)

            fit.append(fit0)
            t2r_est.append(t2r_est0)
            t2r_err.append(t2r_err0)
            f_est.append(f_est0)
            f_err.append(f_err0)
            plot_sig.append(plot_sig0)

        self.plot_stark_shift(gain_sweep, f_est, f_err)

        return  t2r_est, t2r_err, f_est, f_err, I, Q, delay_times, fit, self.config

    from scipy.optimize import curve_fit
    import numpy as np
    import matplotlib.pyplot as plt
    import os, datetime

    def plot_stark_shift_with_t_phi(self,
                                    gain_sweep,
                                    f_est,
                                    f_err,
                                    t2r_est,
                                    t2r_err,
                                    config=None,
                                    bottom_vs_nbar=True):

        """
        Plot Ramsey frequency vs Stark tone gain, fit a quadratic Stark shift,
        convert to nbar, plot nbar vs gain, and finally plot dephasing rate
        vs nbar using T2* data.

        Parameters
        ----------
        gain_sweep : array_like
            Stark tone gains (one per Ramsey experiment).
        f_est, f_err : array_like
            Fitted Ramsey frequencies and their errors (same length as gain_sweep).
        t2r_est, t2r_err : array_like
            Fitted Ramsey decay times T2* and their errors (same length as gain_sweep).
        config : dict, optional
            Config dict containing chi, etc.

        Returns
        -------
        nbar_per_gain : np.ndarray
            Array of nbar values corresponding to each point in gain_sweep.
        """

        gain_sweep = np.asarray(gain_sweep, dtype=float)
        f_est = np.asarray(f_est, dtype=float)
        f_err = np.asarray(f_err, dtype=float)
        t2r_est = np.asarray(t2r_est, dtype=float)
        t2r_err = np.asarray(t2r_err, dtype=float)

        if not (len(gain_sweep) == len(f_est) == len(t2r_est)):
            raise ValueError("gain_sweep, f_est, and t2r_est must have the same length.")

        # --- Dispersive shift chi (same units as f_est, e.g. MHz) ---
        if config is not None:
            chi = float(config['Ramsey_stark']['chi'][self.QubitIndex])
        else:
            chi = float(self.config['chi'][self.QubitIndex])

        # --- Model for fitting: simple quadratic vs gain ---
        # f(gain) = f0 + A * gain^2
        def q_shift_model(g, A, f0):
            return f0 + A * g ** 2

        kappa = float([0.127, 0.109, 0.139, 0.183, 0.178, 0.174][self.QubitIndex])
        f_q = np.asarray([2780, 2980, 2885, 3096, 3043.32, 3093][self.QubitIndex], dtype=float)
        f_r = np.asarray([7149,7171,7204,7228.9, 7264.31,7287.5][self.QubitIndex], dtype=float)
        Delta_r = 0#f_r - f_q

        # Rough initial guess for A and f0
        if np.ptp(gain_sweep) > 0:
            A_guess = (f_est[-1] - f_est[0]) / (
                    max(gain_sweep) ** 2 - min(gain_sweep) ** 2 + 1e-12
            )
        else:
            A_guess = 0.0
        f0_guess = f_est[0]

        # --- Fit frequency vs gain ---
        use_poly_fallback = False
        try:
            popt, pcov = curve_fit(
                q_shift_model,
                gain_sweep,
                f_est,
                sigma=f_err,
                p0=[A_guess, f0_guess],
                absolute_sigma=True
            )
            A_fit, f0_fit = popt
        except Exception:
            # Fallback: full quadratic polynomial fit f = a2*g^2 + a1*g + a0
            coeffs = np.polyfit(gain_sweep, f_est, 2)
            a2, a1, a0 = coeffs
            f0_fit = a0
            use_poly_fallback = True

        # --- Generate smooth fit for plotting ---
        gain_fit = np.linspace(np.min(gain_sweep), np.max(gain_sweep), 200)

        if not use_poly_fallback:
            f_fit = q_shift_model(gain_fit, A_fit, f0_fit)
        else:
            f_fit = np.polyval(coeffs, gain_fit)

        # --- Compute delta f and nbar (pointwise at the measured gains) ---
        delta_f_meas = f_est - f0_fit  # same units as f_est and chi
        chi = np.abs(chi)
        # nbar = delta_f / (2 * chi)
        nbar_per_gain = delta_f_meas / (2.0 * chi)
        nbar_err = f_err / (2.0 * chi)

        # For the smooth curve: convert fitted delta f to nbar as well
        delta_f_fit = f_fit - f0_fit
        nbar_fit = delta_f_fit / (2.0 * chi)

        # --- Dephasing rate from T2* ---
        # Gamma_phi = 1 / T2*
        # (Add tiny offset in denominator to avoid division by zero)

        # Measurement-induced dephasing model:
        #   Gamma_phi(nbar) = Gamma0
        #                   + (2 * nbar * kappa * chi^2) / (kappa^2/4 + chi^2 + Delta_r^2)
        # --- Dephasing rate from T2* corrected to pure dephasing using constant T1 ---
        T1_const = 110.0  # same time units as t2r_est (e.g. microseconds)

        gamma_ramsey = 1.0 / (t2r_est + 1e-30)  # = 1/T2*
        gamma_phi = gamma_ramsey - 1.0 / (2.0 * T1_const)  # = 1/T2* - 1/(2T1)
        gamma_phi_err = t2r_err / (t2r_est ** 2 + 1e-30)  # T1 treated exact

        # Optional: avoid tiny negative values due to noise
        gamma_phi = np.maximum(gamma_phi, 0.0)

        # --- parameters for the formula ---
        Delta = -0.08  # detuning from bare cavity resonance (same units as chi, kappa_m)
        kappa_m = kappa  # bare resonator linewidth
        chi = np.abs(chi)

        # --- Choose bottom axis + fitting model ---
        use_gamma_fit = True

        if bottom_vs_nbar:
            # Use: Omega_d^2 = nbar*(Delta^2 + (kappa/2)^2)  -> Gamma_phi(nbar) is linear in nbar
            def gamma_model_nbar(nbar, Gamma0):
                halfk = kappa_m / 2.0
                denom = ((Delta + chi) ** 2 + halfk ** 2) * ((Delta - chi) ** 2 + halfk ** 2)
                Omega_d_sq = nbar * (Delta ** 2 + halfk ** 2)
                Gamma_m = (2.0 * chi ** 2 * kappa_m * Omega_d_sq) / denom
                return Gamma0 + Gamma_m

            x_data = nbar_per_gain
            x_fit = np.linspace(np.min(nbar_per_gain), np.max(nbar_per_gain), 400)
            Gamma0_guess = gamma_phi[0]

            try:
                popt_g, pcov_g = curve_fit(
                    gamma_model_nbar,
                    x_data,
                    gamma_phi,
                    sigma=gamma_phi_err,
                    p0=[Gamma0_guess],
                    absolute_sigma=True,
                    maxfev=20000
                )
                (Gamma0_fit,) = popt_g
                y_fit = gamma_model_nbar(x_fit, Gamma0_fit)
            except Exception:
                use_gamma_fit = False

        else:
            # Original equation with Omega_d = s*gain (fit Gamma0 and s)
            def gamma_model_gain(gain, Gamma0, s):
                halfk = kappa_m / 2.0
                denom = ((Delta + chi) ** 2 + halfk ** 2) * ((Delta - chi) ** 2 + halfk ** 2)
                Omega_d_sq = (s * gain) ** 2
                Gamma_m = (2.0 * chi ** 2 * kappa_m * Omega_d_sq) / denom
                return Gamma0 + Gamma_m

            x_data = gain_sweep
            x_fit = np.linspace(np.min(gain_sweep), np.max(gain_sweep), 400)
            Gamma0_guess = gamma_phi[0]
            s_guess = 1.0

            try:
                popt_g, pcov_g = curve_fit(
                    gamma_model_gain,
                    x_data,
                    gamma_phi,
                    sigma=gamma_phi_err,
                    p0=[Gamma0_guess, s_guess],
                    absolute_sigma=True,
                    maxfev=20000
                )
                Gamma0_fit, s_fit = popt_g
                y_fit = gamma_model_gain(x_fit, Gamma0_fit, s_fit)
            except Exception:
                use_gamma_fit = False




        # --- Plotting: three panels ---
        fig, (ax_f, ax_n, ax_g) = plt.subplots(3, 1, figsize=(6, 10))
        # Make frequency and nbar panels share the same gain x-axis
        ax_n.sharex(ax_f)

        # Top: Ramsey frequency vs gain
        ax_f.errorbar(gain_sweep, f_est, yerr=f_err, fmt='ko', label='data')
        ax_f.plot(gain_fit, f_fit, 'r:', label='fit')
        ax_f.set_ylabel('Ramsey frequency (MHz)')
        ax_f.set_title(f"Qubit {self.QubitIndex} Stark shift vs gain")
        ax_f.legend()
        ax_f.grid(True, alpha=0.3)

        # Middle: nbar vs gain
        ax_n.errorbar(
            gain_sweep, nbar_per_gain,
            yerr=nbar_err,
            fmt='ko',
            label=r'$\bar{n}$ from data'
        )
        ax_n.plot(gain_fit, nbar_fit, 'r:', label=r'$\bar{n}$ from fit')
        ax_n.set_xlabel('stark tone gain (a.u.)')
        ax_n.set_ylabel(r'$\bar{n}$ (photons)')
        ax_n.grid(True, alpha=0.3)
        ax_n.legend()

        # --- Bottom plot: dephasing rate vs chosen x-axis ---
        ax_g.errorbar(
            x_data,
            gamma_phi,
            yerr=gamma_phi_err,
            fmt='ko',
            label='data'
        )

        if use_gamma_fit:
            if bottom_vs_nbar:
                ax_g.plot(x_fit, y_fit, 'r:', label=r'Eq. fit (via $\bar{n}$ mapping)')
            else:
                ax_g.plot(x_fit, y_fit, 'r:', label=r'Eq. fit ($\Omega_d=s\cdot$gain)')

        ax_g.set_xlabel(r'$\bar{n}$ (photons)' if bottom_vs_nbar else 'stark tone gain (a.u.)')
        ax_g.set_ylabel(r'$\Gamma_\phi$ (1 / $\mu$s)')
        ax_g.grid(True, alpha=0.3)
        ax_g.legend(fontsize=10)

        fig.tight_layout()

        # --- Save figure if requested ---
        if self.save_figs:
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(
                outerFolder_expt,
                f"stark_shift_Q{self.QubitIndex + 1}_{formatted_datetime}_"
                f"{self.expt_name}_q{self.QubitIndex + 1}.png"
            )
            fig.savefig(file_name, dpi=300, bbox_inches='tight')

        return nbar_per_gain

    def plot_stark_shift(self, gain_sweep, f_est, f_err, config=None):
        """
        Plot Ramsey frequency vs Stark tone gain, fit a quadratic Stark shift,
        convert to nbar, and also plot nbar vs gain.

        Returns
        -------
        nbar_per_gain : np.ndarray
            Array of nbar values corresponding to each point in gain_sweep.
        """

        gain_sweep = np.asarray(gain_sweep, dtype=float)
        f_est = np.asarray(f_est, dtype=float)
        f_err = np.asarray(f_err, dtype=float)

        # --- You need chi here: dispersive shift in same units as f_est (e.g. MHz) ---
        # Adjust the key to whatever you use for chi in your config.
        if config is not None:
            chi =float(config['Ramsey_stark']['chi'][self.QubitIndex])
        else:
            chi = self.config['chi'][self.QubitIndex]  # e.g. chi in MHz



        # --- Model for fitting: simple quadratic vs gain ---
        # f(gain) = f0 + A * gain^2
        def q_shift_model(g, A, f0):
            return f0 + A * g ** 2

        # Rough initial guess for A and f0
        # (difference over full range divided by squared gain span)
        if np.ptp(gain_sweep) > 0:
            A_guess = (f_est[-1] - f_est[0]) / (
                    max(gain_sweep) ** 2 - min(gain_sweep) ** 2 + 1e-12
            )
        else:
            A_guess = 0.0
        f0_guess = f_est[0]

        # --- Fit frequency vs gain ---
        try:
            popt, pcov = curve_fit(
                q_shift_model,
                gain_sweep,
                f_est,
                sigma=f_err,
                p0=[A_guess, f0_guess],
                absolute_sigma=True
            )
            A_fit, f0_fit = popt
            use_poly_fallback = False
        except Exception as e:
            # Fallback: full quadratic polynomial fit f = a2*g^2 + a1*g + a0
            coeffs = np.polyfit(gain_sweep, f_est, 2)
            a2, a1, a0 = coeffs
            # For delta f we just need a baseline at gain=0:
            f0_fit = a0
            # Use the polynomial instead of the simple model
            use_poly_fallback = True

        # --- Generate smooth fit for plotting ---
        gain_fit = np.linspace(np.min(gain_sweep), np.max(gain_sweep), 200)

        if not use_poly_fallback:
            f_fit = q_shift_model(gain_fit, A_fit, f0_fit)
        else:
            f_fit = np.polyval(coeffs, gain_fit)

        # --- Compute delta f and nbar (pointwise at the measured gains) ---
        # Stark shift (relative to f0_fit) at each experimental gain
        delta_f_meas = f_est - f0_fit  # same units as f_est and chi
        chi = np.abs(chi)

        # nbar = delta_f / (2 * chi)
        # If chi is negative, this will carry the sign; you can take abs if desired.
        nbar_per_gain = delta_f_meas / (2.0 * chi)

        # Uncertainty on nbar from f_err
        nbar_err = f_err / (2.0 * chi)

        # For the smooth curve: convert fitted delta f to nbar as well
        delta_f_fit = f_fit - f0_fit
        nbar_fit = delta_f_fit / (2.0 * chi)

        # --- Plotting: two panels, frequency and nbar ---
        fig, (ax_f, ax_n) = plt.subplots(2, 1, sharex=True, figsize=(6, 8))

        # Top: Ramsey frequency vs gain
        ax_f.errorbar(gain_sweep, f_est, yerr=f_err, fmt='ko', label='data')
        ax_f.plot(gain_fit, f_fit, 'r:', label='fit')
        ax_f.set_ylabel('Ramsey frequency (MHz)')
        ax_f.set_title(f"Qubit {self.QubitIndex} Stark shift vs gain")
        ax_f.legend()
        ax_f.grid(True, alpha=0.3)

        # Bottom: nbar vs gain
        ax_n.errorbar(
            gain_sweep, nbar_per_gain,
            yerr=nbar_err,
            fmt='ko',
            label=r'$\bar{n}$ from data'
        )
        ax_n.plot(gain_fit, nbar_fit, 'r:', label=r'$\bar{n}$ from fit')
        ax_n.set_xlabel('stark tone gain (a.u.)')
        ax_n.set_ylabel(r'$\bar{n}$ (photons)')
        ax_n.grid(True, alpha=0.3)
        ax_n.legend()

        fig.tight_layout()

        # --- Save figure if requested ---
        if self.save_figs:
            outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
            self.create_folder_if_not_exists(outerFolder_expt)
            now = datetime.datetime.now()
            formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
            file_name = os.path.join(
                outerFolder_expt,
                f"stark_shift_Q{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}.png"
            )
            fig.savefig(file_name, dpi=300, bbox_inches='tight')

        return nbar_per_gain

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

    def plot_results(self, I, Q, delay_times, now, fit, t2r_est, t2r_err, f_est, f_err, plot_sig, config = None, fig_quality = 100):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        plt.rcParams.update({'font.size': 18})

        # Calculate the middle of the plot area
        plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2
        if self.fit_data:
            if 'Amplitude' in plot_sig:
                ax1.plot(delay_times, fit, '-', color='red', linewidth=3, label="Fit")
            # if 'Q' in plot_sig:
            #     ax2.plot(delay_times, fit, '-', color='red', linewidth=3, label="Fit")

            # Add title, centered on the plot area
            if config is not None:
                fig.text(plot_middle, 0.98,
                         f"stark Ramsey Q{self.QubitIndex}: {f_est} +/- {f_err} MHz, gain {config['stark_gain']}, detuning {config['detuning']} MHz",
                         fontsize=24, ha='center', va='top') #, pi gain %.2f" % float(config['pi_amp']) + f", {float(config['sigma']) * 1000} ns sigma
            else:
                try:
                    fig.text(plot_middle, 0.98,
                             f"T2 Q{self.QubitIndex + 1}, T2R %.2f us" % float(
                                 t2r_est) + f", {float(self.config['reps'])}*{float(self.config['rounds'])} avgs,",
                             fontsize=24, ha='center', va='top')
                except:
                    fig.text(plot_middle, 0.98,
                             f"T2 Q{self.QubitIndex + 1}, T2R",
                             fontsize=24, ha='center', va='top')

        else:
            # Add title, centered on the plot area
            if config is not None:
                fig.text(plot_middle, 0.98,
                         f"T2 Q{self.QubitIndex + 1}" + f", {float(config['reps'])}*{float(config['rounds'])} avgs," ,
                         fontsize=24, ha='center', va='top') #, pi gain %.2f" % float(config['pi_amp']) + f", {float(config['sigma']) * 1000} ns sigma
            else:
                fig.text(plot_middle, 0.98,
                         f"T2 Q{self.QubitIndex + 1}, pi gain %.2f" % float(self.config[
                                                                                'pi_amp']) + f", {float(self.config['sigma']) * 1000} ns sigma" + f", {float(self.config['reps'])}*{float(self.config['rounds'])} avgs,",
                         fontsize=24, ha='center', va='top')

        # I subplot
        ax1.plot(delay_times, np.hypot(I, Q), label="Amplitude", linewidth=2)
        ax1.set_ylabel("Amplitude (a.u.)", fontsize=20)
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

