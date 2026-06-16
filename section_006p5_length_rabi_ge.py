from copy import deepcopy
from section_004_qubit_spec_ge import QubitSpectroscopy
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import datetime
from build_task import *
from build_state import *
from expt_config import *
import copy
import visdom
import logging
import math

class LengthRabiExperiment:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, round_num, signal, save_figs, experiment = None,
                 live_plot = None, increase_qubit_reps = False, qubit_to_increase_reps_for = None,
                 multiply_qubit_reps_by = 0, verbose = False, logger = None, qick_verbose=True, QZE=False,
                 projective_readout_pulse_len_us=9,  time_between_projective_readout_pulses=None,
                 zeno_pulse_gain=None,chevron=False, length_rabi_vs_gain=False):
        self.qick_verbose = qick_verbose
        self.QubitIndex = QubitIndex
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder
        self.chevron=chevron
        self.QZE = QZE
        if self.QZE:
            self.expt_name = "length_rabi_ge_qze"

            self.Qubit = 'Q' + str(self.QubitIndex)
            self.exp_cfg = expt_cfg[self.expt_name]
            self.round_num = round_num
            self.live_plot = live_plot
            self.signal = signal
            self.save_figs = save_figs
            self.experiment = experiment
            self.verbose = verbose
            self.zeno_pulse_gain = zeno_pulse_gain

            self.projective_readout_pulse_len_us = projective_readout_pulse_len_us
            self.time_between_projective_readout_pulses=time_between_projective_readout_pulses
            self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")
            qze_mask = np.arange(0, self.number_of_qubits + 1)
            qze_mask = np.delete(qze_mask, QubitIndex)
            self.exp_cfg['qze_mask'] = qze_mask

            self.experiment.readout_cfg['res_gain_qze'] = [self.experiment.readout_cfg['res_gain_ge'][QubitIndex],0,0,0,0,0,self.zeno_pulse_gain]
            self.experiment.readout_cfg['res_freq_qze'] = self.experiment.readout_cfg['res_freq_ge']
            self.experiment.readout_cfg['res_phase_qze'] = self.experiment.readout_cfg['res_phase']
            if len(self.experiment.readout_cfg['res_freq_qze']) <7: #otherise it keeps appending
                self.experiment.readout_cfg['res_freq_qze'].append(experiment.readout_cfg['res_freq_qze'][self.QubitIndex])
                self.experiment.readout_cfg['res_phase_qze'].append(experiment.readout_cfg['res_phase_qze'][self.QubitIndex])

        else:
            self.length_rabi_vs_gain = length_rabi_vs_gain
            if chevron:
                self.expt_name = "length_rabi_ge_chevron"
            elif length_rabi_vs_gain:
                self.expt_name = "length_rabi_vs_gain"
            else:
                self.expt_name = "length_rabi_ge"
            self.Qubit = 'Q' + str(self.QubitIndex)
            self.exp_cfg = expt_cfg[self.expt_name]
            self.round_num = round_num
            self.live_plot = live_plot
            self.signal = signal
            self.save_figs = save_figs
            self.experiment = experiment
            self.verbose = verbose
            self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")
        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            if increase_qubit_reps:
                    if self.QubitIndex==qubit_to_increase_reps_for:
                        if self.verbose: print(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        self.logger.info(f"Increasing reps for {self.Qubit} by {multiply_qubit_reps_by} times")
                        self.config["reps"] *= multiply_qubit_reps_by
            #self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} Rabi configuration: {self.config}')
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} Rabi configuration: ', self.config)

    def run(self, thresholding=False, constant_zeno_pulse=False):
        if not self.chevron:
            if self.length_rabi_vs_gain:
                amp_rabi = LengthVsGainRabiProgram(
                    self.experiment.soccfg,
                    reps=self.config['reps'],
                    final_delay=self.config['relax_delay'],
                    cfg=self.config
                )
            else:
                amp_rabi = LengthRabiProgram(
                    self.experiment.soccfg,
                    reps=self.config['reps'],
                    final_delay=self.config['relax_delay'],
                    cfg=self.config
                )
        else:
            amp_rabi = LengthRabiChevronProgram(
                self.experiment.soccfg,
                reps=self.config['reps'],
                final_delay=self.config['relax_delay'],
                cfg=self.config)

        iq_list = amp_rabi.acquire(
            self.experiment.soc,
            rounds=self.config["rounds"],
            progress=self.qick_verbose
        )
        iq_list = iq_list[0][0].T
        I = (iq_list[0])
        Q = (iq_list[1])

        #get the lens that were used so you can use to plot on the x axis
        lengths = amp_rabi.get_pulse_param('qubit_pulse', "length", as_array=True)
        #lens = amp_rabi.get_pulse_param('qubit_pulse', "gain", as_array=True)
        # x_axis is what gets returned/saved as the swept parameter for the plotted data
        x_axis = lengths
        if self.length_rabi_vs_gain:
            # 2D sweep: a Rabi oscillation (vs pulse length) for each qubit drive gain.
            # I, Q have shape (gain_steps, len_steps); fit a cosine per gain.
            gains = amp_rabi.get_pulse_param('qubit_pulse', "gain", as_array=True)
            q1_fit_cosine, pi_len = self.plot_results_vs_gain(I, Q, lengths, gains, config=self.config)
            x_axis = gains
        elif not self.chevron:
            q1_fit_cosine, pi_len = self.plot_results( I, Q, lengths, config = self.config)
        else:

            freqs = amp_rabi.get_pulse_param('qubit_pulse', "freq", as_array=True)
            mag = self.plot_results_chevron(I, Q, lengths, freqs, config=self.config)
            q1_fit_cosine, pi_len = None, None
        from section_005_single_shot_ge import SingleShotProgram_g, SingleShotProgram_e
        q_config = all_qubit_state(self.experiment, self.number_of_qubits)
        ss_exp_cfg = add_qubit_experiment(expt_cfg, 'Readout_Optimization', self.QubitIndex)
        ss_config = {**q_config[self.Qubit], **ss_exp_cfg}
        print('performing single shot for g-e calibration')

        ssp_g = SingleShotProgram_g(self.experiment.soccfg, reps=1, final_delay=ss_config['relax_delay'], cfg=ss_config)
        iq_list_g = ssp_g.acquire(self.experiment.soc, rounds=1, progress=True)

        ssp_e = SingleShotProgram_e(self.experiment.soccfg, reps=1, final_delay=ss_config['relax_delay'], cfg=ss_config)
        iq_list_e = ssp_e.acquire(self.experiment.soc, rounds=1, progress=True)

        ss_I_g = iq_list_g[0][0].T[0]
        ss_Q_g = iq_list_g[0][0].T[1]
        ss_I_e = iq_list_e[0][0].T[0]
        ss_Q_e = iq_list_e[0][0].T[1]

        raw_0 = amp_rabi.get_raw()  # I,Q data without normalizing to readout window, subtracting readout offset, or rotation/thresholding
        A = np.squeeze(raw_0[0])
        I_shots = A[:, :, 0]  # if you have 4 steps and 3 shots/reps this is like [[1,2,3,4],[1,2,3,4],[1,2,3,4]]
        Q_shots = A[:, :, 1]

        # plot_results_scaled does a 1D cosine fit, so it only applies to the 1D (single-gain) sweeps
        if not self.length_rabi_vs_gain:
            q1_fit_cosine, pi_amp = self.plot_results_scaled(I, Q, lengths, config=self.config,
                                                      scaling=True, Ie=ss_I_e, Ig=ss_I_g, Qe=ss_Q_e, Qg=ss_Q_g)

        return I, Q, x_axis, q1_fit_cosine, pi_len, self.config, ss_Q_e, ss_Q_g,ss_I_e, ss_I_g, I_shots, Q_shots

    def run_QZE(self,constant_zeno_pulse=False,adapt_qubit_freq=False, wait_for_res_ring_up=False, exp=None):
        qubit_length_ge_loop = np.linspace(self.config['start'], self.config['stop'], self.config['steps'])
        lengths = []
        I = []
        Q = []
        Magnitude =[]
        for row_idx, length in enumerate(qubit_length_ge_loop):
            updated_config = deepcopy(self.config)
            updated_config['qubit_length_ge'] = length
            print('updated_config[qubit_length_ge]: ', round(updated_config['qubit_length_ge'],4), ' updated_config[res_gain_qze]: ',[round(float(n),4) for n in updated_config['res_gain_qze']])

            if constant_zeno_pulse:
                if adapt_qubit_freq:
                    if updated_config['qubit_length_ge'] > 0.11:
                        exp_spec=deepcopy(exp)
                        exp_spec.qubit_cfg['qubit_length_ge'] = length #update the length of the qubit pulse drive, this is used for zeno/stark pulse inside of qspec
                        q_spec = QubitSpectroscopy(self.QubitIndex, tot_num_of_qubits, "M:/_Data/20250822 - Olivia/run6/6transmon/QZE/QZE_measurement/Documentation/", 0,
                                                   'None', save_figs=True, experiment=exp_spec,
                                                   live_plot=False, verbose=False,
                                                   qick_verbose=True, zeno_stark=True, zeno_stark_pulse_gain=self.zeno_pulse_gain) #update the zeno gain inside the qspec class when redefining the lists for a 7th channel

                        (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit,
                         qubit_freq, sys_config_qspec) = q_spec.run_with_stark_tone(length, wait_for_res_ring_up=wait_for_res_ring_up)
                        del exp_spec
                        updated_config['qubit_freq_ge_starked'][self.QubitIndex] = qubit_freq
                        print(qubit_freq)
                        print('-------------------------')

                if wait_for_res_ring_up:
                    amp_rabi = QZE_constant_pulse_RabiProgram_WaitForResRingUp(
                        self.experiment.soccfg,
                        reps=updated_config['reps'],
                        final_delay=updated_config['relax_delay'],
                        cfg=updated_config
                    )
                else:
                    amp_rabi = QZE_constant_pulse_RabiProgram(
                        self.experiment.soccfg,
                        reps=updated_config['reps'],
                        final_delay=updated_config['relax_delay'],
                        cfg=updated_config
                    )
            else:
                amp_rabi = QZERabiProgram(
                    self.experiment.soccfg,
                    reps=updated_config['reps'],
                    final_delay=updated_config['relax_delay'],
                    cfg=updated_config
                )
            iq_list = amp_rabi.acquire(
                self.experiment.soc,
                rounds=updated_config["rounds"],
                progress=self.qick_verbose
            )

            I.append(iq_list[self.QubitIndex][:, 0][0]) #just a 2d list here becasue we arent doing a qick loop
            Q.append(iq_list[self.QubitIndex][:, 1][0]) #just a 2d list here becasue we arent doing a qick loop
            Magnitude.append(np.abs(iq_list[0].dot([1, 1j]))[0])
            #now I Q and mag should all by a list with a single float in them

            # get the lengs that were used so you can use to plot on the x axis
            lengths.append(amp_rabi.get_pulse_param('qubit_pulse', "length")) #should be just a float by default

            del updated_config


        I=np.asarray(I)
        Q = np.asarray(Q)
        lengths=np.asarray(lengths)
        Magnitude = np.asarray(Magnitude)
        q1_fit_cosine, pi_len = self.plot_results(I, Q, lengths, config=self.config)
        return I, Q, Magnitude, lengths, q1_fit_cosine, pi_len, self.config

    def run_QZE_one_starked_qfreq(self,constant_zeno_pulse=False,adapt_qubit_freq=False, wait_for_res_ring_up=False,
                                  exp=None,optimizationFolder=None, hold_ground=False,three_pulse_binary=False):
        exp_spec = deepcopy(exp)
        exp_spec.qubit_cfg[
            'qubit_length_ge'] = 0.2   #1us because why not, it shouldnt matter that much what is chosen here
        exp_spec.qubit_cfg['qubit_gain_ge'][
            self.QubitIndex] = 0.13  # turn it down for this qspec finding to lower err bars and minimize broadening

        q_spec = QubitSpectroscopy(self.QubitIndex, tot_num_of_qubits,
                                   "M:/_Data/20250822 - Olivia/run6/6transmon/QZE/QZE_measurement/Documentation/", 0,
                                   'None', save_figs=True, experiment=exp_spec,
                                   live_plot=False, verbose=False,
                                   qick_verbose=True, zeno_stark=True,
                                   zeno_stark_pulse_gain=self.zeno_pulse_gain)  # update the zeno gain inside the qspec class when redefining the lists for a 7th channel

        (qspec_I, qspec_Q, qspec_freqs, qspec_I_fit, qspec_Q_fit,
         qubit_freq, sys_config_qspec, fwhm) = q_spec.run_with_stark_tone(0.2, wait_for_res_ring_up=wait_for_res_ring_up)
        ######################################## save qspec data #####################################
        def create_data_dict(keys, save_r, qs):
            return {Q: {key: np.empty(save_r, dtype=object) for key in keys} for Q in range(len(qs))}
        qspec_keys = ['Dates', 'I', 'Q', 'Frequencies', 'I Fit', 'Q Fit', 'Round Num', 'Batch Num', 'Recycled QFreq',
                      'Exp Config', 'Syst Config']
        qspec_data = create_data_dict(qspec_keys, 1, list_of_all_qubits)
        qspec_data[self.QubitIndex]['Dates'][0] = (
            time.mktime(datetime.datetime.now().timetuple()))
        qspec_data[self.QubitIndex]['I'][0] = qspec_I
        qspec_data[self.QubitIndex]['Q'][0] = qspec_Q
        qspec_data[self.QubitIndex]['Frequencies'][0] = qspec_freqs
        qspec_data[self.QubitIndex]['I Fit'][0] = qspec_I_fit
        qspec_data[self.QubitIndex]['Q Fit'][0] = qspec_Q_fit
        qspec_data[self.QubitIndex]['Round Num'][0] = 0
        qspec_data[self.QubitIndex]['Batch Num'][0] = 0
        qspec_data[self.QubitIndex]['Recycled QFreq'][0] = False  # no rr so no recycling here
        qspec_data[self.QubitIndex]['Exp Config'][0] = expt_cfg
        qspec_data[self.QubitIndex]['Syst Config'][0] = sys_config_qspec
        from section_008_save_data_to_h5 import Data_H5
        saver_qspec = Data_H5(optimizationFolder, qspec_data, 0, 1)
        saver_qspec.save_to_h5('QSpec_starked')
        del saver_qspec
        del qspec_data


        del exp_spec

        self.config['qubit_freq_ge_starked'][self.QubitIndex] = qubit_freq #use for all lengths on x axis

        self.config['fwhm_w01_starked'] = fwhm
        print(qubit_freq)
        print('-------------------------')

        qubit_length_ge_loop = np.linspace(self.config['start'], self.config['stop'], self.config['steps'])
        lengths = []
        I = []
        Q = []
        Magnitude =[]

        for row_idx, length in enumerate(qubit_length_ge_loop):
            updated_config = deepcopy(self.config)
            updated_config['qubit_length_ge'] = length
            print('updated_config[qubit_gain_ge]: ', round(updated_config['qubit_gain_ge'],4), 'updated_config[qubit_length_ge]: ', round(updated_config['qubit_length_ge'],4), ' updated_config[res_gain_qze]: ',[round(float(n),4) for n in updated_config['res_gain_qze']])

            if constant_zeno_pulse:
            #
            #     if wait_for_res_ring_up:
            #         amp_rabi = QZE_constant_pulse_RabiProgram_WaitForResRingUp(
            #             self.experiment.soccfg,
            #             reps=updated_config['reps'],
            #             final_delay=updated_config['relax_delay'],
            #             cfg=updated_config
            #         )
            #     else:
                if adapt_qubit_freq:
                    if hold_ground:
                        if three_pulse_binary:
                            amp_rabi = QZE_constant_pulse_3pulse_RabiProgram(
                                self.experiment.soccfg,
                                reps=updated_config['reps'],
                                final_delay=updated_config['relax_delay'],
                                cfg=updated_config
                            )
                        else:
                            amp_rabi = QZE_constant_pulse_gnd_RabiProgram(
                                self.experiment.soccfg,
                                reps=updated_config['reps'],
                                final_delay=updated_config['relax_delay'],
                                cfg=updated_config
                            )
                    else:
                        amp_rabi = QZE_constant_pulse_RabiProgram(
                            self.experiment.soccfg,
                            reps=updated_config['reps'],
                            final_delay=updated_config['relax_delay'],
                            cfg=updated_config
                        )


                else:
                    amp_rabi = QZE_constant_pulse_RabiProgram_unstarked_freq(
                        self.experiment.soccfg,
                        reps=updated_config['reps'],
                        final_delay=updated_config['relax_delay'],
                        cfg=updated_config
                    )
            else:

                amp_rabi = QZERabiProgram(
                    self.experiment.soccfg,
                    reps=updated_config['reps'],
                    final_delay=updated_config['relax_delay'],
                    cfg=updated_config
                )
            iq_list = amp_rabi.acquire(
                self.experiment.soc,
                rounds=updated_config["rounds"],
                progress=self.qick_verbose
            )

            I.append(iq_list[self.QubitIndex][:, 0][0]) #just a 2d list here becasue we arent doing a qick loop
            Q.append(iq_list[self.QubitIndex][:, 1][0]) #just a 2d list here becasue we arent doing a qick loop
            Magnitude.append(np.abs(iq_list[0].dot([1, 1j]))[0])
            #now I Q and mag should all by a list with a single float in them

            # get the lengs that were used so you can use to plot on the x axis
            lengths.append(amp_rabi.get_pulse_param('qubit_pulse', "length")) #should be just a float by default

            del updated_config


        I=np.asarray(I)
        Q = np.asarray(Q)
        lengths=np.asarray(lengths)
        Magnitude = np.asarray(Magnitude)
        q1_fit_cosine, pi_len = self.plot_results(I, Q, lengths, config=self.config)
        return I, Q, Magnitude, lengths, q1_fit_cosine, pi_len, self.config
    def plot_results_scaled(self, I, Q, gains, config = None, fig_quality = 100, scaling=False, Ie=None, Ig=None, Qe=None, Qg=None, file_ext=''):
        try:
            if scaling:
                e = np.mean((Ie + 1j * Qe))
                g = np.mean((Ig + 1j * Qg))
                ### Normalization ###
                pop_norm = abs(((I + 1j * Q) - g) * (e - g) / abs(e - g) ** 2)
                ydata = pop_norm
                fig, (ax1) = plt.subplots(1, 1, figsize=(10, 5))
                plt.rcParams.update({'font.size': 18})

                plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

                q1_a_guess_ydata = (np.max(ydata) - np.min(ydata)) / 2
                q1_d_guess_ydata = np.mean(ydata)

                q1_b_guess = 1 / gains[-1]
                q1_c_guess = 0

                q1_guess_ydata = [q1_a_guess_ydata, q1_b_guess, q1_c_guess, q1_d_guess_ydata]
                q1_popt_ydata, q1_pcov_ydata = curve_fit(self.cosine, gains, ydata, maxfev=100000, p0=q1_guess_ydata)
                q1_fit_cosine_ydata = self.cosine(gains, *q1_popt_ydata)


                first_three_avg_ydata = np.mean(q1_fit_cosine_ydata[:3])
                last_three_avg_ydata = np.mean(q1_fit_cosine_ydata[-3:])

                best_signal_fit = None
                pi_amp = None

                best_signal_fit = q1_fit_cosine_ydata
                # figure out if you should take the min or the max value of the fit to say where pi_amp should be
                if last_three_avg_ydata > first_three_avg_ydata:
                    pi_amp = gains[np.argmax(best_signal_fit)]
                else:
                    pi_amp = gains[np.argmin(best_signal_fit)]

                ax1.plot(gains, q1_fit_cosine_ydata, '-', color='red', linewidth=3, label="Fit")

                if config is not None:
                    if self.QZE:
                        fig.text(plot_middle, 0.98,
                                 f"Rabi Q{self.QubitIndex + 1}_" + f", {config['reps']}*{config['rounds']} avgs" + f' pi_amp {round(pi_amp, 2)} '
                                                                                                                   f'projective readout pulse length'
                                                                                                                   f': {self.projective_readout_pulse_len_us}'
                                                                                                                   f' readout pulse amp: '
                                                                                                                   f' {self.experiment.readout_cfg["res_gain_ge"][self.QubitIndex]} ',
                                 fontsize=24, ha='center',
                                 va='top')  # f", {config['sigma'] * 1000} ns sigma" need to add in all qqubit sigmas to save exp_cfg before putting htis back

                    else:
                        fig.text(plot_middle, 0.98,
                                 f"Rabi Q{self.QubitIndex + 1}_" + f", {config['reps']}*{config['rounds']} avgs" + f' pi_amp {pi_amp} ',
                                 fontsize=24, ha='center',
                                 va='top')  # f", {config['sigma'] * 1000} ns sigma" need to add in all qqubit sigmas to save exp_cfg before putting htis back

                else:
                    fig.text(plot_middle, 0.98,
                             f' pi_amp {pi_amp} ',
                             fontsize=24, ha='center', va='top')

                ax1.plot(gains, ydata, label="Length (us)", linewidth=2)
                ax1.set_ylabel("Qubit Population", fontsize=20)
                ax1.tick_params(axis='both', which='major', labelsize=16)
                ax1.set_xlabel("Length (us)", fontsize=20)

                plt.tight_layout()
                plt.subplots_adjust(top=0.93)

                if self.save_figs:
                    outerFolder_expt = os.path.join(self.outerFolder, self.expt_name + "_plots")
                    self.create_folder_if_not_exists(outerFolder_expt)
                    now = datetime.datetime.now()
                    formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                    file_name = os.path.join(outerFolder_expt,
                                             f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}"+file_ext+".png")
                    fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
                plt.close(fig)
            else:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
                plt.rcParams.update({'font.size': 18})

                plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

                q1_a_guess_I = (np.max(I) - np.min(I)) / 2
                q1_d_guess_I = np.mean(I)
                q1_a_guess_Q = (np.max(Q) - np.min(Q)) / 2
                q1_d_guess_Q = np.mean(Q)
                q1_b_guess = 1 / gains[-1]
                q1_c_guess = 0

                q1_guess_I = [q1_a_guess_I, q1_b_guess, q1_c_guess, q1_d_guess_I]
                q1_popt_I, q1_pcov_I = curve_fit(self.cosine, gains, I, maxfev=100000, p0=q1_guess_I)
                q1_fit_cosine_I = self.cosine(gains, *q1_popt_I)

                q1_guess_Q = [q1_a_guess_Q, q1_b_guess, q1_c_guess, q1_d_guess_Q]
                q1_popt_Q, q1_pcov_Q = curve_fit(self.cosine, gains, Q, maxfev=100000, p0=q1_guess_Q)
                q1_fit_cosine_Q = self.cosine(gains, *q1_popt_Q)

                first_three_avg_I = np.mean(q1_fit_cosine_I[:3])
                last_three_avg_I = np.mean(q1_fit_cosine_I[-3:])
                first_three_avg_Q = np.mean(q1_fit_cosine_Q[:3])
                last_three_avg_Q = np.mean(q1_fit_cosine_Q[-3:])

                best_signal_fit = None
                pi_amp = None
                if 'Q' in self.signal:
                    best_signal_fit = q1_fit_cosine_Q
                    # figure out if you should take the min or the max value of the fit to say where pi_amp should be
                    if last_three_avg_Q > first_three_avg_Q:
                        pi_amp = gains[np.argmax(best_signal_fit)]
                    else:
                        pi_amp = gains[np.argmin(best_signal_fit)]
                if 'I' in self.signal:
                    best_signal_fit = q1_fit_cosine_I
                    # figure out if you should take the min or the max value of the fit to say where pi_amp should be
                    if last_three_avg_I > first_three_avg_I:
                        pi_amp = gains[np.argmax(best_signal_fit)]
                    else:
                        pi_amp = gains[np.argmin(best_signal_fit)]
                if 'None' in self.signal:
                    # choose the best signal depending on which has a larger magnitude
                    if abs(first_three_avg_Q - last_three_avg_Q) > abs(first_three_avg_I - last_three_avg_I):
                        best_signal_fit = q1_fit_cosine_Q
                        # figure out if you should take the min or the max value of the fit to say where pi_amp should be
                        if last_three_avg_Q > first_three_avg_Q:
                            pi_amp = gains[np.argmax(best_signal_fit)]
                        else:
                            pi_amp = gains[np.argmin(best_signal_fit)]
                    else:
                        best_signal_fit = q1_fit_cosine_I
                        # figure out if you should take the min or the max value of the fit to say where pi_amp should be
                        if last_three_avg_I > first_three_avg_I:
                            pi_amp = gains[np.argmax(best_signal_fit)]
                        else:
                            pi_amp = gains[np.argmin(best_signal_fit)]
                else:
                    print('Invalid signal passed, please do I Q or None')


                ax2.plot(gains, q1_fit_cosine_Q, '-', color='red', linewidth=3, label="Fit")
                ax1.plot(gains, q1_fit_cosine_I, '-', color='red', linewidth=3, label="Fit")

                if config is not None:
                    if self.QZE:
                        fig.text(plot_middle, 0.98,
                                 f"Rabi Q{self.QubitIndex + 1}_" + f", {config['reps']}*{config['rounds']} avgs" + f' pi_amp {round(pi_amp,2)} '
                                                                                                                   f'projective readout pulse length'
                                                                                                                   f': {self.projective_readout_pulse_len_us}'
                                                                                                                   f' readout pulse amp: '
                                                                                                                   f' {self.experiment.readout_cfg["res_gain_ge"][self.QubitIndex]} ',
                                 fontsize=24, ha='center',
                                 va='top')  # f", {config['sigma'] * 1000} ns sigma" need to add in all qqubit sigmas to save exp_cfg before putting htis back

                    else:
                        fig.text(plot_middle, 0.98,
                                 f"Rabi Q{self.QubitIndex + 1}_"  + f", {config['reps']}*{config['rounds']} avgs" + f' pi_amp {pi_amp} ',
                                 fontsize=24, ha='center', va='top') #f", {config['sigma'] * 1000} ns sigma" need to add in all qqubit sigmas to save exp_cfg before putting htis back

                else:
                    fig.text(plot_middle, 0.98,
                             f"Rabi Q{self.QubitIndex + 1}_" f", {self.config['sigma'] * 1000} ns sigma" + f' pi_amp {pi_amp} '+ f", {self.config['reps']}*{self.config['rounds']} avgs",
                             fontsize=24, ha='center', va='top')

                ax1.plot(gains, I, label="Length (us)", linewidth=2)
                ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
                ax1.tick_params(axis='both', which='major', labelsize=16)

                ax2.plot(gains, Q, label="Q", linewidth=2)
                ax2.set_xlabel("Length (us)", fontsize=20)
                ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
                ax2.tick_params(axis='both', which='major', labelsize=16)

                plt.tight_layout()
                plt.subplots_adjust(top=0.93)

                if self.save_figs:
                    if self.correction:
                        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name +'_correction'+ "_plots")
                    else:
                        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name +  "_plots")
                    self.create_folder_if_not_exists(outerFolder_expt)
                    now = datetime.datetime.now()
                    formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                    file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
                    fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
                plt.close(fig)
            return best_signal_fit, pi_amp

        except Exception as e:
            if self.verbose: print("Error fitting cosine:", e)
            self.logger.info(f"Error fitting cosine: {e}")
            # Return None if the fit didn't work
            return None, None
    def run_oscilliscope_simple(self, thresholding=False):

        prog = OscilliscopeExampleProgram(self.experiment.soccfg, reps=1, final_delay=0.1, cfg=self.config)
        iq_list = prog.acquire_decimated(self.experiment.soc, rounds=self.config['soft_avgs'])


        I = iq_list[self.QubitIndex][:, 0]
        Q = iq_list[self.QubitIndex][:, 1]

        t = prog.get_time_axis(ro_index=0)

        plt.plot(t, I, label="I value")
        plt.plot(t, Q, label="Q value")
        plt.plot(t, np.abs(iq_list[0].dot([1, 1j])), label="magnitude")
        plt.legend()
        plt.ylabel("a.u.")
        plt.xlabel("us")
        plt.show()

    def run_oscilliscope_zeno(self, thresholding=False):
        qubit_length_ge_loop = np.linspace(self.config['start'], self.config['stop'], self.config['steps'])
        zeno_gain_ge_loop = np.linspace(0.2, 1, self.config['steps'])

        num_rows = len(qubit_length_ge_loop)
        num_cols = len(zeno_gain_ge_loop)

        fig, axs = plt.subplots(num_rows, num_cols, sharex='col', figsize=(10 * num_cols, 3 * num_rows))

        if num_rows == 1 and num_cols == 1:
            axs = np.array([[axs]])
        elif num_rows == 1:
            axs = np.array([axs])
        elif num_cols == 1:
            axs = np.array([[ax] for ax in axs])

        for col_idx, zeno_gain in enumerate(zeno_gain_ge_loop):
            for row_idx, length in enumerate(qubit_length_ge_loop):
                updated_config = deepcopy(self.config)
                updated_config['qubit_length_ge'] = length
                updated_config['res_gain_qze'][-1] = zeno_gain

                prog = OscilliscopeQZEProgram(self.experiment.soccfg, reps=1, final_delay=0.5, cfg=updated_config)
                iq_list = prog.acquire_decimated(self.experiment.soc, rounds=updated_config['soft_avgs'])

                I = iq_list[self.QubitIndex][:, 0]
                Q = iq_list[self.QubitIndex][:, 1]
                t = prog.get_time_axis(ro_index=0)
                magnitude = np.abs(iq_list[0].dot([1, 1j]))

                ax = axs[row_idx, col_idx]
                ax.plot(t, I, label="I value")
                ax.plot(t, Q, label="Q value")
                ax.plot(t, magnitude, label="magnitude")
                ax.set_title(f"Zeno gain: {round(zeno_gain,6)}, Qubit drive length: {round(length, 6)}")
                ax.set_ylabel("a.u.")
                if row_idx == num_rows - 1:
                    ax.set_xlabel("us")
                else:
                    ax.set_xlabel("")

                if row_idx == 0 and col_idx == num_cols - 1:
                    ax.legend(loc='upper right', prop={'size': 12})

        plt.tight_layout()

        outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
        self.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt,
                                 f"Q{self.QubitIndex + 1}_{formatted_datetime}_pulses.png")
        fig.savefig(file_name, dpi=300, bbox_inches='tight')
        plt.show()



    def live_plotting(self, amp_rabi, thresholding):
        I = Q = expt_mags = expt_phases = expt_pop = None
        viz = visdom.Visdom()
        if not viz.check_connection(timeout_seconds=5):
            raise RuntimeError("Visdom server not connected!")

        for ii in range(self.config["rounds"]):
            if thresholding:
                iq_list = amp_rabi.acquire(self.experiment.soc, rounds=1,
                                           threshold=self.experiment.readout_cfg["threshold"],
                                           angle=self.experiment.readout_cfg["ro_phase"], progress=self.qick_verbose)
            else:
                iq_list = amp_rabi.acquire(self.experiment.soc, rounds=1, progress=self.qick_verbose)
            lens = amp_rabi.get_pulse_param('qubit_pulse', "gain", as_array=True)
            iq_list = iq_list[0][0].T
            this_I = (iq_list[0])
            this_Q = (iq_list[1])

            if I is None:  # ii == 0
                I, Q = this_I, this_Q
            else:
                I = (I * ii + this_I) / (ii + 1.0)
                Q = (Q * ii + this_Q) / (ii + 1.0)

            viz.line(X=lens, Y=I, opts=dict(height=400, width=700, title='Rabi I', showlegend=True, xlabel='expt_pts'),win='Rabi_I')
            viz.line(X=lens, Y=Q, opts=dict(height=400, width=700, title='Rabi Q', showlegend=True, xlabel='expt_pts'),win='Rabi_Q')
        return I, Q, lens

    def cosine(self, x, a, b, c, d):

        return a * np.cos(2. * np.pi * b * x - c * 2 * np.pi) + d

    def plot_results_chevron(self, I, Q, lens, freqs,
                             config=None, fig_quality=100,
                             showfig=False, subtract_center=False):
        """
        Chevron heatmap:
          z-axis (color): sqrt(I^2 + Q^2)
          x-axis: pulse length (lens)
          y-axis: frequency offset (freqs)

        Supports:
          - single 2D arrays: I,Q shape (n_freqs, n_lens)
          - list/tuple of 2D arrays (multiple repeats): averaged before plotting

        Parameters
        ----------
        subtract_center : bool
            If True, subtract median of the magnitude to remove background contrast.
        """

        # ---- helper: normalize inputs into a list of arrays ----
        def _to_stack(x):
            if x is None:
                return None
            # if list of arrays -> keep
            if isinstance(x, (list, tuple)) and len(x) > 0 and isinstance(x[0], (list, tuple, np.ndarray)):
                return [np.asarray(xx) for xx in x]
            # otherwise wrap
            return [np.asarray(x)]

        try:
            lens = np.asarray(lens)
            freqs = np.asarray(freqs)

            I_stack = _to_stack(I)
            Q_stack = _to_stack(Q)

            plt.rcParams.update({'font.size': 18})

            # ---- average across repeats/traces if provided as list ----
            I_arr = np.stack(I_stack, axis=0)  # (n_traces, n_freqs, n_lens) OR (n_traces, ...)
            Q_arr = np.stack(Q_stack, axis=0)

            I_mean = np.mean(I_arr, axis=0)
            Q_mean = np.mean(Q_arr, axis=0)

            if I_mean.ndim != 2 or Q_mean.ndim != 2:
                raise ValueError(
                    f"Chevron plotting expects 2D I/Q after averaging. "
                    f"Got I_mean.ndim={I_mean.ndim}, Q_mean.ndim={Q_mean.ndim}."
                )

            # Expect shape (n_freqs, n_lens)
            n_freqs, n_lens = I_mean.shape

            if freqs.shape[0] != n_freqs:
                raise ValueError(
                    f"freqs length ({freqs.shape[0]}) does not match I/Q freq dimension ({n_freqs})."
                )
            if lens.shape[0] != n_lens:
                raise ValueError(
                    f"lens length ({lens.shape[0]}) does not match I/Q length dimension ({n_lens})."
                )

            mag = np.sqrt(I_mean ** 2 + Q_mean ** 2)  # (n_freqs, n_lens)

            if subtract_center:
                mag = mag - np.median(mag)

            # ---- plot heatmap ----
            fig, ax = plt.subplots(1, 1, figsize=(10, 6))
            plot_middle = (ax.get_position().x0 + ax.get_position().x1) / 2

            # imshow expects (Ny, Nx) = (len(y), len(x)).
            # Our mag is (n_freqs, n_lens) which corresponds to (y, x) already,
            # so we do NOT transpose here.
            im = ax.imshow(
                mag,
                origin='lower',
                aspect='auto',
                extent=(lens[0], lens[-1], freqs[0], freqs[-1]),
                interpolation='nearest',  # <- no smoothing between pixels
                resample=False  # <- avoid extra resampling smoothing
            )

            ax.set_xlabel("Qubit drive pulse length (us)", fontsize=20)
            ax.set_ylabel("Frequency offset (Hz)", fontsize=20)  # change label/units if your freqs are MHz etc.
            ax.tick_params(axis='both', which='major', labelsize=16)

            cbar = fig.colorbar(im, ax=ax)
            cbar.set_label("Signal magnitude sqrt(I^2+Q^2) (avg)", fontsize=18)

            # ---- title ----
            if config is not None:
                fig.text(
                    plot_middle, 0.98,
                    f"Chevron Q{self.QubitIndex + 1}, {float(config['reps'])}*{float(config['rounds'])} avgs",
                    fontsize=16, ha='center', va='top'
                )
            else:
                fig.text(
                    plot_middle, 0.98,
                    f"Chevron Q{self.QubitIndex + 1}",
                    fontsize=16, ha='center', va='top'
                )

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)

            if showfig:
                plt.show()

            if self.save_figs:
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(
                    outerFolder_expt,
                    f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_chevron_q{self.QubitIndex + 1}.png"
                )
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')

            plt.close(fig)

            # For chevron we typically don't return a pi_len from a 1D cosine fit.
            # Return the magnitude map in case you want to analyze it later.
            return mag

        except Exception as e:
            if self.verbose:
                print("Error plotting chevron heatmap:", e)
            self.logger.info(f"Error plotting chevron heatmap: {e}")
            return None

    def plot_results(self, I, Q, lens, config = None, fig_quality = 100, showfig=False):
        try:
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            plt.rcParams.update({'font.size': 18})

            plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

            q1_a_guess_I = (np.max(I) - np.min(I)) / 2
            q1_d_guess_I = np.mean(I)
            q1_a_guess_Q = (np.max(Q) - np.min(Q)) / 2
            q1_d_guess_Q = np.mean(Q)
            q1_b_guess = 1 / lens[-1]
            q1_c_guess = 0

            q1_guess_I = [q1_a_guess_I, q1_b_guess, q1_c_guess, q1_d_guess_I]
            q1_popt_I, q1_pcov_I = curve_fit(self.cosine, lens, I, maxfev=100000, p0=q1_guess_I)
            q1_fit_cosine_I = self.cosine(lens, *q1_popt_I)

            q1_guess_Q = [q1_a_guess_Q, q1_b_guess, q1_c_guess, q1_d_guess_Q]
            q1_popt_Q, q1_pcov_Q = curve_fit(self.cosine, lens, Q, maxfev=100000, p0=q1_guess_Q)
            q1_fit_cosine_Q = self.cosine(lens, *q1_popt_Q)

            first_three_avg_I = np.mean(q1_fit_cosine_I[:3])
            last_three_avg_I = np.mean(q1_fit_cosine_I[-3:])
            first_three_avg_Q = np.mean(q1_fit_cosine_Q[:3])
            last_three_avg_Q = np.mean(q1_fit_cosine_Q[-3:])

            best_signal_fit = None
            pi_len = None
            if 'Q' in self.signal:
                best_signal_fit = q1_fit_cosine_Q
                # figure out if you should take the min or the max value of the fit to say where pi_len should be
                if last_three_avg_Q > first_three_avg_Q:
                    pi_len = lens[np.argmax(best_signal_fit)]
                else:
                    pi_len = lens[np.argmin(best_signal_fit)]
            if 'I' in self.signal:
                best_signal_fit = q1_fit_cosine_I
                # figure out if you should take the min or the max value of the fit to say where pi_len should be
                if last_three_avg_I > first_three_avg_I:
                    pi_len = lens[np.argmax(best_signal_fit)]
                else:
                    pi_len = lens[np.argmin(best_signal_fit)]
            if 'None' in self.signal:
                # choose the best signal depending on which has a larger magnitude
                if abs(first_three_avg_Q - last_three_avg_Q) > abs(first_three_avg_I - last_three_avg_I):
                    best_signal_fit = q1_fit_cosine_Q
                    # figure out if you should take the min or the max value of the fit to say where pi_len should be
                    if last_three_avg_Q > first_three_avg_Q:
                        pi_len = lens[np.argmax(best_signal_fit)]
                    else:
                        pi_len = lens[np.argmin(best_signal_fit)]
                else:
                    best_signal_fit = q1_fit_cosine_I
                    # figure out if you should take the min or the max value of the fit to say where pi_len should be
                    if last_three_avg_I > first_three_avg_I:
                        pi_len = lens[np.argmax(best_signal_fit)]
                    else:
                        pi_len = lens[np.argmin(best_signal_fit)]
            else:
                print('Invalid signal passed, please do I Q or None')


            ax2.plot(lens, q1_fit_cosine_Q, '-', color='red', linewidth=3, label="Fit")
            ax1.plot(lens, q1_fit_cosine_I, '-', color='red', linewidth=3, label="Fit")

            if config is not None:
                if self.QZE:
                    fig.text(plot_middle, 0.98,
                             f"Rabi Q{self.QubitIndex + 1}_" + f", {config['reps']}*{config['rounds']} avgs" 
                                                                                                               f'zenopulse gain'
                                                                                                               f': {self.experiment.readout_cfg["res_gain_ge"][self.QubitIndex]}'
                                                                                                               f' readout pulse amp: '
                                                                                                               f' {self.experiment.readout_cfg["res_gain_ge"][self.QubitIndex]} ',
                             fontsize=24, ha='center',
                             va='top')  # f", {config['sigma'] * 1000} ns sigma" need to add in all qqubit sigmas to save exp_cfg before putting htis back

                else:
                    fig.text(plot_middle, 0.98,
                             f"Rabi Q{self.QubitIndex + 1}_"  + f", {config['reps']}*{config['rounds']} avgs",
                             fontsize=24, ha='center', va='top') #f", {config['sigma'] * 1000} ns sigma" need to add in all qqubit sigmas to save exp_cfg before putting htis back

            else:
                fig.text(plot_middle, 0.98,
                         f"Rabi Q{self.QubitIndex + 1}_" f", {self.config['sigma'] * 1000} ns sigma" + f", {self.config['reps']}*{self.config['rounds']} avgs",
                         fontsize=24, ha='center', va='top')

            ax1.plot(lens, I, label="Qubit drive pulse length (us)", linewidth=2)
            ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=16)

            ax2.plot(lens, Q, label="Q", linewidth=2)
            ax2.set_xlabel("Qubit drive pulse length (us)", fontsize=20)
            ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
            ax2.tick_params(axis='both', which='major', labelsize=16)

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)

            if showfig:
                plt.show()
            if self.save_figs:
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(outerFolder_expt, f"R_{self.round_num}_" + f"Q_{self.QubitIndex + 1}_" + f"{formatted_datetime}_" + self.expt_name + f"_q{self.QubitIndex + 1}.png")
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)
            return best_signal_fit, pi_len

        except Exception as e:
            if self.verbose: print("Error fitting cosine:", e)
            self.logger.info(f"Error fitting cosine: {e}")
            # Return None if the fit didn't work
            return None, None

    def plot_results_vs_gain(self, I, Q, lens, gains, config=None, fig_quality=100, showfig=False):
        """
        Length-Rabi-vs-gain analysis.

        I and Q have shape (n_gains, n_lengths): a Rabi oscillation as a function of
        pulse length for each qubit-drive gain. Fit a cosine to each gain's oscillation,
        print the Rabi frequency for that gain (the cosine 'b' parameter; this is in MHz
        when the pulse length is in us), and plot Rabi frequency vs gain.

        Returns (fits, rabi_freqs):
            fits       : (n_gains, n_lengths) array of the per-gain cosine fits
            rabi_freqs : (n_gains,) array of Rabi frequencies (MHz)
        """
        try:
            I = np.asarray(I)
            Q = np.asarray(Q)
            lens = np.asarray(lens, dtype=float)
            gains = np.asarray(gains, dtype=float)

            # pick the signal we fit on (default to I for 'I' or 'None')
            data = Q if 'Q' in self.signal and 'None' not in self.signal else I

            # frequency limits for the fit (cycles per us == MHz when lens is in us)
            dt = (lens[-1] - lens[0]) / (len(lens) - 1)
            nyquist = 0.5 / dt                       # max resolvable frequency
            f_min = 0.5 / (lens[-1] - lens[0])       # at least ~half a period over the window

            rabi_freqs = []
            fits = []
            print(f"\n--- Rabi frequency vs gain (Q{self.QubitIndex + 1}) ---")
            for idx in range(len(gains)):
                trace = np.asarray(data[idx], dtype=float)
                try:
                    a_guess = (np.max(trace) - np.min(trace)) / 2
                    d_guess = np.mean(trace)
                    # FFT-based frequency guess: robust when there are many oscillations
                    b_guess = self._estimate_freq(lens, trace)
                    b_guess = min(max(b_guess, f_min), nyquist)
                    guess = [a_guess, b_guess, 0, d_guess]
                    # bound b so the optimiser can't run off to a low-frequency alias
                    lower = [0, f_min, -1, -np.inf]
                    upper = [np.inf, nyquist, 1, np.inf]
                    popt, _ = curve_fit(self.cosine, lens, trace, maxfev=100000,
                                        p0=guess, bounds=(lower, upper))
                    rabi_freq = abs(popt[1])  # cycles per us == MHz when lens is in us
                    fits.append(self.cosine(lens, *popt))
                    print(f"  gain = {gains[idx]:.4f}  ->  Rabi frequency = {rabi_freq:.4f} MHz")
                except Exception as e:
                    rabi_freq = np.nan
                    fits.append(np.full(lens.shape, np.nan, dtype=float))
                    print(f"  gain = {gains[idx]:.4f}  ->  fit failed: {e}")
                rabi_freqs.append(rabi_freq)
            print("-------------------------------------------\n")

            rabi_freqs = np.array(rabi_freqs)
            fits = np.array(fits)

            # ---- plot: heatmap of the signal + Rabi frequency vs gain ----
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
            plt.rcParams.update({'font.size': 18})

            mag = np.sqrt(I ** 2 + Q ** 2)  # (n_gains, n_lengths)
            extent = [lens[0], lens[-1], gains[0], gains[-1]]
            im = ax1.imshow(mag, aspect='auto', origin='lower', extent=extent)
            ax1.set_xlabel("Qubit drive pulse length (us)", fontsize=18)
            ax1.set_ylabel("Qubit drive gain (a.u.)", fontsize=18)
            ax1.set_title(f"Rabi magnitude Q{self.QubitIndex + 1}", fontsize=18)
            fig.colorbar(im, ax=ax1)

            ax2.plot(gains, rabi_freqs, 'o-', linewidth=2)
            ax2.set_xlabel("Qubit drive gain (a.u.)", fontsize=18)
            ax2.set_ylabel("Rabi frequency (MHz)", fontsize=18)
            ax2.tick_params(axis='both', which='major', labelsize=16)

            plt.tight_layout()

            # ---- per-gain figure: each Rabi curve with its cosine fit overlaid ----
            n = len(gains)
            ncols = 4
            nrows = int(np.ceil(n / ncols))
            fig2, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows),
                                      squeeze=False)
            for idx in range(nrows * ncols):
                ax = axes[idx // ncols][idx % ncols]
                if idx < n:
                    ax.plot(lens, data[idx], '.', markersize=4, label="data")
                    ax.plot(lens, fits[idx], '-', color='red', linewidth=1.5, label="fit")
                    ax.set_title(f"gain={gains[idx]:.3f}, f={rabi_freqs[idx]:.3f} MHz",
                                 fontsize=10)
                    ax.tick_params(axis='both', which='major', labelsize=8)
                else:
                    ax.axis('off')
            fig2.text(0.5, 0.995, f"Length-Rabi fits per gain  Q{self.QubitIndex + 1}",
                      ha='center', va='top', fontsize=14)
            fig2.tight_layout(rect=[0, 0, 1, 0.98])

            if showfig:
                plt.show()
            if self.save_figs:
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                base = (f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_"
                        f"{self.expt_name}")
                fig.savefig(os.path.join(outerFolder_expt,
                            f"{base}_vs_gain_q{self.QubitIndex + 1}.png"),
                            dpi=fig_quality, bbox_inches='tight')
                fig2.savefig(os.path.join(outerFolder_expt,
                             f"{base}_per_gain_fits_q{self.QubitIndex + 1}.png"),
                             dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)
            plt.close(fig2)

            return fits, rabi_freqs

        except Exception as e:
            if self.verbose: print("Error fitting cosine vs gain:", e)
            self.logger.info(f"Error fitting cosine vs gain: {e}")
            return None, None

    def _estimate_freq(self, x, y):
        """Estimate the dominant oscillation frequency of y(x) via an rFFT.
        Returns cycles per unit of x (MHz when x is in us). Assumes ~uniform x."""
        n = len(x)
        dt = (x[-1] - x[0]) / (n - 1)
        yf = np.abs(np.fft.rfft(y - np.mean(y)))
        xf = np.fft.rfftfreq(n, d=dt)
        if len(yf) <= 1:
            return 1.0 / (x[-1] - x[0])
        peak = np.argmax(yf[1:]) + 1  # skip DC bin
        return xf[peak]

    def plot_QZE(self, I, Q, lens, config=None, fig_quality=100):
        try:
            # Create the figure with two subplots for I and Q
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
            plt.rcParams.update({'font.size': 18})
            plot_middle = (ax1.get_position().x0 + ax1.get_position().x1) / 2

            best_signal_fits = []
            pi_lens = []

            # Loop over each measurement in the list(s)
            for idx in range(len(I)):
                current_I = I[idx]
                current_Q = Q[idx]
                current_lens = lens[idx]

                # Calculate initial guesses from the current measurement
                q1_a_guess_I = (np.max(current_I) - np.min(current_I)) / 2
                q1_d_guess_I = np.mean(current_I)
                q1_a_guess_Q = (np.max(current_Q) - np.min(current_Q)) / 2
                q1_d_guess_Q = np.mean(current_Q)
                q1_b_guess = 1 / current_lens[-1]
                q1_c_guess = 0

                # Fit for I
                q1_guess_I = [q1_a_guess_I, q1_b_guess, q1_c_guess, q1_d_guess_I]
                q1_popt_I, q1_pcov_I = curve_fit(self.cosine, current_lens, current_I,
                                                 maxfev=100000, p0=q1_guess_I)
                q1_fit_cosine_I = self.cosine(current_lens, *q1_popt_I)

                # Fit for Q
                q1_guess_Q = [q1_a_guess_Q, q1_b_guess, q1_c_guess, q1_d_guess_Q]
                q1_popt_Q, q1_pcov_Q = curve_fit(self.cosine, current_lens, current_Q,
                                                 maxfev=100000, p0=q1_guess_Q)
                q1_fit_cosine_Q = self.cosine(current_lens, *q1_popt_Q)

                # Calculate average values from the fits for deciding the best signal
                first_three_avg_I = np.mean(q1_fit_cosine_I[:3])
                last_three_avg_I = np.mean(q1_fit_cosine_I[-3:])
                first_three_avg_Q = np.mean(q1_fit_cosine_Q[:3])
                last_three_avg_Q = np.mean(q1_fit_cosine_Q[-3:])

                best_signal_fit = None
                pi_len = None

                # Determine which signal to use based on self.signal
                if 'Q' in self.signal:
                    best_signal_fit = q1_fit_cosine_Q
                    if last_three_avg_Q > first_three_avg_Q:
                        pi_len = current_lens[np.argmax(best_signal_fit)]
                    else:
                        pi_len = current_lens[np.argmin(best_signal_fit)]
                elif 'I' in self.signal:
                    best_signal_fit = q1_fit_cosine_I
                    if last_three_avg_I > first_three_avg_I:
                        pi_len = current_lens[np.argmax(best_signal_fit)]
                    else:
                        pi_len = current_lens[np.argmin(best_signal_fit)]
                elif 'None' in self.signal:
                    if abs(first_three_avg_Q - last_three_avg_Q) > abs(first_three_avg_I - last_three_avg_I):
                        best_signal_fit = q1_fit_cosine_Q
                        if last_three_avg_Q > first_three_avg_Q:
                            pi_len = current_lens[np.argmax(best_signal_fit)]
                        else:
                            pi_len = current_lens[np.argmin(best_signal_fit)]
                    else:
                        best_signal_fit = q1_fit_cosine_I
                        if last_three_avg_I > first_three_avg_I:
                            pi_len = current_lens[np.argmax(best_signal_fit)]
                        else:
                            pi_len = current_lens[np.argmin(best_signal_fit)]
                else:
                    print('Invalid signal passed, please do I, Q, or None')

                best_signal_fits.append(best_signal_fit)
                pi_lens.append(pi_len)

                # Plot the fits and original data for this measurement.
                # Labels include the measurement index so each can be distinguished.
                ax1.plot(current_lens, q1_fit_cosine_I, '-', linewidth=3, label=f"Fit I {idx + 1}")
                ax2.plot(current_lens, q1_fit_cosine_Q, '-', linewidth=3, label=f"Fit Q {idx + 1}")
                ax1.plot(current_lens, current_I, label=f"I Data {idx + 1}", linewidth=2)
                ax2.plot(current_lens, current_Q, label=f"Q Data {idx + 1}", linewidth=2)

            # Use the last measurement's pi_len for the title/annotation (you can change this logic as needed)
            last_pi_len = pi_lens[-1] if pi_lens else None
            if config is not None:
                if self.QZE:
                    fig.text(plot_middle, 0.98,
                             f"Rabi Q{self.QubitIndex + 1}_, {config['reps']}*{config['rounds']} avgs, pi_len {round(last_pi_len, 2)} "
                             f"projective readout pulse length: {self.projective_readout_pulse_len_us} "
                             f"readout pulse amp: {self.experiment.readout_cfg['res_gain_ge'][self.QubitIndex]}",
                             fontsize=24, ha='center', va='top')
                else:
                    fig.text(plot_middle, 0.98,
                             f"Rabi Q{self.QubitIndex + 1}_, {config['reps']}*{config['rounds']} avgs, pi_len {last_pi_len}",
                             fontsize=24, ha='center', va='top')
            else:
                fig.text(plot_middle, 0.98,
                         f"Rabi Q{self.QubitIndex + 1}_, {self.config['sigma'] * 1000} ns sigma, pi_len {last_pi_len}, {self.config['reps']}*{self.config['rounds']} avgs",
                         fontsize=24, ha='center', va='top')

            ax1.set_ylabel("I Amplitude (a.u.)", fontsize=20)
            ax2.set_ylabel("Q Amplitude (a.u.)", fontsize=20)
            ax2.set_xlabel("Gain (a.u.)", fontsize=20)
            ax1.tick_params(axis='both', which='major', labelsize=16)
            ax2.tick_params(axis='both', which='major', labelsize=16)
            ax1.legend()
            ax2.legend()

            plt.tight_layout()
            plt.subplots_adjust(top=0.93)

            if self.save_figs:
                outerFolder_expt = os.path.join(self.outerFolder, self.expt_name)
                self.create_folder_if_not_exists(outerFolder_expt)
                now = datetime.datetime.now()
                formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
                file_name = os.path.join(outerFolder_expt,
                                         f"R_{self.round_num}_Q_{self.QubitIndex + 1}_{formatted_datetime}_{self.expt_name}_q{self.QubitIndex + 1}.png")
                fig.savefig(file_name, dpi=fig_quality, bbox_inches='tight')
            plt.close(fig)
            return best_signal_fits, pi_lens

        except Exception as e:
            if self.verbose:
                print("Error fitting cosine:", e)
            self.logger.info(f"Error fitting cosine: {e}")
            return None, None

    def roll(self, data: np.ndarray) -> np.ndarray:

        kernel = np.ones(5) / 5
        smoothed = np.convolve(data, kernel, mode='valid')

        # Preserve the original array's shape by padding the edges
        pad_size = (len(data) - len(smoothed)) // 2
        return np.concatenate((data[:pad_size], smoothed, data[-pad_size:]))

    def get_results(self, I, Q, lens, grab_depths = False, rolling_avg=False):
        if rolling_avg:
            I = self.roll(I)
            Q = self.roll(Q)

            first_three_avg_I = np.mean(I[:3])
            last_three_avg_I = np.mean(I[-3:])
            first_three_avg_Q = np.mean(Q[:3])
            last_three_avg_Q = np.mean(Q[-3:])
            if 'Q' in self.signal:
                best_signal = Q
                # figure out if you should take the min or the max value of the fit to say where pi_len should be
                if last_three_avg_Q > first_three_avg_Q:
                    pi_len = lens[np.argmax(best_signal)]
                else:
                    pi_len = lens[np.argmin(best_signal)]
            if 'I' in self.signal:
                best_signal = I
                # figure out if you should take the min or the max value of the fit to say where pi_len should be
                if last_three_avg_I > first_three_avg_I:
                    pi_len = lens[np.argmax(best_signal)]
                else:
                    pi_len = lens[np.argmin(best_signal)]
            if 'None' in self.signal:
                # choose the best signal depending on which has a larger magnitude
                if abs(first_three_avg_Q - last_three_avg_Q) > abs(first_three_avg_I - last_three_avg_I):
                    best_signal = Q
                    # figure out if you should take the min or the max value of the fit to say where pi_len should be
                    if last_three_avg_Q > first_three_avg_Q:
                        pi_len = lens[np.argmax(best_signal)]
                    else:
                        pi_len = lens[np.argmin(best_signal)]
                else:
                    best_signal = I
                    # figure out if you should take the min or the max value of the fit to say where pi_len should be
                    if last_three_avg_I > first_three_avg_I:
                        pi_len = lens[np.argmax(best_signal)]
                    else:
                        pi_len = lens[np.argmin(best_signal)]
                tot_amp = [np.sqrt((ifit) ** 2 + (qfit) ** 2) for ifit, qfit in zip(I, Q)]
                depth = abs(tot_amp[np.argmin(tot_amp)] - tot_amp[np.argmax(tot_amp)])
            else:
                print('Invalid signal passed, please do I Q or None')
            return best_signal, pi_len

        else:
            q1_a_guess_I = (np.max(I) - np.min(I)) / 2
            q1_d_guess_I = np.mean(I)
            q1_a_guess_Q = (np.max(Q) - np.min(Q)) / 2
            q1_d_guess_Q = np.mean(Q)
            q1_b_guess = 1 / lens[-1]
            q1_c_guess = 0

            q1_guess_I = [q1_a_guess_I, q1_b_guess, q1_c_guess, q1_d_guess_I]
            q1_popt_I, q1_pcov_I = curve_fit(self.cosine, lens, I, maxfev=100000, p0=q1_guess_I)
            q1_fit_cosine_I = self.cosine(lens, *q1_popt_I)

            q1_guess_Q = [q1_a_guess_Q, q1_b_guess, q1_c_guess, q1_d_guess_Q]
            q1_popt_Q, q1_pcov_Q = curve_fit(self.cosine, lens, Q, maxfev=100000, p0=q1_guess_Q)
            q1_fit_cosine_Q = self.cosine(lens, *q1_popt_Q)

            first_three_avg_I = np.mean(q1_fit_cosine_I[:3])
            last_three_avg_I = np.mean(q1_fit_cosine_I[-3:])
            first_three_avg_Q = np.mean(q1_fit_cosine_Q[:3])
            last_three_avg_Q = np.mean(q1_fit_cosine_Q[-3:])

            best_signal_fit = None
            pi_len = None
            if 'Q' in self.signal:
                best_signal_fit = q1_fit_cosine_Q
                # figure out if you should take the min or the max value of the fit to say where pi_len should be
                if last_three_avg_Q > first_three_avg_Q:
                    pi_len = lens[np.argmax(best_signal_fit)]
                else:
                    pi_len = lens[np.argmin(best_signal_fit)]
            if 'I' in self.signal:
                best_signal_fit = q1_fit_cosine_I
                # figure out if you should take the min or the max value of the fit to say where pi_len should be
                if last_three_avg_I > first_three_avg_I:
                    pi_len = lens[np.argmax(best_signal_fit)]
                else:
                    pi_len = lens[np.argmin(best_signal_fit)]
            if 'None' in self.signal:
                # choose the best signal depending on which has a larger magnitude
                if abs(first_three_avg_Q - last_three_avg_Q) > abs(first_three_avg_I - last_three_avg_I):
                    best_signal_fit = q1_fit_cosine_Q
                    # figure out if you should take the min or the max value of the fit to say where pi_len should be
                    if last_three_avg_Q > first_three_avg_Q:
                        pi_len = lens[np.argmax(best_signal_fit)]
                    else:
                        pi_len = lens[np.argmin(best_signal_fit)]
                else:
                    best_signal_fit = q1_fit_cosine_I
                    # figure out if you should take the min or the max value of the fit to say where pi_len should be
                    if last_three_avg_I > first_three_avg_I:
                        pi_len = lens[np.argmax(best_signal_fit)]
                    else:
                        pi_len = lens[np.argmin(best_signal_fit)]
                tot_amp = [np.sqrt((ifit)**2 + (qfit)**2) for ifit,qfit in zip(q1_fit_cosine_I, q1_fit_cosine_Q)]
                depth = abs(tot_amp[np.argmin(tot_amp)] - tot_amp[np.argmax(tot_amp)])
            else:
                print('Invalid signal passed, please do I Q or None')
            if grab_depths:
                return best_signal_fit, pi_len, depth
            else:
                return best_signal_fit, pi_len

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)


class LengthRabiProgram(AveragerProgramV2):
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
        # Define a generator for the readout pulses with the gains, phases, and mixer/mux frequencies
        # Configure the hardware to set this sort of pulse that we can trigger later
        # This has a rectangle pulse becuase style="const"
        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])
        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )
        self.add_loop("lenloop", cfg["steps"])

    def _body(self, cfg):
        #self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse

        self.delay_auto(t=0, tag='waiting')

        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])
class LengthVsGainRabiProgram(AveragerProgramV2):
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
        # Define a generator for the readout pulses with the gains, phases, and mixer/mux frequencies
        # Configure the hardware to set this sort of pulse that we can trigger later
        # This has a rectangle pulse becuase style="const"
        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])
        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=QickSweep1D("gainloop", cfg['start_gain'], cfg['end_gain']),
                       )
        self.add_loop("lenloop", cfg["steps"])
        self.add_loop("gainloop", cfg["gain_steps"])

    def _body(self, cfg):
        # self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse

        self.delay_auto(t=0, tag='waiting')

        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class LengthRabiChevronProgram(AveragerProgramV2):
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
        # Define a generator for the readout pulses with the gains, phases, and mixer/mux frequencies
        # Configure the hardware to set this sort of pulse that we can trigger later
        # This has a rectangle pulse becuase style="const"
        self.add_pulse(ch=res_ch, name="res_pulse", ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])
        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'],
                       freq=QickSweep1D("freqloop", cfg['start_freq'], cfg['end_freq']),
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )
        self.add_loop("lenloop", cfg["steps"])
        self.add_loop("freqloop", cfg["freq_steps"])

    def _body(self, cfg):
        #self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  # play probe pulse

        self.delay_auto(t=0, tag='waiting')

        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class QZE_gaus_pulse_RabiProgram(AveragerProgramV2):
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
        # generator for the readout and the resonator pulses (qze and readout, where cfg['res_gain_qze'] should have
        # varying lens in each loop iterationof calling this classfor the zeno pulse on ch 7)

        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],  # 9us as usual, should be same length as readout window above
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        # projection pulse (the short pulse used for projective measurement,7 ns)
        self.add_pulse(ch=res_ch, name="proj_pulse",
                       style="const",
                       length=cfg["zeno_pulse_width"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['qubit_freq_ge'], #[0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        # print('qubit_length_ge(should vary on the inner loop from 0.001-0.7): ', round(cfg['qubit_length_ge'], 4),
        #       'qubit_gain_ge(should be constant) : ', round(cfg['qubit_gain_ge'], 4),
        #       ' res_gain_qze (should vary on the outter loop from 0.1-1): ', [round(float(n), 4) for n in cfg['res_gain_qze']])

    def _body(self, cfg):
        #drive the qubit on the qubit channel (list with length 6)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)

        #length of the qubit pulse (us)
        Tdrive = cfg[
            'qubit_length_ge']  #start in congfig is set to 0.1 so i always start at 0.1 and end at 0.7, so qubit should be in first excited state

        #now we have started pulse at 0.1us
        #each pulse is 9 ns long with a 2 ns gap between pulses
        #schendule pulses as long as the entire pulse fits within the qubit drive pulse time
        if Tdrive>0.11:
            t_pulse = 0.11 #start at 0 (qubits in first excited state because config starts at 0.1us qubit pulse)
            while t_pulse + cfg["zeno_pulse_width"] <= Tdrive: # as long as we are below the qubit drive pulse time for the next short res pulse
                self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=t_pulse) #schedule this pulse
                t_pulse += cfg["zeno_pulse_period"]  # 9 ns pulse + 2 ns wait = 11 ns per cycle

        self.delay_auto(t=0, tag='waiting') #auto wait for those pulses to be done
        #immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class QZERabiProgram(AveragerProgramV2):
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
        # generator for the readout and the resonator pulses (qze and readout, where cfg['res_gain_qze'] should have
        # varying lens in each loop iterationof calling this classfor the zeno pulse on ch 7)
        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],  # 9us as usual, should be same length as readout window above
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        # projection pulse (the short pulse used for projective measurement,7 ns)
        self.add_pulse(ch=res_ch, name="proj_pulse",
                       style="const",
                       length=cfg["zeno_pulse_width"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['qubit_freq_ge'], #[0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        # print('qubit_length_ge(should vary on the inner loop from 0.001-0.7): ', round(cfg['qubit_length_ge'], 4),
        #       'qubit_gain_ge(should be constant) : ', round(cfg['qubit_gain_ge'], 4),
        #       ' res_gain_qze (should vary on the outter loop from 0.1-1): ', [round(float(n), 4) for n in cfg['res_gain_qze']])

    def _body(self, cfg):
        #drive the qubit on the qubit channel (list with length 6)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)

        #length of the qubit pulse (us)
        Tdrive = cfg[
            'qubit_length_ge']  #start in congfig is set to 0.1 so i always start at 0.1 and end at 0.7, so qubit should be in first excited state

        #now we have started pulse at 0.1us
        #each pulse is 9 ns long with a 2 ns gap between pulses
        #schendule pulses as long as the entire pulse fits within the qubit drive pulse time
        if Tdrive>0.11:
            t_pulse = 0.11 #start at 0 (qubits in first excited state because config starts at 0.1us qubit pulse)
            while t_pulse + cfg["zeno_pulse_width"] <= Tdrive: # as long as we are below the qubit drive pulse time for the next short res pulse
                self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=t_pulse) #schedule this pulse
                t_pulse += cfg["zeno_pulse_period"]  # 9 ns pulse + 2 ns wait = 11 ns per cycle

        self.delay_auto(t=0, tag='waiting') #auto wait for those pulses to be done
        #immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class QZE_constant_pulse_RabiProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_ch = cfg['ro_ch']
        res_ch = cfg['res_ch']
        qubit_ch = cfg['qubit_ch']
        self.add_readoutconfig(ch=ro_ch, name="myro",
                               freq=cfg['res_freq_ge'],
                               gen_ch=res_ch,
                               outsel='product')
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        # generator for the readout and the resonator pulses (qze and readout, where cfg['res_gain_qze'] should have
        # varying lens in each loop iterationof calling this classfor the zeno pulse on ch 7)
        self.declare_gen(ch=res_ch, nqz=cfg['nqz_res'], ro_ch=ro_ch,
                         mux_freqs=cfg['res_freq_qze'],
                         mux_gains=cfg['res_gain_qze'],  # has 7 values not just 6, extra one for the zeno
                         mux_phases=cfg['res_phase_qze'],
                         mixer_freq=cfg['mixer_freq'])
        self.declare_readout(ch=cfg['ro_ch'], length=cfg['res_length'])

        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],  # 9us as usual, should be same length as readout window above
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        # projection pulse
        if cfg['qubit_length_ge'] > 0.11:
            self.add_pulse(ch=res_ch, name="proj_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11+ 3,
                           freq=cfg['res_freq_ge'],
                           phase=cfg['ro_phase'],
                           gain=cfg['res_gain_ge']
                           )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", #for before we hit pi pulse len
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['qubit_freq_ge'], #[0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        self.add_pulse(ch=qubit_ch, name="qubit_pulse_pi_len",  #after pi pulse len, this is the normal non zeno freq drive
                       style="const",
                       length=0.11,  # total_drive_length,
                       freq=cfg['qubit_freq_ge'],  # [0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        print('starked: ',cfg['qubit_freq_ge_starked'][0], ' qfreq: ', cfg['qubit_freq_ge'])

        if cfg['qubit_length_ge'] > 0.11:
            self.add_pulse(ch=qubit_ch, name="starked_qubit_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11,  # total_drive_length,
                           freq=cfg['qubit_freq_ge_starked'][0],  # [0] # only should be one value
                           phase=cfg['qubit_phase'],
                           gain=cfg['qubit_gain_ge'])  # ramp it up here

        # print('qubit_length_ge(should vary on the inner loop from 0.001-0.7): ', round(cfg['qubit_length_ge'], 4),
        #       'qubit_gain_ge(should be constant) : ', round(cfg['qubit_gain_ge'], 4),
        #       ' res_gain_qze (should vary on the outter loop from 0.1-1): ', [round(float(n), 4) for n in cfg['res_gain_qze']])

    def _body(self, cfg):
        #drive the qubit on the qubit channel (list with length 6)
        if cfg['qubit_length_ge'] <= 0.11:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  #just drive without zeno/stark
        else:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse_pi_len", t=0) #regular w01 pi pulse first, will last 0.11us
            self.delay_auto(t=0, tag='waiting_pi') #wait to finish
            self.pulse(ch=cfg["qubit_ch"], name="starked_qubit_pulse", t=3) #play starked freq qubit drive for rest of qubit pulse len,, start 3 us after ring up
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0) #play zeno/stark tone in resonator
        self.delay_auto(t=0, tag='waiting') #auto wait for those pulses to be done
        #immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class QZE_constant_pulse_gnd_RabiProgram(AveragerProgramV2):
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
        # generator for the readout and the resonator pulses (qze and readout, where cfg['res_gain_qze'] should have
        # varying lens in each loop iterationof calling this classfor the zeno pulse on ch 7)

        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],  # 9us as usual, should be same length as readout window above
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        # projection pulse
        if cfg['qubit_length_ge'] > 0.11*3:
            self.add_pulse(ch=res_ch, name="proj_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11*2+ 3,
                           freq=cfg['res_freq_ge'],
                           phase=cfg['ro_phase'],
                           gain=cfg['res_gain_ge']
                           )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", #for before we hit pi pulse len
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['qubit_freq_ge'], #[0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        self.add_pulse(ch=qubit_ch, name="qubit_pulse_pi_len",  #after pi pulse len, this is the normal non zeno freq drive
                       style="const",
                       length=0.11*3,  # total_drive_length,
                       freq=cfg['qubit_freq_ge'],  # [0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        print('starked: ',cfg['qubit_freq_ge_starked'][0], ' qfreq: ', cfg['qubit_freq_ge'])

        if cfg['qubit_length_ge'] > 0.11*3+0.01:
            self.add_pulse(ch=qubit_ch, name="starked_qubit_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11*3,  # total_drive_length,
                           freq=cfg['qubit_freq_ge_starked'][0],  # [0] # only should be one value
                           phase=cfg['qubit_phase'],
                           gain=cfg['qubit_gain_ge'])  # ramp it up here

        # print('qubit_length_ge(should vary on the inner loop from 0.001-0.7): ', round(cfg['qubit_length_ge'], 4),
        #       'qubit_gain_ge(should be constant) : ', round(cfg['qubit_gain_ge'], 4),
        #       ' res_gain_qze (should vary on the outter loop from 0.1-1): ', [round(float(n), 4) for n in cfg['res_gain_qze']])

    def _body(self, cfg):
        #drive the qubit on the qubit channel (list with length 6)
        if cfg['qubit_length_ge'] <= 0.11*3+0.01:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  #just drive without zeno/stark
        else:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse_pi_len", t=0) #regular w01 pi pulse first, will last 0.11us
            self.delay_auto(t=0, tag='waiting_pi') #wait to finish
            self.pulse(ch=cfg["qubit_ch"], name="starked_qubit_pulse", t=3) #play starked freq qubit drive for rest of qubit pulse len,, start 3 us after ring up
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0) #play zeno/stark tone in resonator
        self.delay_auto(t=0, tag='waiting') #auto wait for those pulses to be done
        #immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


class QZE_constant_pulse_3pulse_RabiProgram(AveragerProgramV2):
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
        # generator for the readout and the resonator pulses (qze and readout, where cfg['res_gain_qze'] should have
        # varying lens in each loop iterationof calling this classfor the zeno pulse on ch 7)

        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],  # 9us as usual, should be same length as readout window above
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        # projection pulse
        if cfg['qubit_length_ge'] > 0.11*3:
            self.add_pulse(ch=res_ch, name="proj_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11*2+ 3,
                           freq=cfg['res_freq_ge'],
                           phase=cfg['ro_phase'],
                           gain=cfg['res_gain_ge']
                           )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", #for before we hit pi pulse len
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['qubit_freq_ge'], #[0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        self.add_pulse(ch=qubit_ch, name="qubit_pulse_pi_len",  #after pi pulse len, this is the normal non zeno freq drive
                       style="const",
                       length=0.11*3,  # total_drive_length,
                       freq=cfg['qubit_freq_ge'],  # [0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        print('starked: ',cfg['qubit_freq_ge_starked'][0], ' qfreq: ', cfg['qubit_freq_ge'])

        if cfg['qubit_length_ge'] > 0.11*3+0.01:
            self.add_pulse(ch=qubit_ch, name="starked_qubit_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11*3,  # total_drive_length,
                           freq=cfg['qubit_freq_ge_starked'][0],  # [0] # only should be one value
                           phase=cfg['qubit_phase'],
                           gain=cfg['qubit_gain_ge'])  # ramp it up here

        # print('qubit_length_ge(should vary on the inner loop from 0.001-0.7): ', round(cfg['qubit_length_ge'], 4),
        #       'qubit_gain_ge(should be constant) : ', round(cfg['qubit_gain_ge'], 4),
        #       ' res_gain_qze (should vary on the outter loop from 0.1-1): ', [round(float(n), 4) for n in cfg['res_gain_qze']])

    def _body(self, cfg):
        #drive the qubit on the qubit channel (list with length 6)
        if cfg['qubit_length_ge'] <= 0.11*3+0.01:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  #just drive without zeno/stark
        elif 1.11 > cfg['qubit_length_ge'] > 0.11*3+0.01:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse_pi_len", t=0) #regular w01 pi pulse first, will last 0.11us
            self.delay_auto(t=0, tag='waiting_pi') #wait to finish
            self.pulse(ch=cfg["qubit_ch"], name="starked_qubit_pulse", t=3) #play starked freq qubit drive for rest of qubit pulse len,, start 3 us after ring up
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0) #play zeno/stark tone in resonator
        elif 2.2 > cfg['qubit_length_ge'] >= 0.11*3+0.01 + 1.11:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  #just drive without zeno/stark
        elif 3.19 > cfg['qubit_length_ge'] >= 0.11*3+0.01 + 2.2: #odd so should be in gnd here
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse_pi_len",
                       t=0)  # regular w01 pi pulse first, will last 0.11us
            self.delay_auto(t=0, tag='waiting_pi')  # wait to finish
            self.pulse(ch=cfg["qubit_ch"], name="starked_qubit_pulse",
                       t=3)  # play starked freq qubit drive for rest of qubit pulse len,, start 3 us after ring up
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0)  # play zeno/stark tone in resonator
        elif 3.96 > cfg['qubit_length_ge'] >= 0.11*3+0.01 + 3.19:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  #just drive without zeno/stark
        elif 5.06 > cfg['qubit_length_ge'] >= 0.11*3+0.01 + 3.96:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse_pi_len",
                       t=0)  # regular w01 pi pulse first, will last 0.11us
            self.delay_auto(t=0, tag='waiting_pi')  # wait to finish
            self.pulse(ch=cfg["qubit_ch"], name="starked_qubit_pulse",
                       t=3)  # play starked freq qubit drive for rest of qubit pulse len,, start 3 us after ring up
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0)  # play zeno/stark tone in resonator
        else:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)

        self.delay_auto(t=0, tag='waiting') #auto wait for those pulses to be done
        #immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


class QZE_constant_pulse_RabiProgram_unstarked_freq(AveragerProgramV2):
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
        # generator for the readout and the resonator pulses (qze and readout, where cfg['res_gain_qze'] should have
        # varying lens in each loop iterationof calling this classfor the zeno pulse on ch 7)

        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],  # 9us as usual, should be same length as readout window above
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        # projection pulse
        if cfg['qubit_length_ge'] > 0.11:
            self.add_pulse(ch=res_ch, name="proj_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11,
                           freq=cfg['res_freq_ge'],
                           phase=cfg['ro_phase'],
                           gain=cfg['res_gain_ge']
                           )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['qubit_freq_ge'], #[0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here


        # print('qubit_length_ge(should vary on the inner loop from 0.001-0.7): ', round(cfg['qubit_length_ge'], 4),
        #       'qubit_gain_ge(should be constant) : ', round(cfg['qubit_gain_ge'], 4),
        #       ' res_gain_qze (should vary on the outter loop from 0.1-1): ', [round(float(n), 4) for n in cfg['res_gain_qze']])

    def _body(self, cfg):
        #drive the qubit on the qubit channel (list with length 6)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  # just drive without zeno/stark
        if cfg['qubit_length_ge'] > 0.11:
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0.11) #play zeno/stark tone in resonator
        self.delay_auto(t=0, tag='waiting') #auto wait for those pulses to be done
        #immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class QZE_constant_pulse_RabiProgram_WaitForResRingUp(AveragerProgramV2):
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
        # generator for the readout and the resonator pulses (qze and readout, where cfg['res_gain_qze'] should have
        # varying lens in each loop iterationof calling this classfor the zeno pulse on ch 7)

        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",ro_ch=ro_ch,
                       style="const",
                       length=cfg["res_length"],  # 9us as usual, should be same length as readout window above
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        # projection pulse
        if cfg['qubit_length_ge'] > 0.11:
            self.add_pulse(ch=res_ch, name="proj_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11 + 3, #plus res ring up time
                           freq=cfg['res_freq_ge'],
                           phase=cfg['ro_phase'],
                           gain=cfg['res_gain_ge']
                           )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse", #for before we hit pi pulse len
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['qubit_freq_ge'], #[0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        self.add_pulse(ch=qubit_ch, name="qubit_pulse_pi_len",  #after pi pulse len, this is the normal non zeno freq drive
                       style="const",
                       length=0.11,  # total_drive_length,
                       freq=cfg['qubit_freq_ge'],  # [0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here
        if cfg['qubit_length_ge'] > 0.11:
            self.add_pulse(ch=qubit_ch, name="starked_qubit_pulse",
                           style="const",
                           length=cfg['qubit_length_ge']-0.11,  # total_drive_length,
                           freq=cfg['qubit_freq_ge_starked'][0],  # [0] # only should be one value
                           phase=cfg['qubit_phase'],
                           gain=cfg['qubit_gain_ge'])  # ramp it up here

        # print('qubit_length_ge(should vary on the inner loop from 0.001-0.7): ', round(cfg['qubit_length_ge'], 4),
        #       'qubit_gain_ge(should be constant) : ', round(cfg['qubit_gain_ge'], 4),
        #       ' res_gain_qze (should vary on the outter loop from 0.1-1): ', [round(float(n), 4) for n in cfg['res_gain_qze']])

    def _body(self, cfg):
        #drive the qubit on the qubit channel (list with length 6)
        if cfg['qubit_length_ge'] <= 0.11:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)  #just drive without zeno/stark
        else:
            self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse_pi_len", t=0) #regular w01 pi pulse first, will last 0.11us
            self.delay_auto(t=0, tag='waiting_pi') #wait to finish
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=0) #play zeno/stark tone in resonator immediately, it is the qubit length plus the ring up time
            self.pulse(ch=cfg["qubit_ch"], name="starked_qubit_pulse", t=3)  # play starked freq qubit drive after resonator has rung up, wil hold until end of zeno drive

        self.delay_auto(t=0, tag='waiting') #auto wait for those pulses to be done
        #immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


class QZERabiProgram(AveragerProgramV2):
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
        # generator for the readout and the resonator pulses (qze and readout, where cfg['res_gain_qze'] should have
        # varying lens in each loop iterationof calling this classfor the zeno pulse on ch 7)

        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=cfg["res_length"],  # 9us as usual, should be same length as readout window above
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        # projection pulse (the short pulse used for projective measurement,7 ns)
        self.add_pulse(ch=res_ch, name="proj_pulse",
                       style="const",
                       length=cfg["zeno_pulse_width"],
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_qubit'])

        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['qubit_freq_ge'], #[0] # only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'])  # ramp it up here

        # print('qubit_length_ge(should vary on the inner loop from 0.001-0.7): ', round(cfg['qubit_length_ge'], 4),
        #       'qubit_gain_ge(should be constant) : ', round(cfg['qubit_gain_ge'], 4),
        #       ' res_gain_qze (should vary on the outter loop from 0.1-1): ', [round(float(n), 4) for n in cfg['res_gain_qze']])

    def _body(self, cfg):
        #drive the qubit on the qubit channel (list with length 6)
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)

        #length of the qubit pulse (us)
        Tdrive = cfg[
            'qubit_length_ge']  #start in congfig is set to 0.1 so i always start at 0.1 and end at 0.7, so qubit should be in first excited state

        #now we have started pulse at 0.1us
        #each pulse is 9 ns long with a 2 ns gap between pulses
        #schendule pulses as long as the entire pulse fits within the qubit drive pulse time
        if Tdrive>0.11:
            t_pulse = 0.11 #start at 0 (qubits in first excited state because config starts at 0.1us qubit pulse)
            while t_pulse + cfg["zeno_pulse_width"] <= Tdrive: # as long as we are below the qubit drive pulse time for the next short res pulse
                self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=t_pulse) #schedule this pulse
                t_pulse += cfg["zeno_pulse_period"]  # 9 ns pulse + 2 ns wait = 11 ns per cycle

        self.delay_auto(t=0, tag='waiting') #auto wait for those pulses to be done
        #immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])


class OscilliscopeQZEProgram(AveragerProgramV2):
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
        # generator for the readout pulses
        # final readout pulse (to measure the qubit state)
        self.add_pulse(ch=res_ch, name="res_pulse",
                       style="const",
                       length=9,  # 9us as usual  cfg["res_length"]
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )
        # projection pulse (the short pulse used for projective measurement,9 ns)
        self.add_pulse(ch=res_ch, name="proj_pulse",
                       style="const",
                       length=0.05, #0.007
                       freq=cfg['res_freq_ge'],
                       phase=cfg['ro_phase'],
                       gain=cfg['res_gain_ge']
                       )

        # generator for the qubit drive and add the qubit drive pulse.
        # drive pulse is continuous over the full duration:
        self.declare_gen(ch=qubit_ch, nqz=cfg['nqz_res'], mixer_freq=cfg['mixer_freq'])
        # total_drive_length = 0.1 + 0.6

        self.add_pulse(ch=qubit_ch, name="qubit_pulse",
                       style="const",
                       length=cfg['qubit_length_ge'],  # total_drive_length,
                       freq=cfg['res_freq_ge'][0], #only should be one value
                       phase=cfg['qubit_phase'],
                       gain=cfg['qubit_gain_ge'] * 20) #ramp it up here

    def _body(self, cfg):

        # drive the qubit on the qubit channel (list with length 6)
        #self.pulse(ch=cfg["res_ch"], name="qubit_pulse", t=0) #why do i need to send this on the res _ch?

        # length of the qubit pulse (us)
        Tdrive = cfg['qubit_length_ge']  #1.5 #start in congfig is set to 0.1 so i always start at 0.1 and end at 0.7, so qubit should be in first excited state

        # now we have started pulse at 0.1us
        # each pulse is 9 ns long with a 2 ns gap between pulses
        # schendule pulses as long as the entire pulse fits within the qubit drive pulse time

        t_pulse = 0  # start at 0 (qubits in first excited state because config starts at 0.1us qubit pulse)
        while t_pulse + 0.05 <= Tdrive:  # as long as we are below the qubit drive pulse time for the next short res pulse
            self.pulse(ch=cfg['res_ch'], name="proj_pulse", t=t_pulse)  # schedule this pulse
            t_pulse += 0.2  # 9 ns pulse + 2 ns wait = 11 ns per cycle


        #self.delay_auto(t=0.5, tag='waiting')  # auto wait for those pulses to be done
        # # immediately after the qubit pulse ends, trigger the readout resonator pulse (9us long).
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=Tdrive+0.5)
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class OscilliscopeExampleProgram(AveragerProgramV2):
    def _initialize(self, cfg):

        ro_chs = cfg['ro_ch']
        gen_ch = cfg['res_ch']
        self.add_readoutconfig(ch=ro_chs, name="myro",
                               freq=cfg['res_freq_ge'],
                               gen_ch=gen_ch,
                               outsel='product')
        self.send_readoutconfig(ch=cfg['ro_ch'], name="myro", t=0)
        self.declare_gen(
            ch=gen_ch, nqz=cfg['nqz_res'], ro_ch=ro_chs[0],
            mux_freqs=cfg['res_freq_qze'],
            mux_gains=[1,0,0,0,0,0,0], #need to ramp it up here to see a clean signal
            mux_phases=cfg['res_phase'],
            mixer_freq=cfg['mixer_freq']
        )
        for ch, f, ph in zip(cfg['ro_ch'], cfg['res_freq_ge'], cfg['ro_phase']):
            self.declare_readout(
                ch=ch, length=10, freq=f, phase=ph, gen_ch=gen_ch
            )

        self.add_pulse(
            ch=gen_ch, name="mymux",
            style="const",
            length=cfg["res_length"],
            mask=cfg["list_of_all_qubits"],
        )


        self.add_pulse(ch=gen_ch, name="mygaus",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"],
                       )

    def _body(self, cfg):

        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=0, ddr4=True)
        self.pulse(ch=cfg['res_ch'], name="mymux", t=0)
        self.delay_auto(t=3, tag='waiting')
        self.pulse(ch=cfg['res_ch'], name="mygaus", t=0)