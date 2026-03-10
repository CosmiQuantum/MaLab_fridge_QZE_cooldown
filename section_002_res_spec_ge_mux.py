import matplotlib.pyplot as plt
from qick.asm_v2 import AveragerProgramV2
from tqdm import tqdm
from build_state import *
from expt_config import *
import copy
import datetime
import logging
import numpy as np


class SingleToneSpectroscopyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_chs = cfg['ro_ch']
        res_ch = cfg['res_ch']
        #self.reset_gens()
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
        
        self.add_gauss(ch=res_ch, name='ramp2', sigma = 0.01, length = 0.02, even_length=True)
        self.add_pulse(ch=res_ch, name= "Ramp", ro_ch=ro_chs,
                       style="flat_top",
                       envelope = 'ramp2',
                       length = cfg['res_length'],
                       freq = cfg['res_freq_ge'],
                       phase = cfg['ro_phase'],
                       gain = cfg['res_gain_ge']
                       )


        
        v_smag = 16383  ## Max Power for arb signal
        sigma = 0.01  ## Sam Suggest 0.01, TOF plots suggest 0.02 looks a lot smoother                          

        alpha = 0.4   ### Normal Gain Units ## With this setup alpha + beta cannot be more than 0.5
        beta = 0.1    ### Normal Gain units  ## sneaking around this is tough so we will ignore for n
        l_two = 0.03     ##Length of initial peak
        l_base = 0.85     ##Dominated Readout length  



        stepsmall = 10000  ##This is probably over kill but it looks nice                                       
        ti = 4*sigma #0 + StartTime
        deslen = l_base+ti+ (4*sigma)
        alpha =alpha*4
        beta = beta * 4
        fs_gen = self.soccfg['gens'][res_ch]['fs']
        stepscorr = (((int(fs_gen * deslen)) + 15) // 16) * 16

        t_new = np.linspace(0, deslen, stepscorr)

        # Defining the piecewise flat-top Gaussian function f(t, sigma, ti, L)                              
        def f_flat(t_new, sigma, ti, L):
            # Gaussian Rise                                                                                           
            rise = np.exp(-((t_new - ti)**2) / (2 * sigma**2))
            # Flat Top                                                                                                  
            top = np.ones_like(t_new)
            # Gaussian Fall                                                                                       
            fall = np.exp(-((t_new - (ti + L))**2) / (2 * sigma**2))
            # Use np.select to apply conditions element-wise                            
            return np.select(
                [t_new < ti, (t_new >= ti) & (t_new < ti + L), t_new >= ti + L],
                [rise, top, fall]
            )
        pulse_shape = v_smag  * (beta * f_flat(t_new, sigma, ti, l_base) + alpha * f_flat(t_new, sigma, ti, l_two))

        self.add_envelope(ch=res_ch, name="TwoStep",idata = pulse_shape)
        self.add_pulse(ch=res_ch, name="two_step", ro_ch=ro_chs,
                       style="arb",
                       envelope='TwoStep',
                       freq= cfg['res_freq_ge'],#cfg['res_freq_ge'],                                            
                       phase=cfg['ro_phase'],
                       gain= 1.0#cfg['res_gain_ge']+0.6                                                         
                       )

        self.delay(100.0)


        
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

        print(self.exp_cfg["reps"])
        print(self.exp_cfg["reps"])
        print(self.exp_cfg["reps"])
        print(self.exp_cfg["reps"])
        print(self.exp_cfg["reps"])


        
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
            prog = SingleToneSpectroscopyProgram(self.experiment.soccfg, reps=self.exp_cfg["reps"], final_delay=0.5, cfg=self.config)
            print(self.exp_cfg["reps"])
            print(self.exp_cfg["reps"])
            print(self.exp_cfg["reps"])
            print(self.exp_cfg["reps"])
            print(self.exp_cfg["reps"])

            iq_list = prog.acquire(self.experiment.soc, progress=self.qick_verbose)
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

        print(self.exp_cfg["reps"])
        print(self.exp_cfg["reps"])
        print(self.exp_cfg["reps"])
        print(self.exp_cfg["reps"])
        print(self.exp_cfg["reps"])


        

        plt.subplot(2, 3, 1)
        #plt.plot(fpts + fcenter[i], amps[i], '-', linewidth=1.5)
        plt.plot([f + fcenter for f in fpts], amps, '-', linewidth=1.5)
        freq_r = fpts[np.argmin(amps)] + fcenter

        print(freq_r)

        res_freqs.append(freq_r)

        plt.axvline(freq_r, linestyle='--', color='orange', linewidth=1.5)
        plt.title(f"Resonator {self.QubitIndex + 1} {freq_r:.3f} MHz", pad=10)

        plt.xlabel("Frequency (MHz)")
        plt.ylabel("Amplitude (a.u.)")

        plt.ylim(plt.ylim()[0] - 0.05 * (plt.ylim()[1] - plt.ylim()[0]), plt.ylim()[1])

        if self.experiment is not None:
            plt.suptitle(f"MUXed resonator spectroscopy {self.config['reps']}*{self.config['rounds']} avgs", fontsize=24, y=0.95)
        else:
            plt.suptitle(f"MUXed resonator spectroscopy {reloaded_config ['reps']}*{reloaded_config ['rounds']} avgs",
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



