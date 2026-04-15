from scipy.optimize import curve_fit
from build_task import *
from build_state import *
from expt_config import *
from system_config import *
import matplotlib.pyplot as plt
import numpy as np
import logging

class CKPProgram_g(AveragerProgramV2):
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
        self.add_pulse(ch=res_ch, name="stark_tone",
                       style="const",
                       length=cfg['ckp_length'],
                       freq=cfg['res_freq_ckp'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ckp_gain']
                       )
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
                       freq=QickSweep1D("qubit_pulse_loop", cfg['qubit_freq_ge'] + cfg["start_freq"],
                                        cfg['qubit_freq_ge'] + cfg["end_freq"]),
                       phase=cfg['ro_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_loop("qubit_pulse_loop", cfg["qubit_pulse_steps"])  # inner loop

    def _body(self, cfg):
        self.pulse(ch=self.cfg['res_ch'], name="stark_tone", t=0)  # play stark tone
        self.pulse(ch=cfg['qubit_ch'], name="qubit_pulse", t=cfg['qubit_pulse_delay'])  # play qubit pulse with delay
        self.delay(t=cfg['ckp_length'] + cfg[
            'readout_pulse_delay'])  # wait for stark tone to finish and for resonator to reach vacuum
        self.delay_auto(t=0)
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class CKPProgram_e(AveragerProgramV2):
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
        self.add_pulse(ch=res_ch, name="stark_tone",
                       style="const",
                       length=cfg['ckp_length'],
                       freq=cfg['res_freq_ckp'],
                       phase=cfg['ro_phase'],
                       gain=cfg['ckp_gain']
                       )
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
                       freq=QickSweep1D("qubit_pulse_loop", cfg['qubit_freq_ge'] + cfg["start_freq"],
                                        cfg['qubit_freq_ge'] + cfg["end_freq"]),
                       phase=cfg['ro_phase'],
                       gain=cfg['qubit_gain_ge'],
                       )

        self.add_gauss(ch=qubit_ch, name="ramp", sigma=cfg['sigma'], length=cfg['sigma'] * 4, even_length=False)
        self.add_pulse(ch=qubit_ch, name="pi_pulse",
                       style="arb",
                       envelope="ramp",
                       freq=cfg['qubit_freq_ge'],
                       phase=cfg['qubit_phase'],
                       gain=cfg['pi_amp'],
                       )

        self.add_loop("qubit_pulse_loop", cfg["qubit_pulse_steps"])  # inner loop

    def _body(self, cfg):
        self.pulse(ch=cfg['qubit_ch'], name="pi_pulse") # put qubit in e
        self.delay_auto()
        self.pulse(ch=self.cfg['res_ch'], name="stark_tone")  # play stark tone
        self.pulse(ch=cfg['qubit_ch'], name="qubit_pulse", t=cfg['qubit_pulse_delay'])  # play qubit pulse with delay
        self.delay(t=cfg['ckp_length'] + cfg[
            'readout_pulse_delay'])  # wait for stark tone to finish and for resonator to reach vacuum
        self.delay_auto(t=0)
        self.pulse(ch=cfg['res_ch'], name="res_pulse")
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'])

class CKPMeasurement:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, round_num, signal, save_figs,res_freq_ckp, res_phase_ckp, experiment = None,
                 fit_data = None, verbose = False, logger = None, qick_verbose=True):
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
        self.logger = logger if logger is not None else logging.getLogger("custom_logger_for_rr_only")

        if experiment is not None:
            self.q_config = all_qubit_state(self.experiment, self.number_of_qubits)
            self.exp_cfg = add_qubit_experiment(expt_cfg, self.expt_name, self.QubitIndex)
            self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
            self.config['res_freq_ckp'] = res_freq_ckp
            self.config['res_phase_ckp'] = res_phase_ckp
            stark_mask = np.arange(0, self.number_of_qubits + 1)
            stark_mask = np.delete(stark_mask, QubitIndex)
            if self.verbose: print(f'Q {self.QubitIndex + 1} Round {self.round_num} CKP configuration: ', self.config)
            self.logger.info(f'Q {self.QubitIndex + 1} Round {self.round_num} CKP configuration:{self.config}')

    def run(self):
        now = datetime.datetime.now()

        gain_sweep = np.linspace(self.config["start_gain"], self.config["end_gain"],num=self.config["gain_steps"])
        res_freq_sweep = np.linspace(self.config["res_freq_start"], self.config["res_freq_stop"], num=self.config["res_freq_steps"])

        I_g_nested = []  # [gain][res_freq][IQ over qubit-freq sweep]
        Q_g_nested = []
        I_e_nested = []
        Q_e_nested = []

        sweep_points = []  # list of (gain, res_freq) actually used, in iteration order
        qu_freq_sweep = None  # will capture once (assumed identical for g/e and across iterations)

        for g in gain_sweep:
            # per-gain containers (these will be appended to the outer lists)
            Ig_per_gain = []
            Qg_per_gain = []
            Ie_per_gain = []
            Qe_per_gain = []

            for f in res_freq_sweep:
                self.config['ckp_gain'] = float(np.round(g, 3))
                self.config['res_freq_ckp'] = float(np.round(f, 6))  # last channel pulse

                ckp_g = CKPProgram_g(
                    self.experiment.soccfg,
                    reps=self.config['reps'],
                    final_delay=self.config['relax_delay'],
                    cfg=self.config
                )
                ckp_e = CKPProgram_e(
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
                i0_g = iqg[0]
                q0_g = iqg[1]

                iq_list_e = ckp_e.acquire(
                    self.experiment.soc,
                    rounds=self.config['rounds'],
                    progress=self.qick_verbose
                )
                iqe = iq_list_e[0][0].T
                i0_e = iqe[0]
                q0_e = iqe[1]

                # append IQ vectors (over qubit-freq sweep) at the innermost level
                Ig_per_gain.append(i0_g)
                Qg_per_gain.append(q0_g)
                Ie_per_gain.append(i0_e)
                Qe_per_gain.append(q0_e)

                # record the sweep point
                sweep_points.append((float(np.round(g, 3)), float(np.round(f, 6))))

                # capture qubit frequency sweep array once
                if qu_freq_sweep is None:
                    qu_freq_sweep = ckp_g.get_pulse_param("qubit_pulse", "freq", as_array=True)

            # finish this gain
            I_g_nested.append(Ig_per_gain)
            Q_g_nested.append(Qg_per_gain)
            I_e_nested.append(Ie_per_gain)
            Q_e_nested.append(Qe_per_gain)

        if self.save_figs:
            mid_gain_index = len(gain_sweep) // 2
            save_path = os.path.join(
                self.outerFolder,
                f"Q{self.QubitIndex + 1}_ckp_slice_gain_{mid_gain_index}.png"
            )
            self.plot_ckp_slice(
                I_g_nested, Q_g_nested, I_e_nested, Q_e_nested,
                qu_freq_sweep, gain_sweep, res_freq_sweep,
                gain_index=mid_gain_index,
                save_path=save_path,
                show=False
            )

        return (I_g_nested, Q_g_nested, I_e_nested, Q_e_nested, qu_freq_sweep, gain_sweep, res_freq_sweep, sweep_points, self.config)

    def plot_ckp_slice(self, I_g_nested, Q_g_nested, I_e_nested, Q_e_nested,
                       qu_freq_sweep, gain_sweep, res_freq_sweep,
                       gain_index=0, save_path=None, show=False):
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

        fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)

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
        axes[0].set_title(f'|g> prep, gain={gain_sweep[gain_index]:.3f}')
        axes[0].set_xlabel('Qubit probe frequency')
        axes[0].set_ylabel('Resonator CKP frequency')
        fig.colorbar(im0, ax=axes[0])

        im1 = axes[1].imshow(
            mag_e,
            aspect='auto',
            origin='lower',
            extent=extent
        )
        axes[1].set_title(f'|e> prep, gain={gain_sweep[gain_index]:.3f}')
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
        if 0 <= QUBIT_INDEX < num_qubits:  # makes sure you are within the range of options
            res_gain_ge[QUBIT_INDEX] = 1  # Set the gain for the selected qubit
        return res_gain_ge

    def exponential(self, x, a, b, c, d):
        return a * np.exp(-(x - b) / c) + d

    def create_folder_if_not_exists(self, folder):
        """Creates a folder at the given path if it doesn't already exist."""
        if not os.path.exists(folder):
            os.makedirs(folder)