import matplotlib.pyplot as plt
from qick.asm_v2 import AveragerProgramV2
from tqdm import tqdm
from build_state import *
from expt_config import *
import matplotlib.pyplot as plt
import os, datetime
import datetime
import time
from windfreak import SynthHD

class SingleToneSpectroscopyProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        ro_chs = cfg['ro_ch']
        res_ch = cfg['res_ch']

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

    def _body(self, cfg):
        self.trigger(ros=[cfg['ro_ch']], pins=[0], t=cfg['trig_time'], ddr4=True)
        self.pulse(ch=cfg['res_ch'], name="res_pulse", t=0)


class PunchOut:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, experiment, unmasking_resgain=False):
        self.outerFolder = outerFolder
        self.expt_name = "res_spec"
        self.number_of_qubits = number_of_qubits
        self.experiment = experiment
        self.Qubit = 'Q' + str(1)
        self.QubitIndex = QubitIndex
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]

        if unmasking_resgain:
            self.exp_cfg["list_of_all_qubits"] = [self.QubitIndex]

        self.q_config = all_qubit_state(experiment, self.number_of_qubits)
        self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
        print(f'Punch Out configuration: ', self.config)

    def run(self, soccfg, soc, start_gain, stop_gain, num_points, DAC_att, ADC_att,
            plot_Center_shift=True, plot_res_sweeps=True):

        fpts = self.exp_cfg["start"] + self.exp_cfg["step_size"] * np.arange(self.exp_cfg["steps"])

        fcenter = np.asarray(self.config['res_freq_ge']).astype(float)

        resonance_vals, power_sweep, frequency_sweeps = self.sweep_power(
            soccfg, soc, fpts, fcenter, start_gain, stop_gain, num_points
        )

        if plot_Center_shift:
          
            self.plot_center_shift(resonance_vals, power_sweep, DAC_att, ADC_att)

        if plot_res_sweeps:
            self.plot_res_sweeps(fpts, fcenter, frequency_sweeps, power_sweep, DAC_att, ADC_att)

        return

    def _amp_from_iq(self,iq_list):
        arr = np.asarray(iq_list)
        try:
            if arr.shape[-1] == 2:
                I = arr[..., 0].mean()
                Q = arr[..., 1].mean()
                return float(np.abs(I + 1j * Q))
            # Fallback: try to interpret as [I, Q] nested somewhere
            I = float(arr.flat[0])
            Q = float(arr.flat[1])
            return float(np.abs(I + 1j * Q))
        except Exception:
            return float(np.abs(arr).mean())

    def sweep_power(self, soccfg, soc, fpts, fcenter, start_gain, stop_gain, num_points):
        power_sweep = np.linspace(start_gain, stop_gain, num_points)
        N = int(self.number_of_qubits)
        F = len(fpts)
        P = len(power_sweep)

        frequency_sweeps = np.zeros((P, N, F), dtype=float)
        resonance_vals = []
        original_qindex = getattr(self, "QubitIndex", None)

        for pi, p in enumerate(power_sweep):
            power = round(float(p), 3)
            self.config['res_gain_ge'] = power

            freq_res_for_power = []
            for qi in range(N):
                self.QubitIndex = qi

                amps_q = np.zeros(F, dtype=float)
                center_q = float(fcenter[qi])

                for fi, df in enumerate(fpts):
                    self.config["res_freq_ge"] = float(center_q + df)

                    prog = SingleToneSpectroscopyProgram(
                        soccfg, reps=self.exp_cfg["reps"], final_delay=0.5, cfg=self.config
                    )
                    iq_list = prog.acquire(soc, rounds=self.exp_cfg["rounds"], progress=True)
                    amps_q[fi] = self._amp_from_iq(iq_list)

                frequency_sweeps[pi, qi, :] = amps_q

                min_idx = int(np.argmin(amps_q))
                freq_res_for_power.append(round(float(fpts[min_idx] + center_q), 3))

            resonance_vals.append(freq_res_for_power)

        if original_qindex is not None:
            self.QubitIndex = original_qindex

        return resonance_vals, power_sweep, frequency_sweeps

    def plot_res_sweeps(self, fpts, fcenter, frequency_sweeps, power_sweep, DAC_att, ADC_att):
        P, N, F = frequency_sweeps.shape

        plt.figure(figsize=(12, 8))
        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 10,
        })

        for qi in range(N):
            plt.subplot(2, 3, qi + 1)
            x = fpts + float(fcenter[qi])
            for pi in range(P):
                plt.plot(x, frequency_sweeps[pi, qi, :], '-', linewidth=1.5,
                         label=str(round(power_sweep[pi], 3)))
            plt.xlabel("Frequency (MHz)", fontweight='normal')
            plt.ylabel("Amplitude (a.u)", fontweight='normal')
            plt.title(f"Resonator {qi + 1}", pad=10)
            if qi == 0:
                plt.legend(loc='upper left', title='Gain')

        plt.suptitle(f"Resonance At Various Probe Gains DAC_Att_{DAC_att}, ADC_ATT_{ADC_att}",
                     fontsize=24, y=0.95)

        plt.tight_layout(pad=2.0)
        outerFolder_expt = os.path.join(self.outerFolder, "punch_out")
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(
            outerFolder_expt,
            f"{formatted_datetime}_punch_out_res_sweep_DAC_Att_{DAC_att}_ADC_ATT_{ADC_att}.png"
        )
        plt.savefig(file_name, dpi=300)
        from section_008_save_data_to_h5 import Data_H5
        import numpy as np, time, copy

        sweeps = np.asarray(frequency_sweeps)
        P, N, F = sweeps.shape
        fpts_arr = np.asarray(fpts, dtype=float)
        fcenter_arr = np.asarray(fcenter, dtype=float)
        power_arr = np.asarray(power_sweep, float)

        resonance_vals = np.empty((P, N), dtype=float)
        for pi in range(P):
            for qi in range(N):
                min_idx = int(np.argmin(sweeps[pi, qi, :]))
                resonance_vals[pi, qi] = float(fpts_arr[min_idx] + fcenter_arr[qi])


        def _create_data_dict(keys, num_qubits):
            d = {Q: {k: np.empty(1, dtype=object) for k in keys} for Q in range(num_qubits)}
            return d

        punchout_keys = ['Dates', 'freq_pts', 'freq_center', 'Amps', 'Found Freqs',
                         'Power Sweep', 'Round Num', 'Batch Num', 'Exp Config', 'Syst Config']
        punchout_data = _create_data_dict(punchout_keys, N)

        now_epoch = time.mktime(datetime.datetime.now().timetuple())

        for Q in range(N):
            punchout_data[Q]['Dates'][0] = now_epoch
            punchout_data[Q]['freq_pts'][0] = fpts_arr
            punchout_data[Q]['freq_center'][0] = fcenter_arr[Q]
            punchout_data[Q]['Amps'][0] = sweeps[:, Q, :]
            punchout_data[Q]['Found Freqs'][0] = resonance_vals[:, Q]  # (P,)
            punchout_data[Q]['Power Sweep'][0] = power_arr  # (P,)
            punchout_data[Q]['Round Num'][0] = 0
            punchout_data[Q]['Batch Num'][0] = 0
            punchout_data[Q]['Exp Config'][0] = copy.deepcopy(self.exp_cfg)
            punchout_data[Q]['Syst Config'][0] = copy.deepcopy(self.config)

        base = f"{formatted_datetime}_punch_out_res_sweep_DAC_Att_{DAC_att}_ADC_ATT_{ADC_att}"
        saver_punch = Data_H5(outerFolder_expt, punchout_data, 0, 1)
        saver_punch.save_to_h5('punch_out_ge')
        del saver_punch
        del punchout_data
        plt.close()
        return
    def plot_center_shift(self, resonance_vals, power_sweep,DAC_att, ADC_att ):
        plt.figure(figsize=(12, 8))

        plt.rcParams.update({
            'font.size': 14,
            'axes.titlesize': 18,
            'axes.labelsize': 16,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
        })

        for i in range(self.number_of_qubits):
            plt.subplot(2, 3, i + 1)
            plt.plot(power_sweep, [six_resonance_vals[i] for six_resonance_vals in resonance_vals], '-', linewidth=1.5)

            plt.xlabel("Probe Gain", fontweight='normal')
            plt.ylabel("Freq (MHz)", fontweight='normal')
            plt.title(f"Resonator {i + 1}", pad=10)

        plt.suptitle(f"Frequency vs Probe Gain, _DAC_Att_{DAC_att}, ADC_ATT_{ADC_att}", fontsize=24, y=0.95)

        plt.tight_layout(pad=2.0)

        outerFolder_expt = os.path.join(self.outerFolder, 'punch_out')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt, f"{formatted_datetime}_punch_out_center_shift_DAC_Att_{DAC_att}_ADC_ATT_{ADC_att}.png")
        plt.savefig(file_name, dpi=300)
        plt.close()
        return



class TWPAConsistency:
    def __init__(self, outerFolder, experiment):
        self.outerFolder = outerFolder
        self.expt_name = "res_spec"

        self.experiment = experiment
        self.Qubit = 'Q' + str(1)
        self.experiment = experiment
        self.exp_cfg = expt_cfg[self.expt_name]
        self.q_config = all_qubit_state(experiment)
        self.config = {**self.q_config[self.Qubit], **self.exp_cfg}
        print(f'TWPA Consistency configuration: ', self.config)

    def run(self, soccfg, soc, pump_power, pump_freq, num_points, plot_res_sweeps=True, plot_gains=True):
        fpts = (np.linspace(self.exp_cfg["start"], self.exp_cfg["stop"], self.exp_cfg["steps"]))

        resonance_vals, frequency_sweeps, gains = self.repeat_TWPA(soccfg, soc, fpts, pump_power, pump_freq, num_points)

        if plot_res_sweeps:
            self.plot_res_sweeps(fpts, frequency_sweeps, pump_power, pump_freq, num_points)

        if plot_gains:
            self.plot_TWPA_gains(pump_power, pump_freq, num_points, gains)

        return

    def repeat_TWPA(self, soccfg, soc, fpts, pump_power, pump_freq, num_points):
        #power_sweep = np.linspace(start_power, stop_power, num_points)

        resonance_vals = []
        frequency_sweeps = []
        gains = np.zeros((len(self.config['res_freq_ge']), num_points))

        synth = SynthHD('/dev/ttyACM0')
        synth[0].frequency = pump_freq
        synth[0].power = pump_power
        synth[0].enable = True
        time.sleep(2)

        for n in range(num_points):
            amps = np.zeros((len(self.config['res_freq_ge']), len(fpts)))
            for index, f in enumerate(tqdm(fpts)):
                self.config["res_freq_ge"] = f
                prog = SingleToneSpectroscopyProgram(soccfg, reps=self.exp_cfg["reps"], final_delay=0.5,
                                                     cfg=self.config)
                iq_list = prog.acquire(soc, rounds=self.exp_cfg["rounds"], progress=False)
                for i in range(len(self.config['res_freq_ge'])):
                    amps[i][index] = np.abs(iq_list[i][:, 0] + 1j * iq_list[i][:, 1])
                    gains[i][n] = np.max(amps[i]) - np.min(amps[i])
            amps = np.array(amps)
            frequency_sweeps.append(amps)

            freq_res = []
            for i in range(len(self.config['res_freq_ge'])):
                freq_res.append(fpts[np.argmin(amps[i])])
            resonance_vals.append(freq_res)
        synth[0].enable = False

        return resonance_vals, frequency_sweeps, gains

    def plot_res_sweeps(self, fpts, frequency_sweeps, pump_power, pump_freq, num_points):
        plt.figure(figsize=(12, 8))

        # Set larger font sizes
        plt.rcParams.update({
            'font.size': 14,  # Base font size
            'axes.titlesize': 18,  # Title font size
            'axes.labelsize': 16,  # Axis label font size
            'xtick.labelsize': 14,  # X-axis tick label size
            'ytick.labelsize': 14,  # Y-axis tick label size
            'legend.fontsize': 14,  # Legend font size
        })
        for meas_index in range(num_points):
            for i in range(len(self.config['res_freq_ge'])):
                plt.subplot(2, 2, i + 1)
                plt.plot(fpts.T[i], frequency_sweeps[meas_index][i], '-', linewidth=1.5,
                         label=(meas_index+1))

                plt.xlabel("Frequency (MHz)", fontweight='normal')
                plt.ylabel("Amplitude (a.u)", fontweight='normal')
                plt.title(f"Resonator {i + 1}", pad=10)
                plt.legend(loc='upper left', fontsize='6', title='Meas Num')

        # Add a main title to the figure
        plt.suptitle(f"Resonance With TWPA: {pump_freq/1e9} GHz, {pump_power} dBm", fontsize=24, y=0.95)

        plt.tight_layout(pad=2.0)
        outerFolder_expt = os.path.join(self.outerFolder, 'TWPA_opt')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt, f"{formatted_datetime}_TWPAcons_res_sweep.png")
        plt.savefig(file_name, dpi=300)
        plt.close()
        return

    def plot_TWPA_gains(self, pump_power, pump_freq, num_points, gains):
        plt.figure(figsize=(12, 8))

        # Set larger font sizes
        plt.rcParams.update({
            'font.size': 14,  # Base font size
            'axes.titlesize': 18,  # Title font size
            'axes.labelsize': 16,  # Axis label font size
            'xtick.labelsize': 14,  # X-axis tick label size
            'ytick.labelsize': 14,  # Y-axis tick label size
            'legend.fontsize': 14,  # Legend font size
        })
        for i in range(len(self.config['res_freq_ge'])):
            plt.subplot(2, 2, i + 1)
            plt.scatter(range(1, num_points+1), gains[i])

            plt.xlabel("Measurement Number", fontweight='normal')
            plt.ylabel("Peak height (a.u)", fontweight='normal')
            plt.title(f"Resonator {i + 1}", pad=10)

        # Add a main title to the figure
        plt.suptitle(f"Amplitude Heights With TWPA: {pump_freq/1e9} GHz, {pump_power} dBm", fontsize=24, y=0.95)

        plt.tight_layout(pad=2.0)
        outerFolder_expt = os.path.join(self.outerFolder, 'TWPA_opt')
        self.experiment.create_folder_if_not_exists(outerFolder_expt)
        now = datetime.datetime.now()
        formatted_datetime = now.strftime("%Y-%m-%d_%H-%M-%S")
        file_name = os.path.join(outerFolder_expt, f"{formatted_datetime}_TWPAcons_heights.png")
        plt.savefig(file_name, dpi=300)
        plt.close()
        print(f"For TWPA at {pump_freq/1e9} GHz, {pump_power} dBm:")
        for q in range(len(self.config['res_freq_ge'])):
            max_gain = np.max(gains[q])
            min_gain = np.min(gains[q])
            print(f"Q{q + 1} Max Height {max_gain}")
            print(f"Q{q + 1} Min Height {min_gain}")
        return