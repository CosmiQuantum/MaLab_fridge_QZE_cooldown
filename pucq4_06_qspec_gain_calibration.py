"""Find the lowest reliable PUCQ4 spectroscopy gain for 0.5-1 MHz lines."""

import datetime
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from qick.asm_v2 import AveragerProgramV2, QickSweep1D
from scipy.optimize import curve_fit

from socProxy import makeProxy
import pucq4_config as P


MODES = [0, 1, 2, 4, 5]  # M4 is deliberately excluded: no validated transition.
CENTERS_MHZ = np.array([5515.3, 5439.6, 5557.5, np.nan, 7083.6, 6523.5])
GAINS_BY_MODE = {
    0: np.array([0.03, 0.03]),
    1: np.array([0.05, 0.05]),
    2: np.array([0.08, 0.08]),
    4: np.array([0.05, 0.05]),
    5: np.array([0.03, 0.03]),
}
SPAN_MHZ = 6.0
STEPS = 121
SPEC_LENGTH_US = 10.0
REPS = 2048
ROUNDS = 1
RES_GAIN = 0.3
RES_LENGTH_US = 10.0
RELAX_DELAY_US = 100.0

study = "pucq4_first_light"
sub_study = "qspec_gain_repeatability_0mA"
substudy_txt_notes = (
    "Descending qubit spectroscopy gain calibration on M1, M2, M3, M5, and "
    "M6 at 0 mA. M4 excluded because no transition was reproducibly found. "
    "Repeatability check for candidate gains. Target fitted FWHM is approximately "
    "0.5-1.0 MHz with at least 5-sigma contrast.")


class GainCalibrationProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        P.declare_res_single(self, cfg, cfg["_res_freq"], cfg["_res_gain"],
                             cfg["res_length"])
        self.declare_gen(ch=cfg["qubit_ch"], nqz=cfg["nqz_qubit"])
        self.add_loop("freqloop", cfg["steps"])
        self.add_pulse(ch=cfg["qubit_ch"], name="qubit_pulse", style="const",
                       length=cfg["qubit_length_ge"],
                       freq=cfg["qubit_freq_ge"], phase=0,
                       gain=cfg["qubit_gain_ge"])

    def _body(self, cfg):
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)
        self.delay_auto(0.01)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0],
                     t=cfg.get("trig_time", 0.0))


def lorentzian(freq, center, fwhm, height, offset):
    return offset + height / (1.0 + 4.0 * ((freq - center) / fwhm) ** 2)


def fit_line(freqs, values):
    baseline = float(np.median(np.r_[values[:20], values[-20:]]))
    deviations = values - baseline
    peak_index = int(np.argmax(np.abs(deviations)))
    noise = float(1.4826 * np.median(np.abs(deviations - np.median(deviations))))
    height = float(deviations[peak_index])
    try:
        params, _ = curve_fit(
            lorentzian, freqs, values,
            p0=[freqs[peak_index], 0.75, height, baseline],
            bounds=([freqs[0], 0.05, -np.inf, -np.inf],
                    [freqs[-1], 4.0, np.inf, np.inf]), maxfev=20000)
        center, fwhm, fit_height, offset = [float(value) for value in params]
        fit = lorentzian(freqs, *params)
        snr = abs(fit_height) / max(noise, np.finfo(float).eps)
        reliable = 0.45 <= fwhm <= 1.05 and snr >= 5.0
        return center, fwhm, snr, reliable, fit
    except (RuntimeError, ValueError):
        return np.nan, np.nan, 0.0, False, np.full_like(values, np.nan)


def main(experiment=None):
    soc, soccfg = ((experiment.soc, experiment.soccfg)
                   if experiment is not None else makeProxy())
    folders = P.setup_data_folders(study, sub_study, substudy_txt_notes)
    plotdir = folders["studyDocumentationFolder"]
    datadir = folders["subStudyDataFolder"]
    results = {}

    for mode in MODES:
        mode_results = []
        center_guess = float(CENTERS_MHZ[mode])
        found_reliable = False
        for gain in GAINS_BY_MODE[mode]:
            cfg = P.base_cfg(experiment)
            cfg.update({
                "steps": STEPS,
                "qubit_length_ge": SPEC_LENGTH_US,
                "qubit_gain_ge": float(gain),
                "qubit_freq_ge": QickSweep1D(
                    "freqloop", center_guess - SPAN_MHZ / 2,
                    center_guess + SPAN_MHZ / 2),
                "_res_freq": float(P.RES_FREQS_MEASURED[mode]
                                   + P.RES_READOUT_OFFSETS[mode]),
                "_res_gain": RES_GAIN,
                "res_length": RES_LENGTH_US,
                "relax_delay": RELAX_DELAY_US,
            })
            prog = GainCalibrationProgram(
                soccfg, reps=REPS, final_delay=RELAX_DELAY_US, cfg=cfg)
            iq = prog.acquire(soc, rounds=ROUNDS, progress=True)
            data = np.asarray(iq[0], dtype=float).reshape(-1, 2)
            freqs = np.linspace(center_guess - SPAN_MHZ / 2,
                                center_guess + SPAN_MHZ / 2, len(data))
            # Fit the component with the larger robust excursion.
            candidates = []
            for component in (data[:, 0], data[:, 1],
                              np.hypot(data[:, 0], data[:, 1])):
                candidates.append(fit_line(freqs, component))
            fit_result = max(candidates, key=lambda result: result[2])
            mode_results.append((float(gain), freqs, data[:, 0], data[:, 1],
                                 *fit_result))
            print(f"M{mode + 1} gain {gain:.3f}: center={fit_result[0]:.4f} "
                  f"MHz, FWHM={fit_result[1]:.3f} MHz, SNR={fit_result[2]:.1f}, "
                  f"accepted={fit_result[3]}")
            if fit_result[3]:
                found_reliable = True
            elif found_reliable:
                # Gains are descending. Once contrast is lost after an accepted
                # point, lower gains will not be more reliable at fixed averaging.
                break
        results[mode] = mode_results

    fig, axes = plt.subplots(len(MODES), 1, figsize=(12, 3 * len(MODES)),
                             squeeze=False)
    selected = {}
    for axis, mode in zip(axes[:, 0], MODES):
        accepted = [row for row in results[mode] if row[7]]
        selected[mode] = accepted[-1] if accepted else None
        for row in results[mode]:
            gain, freqs, i_values, q_values, center, fwhm, snr, reliable, fit = row
            values = i_values if np.ptp(i_values) >= np.ptp(q_values) else q_values
            axis.plot(freqs, values, linewidth=0.7,
                      label=f"g={gain:.3f}, w={fwhm:.2f}, SNR={snr:.1f}")
        axis.set_title(f"M{mode + 1}; selected gain: "
                       f"{selected[mode][0] if selected[mode] else 'none'}")
        axis.set_ylabel("I or Q")
        axis.legend(fontsize=7, ncol=2)
    axes[-1, 0].set_xlabel("Qubit frequency (MHz)")
    fig.tight_layout()
    fig.savefig(os.path.join(plotdir, "qspec_gain_calibration.png"), dpi=200)
    plt.close(fig)

    keys = ["Dates", "Gains", "I", "Q", "Frequencies", "Centers",
            "FWHM", "SNR", "Accepted", "Selected Gain", "Round Num",
            "Batch Num", "Exp Config", "Syst Config"]
    saved = P.create_data_dict(keys)
    now = time.mktime(datetime.datetime.now().timetuple())
    for mode in MODES:
        rows = results[mode]
        saved[mode]["Dates"][0] = now
        saved[mode]["Gains"][0] = np.array([row[0] for row in rows])
        saved[mode]["I"][0] = np.array([row[2] for row in rows])
        saved[mode]["Q"][0] = np.array([row[3] for row in rows])
        saved[mode]["Frequencies"][0] = np.array([row[1] for row in rows])
        saved[mode]["Centers"][0] = np.array([row[4] for row in rows])
        saved[mode]["FWHM"][0] = np.array([row[5] for row in rows])
        saved[mode]["SNR"][0] = np.array([row[6] for row in rows])
        saved[mode]["Accepted"][0] = np.array([row[7] for row in rows])
        saved[mode]["Selected Gain"][0] = (
            selected[mode][0] if selected[mode] else np.nan)
        saved[mode]["Round Num"][0] = 1
        saved[mode]["Batch Num"][0] = 1
        saved[mode]["Exp Config"][0] = {
            "span_mhz": SPAN_MHZ, "steps": STEPS,
            "pulse_length_us": SPEC_LENGTH_US, "reps": REPS,
            "rounds": ROUNDS, "gains": GAINS_BY_MODE[mode].tolist(),
        }
        saved[mode]["Syst Config"][0] = P.base_cfg(experiment)
    P.save_h5(datadir, saved, "qspec_gain")
    return selected


class PUCQ4QSpecGainCalibration:
    def __init__(self, QubitIndex, number_of_qubits, outerFolder, round_num,
                 signal, save_figs=True, experiment=None):
        self.QubitIndex = QubitIndex
        self.number_of_qubits = number_of_qubits
        self.outerFolder = outerFolder
        self.round_num = round_num
        self.signal = signal
        self.save_figs = save_figs
        self.experiment = experiment
        self.expt_name = "qubit_spec_ge"

    def run(self):
        return main(self.experiment)


if __name__ == "__main__":
    main()
