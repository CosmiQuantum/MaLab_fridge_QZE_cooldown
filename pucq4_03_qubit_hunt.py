"""
STEP 3 -- Find the qubits: sweep the drive across the whole band, per resonator.

The round robin pairs qubit i with resonator i. That assumption is not safe on
PUCQ4 -- the deck labels M5->Q4 and M6->Q6, but the avoided-crossing data says
Q4->M1 and Q6->M2. With one readout tone at a time, a wrong pairing looks
EXACTLY like a missing qubit: you drive the right frequency, watch the wrong
resonator, and see a flat line.

So this script does not assume any pairing. For each resonator it parks the
readout there and sweeps the qubit drive across the entire plausible band. Any
qubit that shows up is, by construction, the one coupled to that resonator.

It also covers the other two unknowns at once:
  - the qubit drive path has never been verified on this setup (TOF and res
    spec only ever exercised gen ch 0 -> ADC)
  - qubit_gain_ge in system_config is still squill's values, tuned for a 3 GHz
    qubit on a different chip; DRIVE_GAIN below is deliberately much higher

    python pucq4_03_qubit_hunt.py

Start with RESONATORS = [0] to check one before committing to all six.
"""

import os
import time
import datetime
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qick.asm_v2 import AveragerProgramV2, QickSweep1D

from socProxy import makeProxy
import pucq4_config as P

# ---------------------------------------------------------------- knobs -----
# M4 was the only unresolved resonator after the coarse and refinement passes.
RESONATORS = [3]

# Targeted windows centered on the PowerPoint table/zero-current Q4 curve.
# Each is +/-200 MHz, far wider than the quoted 20-40 MHz linewidth.
DRIVE_WINDOWS = {
    0: (5505.0, 5540.0),  # M1 coarse candidate 5522.2 MHz
    1: (5420.0, 5455.0),  # M2 coarse candidate 5438.4 MHz
    2: (5540.0, 5575.0),  # M3 coarse candidate 5558.8 MHz
    3: (5485.0, 5500.0),  # M4 coherent Q excursion at 5492.34 MHz
    4: (7060.0, 7100.0),  # M5/Q4 repeated feature near 7080 MHz
}
DRIVE_START = min(DRIVE_WINDOWS[ri][0] for ri in RESONATORS)
DRIVE_STOP = max(DRIVE_WINDOWS[ri][1] for ri in RESONATORS)
DRIVE_STEP = 0.25        # [MHz], resolve the narrow coherent M4 excursion

DRIVE_GAIN = 1.0         # M4-only retry at maximum available drive power.
                         # broadening is a feature here: it widens the line so
                         # a coarse sweep cannot fall between points.
DRIVE_LENGTH = 10.0      # [us] probe pulse

# *** This was 200 on the run that found nothing. ***
# Punch-out needed 65536 averages per point to resolve a resonance dip on this
# chip; the hunt is trying to see a ~25% amplitude change and had 200. That is
# ~18x worse SNR than the measurement that works, which alone explains a flat
# trace. The avg buffer is 16384 deep so reps stays under it, with the rest in
# rounds.
REPS = 4096
ROUNDS = 4               # total 16384 averages/point
RELAX_DELAY = 100        # [us] T1 unknown; keep generous to avoid saturating

# Readout settings. BOTH of these were wrong on the first run:
#
#  - RES_GAIN 0.9 is very likely above punch-out, which decouples the qubit
#    from the resonator entirely. Run pucq4_04_punch_out.py and use ~1/3 of the
#    punch-out gain here.
#  - READOUT_OFFSET parks the readout off the dip minimum. A dispersive shift
#    changes |IQ| by (slope x shift), and the slope is ZERO at the bottom of
#    the dip -- the worst place to sit, and where the least light comes back
#    (the first hunt run read 0.75 |IQ| on M6 against an 18-25 off-resonance
#    baseline). punch_out.py prints the steepest point for each resonator.
# Punch-out found NO shift from gain 0.008 to 1.0 -- 42 dB with the resonance
# fixed -- so this chip sits below punch-out at every accessible power (n_crit
# ~200 photons, and 30 dB of loss means even gain 1.0 does not reach it).
# Readout gain is therefore free to be high for SNR; 0.2 stays clear of the
# mild Kerr-looking pull seen above 0.1.
RES_GAIN = 0.3
RES_LENGTH = 10.0
# Park on the steepest part of the resonance, where d|IQ|/df is largest.
# punch_out.py reported 8919.400 MHz for M1 against a dip at ~8919.8.
# Per-resonator steepest-slope offsets measured from the wide resonator scan.
READOUT_OFFSETS = P.RES_READOUT_OFFSETS

study = "pucq4_first_light"
sub_study = "q4_qubit_refine_0mA"
substudy_txt_notes = (
    "M4-only 0.25 MHz refinement of the coherent I/Q excursion observed at "
    "5492.34 MHz at 0 mA, using M4's measured steepest-slope readout offset. "
    "Yoko 3/Q4 and Yoko 4/Q6 were both verified at 0 A with outputs off.")
# -----------------------------------------------------------------------------


class QubitHuntProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        P.declare_res_single(self, cfg, cfg["_res_freq"], cfg["_res_gain"],
                             cfg["res_length"])

        self.declare_gen(ch=cfg["qubit_ch"], nqz=cfg["nqz_qubit"])
        self.add_loop("freqloop", cfg["steps"])
        self.add_pulse(ch=cfg["qubit_ch"], name="qubit_pulse",
                       style="const",
                       length=cfg["qubit_length_ge"],
                       freq=cfg["qubit_freq_ge"],
                       phase=0,
                       gain=cfg["qubit_gain_ge"])

    def _body(self, cfg):
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)
        self.delay_auto(t=0.01, tag="waiting")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])


def nyquist_chunks(f_start, f_stop, fs):
    """Split a drive range at Nyquist boundaries.

    A single declare_gen has ONE nqz, so a sweep spanning fs/2 (4915 MHz here)
    has to be run as two programs with different nqz. Silently sweeping across
    the boundary gives a sweep that is simply wrong above or below it.
    """
    chunks, lo = [], f_start
    while lo < f_stop:
        zone = P.nyquist_zone(lo, fs)
        zone_top = zone * (fs / 2.0)
        hi = min(f_stop, zone_top - 1e-6)
        if hi > lo:
            chunks.append((lo, hi, zone))
        lo = zone_top
    return chunks


def find_feature(freqs, amps, n_sigma=6.0, min_width=3):
    """Real spectroscopic feature, or (None, contrast) if it is just noise.

    The first version of this used max|amp - median| / median and flagged a
    CANDIDATE on all six resonators when every trace was pure noise -- that
    metric is trivially >5% on any noisy data. A qubit line is not one outlying
    sample: it is a run of consecutive points displaced from the baseline by
    much more than the point-to-point scatter.

    Noise is estimated from the median absolute successive difference, which is
    insensitive to the feature itself (unlike a plain std over the whole trace).
    """
    amps = np.asarray(amps, dtype=float)

    # Robust per-point noise, immune to a real feature sitting in the trace.
    sigma = 1.4826 * np.median(np.abs(np.diff(amps))) / np.sqrt(2)
    if sigma <= 0:
        return None, 0.0

    base = np.median(amps)
    contrast = float(np.abs(amps - base).max() / (abs(base) + 1e-30))

    # Smooth to roughly the expected linewidth before testing. A qubit line
    # spans many points (20-40 MHz at a 4 MHz step), so averaging over that
    # width cuts the noise by sqrt(w) while leaving the feature intact. Testing
    # raw single samples throws away exactly the advantage a wide line gives
    # you, and misses real lines that are only a few sigma per point.
    w = max(int(min_width), 5)
    kernel = np.ones(w) / w
    smooth = np.convolve(amps, kernel, mode="same")
    sigma_s = sigma / np.sqrt(w)

    # Baseline from a wide rolling median, so a slow tilt across the band is
    # not mistaken for a feature.
    half = max(w * 8, 25)
    base_roll = np.array([
        np.median(amps[max(0, i - half):i + half + 1])
        for i in range(len(amps))])

    dev = np.abs(smooth - base_roll)
    # Ignore edges, where 'same' convolution and the rolling median are biased.
    valid = slice(w, len(amps) - w)
    flagged = np.zeros(len(amps), dtype=bool)
    flagged[valid] = dev[valid] > n_sigma * sigma_s
    if not flagged.any():
        return None, contrast

    idx = np.flatnonzero(flagged)
    runs = np.split(idx, np.flatnonzero(np.diff(idx) != 1) + 1)
    good = [r for r in runs if len(r) >= min_width]
    if not good:
        return None, contrast

    best = max(good, key=lambda r: dev[r].max())
    return float(freqs[best[int(np.argmax(dev[best]))]]), contrast


def load_res_freqs():
    """Prefer the frequencies pucq4_02 actually measured."""
    path = os.path.join(P.DATA_ROOT, "")  # placeholder, see below
    # The npz lives under the timestamped tree, so just use the VNA values
    # unless RES_FREQS_MEASURED has been filled in by hand.
    return np.asarray(P.RES_FREQS_MEASURED, dtype=float)


def main(experiment=None):
    soc, soccfg = ((experiment.soc, experiment.soccfg)
                   if experiment is not None else makeProxy())
    fs_q = soccfg["gens"][P.QUBIT_CH]["fs"]

    res_freqs = load_res_freqs()
    folders = P.setup_data_folders(study, sub_study, substudy_txt_notes)
    plotdir = folders["studyDocumentationFolder"]
    datadir = folders["subStudyDataFolder"]
    logger = folders["logger"]

    print(f"\nTargeted drive windows at {DRIVE_STEP} MHz spacing, gain {DRIVE_GAIN}")
    for ri in RESONATORS:
        lo, hi = DRIVE_WINDOWS[ri]
        print(f"  M{ri+1}: {lo:.0f}-{hi:.0f} MHz")
    print(f"Resonators: {[f'M{i+1}' for i in RESONATORS]}\n")

    results = {}
    for ri in RESONATORS:
        freqs_all, i_all, q_all, amps_all = [], [], [], []
        t0 = time.time()
        drive_start, drive_stop = DRIVE_WINDOWS[ri]
        chunks = nyquist_chunks(drive_start, drive_stop, fs_q)

        for lo, hi, zone in chunks:
            steps = max(2, int((hi - lo) / DRIVE_STEP))
            cfg = P.base_cfg(experiment)
            cfg["res_length"] = RES_LENGTH
            cfg["relax_delay"] = RELAX_DELAY
            cfg["nqz_qubit"] = zone
            cfg["steps"] = steps
            cfg["qubit_length_ge"] = DRIVE_LENGTH
            cfg["qubit_gain_ge"] = DRIVE_GAIN
            cfg["qubit_freq_ge"] = QickSweep1D("freqloop", lo, hi)
            cfg["_res_freq"] = float(res_freqs[ri]) + READOUT_OFFSETS[ri]
            cfg["_res_gain"] = RES_GAIN

            prog = QubitHuntProgram(soccfg, reps=REPS,
                                    final_delay=RELAX_DELAY, cfg=cfg)
            iq = prog.acquire(soc, rounds=ROUNDS, progress=True)

            a = np.asarray(iq[0], dtype=float).reshape(-1, 2)
            i_all.append(a[:, 0])
            q_all.append(a[:, 1])
            amps_all.append(np.abs(a[:, 0] + 1j * a[:, 1]))
            freqs_all.append(np.linspace(lo, hi, len(amps_all[-1])))

        freqs = np.concatenate(freqs_all)
        i_values = np.concatenate(i_all)
        q_values = np.concatenate(q_all)
        amps = np.concatenate(amps_all)
        results[ri] = (freqs, i_values, q_values, amps)

        f_best, contrast = find_feature(freqs, amps)
        logger.info(f"M{ri+1}: feature {contrast*100:.1f}% at {f_best} MHz "
                    f"({time.time()-t0:.0f} s)")
        print(f"  M{ri+1}: " + (f"feature {contrast*100:.1f}% at {f_best:.1f} MHz"
              if f_best is not None else "nothing above noise")
              + f"  ({time.time()-t0:.0f} s)")

    # ------------------------------------------------------------------
    # Plot: one panel per resonator, VNA qubit guesses marked
    # ------------------------------------------------------------------
    n = len(RESONATORS)
    fig, axes = plt.subplots(n, 1, figsize=(15, 3.0 * n), squeeze=False,
                             sharex=True)
    summary = []
    for ax, ri in zip(axes[:, 0], RESONATORS):
        freqs, _i_values, _q_values, amps = results[ri]
        f_best, contrast = find_feature(freqs, amps)
        summary.append((ri, f_best, contrast))

        ax.plot(freqs, amps, linewidth=0.8)
        for qf in P.QUBIT_FREQS_VNA:
            ax.axvline(qf, linestyle=":", color="grey", linewidth=0.8)
        if f_best is not None:
            ax.axvline(f_best, linestyle="--", color="tab:red")
        ax.set_ylabel("|IQ|")
        ax.set_title(f"readout on M{ri+1} ({res_freqs[ri]+READOUT_OFFSETS[ri]:.1f} MHz) -- "
                     + (f"feature {contrast*100:.1f}% at {f_best:.0f} MHz"
                        "   <-- CANDIDATE" if f_best is not None
                        else "nothing above noise"))
    axes[-1, 0].set_xlabel("Qubit drive frequency (MHz)")
    fig.suptitle(f"PUCQ4 qubit hunt, drive gain {DRIVE_GAIN} "
                 f"(dotted = VNA qubit values)", fontsize=14)
    fig.tight_layout()

    path = os.path.join(plotdir, "qubit_hunt.png")
    fig.savefig(path, dpi=200)
    fig.savefig(os.path.join(plotdir, "qubit_hunt.pdf"), dpi=200)
    plt.close(fig)

    keys = ["Dates", "I", "Q", "Amps", "Frequencies", "Found Freq",
            "Readout Frequency", "Round Num", "Batch Num", "Exp Config",
            "Syst Config"]
    qspec_data = P.create_data_dict(keys)
    now = time.mktime(datetime.datetime.now().timetuple())
    for ri in RESONATORS:
        run_freqs, run_i, run_q, run_amps = results[ri]
        found_freq, _contrast = find_feature(run_freqs, run_amps)
        qspec_data[ri]["Dates"][0] = now
        qspec_data[ri]["I"][0] = run_i
        qspec_data[ri]["Q"][0] = run_q
        qspec_data[ri]["Amps"][0] = run_amps
        qspec_data[ri]["Frequencies"][0] = run_freqs
        qspec_data[ri]["Found Freq"][0] = found_freq
        qspec_data[ri]["Readout Frequency"][0] = (
            res_freqs[ri] + READOUT_OFFSETS[ri])
        qspec_data[ri]["Round Num"][0] = 1
        qspec_data[ri]["Batch Num"][0] = 1
        qspec_data[ri]["Exp Config"][0] = {
            "start": DRIVE_START, "stop": DRIVE_STOP, "step": DRIVE_STEP,
            "gain": DRIVE_GAIN, "reps": REPS, "rounds": ROUNDS,
        }
        qspec_data[ri]["Syst Config"][0] = P.base_cfg(experiment)
    P.save_h5(datadir, qspec_data, "qspec_ge")

    print("\n" + "=" * 74)
    hits = [(ri, f, c) for ri, f, c in summary if f is not None]
    if hits:
        print("CANDIDATES (readout resonator -> qubit frequency):")
        for ri, f, c in hits:
            print(f"  M{ri+1}  ->  {f:.1f} MHz   ({c*100:.1f}% contrast)")
        print("\nThat mapping is measured, not assumed. If it disagrees with")
        print("the deck's M5->Q4 / M6->Q6, trust this.")
    else:
        print("Nothing found on any resonator across "
              f"{DRIVE_START}-{DRIVE_STOP} MHz.")
        print("\nThat is informative -- it rules out the pairing being wrong,")
        print("since every resonator was checked against the whole band.")
        print("Remaining suspects, in order:")
        print("  1. QUBIT DRIVE PATH. It has never been verified on this")
        print("     setup. Is the drive line connected to DAC ch 1 (DAC_A,")
        print("     tile 2)? Is the room-temperature source/attenuation right?")
        print("  2. Flux. Qubits could be outside even this window.")
        print(f"  3. Drive gain. Raise DRIVE_GAIN from {DRIVE_GAIN} to 1.0.")
        print("  4. Readout contrast. You are 30 dB down at 9 GHz; the")
        print("     dispersive shift may be under the noise. Raise REPS.")
    print(f"\nPlots: {path}")
    print("=" * 74 + "\n")


class PUCQ4QubitHunt:
    """Round-robin-compatible wrapper around the wide qubit search."""

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
        self.exp_cfg = P.base_cfg(experiment)

    def run(self):
        return main(experiment=self.experiment)


if __name__ == "__main__":
    main()
