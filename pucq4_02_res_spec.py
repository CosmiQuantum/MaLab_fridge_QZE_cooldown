"""
STEP 2 -- Resonator spectroscopy across the whole PUCQ4 band.

Sweeps one continuous window (default 8900-9080 MHz) rather than six separate
+/-15 MHz windows, because when you are not yet sure the frequencies are right,
one wide pass that finds whatever is there beats six guesses.

WHY THIS SHOULD WORK WHERE TOF DID NOT
--------------------------------------
pucq4_01_tof.py saw nothing at 9 GHz, but that measurement is the worst case:
acquire_decimated captures a raw trace at the full 553 MHz decimated bandwidth.
This script uses the accumulated readout, which integrates over RES_LENGTH and
then averages REPS*ROUNDS times. Against the TOF run that is roughly:

    10*log10(553e6 / (1/RES_LENGTH))   narrower noise bandwidth
  + 10*log10(REPS*ROUNDS / 400)        relative averaging

which at the defaults below is ~40 dB. The gap TOF revealed was ~18 dB, and the
VNA that DID see these resonators was only ~31 dB ahead of the TOF run. So
there is comfortable margin -- but if you still see nothing, raise REPS and
RES_LENGTH before concluding anything. The script prints its own budget.

    python pucq4_02_res_spec.py
"""

import os
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from qick.asm_v2 import AveragerProgramV2

from socProxy import makeProxy
import pucq4_config as P

# ---------------------------------------------------------------- knobs -----
F_START = 8900.0        # [MHz] band to sweep
F_STOP = 9080.0         # [MHz]
F_STEP = 0.2            # [MHz] coarse first pass; resonators are <1 MHz wide,
                        # so drop to 0.02 once you know roughly where they are.

RES_LENGTH = 10.0       # [us] integration window. Max is 29.6 us on this
                        # firmware (16384 samples at 552.96 MHz). Every
                        # doubling buys 3 dB.
# Averaging. REPS is a hardware loop inside the tProc: the whole average costs
# ONE Pyro round trip. ROUNDS is a software loop in QICK's acquire(), costing a
# round trip EACH -- at ~8 ms of network latency apiece, rounds=1000 made every
# frequency point take 8.4 s to do 30 ms of measuring. Put the averaging in
# REPS and leave ROUNDS at 1.
REPS = 1000
ROUNDS = 1
GAIN = 0.9              # near max; PUCQ4 sits ~11 dB down the sin(x)/x curve
RELAX_DELAY = 20        # [us] no qubit is being excited, so this can be short

# Control sweep in the old chip's band, to prove the method end to end.
# Set to None once you trust it.
CONTROL_BAND = (7140.0, 7300.0)

study = "pucq4_first_light"
sub_study = "res_spec"
substudy_txt_notes = ("PUCQ4 wide resonator spectroscopy, 8900-9080 MHz. "
                      "Accumulated readout with long integration, after TOF "
                      "saw nothing at 9 GHz. Includes a control sweep over the "
                      "previous chip's band.")
# -----------------------------------------------------------------------------


class ResSpecProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        P.declare_res_single(self, cfg, cfg["_freq"], cfg["_gain"],
                             cfg["res_length"])

    def _body(self, cfg):
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)


def sweep(soc, soccfg, cfg, freqs, label):
    amps = np.zeros(len(freqs))
    for i, f in enumerate(tqdm(freqs, desc=label)):
        c = dict(cfg)
        c["_freq"] = float(f)
        c["_gain"] = float(GAIN)
        prog = ResSpecProgram(soccfg, reps=REPS,
                              final_delay=c["relax_delay"], cfg=c)
        amps[i] = P.amp_from_iq(prog.acquire(soc, rounds=ROUNDS,
                                             progress=False))
    return amps


def find_dips(freqs, amps, expected=None, n_expect=P.NUM_RES):
    """Locate resonator dips by prominence.

    The baseline slopes ~25 -> ~11 across the band (analog rolloff), so a flat
    amplitude threshold would find dips at the low-frequency end and miss them
    at the high end. Prominence is measured relative to the local baseline, so
    it is immune to that tilt.
    """
    from scipy.signal import find_peaks

    inverted = -amps
    # Require each dip to stand out by at least 10% of the full amplitude span.
    prom = 0.10 * (np.max(amps) - np.min(amps))
    idx, props = find_peaks(inverted, prominence=prom)

    # Keep the n_expect most prominent, then restore frequency order.
    if len(idx) > n_expect:
        keep = np.argsort(props["prominences"])[-n_expect:]
        idx = np.sort(idx[keep])

    found = freqs[idx]
    depths = 1.0 - amps[idx] / np.array(
        [np.median(amps[max(0, i - 40):i + 40]) for i in idx])

    matched = None
    if expected is not None and len(found):
        # Nearest measured dip to each expected frequency.
        matched = np.array([found[np.argmin(np.abs(found - e))]
                            for e in expected])
    return found, depths, matched


def sensitivity_budget():
    """dB improvement over the pucq4_01_tof.py run, printed up front.

    REPS and ROUNDS both average, so the total is their product -- they differ
    only in where the loop runs, not in the noise they remove.
    """
    tof_bw, tof_avgs = 552.96e6, 400.0
    this_bw = 1.0 / (RES_LENGTH * 1e-6)
    return (10 * np.log10(tof_bw / this_bw)
            + 10 * np.log10((REPS * ROUNDS) / tof_avgs))


def main():
    soc, soccfg = makeProxy()

    cfg = P.base_cfg()
    cfg["res_length"] = RES_LENGTH
    cfg["relax_delay"] = RELAX_DELAY

    folders = P.setup_data_folders(study, sub_study, substudy_txt_notes)
    plotdir = folders["studyDocumentationFolder"]
    datadir = folders["studyDataFolder"]
    logger = folders["logger"]

    gain_db = sensitivity_budget()
    freqs = np.arange(F_START, F_STOP + F_STEP / 2, F_STEP)

    print(f"\nSensitivity vs the TOF run: +{gain_db:.0f} dB")
    print(f"  ({RES_LENGTH} us integration, {REPS} reps x {ROUNDS} rounds)")
    print(f"PUCQ4 sweep: {len(freqs)} points, {F_START}-{F_STOP} MHz")
    if CONTROL_BAND:
        n_ctl = int((CONTROL_BAND[1] - CONTROL_BAND[0]) / F_STEP) + 1
        print(f"Control sweep: {n_ctl} points, "
              f"{CONTROL_BAND[0]}-{CONTROL_BAND[1]} MHz")
    # Time one real point rather than guessing. Per-point cost is dominated by
    # Pyro latency and program compilation, not by the measurement, so it is
    # not something to estimate from first principles.
    t0 = time.time()
    sweep(soc, soccfg, cfg, freqs[:1], "timing")
    per_point = time.time() - t0
    total = per_point * len(freqs)
    if CONTROL_BAND is not None:
        total += per_point * (int((CONTROL_BAND[1] - CONTROL_BAND[0]) / F_STEP) + 1)
    print(f"\nMeasured {per_point:.2f} s/point "
          f"(measurement itself is {REPS * (RES_LENGTH + RELAX_DELAY) / 1e3:.0f} ms)")
    print(f"Estimated total: {total / 60:.0f} min\n")
    if per_point > 2.0:
        print("  That is slow. If ROUNDS > 1, move the averaging into REPS --")
        print("  ROUNDS costs a network round trip each, REPS does not.\n")
    logger.info(f"res spec: {F_START}-{F_STOP} MHz step {F_STEP}, "
                f"res_length={RES_LENGTH}, reps={REPS}, rounds={ROUNDS}, gain={GAIN}, "
                f"budget=+{gain_db:.1f} dB vs TOF")

    amps = sweep(soc, soccfg, cfg, freqs, "PUCQ4 band")

    ctl_freqs = ctl_amps = None
    if CONTROL_BAND is not None:
        ctl_freqs = np.arange(CONTROL_BAND[0], CONTROL_BAND[1] + F_STEP / 2,
                              F_STEP)
        ctl_amps = sweep(soc, soccfg, cfg, ctl_freqs, "control band")

    # ------------------------------------------------------------------
    # Plot
    # ------------------------------------------------------------------
    nrow = 2 if ctl_amps is not None else 1
    fig, axes = plt.subplots(nrow, 1, figsize=(14, 5 * nrow), squeeze=False)

    found, depths, matched = find_dips(freqs, amps, P.RES_FREQS_VNA)

    ax = axes[0][0]
    ax.plot(freqs, amps, linewidth=1.0)
    for i, f in enumerate(P.RES_FREQS_VNA):
        ax.axvline(f, linestyle=":", color="grey", linewidth=1.0)
        ax.text(f, ax.get_ylim()[1], f" M{i + 1}", fontsize=8,
                va="top", color="grey")
    for f in found:
        ax.axvline(f, linestyle="--", color="tab:red", linewidth=0.9, alpha=0.7)
    contrast = (np.median(amps) - amps.min()) / (np.median(amps) + 1e-30)
    ax.set_title(f"PUCQ4 band, gain={GAIN}, {RES_LENGTH} us x {REPS} reps "
                 f"-- depth {contrast * 100:.1f}%  (dotted = VNA values)")
    ax.set_xlabel("Frequency (MHz)")
    ax.set_ylabel("|IQ| (a.u.)")

    if ctl_amps is not None:
        ax2 = axes[1][0]
        ax2.plot(ctl_freqs, ctl_amps, linewidth=1.0, color="tab:green")
        c2 = (np.median(ctl_amps) - ctl_amps.min()) / (np.median(ctl_amps) + 1e-30)
        ax2.set_title(f"CONTROL: previous chip's band -- depth {c2 * 100:.1f}%")
        ax2.set_xlabel("Frequency (MHz)")
        ax2.set_ylabel("|IQ| (a.u.)")

    fig.tight_layout()
    path = os.path.join(plotdir, "res_spec_wide.png")
    fig.savefig(path, dpi=200)
    fig.savefig(os.path.join(plotdir, "res_spec_wide.pdf"), dpi=200)
    plt.close(fig)

    np.savez(os.path.join(datadir, "res_spec_wide.npz"),
             freqs=freqs, amps=amps,
             ctl_freqs=ctl_freqs if ctl_freqs is not None else np.array([]),
             ctl_amps=ctl_amps if ctl_amps is not None else np.array([]),
             gain=GAIN, rounds=ROUNDS, res_length=RES_LENGTH)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 74)
    print(f"Found {len(found)} dips (expected {P.NUM_RES})")
    print(f"{'':4} {'VNA (MHz)':>12} {'measured':>12} {'shift':>10} {'depth':>9}")
    for i, vna in enumerate(P.RES_FREQS_VNA):
        if matched is not None and i < len(matched):
            m = matched[i]
            d = depths[int(np.argmin(np.abs(found - m)))]
            print(f"M{i + 1:<3} {vna:>12.1f} {m:>12.3f} "
                  f"{m - vna:>+10.3f} {d * 100:>8.1f}%")
        else:
            print(f"M{i + 1:<3} {vna:>12.1f} {'--':>12}")

    if len(found) == P.NUM_RES:
        print("\nPaste into system_config.py as res_freq_ge:")
        print("  [" + ", ".join(f"{f:.3f}" for f in np.sort(matched)) + "]")
        print("\nThese are still COARSE (F_STEP = "
              f"{F_STEP} MHz) and taken at high power. Next:")
        print("  1. Narrow F_START/F_STOP around each dip, F_STEP ~0.02")
        print("  2. Sweep GAIN down to find the punched-out frequencies")

    if ctl_amps is not None:
        print(f"Control band deepest feature: "
              f"{ctl_freqs[int(np.argmin(ctl_amps))]:.3f} MHz, "
              f"{(np.median(ctl_amps) - ctl_amps.min()) / np.median(ctl_amps) * 100:.1f}%")
    print()
    if contrast > 0.05:
        print("Resonators are visible. Narrow F_START/F_STOP around each dip")
        print("and drop F_STEP to ~0.02 MHz for the real frequencies.")
    else:
        print("Nothing clear in the PUCQ4 band yet. Before concluding the")
        print("chain cannot reach 9 GHz, spend the cheap dB:")
        print(f"  - REPS {REPS} -> {REPS * 4} buys 6 dB (cheap: it is a board-side loop)")
        print(f"  - RES_LENGTH {RES_LENGTH} -> 25 us buys "
              f"{10 * np.log10(25 / RES_LENGTH):.0f} dB (29.6 us is the max)")
        print("  - GAIN 0.9 -> 1.0 buys 1 dB")
        print("If the control band shows clean dips and PUCQ4 stays flat after")
        print("all that, the answer is the RF chain and it needs hardware.")
    print(f"\nPlots: {path}")
    print(f"Data:  {os.path.join(datadir, 'res_spec_wide.npz')}")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
