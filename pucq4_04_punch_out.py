"""
STEP 4 -- Punch out. Find the readout power and frequency that can see a qubit.

WHY THIS IS THE BLOCKER
-----------------------
pucq4_02 measured the resonators at gain 0.9. The deck says all six PUCQ4
resonators punch out, which means 0.9 is very likely ABOVE the punch-out power
-- so those frequencies are the BARE resonances, with the qubit decoupled.

You cannot do dispersive readout through a punched-out resonator at any drive
power. That, not the qubit drive, is the most likely reason pucq4_03 found
nothing: it parked the readout at a bare frequency, at saturating power, at the
bottom of the dip where the least light comes back.

This script sweeps drive power against frequency for each resonator and finds:
  - the LOW-POWER (dressed) frequency, which is what qubit experiments need
  - the punch-out power, so you can sit just below it
  - the steepest point on the low-power resonance, which is where a dispersive
    shift produces the biggest change in |IQ| -- NOT the dip minimum

    python pucq4_04_punch_out.py

The tension on this chip: dispersive readout wants LOW power, but you are 30 dB
down at 9 GHz. Expect to trade integration time for it.
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
RESONATORS = [0, 1, 2, 3, 4, 5]

SPAN = 4.0             # [MHz] +/- around each measured resonator frequency
STEP = 0.05            # [MHz]

GAIN_START = 0.02      # punch-out usually happens well below 1.0
GAIN_STOP = 1.0
N_GAINS = 12           # log spaced

REPS = 600             # hardware averaging -- one round trip per point
ROUNDS = 1
RES_LENGTH = 10.0      # [us]
RELAX_DELAY = 50       # [us]

study = "pucq4_first_light"
sub_study = "punch_out"
substudy_txt_notes = (
    "PUCQ4 punch out: drive power vs frequency for all six resonators. Run "
    "after the qubit hunt found nothing, on the suspicion that gain 0.9 was "
    "above punch-out and pucq4_02's frequencies are bare, not dressed.")
# -----------------------------------------------------------------------------


class ResSpecProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        P.declare_res_single(self, cfg, cfg["_freq"], cfg["_gain"],
                             cfg["res_length"])

    def _body(self, cfg):
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=cfg["trig_time"])
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)


def steepest_point(freqs, amps):
    """Frequency of maximum |d|IQ|/df| -- the best readout point.

    A dispersive shift moves the resonance sideways, so the change in measured
    amplitude is (slope x shift). At the dip minimum the slope is zero by
    definition, which is the worst possible place to sit.
    """
    d = np.abs(np.gradient(amps, freqs))
    return float(freqs[int(np.argmax(d))]), float(np.max(d))


def main():
    soc, soccfg = makeProxy()

    centers = np.asarray(P.RES_FREQS_VNA, dtype=float)
    offsets = np.arange(-SPAN, SPAN + STEP / 2, STEP)
    gains = np.logspace(np.log10(GAIN_START), np.log10(GAIN_STOP), N_GAINS)

    folders = P.setup_data_folders(study, sub_study, substudy_txt_notes)
    plotdir = folders["studyDocumentationFolder"]
    datadir = folders["studyDataFolder"]
    logger = folders["logger"]

    n = len(offsets) * len(gains) * len(RESONATORS)
    print(f"\n{len(gains)} gains x {len(offsets)} freqs x "
          f"{len(RESONATORS)} resonators = {n} points")
    print(f"Gains: {', '.join(f'{g:.3f}' for g in gains)}\n")

    cfg = P.base_cfg()
    cfg["res_length"] = RES_LENGTH
    cfg["relax_delay"] = RELAX_DELAY

    all_amps = {}
    for ri in RESONATORS:
        t0 = time.time()
        block = np.zeros((len(gains), len(offsets)))
        for gi, g in enumerate(gains):
            for fi, df in enumerate(tqdm(offsets, desc=f"M{ri+1} g={g:.3f}",
                                         leave=False)):
                c = dict(cfg)
                c["_freq"] = float(centers[ri] + df)
                c["_gain"] = float(g)
                prog = ResSpecProgram(soccfg, reps=REPS,
                                      final_delay=RELAX_DELAY, cfg=c)
                block[gi, fi] = P.amp_from_iq(
                    prog.acquire(soc, rounds=ROUNDS, progress=False))
        all_amps[ri] = block
        print(f"  M{ri+1} done ({time.time()-t0:.0f} s)")

    # ------------------------------------------------------------------
    # Analyse: dip frequency vs power, and where punch-out happens
    # ------------------------------------------------------------------
    summary = []
    ncol = 2
    nrow = int(np.ceil(len(RESONATORS) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(15, 4.0 * nrow),
                             squeeze=False)

    for ax, ri in zip(axes.flat, RESONATORS):
        block = all_amps[ri]
        freqs = centers[ri] + offsets
        # Normalise each power row so low and high power are comparable.
        norm = block / (np.median(block, axis=1, keepdims=True) + 1e-30)
        dips = freqs[np.argmin(block, axis=1)]

        im = ax.pcolormesh(freqs, gains, norm, shading="auto")
        ax.plot(dips, gains, "r.-", linewidth=1.2, markersize=5)
        ax.set_yscale("log")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Drive gain")
        fig.colorbar(im, ax=ax, label="|IQ| / row median")

        f_lo, f_hi = float(dips[0]), float(dips[-1])
        shift = f_lo - f_hi
        # Punch-out power: the gain where the dip has moved halfway.
        half = f_hi + 0.5 * shift
        pidx = int(np.argmin(np.abs(dips - half)))
        g_punch = float(gains[pidx])

        # Best readout point on the LOWEST power trace.
        f_steep, slope = steepest_point(freqs, block[0])
        summary.append((ri, f_lo, f_hi, shift, g_punch, f_steep))

        ax.set_title(f"M{ri+1}: low-P {f_lo:.3f}  high-P {f_hi:.3f}  "
                     f"shift {shift:+.3f} MHz\npunch-out near gain "
                     f"{g_punch:.3f}")
        logger.info(f"M{ri+1}: dressed {f_lo:.3f}, bare {f_hi:.3f}, "
                    f"shift {shift:+.3f} MHz, punch-out gain {g_punch:.3f}, "
                    f"steepest {f_steep:.3f}")

    for ax in axes.flat[len(RESONATORS):]:
        ax.set_visible(False)

    fig.suptitle("PUCQ4 punch out", fontsize=15)
    fig.tight_layout()
    path = os.path.join(plotdir, "punch_out.png")
    fig.savefig(path, dpi=200)
    fig.savefig(os.path.join(plotdir, "punch_out.pdf"), dpi=200)
    plt.close(fig)

    np.savez(os.path.join(datadir, "punch_out.npz"),
             gains=gains, offsets=offsets, centers=centers,
             **{f"amps_M{ri+1}": all_amps[ri] for ri in RESONATORS})

    # ------------------------------------------------------------------
    print("\n" + "=" * 78)
    print(f"{'':4} {'dressed':>10} {'bare':>10} {'shift':>9} "
          f"{'punch gain':>11} {'read here':>11}")
    for ri, f_lo, f_hi, shift, g_punch, f_steep in summary:
        print(f"M{ri+1:<3} {f_lo:>10.3f} {f_hi:>10.3f} {shift:>+9.3f} "
              f"{g_punch:>11.3f} {f_steep:>11.3f}")

    moved = [s for s in summary if abs(s[3]) > 0.05]
    print()
    if moved:
        print(f"{len(moved)}/{len(summary)} resonators shifted with power, so "
              f"they DO punch out.")
        print("\nFor qubit experiments use:")
        print("  res_freq_ge = [" + ", ".join(
            f"{s[5]:.3f}" for s in summary) + "]")
        print("    (the STEEPEST point on the low-power resonance, not the dip")
        print("     minimum -- a dispersive shift changes |IQ| by slope*shift,")
        print("     and the slope is zero at the bottom of the dip)")
        print("  res_gain  = about " + ", ".join(
            f"{s[4]/3:.3f}" for s in summary))
        print("    (roughly 1/3 of the punch-out gain, to stay well below it)")
        print("\nThen re-run pucq4_03_qubit_hunt.py with those values.")
    else:
        print("No resonator moved appreciably with power over this range.")
        print("Either the punch-out power is outside GAIN_START..GAIN_STOP,")
        print("or these resonators are not qubit-coupled at this flux bias.")
        print("Try GAIN_START = 0.002 before concluding anything.")
    print(f"\nPlots: {path}")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
