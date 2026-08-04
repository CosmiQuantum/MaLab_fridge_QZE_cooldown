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

# The punch-out shift is the LAMB SHIFT, not chi. The deck lists Lamb shifts of
# 4, 4, 4, 4, 10, 5 MHz for M1..M6, so the dressed resonance can sit up to
# ~10 MHz from the bare one (M5 is the big one). Do not narrow this below ~8 --
# chi (~250 kHz) is the qubit-STATE-dependent shift and a different quantity.
# Zoomed in. The wide sweeps established the resonance does not move over
# 42 dB of power, so there is no point scanning +/-6 MHz of empty spectrum --
# spend the points resolving the lineshape instead.
SPAN = 3.0             # [MHz]
STEP = 0.1             # [MHz] twice the resolution of the +/-6 MHz sweep

# The first sweep (0.02-1.0) showed the dip dead vertical from gain 1.0 down to
# ~0.1 -- that whole decade is wasted. It only began moving below 0.06, and the
# +0.4 MHz seen by gain 0.02 is the WRONG SIGN for punch-out: with the qubit
# ~3.4 GHz below the resonator and g/2pi ~120 MHz, the dressed resonance should
# sit g^2/Delta ~ -4.2 MHz BELOW the bare one (matching the deck's 4 MHz Lamb
# shift for M1). A small positive shift is more likely Kerr at high power. So
# the real transition is below 0.02 and this sweep goes after it.
#
# 0.005 is about the floor: even at MAX_TOTAL_AVERAGES it gives only ~half the
# SNR of the gain-0.02 row that worked. Rows below the noise are gated out and
# drawn as grey x, so pushing lower just wastes time rather than misleading you.
# 0.006 because the 0.005 row was gated out as below the noise floor last run.
# Top raised past 0.1 to re-cover the region where the mild Kerr-looking pull
# appeared, now with proper averaging behind it.
GAIN_START = 0.006
GAIN_STOP = 0.3
N_GAINS = 6            # log spaced

# Adaptive averaging. Signal scales with gain, noise with 1/sqrt(reps), so
# reps ~ 1/gain^2 holds SNR constant across the power sweep. Without this the
# low-power rows are pure noise -- which is exactly what happened on the first
# run: every row below gain ~0.04 was speckle, and the analysis then read the
# bottom row as the "low power" resonance and reported multi-MHz shifts that
# were not real.
REPS_AT_FULL_GAIN = 600
# The avg buffer holds 16384 accumulated samples and each rep writes one, so
# reps CANNOT exceed that -- the previous cap of 20000 would have overflowed it.
# Averaging beyond MAX_REPS is done with rounds instead (a few extra Pyro round
# trips, negligible next to seconds of measurement at these rep counts).
# The avg buffer is 16384 accumulated samples deep, so reps can go right up to
# it. Using the full depth halves the number of rounds -- and therefore the
# Pyro round trips -- at these averaging levels.
MAX_REPS = 16384
MAX_TOTAL_AVERAGES = 65536
# Longer readout rather than more averages. SNR goes as sqrt(N_avg * T_int),
# but each average also pays RELAX_DELAY of dead time -- so stretching the
# integration is cheaper than adding averages for the same gain: 25 us here is
# 1.58x the SNR of the 10 us run for 1.75x the time, where doubling the
# averages would give only 1.41x for 2.0x. Buffer max is 29.6 us.
RES_LENGTH = 25.0      # [us]
# No qubit is driven here and the resonator rings down in ~1 us (kappa <1 MHz),
# so this only needs to be a few ring-down times. It was 50 us, which at 16384
# averages was costing more than the measurement itself.
RELAX_DELAY = 10       # [us]

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


def averaging_for_gain(gain):
    """(reps, rounds) giving ~constant SNR across the power sweep.

    Signal scales with gain and noise with 1/sqrt(total averages), so the total
    goes as 1/gain^2. Split it so reps stays within the 16384-sample avg buffer
    and the remainder goes to rounds.
    """
    total = REPS_AT_FULL_GAIN / max(gain, 1e-6) ** 2
    total = int(np.clip(total, REPS_AT_FULL_GAIN, MAX_TOTAL_AVERAGES))
    reps = min(total, MAX_REPS)
    rounds = max(1, int(np.ceil(total / reps)))
    return reps, rounds


def dip_if_significant(freqs, amps, n_sigma=5.0):
    """(dip frequency, depth, snr) or (None, ...) if the row is just noise.

    Rows below the noise floor still have a minimum, and taking it blindly is
    what produced the bogus multi-MHz punch-out shifts. A row only counts if
    its dip is deep compared with the point-to-point scatter.
    """
    amps = np.asarray(amps, float)
    sigma = 1.4826 * np.median(np.abs(np.diff(amps))) / np.sqrt(2)
    base = np.median(amps)
    i = int(np.argmin(amps))
    depth = base - amps[i]
    if sigma <= 0:
        return None, 0.0, 0.0
    snr = depth / sigma
    if snr < n_sigma:
        return None, float(depth), float(snr)
    return float(freqs[i]), float(depth), float(snr)


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

    centers = np.asarray(P.RES_FREQS_MEASURED, dtype=float)
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
            for fi, df in enumerate(tqdm(offsets,
                                         desc=f"M{ri+1} g={g:.3f} "
                                              f"({averaging_for_gain(g)[0]}x"
                                              f"{averaging_for_gain(g)[1]})",
                                         leave=False)):
                c = dict(cfg)
                c["_freq"] = float(centers[ri] + df)
                c["_gain"] = float(g)
                reps, rounds = averaging_for_gain(g)
                prog = ResSpecProgram(soccfg, reps=reps,
                                      final_delay=RELAX_DELAY, cfg=c)
                block[gi, fi] = P.amp_from_iq(
                    prog.acquire(soc, rounds=rounds, progress=False))
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

        # Only rows whose dip beats the noise are usable.
        dips, usable = [], []
        for gi in range(len(gains)):
            f_d, depth, snr = dip_if_significant(freqs, block[gi])
            dips.append(f_d if f_d is not None else np.nan)
            usable.append(f_d is not None)
        dips = np.array(dips, dtype=float)
        usable = np.array(usable)

        im = ax.pcolormesh(freqs, gains, norm, shading="auto")
        ax.plot(dips[usable], gains[usable], "r.-", linewidth=1.4,
                markersize=6, label="resolved dip")
        if (~usable).any():
            ax.plot(np.full((~usable).sum(), np.nanmedian(dips[usable])
                            if usable.any() else freqs[len(freqs)//2]),
                    gains[~usable], "x", color="0.6", markersize=5,
                    label="below noise floor")
        ax.legend(fontsize=7, loc="upper left")
        ax.set_yscale("log")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Drive gain")
        fig.colorbar(im, ax=ax, label="|IQ| / row median")

        if usable.sum() >= 2:
            g_ok = gains[usable]
            d_ok = dips[usable]
            f_lo, f_hi = float(d_ok[0]), float(d_ok[-1])   # lowest usable power
            shift = f_lo - f_hi
            g_floor = float(g_ok.min())
            f_steep, slope = steepest_point(freqs, block[np.flatnonzero(usable)[0]])
        else:
            f_lo = f_hi = shift = np.nan
            g_floor = np.nan
            f_steep, slope = steepest_point(freqs, block[-1])

        summary.append((ri, f_lo, f_hi, shift, g_floor, f_steep))
        ax.set_title(f"M{ri+1}: usable down to gain {g_floor:.3f}  "
                     f"shift over usable range {shift:+.3f} MHz\n"
                     f"read at {f_steep:.3f} MHz")
        logger.info(f"M{ri+1}: lowest usable gain {g_floor:.4f}, "
                    f"dip {f_lo:.3f} -> {f_hi:.3f}, shift {shift:+.3f} MHz, "
                    f"steepest {f_steep:.3f}, "
                    f"{usable.sum()}/{len(gains)} rows usable")

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
