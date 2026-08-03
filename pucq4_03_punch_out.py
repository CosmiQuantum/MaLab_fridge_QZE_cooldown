"""
STEP 3 -- Punch out. Find the low-power (dressed) resonator frequencies.

At high drive power the resonator sits at its bare frequency; at low power it
shifts by the dispersive coupling to the qubit. The deck says all six PUCQ4
resonators punch out. You need the LOW-POWER frequency for real measurements --
the high-power VNA number will give you no qubit signal.

This repeats the step-2 sweep at a series of gains and plots frequency vs
power for each resonator.

    python pucq4_03_punch_out.py

Picks up the frequencies found by pucq4_02_res_spec_wide.py automatically if
that script has been run; otherwise falls back to the VNA values.

Runtime is N_GAINS times step 2, so start coarse.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm
from qick.asm_v2 import AveragerProgramV2

from socProxy import makeProxy
import pucq4_config as P
from pucq4_02_res_spec_wide import ResSpecProgram

# ---------------------------------------------------------------- knobs -----
SPAN = 6.0          # [MHz] narrower than step 2 -- you know roughly where they are
STEP = 0.10         # [MHz]
GAIN_START = 0.01
GAIN_STOP = 1.0
N_GAINS = 10        # log-spaced
ROUNDS = 100
RES_LENGTH = 2.0
RELAX_DELAY = 50
# -----------------------------------------------------------------------------


def load_centers():
    """Prefer step 2's measured frequencies over the VNA table."""
    path = os.path.join(P.OUTPUT_FOLDER, "02_res_spec_wide", "res_spec.npz")
    if os.path.exists(path):
        d = np.load(path)
        print(f"Using frequencies measured by step 2: {list(d['found'])}")
        return np.asarray(d["found"], dtype=float)
    print("No step-2 results found, falling back to the VNA table.")
    print("Run pucq4_02_res_spec_wide.py first for better centering.")
    return np.asarray(P.RES_FREQS_VNA, dtype=float)


def main():
    soc, soccfg = makeProxy()

    centers = load_centers()
    offsets = np.arange(-SPAN, SPAN + STEP / 2, STEP)
    gains = np.logspace(np.log10(GAIN_START), np.log10(GAIN_STOP), N_GAINS)

    cfg = P.base_cfg()
    cfg["res_length"] = RES_LENGTH
    cfg["relax_delay"] = RELAX_DELAY

    n = len(offsets) * len(gains)
    print(f"\n{len(gains)} gains x {len(offsets)} frequencies = {n} points.")
    print(f"Rough runtime: {n * 0.5 / 60:.0f}-{n * 1.5 / 60:.0f} min\n")

    amps = np.zeros((len(gains), len(offsets), P.NUM_RES))

    for gi, g in enumerate(gains):
        cfg["res_gain_ge"] = [float(g)] * P.NUM_RES
        for fi, df in enumerate(tqdm(offsets, desc=f"gain {g:.3f}", leave=False)):
            cfg["res_freq_ge"] = list(centers + df)
            prog = ResSpecProgram(soccfg, reps=1,
                                  final_delay=cfg["relax_delay"], cfg=cfg)
            iq_list = prog.acquire(soc, rounds=ROUNDS, progress=False)
            amps[gi, fi, :] = P.amps_from_iq(iq_list)

    outdir = P.make_output_folder("03_punch_out")
    np.savez(os.path.join(outdir, "punch_out.npz"),
             gains=gains, offsets=offsets, centers=centers, amps=amps)

    # ------------------------------------------------------------------
    # 2D maps: each row normalized so low- and high-power rows are comparable
    # ------------------------------------------------------------------
    fig, axes = plt.subplots(2, 3, figsize=(17, 9))
    dips = np.zeros((len(gains), P.NUM_RES))

    for i, ax in enumerate(axes.flat):
        block = amps[:, :, i]
        norm = block / (np.median(block, axis=1, keepdims=True) + 1e-30)
        freqs = centers[i] + offsets

        im = ax.pcolormesh(freqs, gains, norm, shading="auto")
        dips[:, i] = freqs[np.argmin(block, axis=1)]
        ax.plot(dips[:, i], gains, "r.-", linewidth=1.2, markersize=5)

        ax.set_yscale("log")
        ax.set_title(f"M{i + 1}")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("Mux gain (DAC units)")
        fig.colorbar(im, ax=ax, label="|IQ| / median")

    fig.suptitle("PUCQ4 punch out", fontsize=16)
    fig.tight_layout()
    path = os.path.join(outdir, "punch_out_2d.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------------
    # Centre shift vs power
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 6))
    for i in range(P.NUM_RES):
        ax.semilogx(gains, dips[:, i] - dips[-1, i], ".-", label=f"M{i + 1}")
    ax.axhline(0, color="k", linewidth=0.8)
    ax.set_xlabel("Mux gain (DAC units)")
    ax.set_ylabel("Frequency shift from highest power (MHz)")
    ax.set_title("Punch out: shift relative to the bare (high-power) frequency")
    ax.legend()
    fig.tight_layout()
    path2 = os.path.join(outdir, "punch_out_shift.png")
    fig.savefig(path2, dpi=150)
    plt.close(fig)

    print("\n" + "=" * 74)
    print("PUNCH OUT RESULTS")
    print("=" * 74)
    print(f"{'':4} {'low power':>12} {'high power':>12} {'shift':>10}")
    for i in range(P.NUM_RES):
        print(f"M{i + 1:<3} {dips[0, i]:>12.3f} {dips[-1, i]:>12.3f} "
              f"{dips[0, i] - dips[-1, i]:>+10.3f}")

    print("\nUse the LOW-POWER frequencies for everything downstream:")
    print("  np.array([" + ", ".join(f"{f:.3f}" for f in dips[0]) + "])")
    print("\nPick a working gain from the 2D plot: the lowest power at which")
    print("the dip is still clearly visible, ideally just below where the")
    print("frequency stops moving.")
    print(f"\nSaved: {path}")
    print(f"       {path2}")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
