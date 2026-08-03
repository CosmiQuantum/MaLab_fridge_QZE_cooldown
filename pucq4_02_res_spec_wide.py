"""
STEP 2 -- Wide resonator spectroscopy. Find all six resonators.

This is the script that answers "can I see the frequencies on QICK". It works
at ANY flux bias: the student's data shows the resonators move only ~7 MHz
across the full +/-10 mA range, so a +/-15 MHz window catches them wherever the
DC lines happen to be sitting.

All six mux tones are stepped together by the same offset, so one sweep
produces all six traces at once.

    python pucq4_02_res_spec_wide.py

Do a COARSE pass first, then narrow SPAN and STEP for a fine pass.

Why this does not reuse ResonanceSpectroscopy from section_002_res_spec_ge_mux:
that class computes amplitude as iq_list[0][0][0] + 1j*iq_list[0][0][1], which
reads readout channel 0's I against channel 0's Q only in the single-tone case
-- for a 6-tone mux it mixes tones together and returns one meaningless trace.
The mux conversion of that file was never finished on this branch.
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

# ---------------------------------------------------------------- knobs -----
SPAN = 15.0        # [MHz] sweep +/- this around each VNA frequency
STEP = 0.10        # [MHz] coarse. Drop to 0.02 for a fine pass.
ROUNDS = 100       # averages per frequency point. Raise if traces are noisy.
GAIN = 0.3         # mux gain per tone. See pucq4_03_punch_out.py.
RES_LENGTH = 2.0   # [us]
RELAX_DELAY = 50   # [us]
# -----------------------------------------------------------------------------


class ResSpecProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        P.declare_res_mux(self, cfg)
        self.add_pulse(ch=cfg["res_ch"], name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"])

    def _body(self, cfg):
        self.trigger(ros=cfg["ro_ch"], pins=[0], t=cfg["trig_time"])
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)


def main():
    soc, soccfg = makeProxy()

    cfg = P.base_cfg()
    cfg["res_gain_ge"] = [GAIN] * P.NUM_RES
    cfg["res_length"] = RES_LENGTH
    cfg["relax_delay"] = RELAX_DELAY

    centers = np.asarray(P.RES_FREQS_VNA, dtype=float)
    offsets = np.arange(-SPAN, SPAN + STEP / 2, STEP)

    print(f"\nSweeping {len(offsets)} points, +/-{SPAN} MHz at {STEP} MHz steps.")
    print(f"Centers (MHz): {list(centers)}")
    print(f"Rough runtime: {len(offsets) * 0.5 / 60:.1f}-"
          f"{len(offsets) * 1.5 / 60:.1f} min\n")

    amps = np.zeros((len(offsets), P.NUM_RES))
    for i, df in enumerate(tqdm(offsets, desc="res spec")):
        cfg["res_freq_ge"] = list(centers + df)
        prog = ResSpecProgram(soccfg, reps=1,
                              final_delay=cfg["relax_delay"], cfg=cfg)
        iq_list = prog.acquire(soc, rounds=ROUNDS, progress=False)
        amps[i, :] = P.amps_from_iq(iq_list)

    # ------------------------------------------------------------------
    # Find each dip
    # ------------------------------------------------------------------
    outdir = P.make_output_folder("02_res_spec_wide")
    found, warnings = [], []

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    for i, ax in enumerate(axes.flat):
        trace = amps[:, i]
        freqs = centers[i] + offsets
        idx = int(np.argmin(trace))
        f_res = float(freqs[idx])
        found.append(f_res)

        ax.plot(freqs, trace, linewidth=1.4)
        ax.axvline(f_res, linestyle="--", color="orange",
                   label=f"{f_res:.3f} MHz")
        ax.axvline(centers[i], linestyle=":", color="grey",
                   label=f"VNA {centers[i]:.1f}")

        depth = (np.median(trace) - trace[idx]) / (np.median(trace) + 1e-30)
        edge = idx < 3 or idx > len(offsets) - 4

        if edge:
            warnings.append(f"M{i + 1}: minimum sits at the edge of the scan -- "
                            f"the real resonance is probably outside +/-{SPAN} MHz")
        if depth < 0.02:
            warnings.append(f"M{i + 1}: dip is only {depth * 100:.1f}% deep -- "
                            f"may be noise, not a resonator")

        flag = "  <-- CHECK" if (edge or depth < 0.02) else ""
        ax.set_title(f"M{i + 1}  {f_res:.3f} MHz  "
                     f"({f_res - centers[i]:+.2f} from VNA){flag}")
        ax.set_xlabel("Frequency (MHz)")
        ax.set_ylabel("|IQ| (a.u.)")
        ax.legend(fontsize=8)

    fig.suptitle(f"PUCQ4 resonator spectroscopy, gain={GAIN}, "
                 f"{ROUNDS} rounds", fontsize=16)
    fig.tight_layout()
    path = os.path.join(outdir, f"res_spec_span{SPAN:g}_step{STEP:g}.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    np.savez(os.path.join(outdir, "res_spec.npz"),
             offsets=offsets, centers=centers, amps=amps,
             found=np.array(found), gain=GAIN, rounds=ROUNDS)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    found = np.array(found)
    shifts = found - centers

    print("\n" + "=" * 74)
    print("RESULTS")
    print("=" * 74)
    print(f"{'':4} {'VNA (MHz)':>12} {'QICK (MHz)':>12} {'shift (MHz)':>13}")
    for i in range(P.NUM_RES):
        print(f"M{i + 1:<3} {centers[i]:>12.1f} {found[i]:>12.3f} "
              f"{shifts[i]:>+13.3f}")

    print(f"\nPaste into pucq4_config.py as RES_FREQS_VNA:")
    print("  np.array([" + ", ".join(f"{f:.3f}" for f in found) + "])")

    if warnings:
        print("\nWARNINGS:")
        for w in warnings:
            print(f"  - {w}")

    # The resonators are a crude flux meter: res 6 spans 9055.5 MHz at -10 mA
    # to 9063.0 MHz at +10 mA, so where it sits tells you roughly where the
    # DC bias is even before you have any flux control wired up.
    print("\nFLUX HINT")
    print("-" * 74)
    print("  From the student's scan, M6 runs 9055.5 MHz (-10 mA) to")
    print(f"  9063.0 MHz (+10 mA), and the table value is 9059 MHz.")
    print(f"  You measured M6 at {found[5]:.3f} MHz.")
    if found[5] < 9056.5:
        print("  -> that is near the bottom of its range: bias is strongly negative.")
    elif found[5] > 9062.0:
        print("  -> that is near the top of its range: bias is strongly positive.")
    else:
        print("  -> mid-range, consistent with a bias somewhere near the table's.")
    print("  This is a sanity check, not a calibration. Confirm the actual")
    print("  current setting with whoever ran the VNA.")

    print(f"\nSaved: {path}")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()
