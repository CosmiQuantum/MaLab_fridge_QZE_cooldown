"""
STEP 1 -- Time of flight, one resonator at a time.

Written for the NON-MUX firmware: two axis_signal_gen_v6 generators, two
axis_dyn_readout_v1 readouts. Only one resonator tone can be driven at a time,
so this loops over the six PUCQ4 frequencies and captures a raw trace at each.

Two purposes:

  1. The honest "does 9 GHz come back at all?" test. PUCQ4's resonators sit in
     ADC Nyquist zone 5 and ~11 dB further down the DAC's sin(x)/x curve than
     the previous 7.2 GHz chip. If nothing shows up here, the problem is the RF
     chain -- HEMT/TWPA/circulators are typically 4-8 GHz parts -- and no
     config change will fix it.
  2. It measures the cable delay, which becomes trig_time in system_config.py.

    python pucq4_01_tof.py

The COMPARE_FREQ knob below is the useful diagnostic: it also captures at the
old chip's 7200 MHz. If 7200 comes back and 9000 does not, that is a clean
answer -- the board is fine and the signal is dying in the analog chain.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qick.asm_v2 import AveragerProgramV2

from socProxy import makeProxy
import pucq4_config as P

# ---------------------------------------------------------------- knobs -----
PULSE_LENGTH = 0.5      # [us] short pulse so the leading edge is sharp
CAPTURE_LENGTH = 1.5    # [us] capture window, must exceed the cable delay
SOFT_AVGS = 400
GAIN = 0.9              # near max -- PUCQ4 needs it, see 00_check_board
COMPARE_FREQ = 7200.0   # [MHz] old chip's band, as a control. None to skip.
# -----------------------------------------------------------------------------


class TOFProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        P.declare_res_single(self, cfg, cfg["_freq"], cfg["_gain"],
                             PULSE_LENGTH)

    def _body(self, cfg):
        # Trigger at t=0 so capture starts before the pulse returns; whatever
        # offset appears IS the time of flight.
        self.trigger(ros=[cfg["ro_ch"]], pins=[0], t=0)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)


def capture(soc, soccfg, cfg, freq, gain):
    cfg = dict(cfg)
    cfg["_freq"] = float(freq)
    cfg["_gain"] = float(gain)
    prog = TOFProgram(soccfg, reps=1, final_delay=1.0, cfg=cfg)
    # qick 0.2.363 spells the averaging argument 'rounds', not 'soft_avgs' --
    # same as section_001_time_of_flight.py.
    iq = prog.acquire_decimated(soc, rounds=SOFT_AVGS)
    trace = np.asarray(iq[0], dtype=float)
    # get_time_axis already accounts for decimation; safer than cycles2us here.
    t = prog.get_time_axis(ro_index=0)[: trace.shape[0]]
    return t, trace[:, 0], trace[:, 1]


def edge_estimate(t, mag):
    """First sample above halfway to the peak, or None if there is no pulse."""
    floor = np.median(mag[: max(4, len(mag) // 10)])
    peak = mag.max()
    if peak < 2 * floor:
        return None, floor, peak
    above = np.flatnonzero(mag > 0.5 * (peak + floor))
    if not len(above):
        return None, floor, peak
    return float(t[above[0]]), floor, peak


def main():
    soc, soccfg = makeProxy()

    cfg = P.base_cfg()
    cfg["res_length"] = CAPTURE_LENGTH   # declare_readout capture window

    targets = [(f"M{i + 1}", f) for i, f in enumerate(P.RES_FREQS_VNA)]
    if COMPARE_FREQ is not None:
        targets.append(("control", float(COMPARE_FREQ)))

    outdir = P.make_output_folder("01_tof")
    results, estimates = [], []

    ncol = 2
    nrow = int(np.ceil(len(targets) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(13, 3 * nrow), sharex=True)

    for ax, (label, freq) in zip(np.atleast_1d(axes).flat, targets):
        t, I, Q = capture(soc, soccfg, cfg, freq, GAIN)
        mag = np.abs(I + 1j * Q)
        tof, floor, peak = edge_estimate(t, mag)
        snr = peak / (floor + 1e-30)
        results.append((label, freq, tof, snr))

        ax.plot(t, I, linewidth=0.9, label="I")
        ax.plot(t, Q, linewidth=0.9, label="Q")
        ax.plot(t, mag, linewidth=1.5, color="k", label="|IQ|")
        if tof is not None:
            estimates.append(tof)
            ax.axvline(tof, linestyle="--", color="orange")
            ax.set_title(f"{label}  {freq:.0f} MHz   edge ~{tof:.3f} us   "
                         f"peak/floor {snr:.1f}")
        else:
            ax.set_title(f"{label}  {freq:.0f} MHz   NO SIGNAL "
                         f"(peak/floor {snr:.2f})")
        ax.set_ylabel("ADC units")
        ax.legend(fontsize=7, loc="upper right")

    for ax in np.atleast_1d(axes).flat[-ncol:]:
        ax.set_xlabel("Time (us)")

    fig.suptitle(f"PUCQ4 time of flight, gain={GAIN}", fontsize=15)
    fig.tight_layout()
    path = os.path.join(outdir, "tof.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    print("\n" + "=" * 70)
    print(f"{'':10}{'freq (MHz)':>12}{'TOF (us)':>12}{'peak/floor':>13}")
    for label, freq, tof, snr in results:
        tof_s = f"{tof:.3f}" if tof is not None else "--"
        print(f"{label:<10}{freq:>12.0f}{tof_s:>12}{snr:>13.2f}")

    pucq4 = [r for r in results if r[0] != "control"]
    control = [r for r in results if r[0] == "control"]
    got_pucq4 = any(r[2] is not None for r in pucq4)
    got_control = any(r[2] is not None for r in control)

    print()
    if got_pucq4:
        print(f"Signal seen at 9 GHz. Estimated time of flight: "
              f"{np.median(estimates):.3f} us")
        print(f"  -> set trig_time = {np.median(estimates):.3f} in "
              f"system_config.py (currently 0.4)")
        print("Look at the plot before trusting that number.")
    elif got_control and not got_pucq4:
        print("The 7200 MHz control came back but 9 GHz did not.")
        print("That is a clean result: the board and RF path work, and the")
        print("PUCQ4 band is dying somewhere analog. Check the HEMT, TWPA and")
        print("circulator bands -- 4-8 GHz parts do not pass 9 GHz. This is")
        print("not fixable in software.")
    else:
        print("NO PULSE ANYWHERE, including the control tone.")
        print("Check in this order:")
        print("  1. Is the readout line connected, HEMT/TWPA powered?")
        print("  2. Raise GAIN (already near max) and SOFT_AVGS.")
        print("  3. Re-run pucq4_00_check_board.py and confirm nqz_res = 2.")
    print(f"\nSaved: {path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
