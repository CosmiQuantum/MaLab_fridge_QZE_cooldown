"""
STEP 1 -- Time of flight.

Fires the readout comb and captures the raw ADC trace. Two purposes:

  1. It is the honest "is there any signal at 9 GHz?" test. If the pulse never
     shows up on any channel, stop -- the problem is the RF chain or the analog
     bandwidth of the ADC, and no amount of config fiddling will fix it.
  2. It measures the cable delay, which becomes TRIG_TIME in pucq4_config.py.
     Every later measurement depends on that number being right.

    python pucq4_01_tof.py

Read the plot: find where the pulse starts on each channel and set
pucq4_config.TRIG_TIME to that time in microseconds. The script prints its own
estimate, but trust your eyes over the estimate.
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
CAPTURE_LENGTH = 1.5    # [us] capture window, must be longer than the delay
SOFT_AVGS = 200         # raise if the trace is noisy
GAIN = 0.5              # crank this up if you see nothing
# -----------------------------------------------------------------------------


class TOFProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        P.declare_res_mux(self, cfg)
        self.add_pulse(ch=cfg["res_ch"], name="res_pulse",
                       style="const",
                       length=PULSE_LENGTH,
                       mask=cfg["list_of_all_qubits"])

    def _body(self, cfg):
        # Trigger at t=0 so the captured trace starts before the pulse arrives;
        # whatever offset we see IS the time of flight.
        self.trigger(ros=cfg["ro_ch"], pins=[0], t=0)
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)


def main():
    soc, soccfg = makeProxy()

    cfg = P.base_cfg()
    cfg["res_gain_ge"] = [GAIN] * P.NUM_RES
    cfg["res_length"] = CAPTURE_LENGTH   # declare_readout uses this

    prog = TOFProgram(soccfg, reps=1, final_delay=1.0, cfg=cfg)
    iq_list = prog.acquire_decimated(soc, soft_avgs=SOFT_AVGS)

    outdir = P.make_output_folder("01_tof")

    fig, axes = plt.subplots(3, 2, figsize=(13, 10), sharex=True)
    estimates = []

    for i, ax in enumerate(axes.flat):
        trace = np.asarray(iq_list[i], dtype=float)
        I, Q = trace[:, 0], trace[:, 1]
        t = soccfg.cycles2us(np.arange(len(I)), ro_ch=cfg["ro_ch"][i])
        mag = np.abs(I + 1j * Q)

        ax.plot(t, I, linewidth=1.0, label="I")
        ax.plot(t, Q, linewidth=1.0, label="Q")
        ax.plot(t, mag, linewidth=1.5, color="k", label="|IQ|")

        # Crude leading-edge estimate: first sample above halfway to the peak.
        floor = np.median(mag[: max(4, len(mag) // 10)])
        peak = mag.max()
        if peak > 2 * floor:
            above = np.flatnonzero(mag > 0.5 * (peak + floor))
            if len(above):
                tof = float(t[above[0]])
                estimates.append(tof)
                ax.axvline(tof, linestyle="--", color="orange")
                ax.set_title(f"RO ch {cfg['ro_ch'][i]}  (M{i + 1})  "
                             f"edge ~{tof:.3f} us")
            else:
                ax.set_title(f"RO ch {cfg['ro_ch'][i]}  (M{i + 1})  no edge")
        else:
            ax.set_title(f"RO ch {cfg['ro_ch'][i]}  (M{i + 1})  NO SIGNAL")

        ax.set_ylabel("ADC units")
        ax.legend(fontsize=8, loc="upper right")

    for ax in axes[-1]:
        ax.set_xlabel("Time (us)")

    fig.suptitle("PUCQ4 time of flight", fontsize=16)
    fig.tight_layout()
    path = os.path.join(outdir, "tof.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)

    np.savez(os.path.join(outdir, "tof.npz"),
             traces=np.asarray(iq_list, dtype=float), cfg=str(cfg))

    print("\n" + "=" * 70)
    if estimates:
        print(f"Leading edge found on {len(estimates)}/{P.NUM_RES} channels.")
        print(f"Estimated time of flight: {np.median(estimates):.3f} us")
        print(f"\n  -> set TRIG_TIME = {np.median(estimates):.3f} "
              f"in pucq4_config.py")
        print("\nLook at the plot before trusting that number.")
    else:
        print("NO PULSE SEEN ON ANY CHANNEL.")
        print("\nBefore blaming the config, check in this order:")
        print("  1. Is the readout line actually connected and the TWPA/HEMT on?")
        print("  2. Raise GAIN at the top of this script toward 1.0.")
        print("  3. Re-run pucq4_00_check_board.py and confirm NQZ_RES.")
        print("  4. 9 GHz may simply be past the analog bandwidth of the ADC")
        print("     input. If a 7 GHz tone shows up and 9 GHz does not, that")
        print("     is your answer and you need an upconverter in the line.")
    print(f"\nSaved: {path}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
