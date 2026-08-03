"""
STEP 4 -- Wide qubit spectroscopy.

Sweeps a probe tone and watches all six readout resonators at once. Two things
come out of this:

  1. The qubit frequency at the current flux bias.
  2. WHICH RESONATOR THAT QUBIT IS COUPLED TO -- because all six readout tones
     stay on, and only the resonator coupled to the qubit being driven will
     respond. That is a direct measurement of the indexing your collaborator is
     currently inferring from avoided crossings.

    python pucq4_04_qubit_spec_wide.py

UNLIKE STEPS 1-3, THIS ONE CARES ABOUT FLUX. The qubits move GHz with bias
(Q4 runs 5.85-7.11 GHz over +/-10 mA), so the VNA frequencies are only a
starting guess. If a wide scan finds nothing, the bias is the first suspect,
not the code.

This script refuses to run on readout-only firmware, which has no drive
generator at all.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from qick.asm_v2 import AveragerProgramV2, QickSweep1D

from socProxy import makeProxy
import pucq4_config as P

# ---------------------------------------------------------------- knobs -----
QUBITS_TO_SCAN = [4]     # indices into P.QUBIT_FREQS_VNA (0-5). Start with one.
SPAN = 150.0             # [MHz] +/- around the VNA guess. Widen to 400+ if the
                         # flux bias is unknown.
STEPS = 400              # frequency points
QUBIT_GAIN = 0.05        # probe gain. Lines were 20-40 MHz wide, so you can
                         # afford to drive fairly hard on a first pass.
QUBIT_LENGTH = 20.0      # [us] probe pulse
REPS = 500
ROUNDS = 4
RELAX_DELAY = 200        # [us]
RES_GAIN = 0.3           # use the value you picked from punch out
# -----------------------------------------------------------------------------


class QubitSpecProgram(AveragerProgramV2):
    def _initialize(self, cfg):
        P.declare_res_mux(self, cfg)
        self.add_pulse(ch=cfg["res_ch"], name="res_pulse",
                       style="const",
                       length=cfg["res_length"],
                       mask=cfg["list_of_all_qubits"])

        self.declare_gen(ch=cfg["qubit_ch"], nqz=cfg["nqz_qubit"])
        self.add_loop("freqloop", cfg["steps"])
        self.add_pulse(ch=cfg["qubit_ch"], name="qubit_pulse",
                       ro_ch=cfg["ro_ch"][0],
                       style="const",
                       length=cfg["qubit_length_ge"],
                       freq=cfg["qubit_freq_ge"],
                       phase=0,
                       gain=cfg["qubit_gain_ge"])

    def _body(self, cfg):
        self.pulse(ch=cfg["qubit_ch"], name="qubit_pulse", t=0)
        self.delay_auto(t=0.01, tag="waiting")
        self.pulse(ch=cfg["res_ch"], name="res_pulse", t=0)
        self.trigger(ros=cfg["ro_ch"], pins=[0], t=cfg["trig_time"])


def swept_amps(iq_list, steps):
    """Amplitude vs sweep point, per readout channel -> (steps, n_ch)."""
    out = []
    for ch_data in iq_list:
        a = np.asarray(ch_data, dtype=float).reshape(-1, 2)
        a = a[-steps:] if a.shape[0] >= steps else a
        out.append(np.abs(a[:, 0] + 1j * a[:, 1]))
    return np.stack(out, axis=1)


def check_drive_generator(soccfg):
    drive = [i for i, g in enumerate(soccfg["gens"])
             if "mux" not in str(g.get("type", "")).lower()]
    if not drive:
        raise SystemExit(
            "\nThis firmware has NO drive generator -- it is readout-only.\n"
            "Qubit spectroscopy, Rabi, T1 and the round robin benchmark cannot\n"
            "run until a firmware with a full-speed generator is loaded.\n"
            "Steps 0-3 (check_board, tof, res_spec, punch_out) still work.\n"
            "Run pucq4_00_check_board.py for the full picture.\n")
    if P.QUBIT_CH not in drive:
        raise SystemExit(
            f"\npucq4_config.QUBIT_CH = {P.QUBIT_CH} is not a drive generator.\n"
            f"Drive generators on this firmware: {drive}\n")
    return drive


def main():
    soc, soccfg = makeProxy()
    check_drive_generator(soccfg)

    # Get the qubit Nyquist zone from the real sample rate rather than trusting
    # the default -- 5.4 and 7.0 GHz could land in different zones.
    fs_q = soccfg["gens"][P.QUBIT_CH]["fs"]
    outdir = P.make_output_folder("04_qubit_spec_wide")

    for qi in QUBITS_TO_SCAN:
        f_guess = float(P.QUBIT_FREQS_VNA[qi])
        start, stop = f_guess - SPAN, f_guess + SPAN

        nqz_lo = P.nyquist_zone(start, fs_q)
        nqz_hi = P.nyquist_zone(stop, fs_q)
        if nqz_lo != nqz_hi:
            print(f"\n*** Q{qi + 1}: the scan window {start:.0f}-{stop:.0f} MHz "
                  f"crosses a Nyquist boundary (zones {nqz_lo}/{nqz_hi}).")
            print("    Narrow SPAN or run the two halves separately.\n")
            continue

        cfg = P.base_cfg()
        cfg["nqz_qubit"] = nqz_lo
        cfg["res_gain_ge"] = [RES_GAIN] * P.NUM_RES
        cfg["relax_delay"] = RELAX_DELAY
        cfg["steps"] = STEPS
        cfg["qubit_length_ge"] = QUBIT_LENGTH
        cfg["qubit_gain_ge"] = QUBIT_GAIN
        cfg["qubit_freq_ge"] = QickSweep1D("freqloop", start, stop)

        print(f"\nQ{qi + 1}: sweeping {start:.1f} - {stop:.1f} MHz "
              f"({STEPS} points, nqz {nqz_lo})")
        print(f"  VNA guess was {f_guess:.1f} MHz -- only valid at the bias the "
              f"VNA data was taken at.")

        prog = QubitSpecProgram(soccfg, reps=REPS,
                                final_delay=RELAX_DELAY, cfg=cfg)
        iq_list = prog.acquire(soc, rounds=ROUNDS, progress=True)

        amps = swept_amps(iq_list, STEPS)
        freqs = np.linspace(start, stop, amps.shape[0])

        # Which resonator actually responded?
        contrast = []
        for i in range(P.NUM_RES):
            t = amps[:, i]
            contrast.append((t.max() - t.min()) / (np.median(t) + 1e-30))
        contrast = np.array(contrast)
        best = int(np.argmax(contrast))

        fig, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True)
        for i, ax in enumerate(axes.flat):
            ax.plot(freqs, amps[:, i], linewidth=1.2)
            marker = "  <-- RESPONDS" if i == best and contrast[i] > 0.05 else ""
            ax.set_title(f"M{i + 1}   contrast {contrast[i] * 100:.1f}%{marker}")
            ax.set_xlabel("Probe frequency (MHz)")
            ax.set_ylabel("|IQ| (a.u.)")
        fig.suptitle(f"PUCQ4 qubit spec, driving near Q{qi + 1} "
                     f"({f_guess:.0f} MHz), gain={QUBIT_GAIN}", fontsize=15)
        fig.tight_layout()
        path = os.path.join(outdir, f"qspec_Q{qi + 1}_span{SPAN:g}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)

        np.savez(os.path.join(outdir, f"qspec_Q{qi + 1}.npz"),
                 freqs=freqs, amps=amps, contrast=contrast, guess=f_guess)

        print("\n" + "=" * 70)
        if contrast[best] > 0.05:
            trace = amps[:, best]
            f_q = float(freqs[np.argmax(np.abs(trace - np.median(trace)))])
            print(f"Q{qi + 1} responds on resonator M{best + 1} "
                  f"(contrast {contrast[best] * 100:.1f}%)")
            print(f"Qubit frequency: {f_q:.2f} MHz "
                  f"({f_q - f_guess:+.1f} from the VNA value)")
            print(f"\n  -> Q{qi + 1} is coupled to M{best + 1}. That is your "
                  f"indexing, measured directly.")
        else:
            print(f"Q{qi + 1}: no clear response on any resonator.")
            print("\nMost likely causes, in order:")
            print("  1. FLUX. The qubit has moved out of the scan window.")
            print(f"     Widen SPAN (try 500) or set the DC bias to whatever")
            print("     the VNA characterization used.")
            print("  2. Probe gain too low -- raise QUBIT_GAIN.")
            print("  3. Readout not on the punched-out frequency -- run step 3.")
        print(f"\nSaved: {path}")
        print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
