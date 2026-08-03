"""
STEP 0 -- Can this board even reach PUCQ4's frequencies?

Sends NO pulses. Connects to the QICK, prints the firmware description, and
answers three questions:

  1. Is there a qubit-drive generator, or is this readout-only firmware?
  2. What Nyquist zone do the 9 GHz resonators land in, on both the DAC and
     the ADC? (Setting nqz wrong is the #1 reason you see nothing.)
  3. Does the readout generator have a digital mixer?

Run this first. It is safe with the fridge cold and takes seconds.

    python pucq4_00_check_board.py
"""

import numpy as np

from socProxy import makeProxy
import pucq4_config as P


def describe_converters(soccfg):
    """Pull generator/readout tables out of soccfg without assuming key names."""
    gens, ros = [], []
    for i, g in enumerate(soccfg["gens"]):
        gens.append({
            "ch": i,
            "type": g.get("type", "?"),
            "fs": g.get("fs", float("nan")),
            "has_mixer": g.get("has_mixer", "mixer_freq" in g),
            "maxlen": g.get("maxlen", None),
        })
    for i, r in enumerate(soccfg["readouts"]):
        ros.append({
            "ch": i,
            "type": r.get("type", "?"),
            "fs": r.get("fs", float("nan")),
        })
    return gens, ros


def main():
    soc, soccfg = makeProxy()

    print("\n" + "=" * 78)
    print("FIRMWARE SUMMARY")
    print("=" * 78)

    gens, ros = describe_converters(soccfg)

    print("\nGenerators (DACs):")
    for g in gens:
        print(f"  ch {g['ch']}: {g['type']:<22} fs = {g['fs']:>9.1f} MHz"
              f"   mixer: {g['has_mixer']}")

    print("\nReadouts (ADCs):")
    for r in ros:
        print(f"  ch {r['ch']}: {r['type']:<22} fs = {r['fs']:>9.1f} MHz")

    # ------------------------------------------------------------------
    # 1. Is there a qubit-drive generator?
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("QUBIT DRIVE")
    print("-" * 78)
    mux_gens = [g for g in gens if "mux" in str(g["type"]).lower()]
    drive_gens = [g for g in gens if "mux" not in str(g["type"]).lower()]

    if not drive_gens:
        print("  *** NO full-speed drive generator in this firmware. ***")
        print("  This is READOUT-ONLY. Steps 1-3 (TOF, res spec, punch out)")
        print("  will work. Qubit spectroscopy, Rabi, T1 and the round robin")
        print("  benchmark CANNOT run until a firmware with a drive generator")
        print("  is loaded.")
    else:
        print(f"  Found {len(drive_gens)} drive generator(s): "
              f"channels {[g['ch'] for g in drive_gens]}")
        print(f"  pucq4_config.QUBIT_CH is set to {P.QUBIT_CH} -- make sure "
              f"that is one of them.")

    if mux_gens:
        m = mux_gens[0]
        print(f"\n  Readout mux generator is channel {m['ch']} ({m['type']}).")
        print(f"  pucq4_config.RES_CH is set to {P.RES_CH}.")
        if m["has_mixer"]:
            print("  It HAS a digital mixer -> set MIXER_FREQ in pucq4_config.py")
            print(f"  to about {np.mean(P.RES_FREQS_VNA):.0f} MHz.")
        else:
            print("  It has NO digital mixer -> leave MIXER_FREQ = None.")
            print("  Tones are synthesized directly, so nqz_res must be right.")

    # ------------------------------------------------------------------
    # 2. Nyquist zones for the PUCQ4 frequencies
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("NYQUIST ZONES FOR PUCQ4")
    print("-" * 78)

    res_gen = mux_gens[0] if mux_gens else gens[P.RES_CH]
    ro = ros[P.RO_CHS[0]]

    print(f"\nResonators 8920-9059 MHz, on DAC ch {res_gen['ch']} "
          f"(fs = {res_gen['fs']:.1f}) and ADC ch {ro['ch']} (fs = {ro['fs']:.1f}):")
    for i, f in enumerate(P.RES_FREQS_VNA):
        zd = P.nyquist_zone(f, res_gen["fs"])
        za = P.nyquist_zone(f, ro["fs"])
        print(f"  M{i + 1}  {f:8.1f} MHz   DAC zone {zd}   ADC zone {za}")

    dac_zones = {P.nyquist_zone(f, res_gen["fs"]) for f in P.RES_FREQS_VNA}
    adc_zones = {P.nyquist_zone(f, ro["fs"]) for f in P.RES_FREQS_VNA}

    print()
    if len(dac_zones) > 1:
        print("  *** The resonators straddle a DAC Nyquist boundary. They cannot")
        print("      all be driven by one mux comb. ***")
    else:
        zone = dac_zones.pop()
        print(f"  All resonators are in DAC Nyquist zone {zone}.")
        if zone != P.NQZ_RES:
            print(f"  *** pucq4_config.NQZ_RES is {P.NQZ_RES} -- change it to "
                  f"{zone}. ***")
        else:
            print(f"  pucq4_config.NQZ_RES = {P.NQZ_RES} is correct.")

    if len(adc_zones) > 1:
        print("  *** The resonators straddle an ADC Nyquist boundary. ***")
    else:
        print(f"  All resonators are in ADC Nyquist zone {adc_zones.pop()}.")

    # The mux comb has to fit inside one zone with room to spare.
    span = P.RES_FREQS_VNA.max() - P.RES_FREQS_VNA.min()
    print(f"\n  Mux comb span: {span:.1f} MHz "
          f"(centre {np.mean(P.RES_FREQS_VNA):.1f} MHz).")

    if drive_gens:
        qg = drive_gens[0]
        print(f"\nQubits 5411-7036 MHz, on DAC ch {qg['ch']} "
              f"(fs = {qg['fs']:.1f}):")
        qz = set()
        for i, f in enumerate(P.QUBIT_FREQS_VNA):
            z = P.nyquist_zone(f, qg["fs"])
            qz.add(z)
            print(f"  Q{i + 1}  {f:8.1f} MHz   DAC zone {z}")
        if len(qz) > 1:
            print(f"\n  *** Qubits span DAC zones {sorted(qz)} -- you will need "
                  f"to change nqz_qubit between qubits. ***")
        else:
            z = qz.pop()
            print(f"\n  All qubits are in DAC zone {z}. "
                  f"pucq4_config.NQZ_QUBIT is {P.NQZ_QUBIT}"
                  f"{' -- correct.' if z == P.NQZ_QUBIT else f' -- change it to {z}.'}")

    # ------------------------------------------------------------------
    # 3. Analog roll-off warning
    # ------------------------------------------------------------------
    print("\n" + "-" * 78)
    print("ANALOG BANDWIDTH")
    print("-" * 78)
    print("  Nyquist zone arithmetic only tells you the tone is representable.")
    print("  It says nothing about how much signal actually survives the")
    print("  analog front end at 9 GHz, which is high for an RFSoC4x2 ADC.")
    print("  pucq4_01_tof.py is the real test -- if you see no ringdown there,")
    print("  the problem is analog, not configuration.")
    print("=" * 78 + "\n")


if __name__ == "__main__":
    main()
