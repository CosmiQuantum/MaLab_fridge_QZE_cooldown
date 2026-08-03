"""
PUCQ4 device parameters and shared helpers.

*** THESE SCRIPTS ARE FOR THE MUX FIRMWARE ONLY. ***
pucq4_01_tof / 02_res_spec_wide / 03_punch_out / 04_qubit_spec_wide all call
declare_gen with mux_freqs/mux_gains/mux_phases, which the non-mux 4x2 firmware
does not accept. On the old firmware use the round robin path instead --
system_config.py and expt_config.py now carry the PUCQ4 numbers.
pucq4_00_check_board.py is firmware-agnostic and still worth running.

Numbers come from "PUCQ4 Initial Characterization.pptx" (VNA characterization).
NOTE: PUCQ4 is FLUX TUNABLE. Every qubit frequency below is only valid at the
DC bias the VNA sweep was taken at. The resonators barely move with flux
(~7 MHz across +/-10 mA), so resonator spectroscopy works at any bias -- but
the qubits move GHz, so treat QUBIT_FREQS_VNA as a starting guess only.

Nothing in this file touches system_config.py or expt_config.py. These scripts
are self-contained so they can't break the config for the other chip.
"""

import os
import numpy as np

# ----------------------------------------------------------------------------
# Device: measured values from the pptx table
# ----------------------------------------------------------------------------
# Resonator index in the deck is M1..M6; here they are list positions 0..5.
RES_FREQS_VNA = np.array([8920.0, 8951.0, 8975.0, 9000.0, 9015.0, 9059.0])  # MHz
RES_FREQS_DESIGN = np.array([7437.0, 7467.0, 7501.0, 7528.0, 7562.0, 7598.0])  # MHz

# Qubit g-e frequencies, same row order as the resonators above.
# WARNING: the resonator <-> qubit mapping is NOT settled. The deck labels
# M5->Q4 and M6->Q6, but the avoided-crossing data suggests Q4 couples to M1
# and Q6 to M2. Do not build a mapping into anything downstream yet.
QUBIT_FREQS_VNA = np.array([5507.0, 5411.0, 5487.0, 5575.0, 7036.0, 6401.0])  # MHz
QUBIT_FREQS_GF2 = np.array([5397.0, 5306.0, 5378.0, 5466.0, 6929.0, 6293.0])  # MHz
ANHARMONICITY = np.array([220.0, 210.0, 218.0, 218.0, 214.0, 216.0])  # MHz

NUM_RES = 6

# Qubit linewidths were broad on the VNA (20-40 MHz), so qubit spec wants a
# wide span and does not need fine steps on the first pass.
QUBIT_LINEWIDTH_GUESS = np.array([40.0, 20.0, 25.0, 25.0, 40.0, 40.0])  # MHz

# ----------------------------------------------------------------------------
# Hardware channels
# ----------------------------------------------------------------------------
RES_CH = 0        # axis_sg_mux8_v1, the readout mux DAC
RO_CHS = [0, 1, 2, 3, 4, 5]
QUBIT_CH = 1      # full-speed DAC -- may not exist in readout-only firmware

# Nyquist zones. With fs_dac ~9830 MHz, zone 2 spans ~4915-9830 MHz, so both
# the 9 GHz resonators and the 5.4-7.0 GHz qubits land in zone 2.
# pucq4_00_check_board.py recomputes this from the real soccfg -- trust that
# over these defaults.
NQZ_RES = 2
NQZ_QUBIT = 2

# The mux8 generator on this board has no digital mixer, so declare_gen must
# NOT be given a mixer_freq. Set to a number only if 00_check_board says the
# generator reports a mixer.
MIXER_FREQ = None

# ----------------------------------------------------------------------------
# Readout defaults
# ----------------------------------------------------------------------------
RES_GAIN = [0.3] * NUM_RES   # mux gains, DAC units
RES_PHASE = [0.0] * NUM_RES
RO_PHASE = [0.0] * NUM_RES
RES_LENGTH = 2.0             # [us] readout pulse / capture length
TRIG_TIME = 0.4              # [us] ADC trigger delay -- CONFIRM with 01_tof
RELAX_DELAY = 100.0          # [us]

OUTPUT_FOLDER = "pucq4_data"


def base_cfg():
    """Config dict shared by every PUCQ4 mux program."""
    return {
        "res_ch": RES_CH,
        "ro_ch": list(RO_CHS),
        "qubit_ch": QUBIT_CH,
        "nqz_res": NQZ_RES,
        "nqz_qubit": NQZ_QUBIT,
        "res_freq_ge": list(RES_FREQS_VNA),
        "res_gain_ge": list(RES_GAIN),
        "res_phase": list(RES_PHASE),
        "ro_phase": list(RO_PHASE),
        "res_length": RES_LENGTH,
        "trig_time": TRIG_TIME,
        "relax_delay": RELAX_DELAY,
        "list_of_all_qubits": list(range(NUM_RES)),
    }


def declare_res_mux(prog, cfg):
    """Declare the readout mux generator + one readout channel per tone.

    Kept in one place so every script declares the mux identically.
    """
    kwargs = dict(
        ch=cfg["res_ch"],
        nqz=cfg["nqz_res"],
        ro_ch=cfg["ro_ch"][0],
        mux_freqs=cfg["res_freq_ge"],
        mux_gains=cfg["res_gain_ge"],
        mux_phases=cfg["res_phase"],
    )
    if MIXER_FREQ is not None:
        kwargs["mixer_freq"] = MIXER_FREQ
    prog.declare_gen(**kwargs)

    for ch, f, ph in zip(cfg["ro_ch"], cfg["res_freq_ge"], cfg["ro_phase"]):
        prog.declare_readout(ch=ch, length=cfg["res_length"], freq=f,
                             phase=ph, gen_ch=cfg["res_ch"])


def amps_from_iq(iq_list):
    """One amplitude per readout channel.

    QICK's acquire() shape varies with version and with how many sweep axes are
    declared, so this flattens everything except the trailing [I, Q] axis
    instead of hard-coding indices. That indexing is exactly what is broken in
    section_002_res_spec_ge_mux.py on the 4x2_board branch, which reads tone 0's
    I against tone 1's I and silently returns nonsense for a 6-tone mux.
    """
    out = []
    for ch_data in iq_list:
        a = np.asarray(ch_data, dtype=float).reshape(-1, 2).mean(axis=0)
        out.append(float(np.abs(a[0] + 1j * a[1])))
    return np.array(out)


def nyquist_zone(freq_mhz, fs_mhz):
    """Which Nyquist zone freq lands in for a converter running at fs."""
    return int(np.floor(freq_mhz / (fs_mhz / 2.0))) + 1


def make_output_folder(name):
    path = os.path.join(OUTPUT_FOLDER, name)
    os.makedirs(path, exist_ok=True)
    return path
