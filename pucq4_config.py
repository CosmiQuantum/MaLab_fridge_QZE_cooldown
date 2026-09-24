"""
PUCQ4 device parameters and shared helpers.

Written for the NON-MUX 4x2 firmware (built 2024-09-29): two plain
axis_signal_gen_v6 generators and two axis_dyn_readout_v1 readouts. That
readout is tProc-configured, so its frequency is set with add_readoutconfig /
send_readoutconfig rather than in declare_readout, and only ONE resonator tone
can be driven at a time.

Everything else goes through the round robin path -- system_config.py and
expt_config.py now carry the PUCQ4 numbers.

The standalone defaults below are for 0 mA. Nonzero-bias production work uses
the validated current table and selector in system_config/round_robin instead
of duplicating that table here. QUBIT_FREQS_VNA remains a historical starting
guess only.

Nothing in this file touches system_config.py or expt_config.py. These scripts
are self-contained so they can't break the config for the other chip.
"""

import os
import datetime
import logging

import numpy as np

from section_008_save_data_to_h5 import Data_H5

# ----------------------------------------------------------------------------
# Device: measured values from the pptx table
# ----------------------------------------------------------------------------
# Resonator index in the deck is M1..M6; here they are list positions 0..5.
RES_FREQS_VNA = np.array([8920.0, 8951.0, 8975.0, 9000.0, 9015.0, 9059.0])  # MHz
RES_FREQS_DESIGN = np.array([7437.0, 7467.0, 7501.0, 7528.0, 7562.0, 7598.0])  # MHz

# Measured on QICK by pucq4_04_punch_out.py, read off the high-gain rows where
# the dip is well resolved (the low-gain rows were below the noise floor).
# Up to 0.7 MHz from the VNA values -- which matters, because the resonators
# are under 1 MHz wide, so using the VNA numbers puts you most of a linewidth
# off. Use THESE for parking the readout.
# M1-M4 retain their established centers. M5/Q4 and M6/Q6 use the final
# optimized 0 mA resonator bases from the current campaign.
RES_FREQS_MEASURED = np.array([8919.77, 8951.61, 8974.80, 8999.92, 9014.66, 9058.88])
# Optimized physical readout tones are base + offset.
RES_READOUT_OFFSETS = np.array([-0.52, -0.40, -0.21, -0.22999999999956344, 0.2000000000007276, -0.41999999999825377])

# Qubit g-e frequencies, same row order as the resonators above.
# Accepted physical assignments are M5->Q4 and M6->Q6. Other physical numbers
# remain unassigned; the September 8 M4 discovery does not assign it physical Q4.
QUBIT_FREQS_VNA = np.array([5507.0, 5411.0, 5487.0, 5575.0, 7036.0, 6401.0])  # MHz
# Zero-current spectroscopy remeasured by the September 8--9 round robin.
# The deck maps physical Q4 to M5 and physical Q6 to M6; M4 remains unassigned.
QUBIT_FREQS_MEASURED_0MA = np.array(
    [5517.691628432926, 5438.710887054151, 5557.335688295514, 5594.541902899418,
     7084.548595174133, 6522.489876246844]
)
QUBIT_FREQS_GF2 = np.array([5397.0, 5306.0, 5378.0, 5466.0, 6929.0, 6293.0])  # MHz
ANHARMONICITY = np.array([220.0, 210.0, 218.0, 218.0, 214.0, 216.0])  # MHz

NUM_RES = 6

# Qubit linewidths were broad on the VNA (20-40 MHz), so qubit spec wants a
# wide span and does not need fine steps on the first pass.
QUBIT_LINEWIDTH_GUESS = np.array([40.0, 20.0, 25.0, 25.0, 40.0, 40.0])  # MHz

# ----------------------------------------------------------------------------
# Hardware channels
# ----------------------------------------------------------------------------
RES_CH = 0        # axis_signal_gen_v6, resonator drive
RO_CH = 0         # axis_dyn_readout_v1. Only 2 exist (0 and 1), NOT 6 --
                  # resonators are measured one at a time, not as a comb.
QUBIT_CH = 1      # axis_signal_gen_v6, qubit drive

# Nyquist zones. With fs_dac ~9830 MHz, zone 2 spans ~4915-9830 MHz, so both
# the 9 GHz resonators and the 5.4-7.0 GHz qubits land in zone 2.
# pucq4_00_check_board.py recomputes this from the real soccfg -- trust that
# over these defaults.
NQZ_RES = 2
NQZ_QUBIT = 2

# Neither generator has a digital mixer (soccfg reports mixer: False, and the
# firmware printout shows a bare 32-bit DDS with range = fs). declare_gen must
# NOT be given a mixer_freq -- QICK warns "doesn't have a mixer, so it will do
# nothing". The mixer_freq entry in system_config.py is vestigial on this
# firmware.
MIXER_FREQ = None

# ----------------------------------------------------------------------------
# Readout defaults
# ----------------------------------------------------------------------------
# Gain per resonator, DAC units. PUCQ4 sits ~11 dB further down the DAC's
# sin(x)/x curve than the 7.2 GHz chip did, so expect to need much more gain
# than the 0.25 that worked before. See pucq4_00_check_board.py.
RES_GAIN = [0.48, 0.56, 0.95, 0.95, 0.65, 0.95]
RES_LENGTHS = [10.0, 5.0, 9.0, 5.0, 6.0, 10.0]  # [us], per M1..M6 at 0 mA
RES_PHASE = [0.0] * NUM_RES
RO_PHASE = [0.0] * NUM_RES
RES_LENGTH = 2.0             # [us] readout pulse / capture length
TRIG_TIME = 0.4              # [us] ADC trigger delay -- CONFIRM with 01_tof
RELAX_DELAY = 100.0          # [us]

# ----------------------------------------------------------------------------
# Data saving -- mirrors round_robin_benchmark_res_spec_simple.py exactly:
#
#   M:/_Data/20250822 - Olivia/{RUN_NAME}/{DEVICE_NAME}/{study}/{sub_study}/
#       {timestamp}/
#           optimization/
#           study_data/          <- raw data (.npz, .h5) ONLY
#           documentation/       <- ALL plots (.png/.pdf), plus
#                                   sub_study_notes.txt and RR_script.log
#
# Convention: plots always go in documentation/, data always in study_data/.
# ----------------------------------------------------------------------------
DATA_ROOT = "M:/_Data/20250822 - Olivia"
RUN_NAME = "pucq4_run_started_Aug_3"
DEVICE_NAME = "PUCQ4"


def base_cfg(experiment=None):
    """Config dict shared by PUCQ4 scripts and round-robin wrappers.

    When a standard ``QICK_experiment`` is supplied, use its hardware and
    readout configuration as the source of truth. Standalone characterization
    scripts retain the PUCQ4 defaults below so they can bootstrap a new device.
    """
    cfg = {
        "res_ch": RES_CH,
        "ro_ch": RO_CH,
        "qubit_ch": QUBIT_CH,
        "nqz_res": NQZ_RES,
        "nqz_qubit": NQZ_QUBIT,
        "res_freq_ge": list(RES_FREQS_MEASURED),
        "res_gain_ge": list(RES_GAIN),
        "res_length_ge": list(RES_LENGTHS),
        "res_freq_offset_ge": list(RES_READOUT_OFFSETS),
        "res_phase": list(RES_PHASE),
        "ro_phase": list(RO_PHASE),
        "res_length": RES_LENGTH,
        "trig_time": TRIG_TIME,
        "relax_delay": RELAX_DELAY,
        "list_of_all_qubits": list(range(NUM_RES)),
    }
    if experiment is not None:
        hw = experiment.hw_cfg
        readout = experiment.readout_cfg
        cfg.update({
            "res_ch": hw["res_ch"],
            "ro_ch": hw["ro_ch"][0] if isinstance(hw["ro_ch"], list) else hw["ro_ch"],
            "qubit_ch": hw["qubit_ch"],
            "nqz_res": hw["nqz_res"],
            "nqz_qubit": hw["nqz_qubit"],
            "res_freq_ge": readout["res_freq_ge"],
            "res_gain_ge": readout["res_gain_ge"],
            "res_length_ge": readout.get("res_length_ge", list(RES_LENGTHS)),
            "res_freq_offset_ge": readout.get(
                "res_freq_offset_ge", list(RES_READOUT_OFFSETS)
            ),
            "res_length": readout["res_length"],
            "trig_time": readout["trig_time"],
        })
    return cfg


def declare_res_single(prog, cfg, freq, gain, pulse_length):
    """Declare the resonator generator + dynamic readout for ONE tone.

    axis_dyn_readout_v1 is tProc-configured: passing freq/phase to
    declare_readout raises "readout N is dynamic". The frequency goes through
    add_readoutconfig/send_readoutconfig instead. This mirrors what
    section_001_time_of_flight.py already does, so it stays consistent with the
    round robin code.
    """
    prog.declare_gen(ch=cfg["res_ch"], nqz=cfg["nqz_res"])
    prog.declare_readout(ch=cfg["ro_ch"], length=cfg["res_length"])

    prog.add_readoutconfig(ch=cfg["ro_ch"], name="myro",
                           freq=freq, gen_ch=cfg["res_ch"], outsel="product")
    prog.send_readoutconfig(ch=cfg["ro_ch"], name="myro", t=0)

    prog.add_pulse(ch=cfg["res_ch"], name="res_pulse", ro_ch=cfg["ro_ch"],
                   style="const",
                   length=pulse_length,
                   freq=freq,
                   phase=cfg["ro_phase"][0] if isinstance(cfg["ro_phase"], list)
                   else cfg["ro_phase"],
                   gain=gain)


def amp_from_iq(iq_list, ro_index=0):
    """Scalar amplitude from an accumulated acquire().

    Flattens everything except the trailing [I, Q] axis rather than hard-coding
    index depth, which varies with QICK version and sweep count.
    """
    a = np.asarray(iq_list[ro_index], dtype=float).reshape(-1, 2).mean(axis=0)
    return float(np.abs(a[0] + 1j * a[1]))


def create_data_dict(keys, save_r=1, qubits=NUM_RES):
    """Create the object-array structure used by round_robin_benchmark.py."""
    return {
        qubit: {key: np.empty(save_r, dtype=object) for key in keys}
        for qubit in range(qubits)
    }


def save_h5(subStudyDataFolder, data, data_type, batch_num=1, save_r=1):
    """Save through the repository's canonical round-robin HDF5 writer."""
    saver = Data_H5(subStudyDataFolder, data, batch_num, save_r)
    saver.save_to_h5(data_type, save_dataset_clean=True)


def nyquist_zone(freq_mhz, fs_mhz):
    """Which Nyquist zone freq lands in for a converter running at fs."""
    return int(np.floor(freq_mhz / (fs_mhz / 2.0))) + 1


def setup_data_folders(study, sub_study, substudy_txt_notes, log_name="RR_script.log"):
    """Build the round robin folder tree and return the folders + a logger.

    Same layout and the same variable names as
    round_robin_benchmark_res_spec_simple.py, so PUCQ4 helper output lands
    alongside the round robin data instead of in the repo directory.

    Returns a dict with dataSetFolder, optimizationFolder, subStudyDataFolder,
    studyDocumentationFolder, and logger.
    """
    data_set = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    runFolder = os.path.join(DATA_ROOT, RUN_NAME)
    deviceFolder = os.path.join(runFolder, DEVICE_NAME)
    studyFolder = os.path.join(deviceFolder, study)
    subStudyFolder = os.path.join(studyFolder, sub_study)

    dataSetFolder = os.path.join(subStudyFolder, data_set)
    optimizationFolder = os.path.join(dataSetFolder, "optimization")
    studyDataFolder = os.path.join(dataSetFolder, "study_data")
    studyDocumentationFolder = os.path.join(dataSetFolder, "documentation")

    for folder in (runFolder, deviceFolder, studyFolder, subStudyFolder,
                   dataSetFolder, optimizationFolder, studyDataFolder,
                   studyDocumentationFolder):
        os.makedirs(folder, exist_ok=True)

    with open(os.path.join(studyDocumentationFolder, "sub_study_notes.txt"),
              "w", encoding="utf-8") as f:
        f.write(substudy_txt_notes)

    # Custom logger with propagation disabled, so the underlying qick package's
    # own logs stay out of this file -- same reasoning as the RR script.
    logger = logging.getLogger("custom_logger_for_rr_only")
    logger.setLevel(logging.DEBUG)
    handler = logging.FileHandler(
        os.path.join(studyDocumentationFolder, log_name), mode="a")
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    print(f"\nSaving to: {dataSetFolder}\n")
    return {
        "dataSetFolder": dataSetFolder,
        "optimizationFolder": optimizationFolder,
        "studyDataFolder": studyDataFolder,  # compatibility with older runs
        "subStudyDataFolder": studyDataFolder,
        "studyDocumentationFolder": studyDocumentationFolder,
        "logger": logger,
    }
