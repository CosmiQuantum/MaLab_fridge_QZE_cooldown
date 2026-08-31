"""Disposable high-resolution M4 qspec diagnostic at low readout power."""
import datetime
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import find_peaks, savgol_filter

from build_state import all_qubit_state
from build_task import add_qubit_experiment
from expt_config import expt_cfg
from section_004_qubit_spec_ge import PulseProbeSpectroscopyProgram
from section_008_save_data_to_h5 import Data_H5
from system_config import QICK_experiment


def _project_iq(i_values, q_values):
    """Linearly detrend IQ, project onto its strongest axis, and estimate noise."""
    x = np.linspace(-1.0, 1.0, i_values.size)
    i_residual = i_values - np.polyval(np.polyfit(x, i_values, 1), x)
    q_residual = q_values - np.polyval(np.polyfit(x, q_values, 1), x)
    samples = np.column_stack((i_residual, q_residual))
    _, _, vh = np.linalg.svd(samples, full_matrices=False)
    projected = samples @ vh[0]
    smooth = savgol_filter(projected, 21, 2)
    noise = np.median(np.abs(projected - smooth)) / 0.6744897501960817
    if not np.isfinite(noise) or noise <= 0:
        noise = max(float(np.std(projected - smooth)), 1e-12)
    normalized = projected / noise
    smooth_normalized = savgol_filter(normalized, 21, 2)
    if abs(np.min(smooth_normalized)) > abs(np.max(smooth_normalized)):
        normalized = -normalized
        smooth_normalized = -smooth_normalized
    return normalized, smooth_normalized


def main():
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = os.path.join(
        "M:/_Data/20250822 - Olivia",
        "pucq4_run_started_Aug_3",
        "PUCQ4",
        "pucq4_first_light",
        "tmp_m4_hr_qspec_0mA",
        stamp,
    )
    optimization = os.path.join(root, "optimization")
    study_data = os.path.join(root, "study_data")
    documentation = os.path.join(root, "documentation")
    for folder in (optimization, study_data, documentation):
        os.makedirs(folder, exist_ok=True)

    qubit = int(os.environ.get("PUCQ4_M4_ROW", "3"))
    drive_start = float(os.environ.get("PUCQ4_M4_START_MHZ", "5525.0"))
    drive_stop = float(os.environ.get("PUCQ4_M4_STOP_MHZ", "5625.0"))
    drive_steps = int(os.environ.get("PUCQ4_M4_STEPS", "2001"))
    reps = int(os.environ.get("PUCQ4_M4_REPS", "2048"))
    rounds = int(os.environ.get("PUCQ4_M4_ROUNDS", "1"))
    qubit_gain = float(os.environ.get("PUCQ4_M4_QGAIN", "0.30"))
    qubit_length_us = float(os.environ.get("PUCQ4_M4_QLENGTH_US", "40.0"))
    readout_frequency_mhz = float(
        os.environ.get("PUCQ4_M4_READOUT_MHZ", "9000.26")
    )
    readout_gain = float(os.environ.get("PUCQ4_M4_READOUT_GAIN", "0.03"))
    readout_length_us = float(
        os.environ.get("PUCQ4_M4_READOUT_LENGTH_US", "25.0")
    )
    qubit_nqz = int(os.environ.get("PUCQ4_M4_NQZ", "2"))

    qspec_cfg = expt_cfg["qubit_spec_ge"]
    qspec_cfg.update(
        {
            "reps": reps,
            "rounds": rounds,
            "start": drive_start,
            "stop": drive_stop,
            "steps": drive_steps,
            "relax_delay": 100,
            "list_of_all_qubits": [qubit],
        }
    )

    experiment = QICK_experiment(
        root,
        DAC_attenuator1=10,
        DAC_attenuator2=15,
        qubit_DAC_attenuator1=5,
        qubit_DAC_attenuator2=4,
        ADC_attenuator=17,
        fridge="QUIET",
    )
    experiment.readout_cfg["res_freq_ge"] = readout_frequency_mhz
    experiment.readout_cfg["res_gain_ge"] = readout_gain
    experiment.readout_cfg["res_gain_ef"] = readout_gain
    experiment.readout_cfg["res_length"] = readout_length_us
    experiment.qubit_cfg["qubit_gain_ge"][qubit] = qubit_gain
    experiment.qubit_cfg["qubit_length_ge"] = qubit_length_us
    experiment.qubit_cfg["nqz_qubit"] = qubit_nqz

    q_state = all_qubit_state(experiment, 6)
    sweep_cfg = add_qubit_experiment(expt_cfg, "qubit_spec_ge", qubit)
    config = {**q_state[f"Q{qubit}"], **sweep_cfg}
    print(
        "M4 high-resolution qspec: "
        f"{drive_start:.2f}--{drive_stop:.2f} MHz, "
        f"{drive_steps} points, gain {qubit_gain:.2f}, "
        f"length {qubit_length_us:.1f} us"
    )
    program = PulseProbeSpectroscopyProgram(
        experiment.soccfg,
        reps=reps,
        final_delay=config["relax_delay"],
        cfg=config,
    )
    iq_list = program.acquire(
        experiment.soc,
        rounds=rounds,
        progress=True,
    )
    iq = iq_list[0][0].T
    i_values = np.asarray(iq[0], dtype=float)
    q_values = np.asarray(iq[1], dtype=float)
    frequencies = np.asarray(
        program.get_pulse_param("qubit_pulse", "freq", as_array=True),
        dtype=float,
    )
    projected, projected_smooth = _project_iq(i_values, q_values)
    magnitude = np.hypot(i_values, q_values)

    absolute_response = np.abs(projected_smooth)
    candidate_indices, properties = find_peaks(
        absolute_response,
        distance=20,
        prominence=max(float(np.std(absolute_response)), 1.0),
    )
    if candidate_indices.size:
        order = candidate_indices[
            np.argsort(properties["prominences"])[::-1]
        ][:10]
    else:
        order = np.argsort(absolute_response)[-10:][::-1]

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)
    axes[0].plot(frequencies, i_values - np.median(i_values), label="I")
    axes[0].plot(
        frequencies,
        q_values - np.median(q_values),
        label="Q",
        alpha=0.8,
    )
    axes[0].set_ylabel("Centered IQ (a.u.)")
    axes[0].legend(loc="best")
    axes[1].plot(frequencies, projected, alpha=0.25, label="projected")
    axes[1].plot(
        frequencies,
        projected_smooth,
        color="black",
        linewidth=1.2,
        label="21-point smooth",
    )
    axes[1].set_ylabel("Projected / local noise")
    axes[1].legend(loc="best")
    axes[2].plot(frequencies, magnitude)
    axes[2].set_ylabel("|IQ| (a.u.)")
    axes[2].set_xlabel("Qubit-drive frequency (MHz)")
    for axis in axes:
        axis.axvline(
            5575.0,
            color="tab:red",
            linestyle="--",
            label="deck M4 5575 MHz",
        )
    axes[0].legend(loc="best")
    fig.suptitle(
        f"M4 qspec: q gain {qubit_gain:.3f} / {qubit_length_us:g} us; "
        f"readout {readout_frequency_mhz:.3f} MHz, "
        f"gain {readout_gain:.3f} / {readout_length_us:g} us"
    )
    fig.tight_layout()
    fig.savefig(
        os.path.join(documentation, "m4_high_resolution_qspec.png"),
        dpi=200,
        bbox_inches="tight",
    )
    plt.close(fig)

    saved_config = {
        "qubit_index": qubit,
        "qubit_drive_start_MHz": drive_start,
        "qubit_drive_stop_MHz": drive_stop,
        "qubit_drive_steps": drive_steps,
        "qubit_drive_step_MHz": (drive_stop - drive_start) / (drive_steps - 1),
        "qubit_gain": qubit_gain,
        "qubit_length_us": qubit_length_us,
        "qubit_nyquist_zone": qubit_nqz,
        "readout_frequency_MHz": readout_frequency_mhz,
        "readout_gain": readout_gain,
        "readout_length_us": readout_length_us,
        "reps": reps,
        "rounds": rounds,
        "relax_delay_us": 100.0,
        "res_DAC_attenuator_1_dB": 10,
        "res_DAC_attenuator_2_dB": 15,
        "qubit_DAC_attenuator_1_dB": 5,
        "qubit_DAC_attenuator_2_dB": 4,
        "ADC_attenuator_dB": 17,
        "yokogawa_current_A": 0.0,
    }
    data = {
        qubit: {
            "Dates": np.asarray([time.time()]),
            "Round Num": np.asarray([1]),
            "Batch Num": np.asarray([0]),
            "Frequencies": frequencies,
            "I": i_values,
            "Q": q_values,
            "Magnitude": magnitude,
            "Projected Response": projected,
            "Smoothed Response": projected_smooth,
            "Exp Config": json.dumps(saved_config, sort_keys=True),
            "Syst Config": json.dumps(
                {**saved_config, "program_config": str(config)},
                sort_keys=True,
            ),
        }
    }
    saver = Data_H5(study_data, data, batch_num=0, save_r=1)
    saver.save_to_h5("m4_qspec", save_dataset_clean=True)

    print("Largest high-resolution responses:")
    for index in order:
        print(
            f"  {frequencies[index]:.5f} MHz: "
            f"{projected_smooth[index]:+.2f} local sigma"
        )
    print(f"Output: {root}")


if __name__ == "__main__":
    main()
