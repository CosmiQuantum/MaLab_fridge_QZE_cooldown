"""Cross-resonator Q6 spectroscopy diagnostic for the PUCQ4 flux campaign.

The qubit drive remains physical Q6 (M6/Python index 5).  Only the readout
resonator is changed, which tests whether a flux-shifted transition is visible
through another resonator without changing the established QICK pulse program.
"""

import argparse
import datetime
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from build_state import all_qubit_state
from build_task import add_qubit_experiment
from expt_config import expt_cfg
from malab_yokogawa_gs200 import YokogawaGS200
from section_004_qubit_spec_ge import PulseProbeSpectroscopyProgram
from section_008_save_data_to_h5 import Data_H5
from system_config import QICK_experiment


DATA_ROOT = os.path.join(
    "M:/_Data/20250822 - Olivia",
    "pucq4_run_started_Aug_3",
    "PUCQ4",
    "pucq4_first_light",
)
CURRENT_GRID_MA = {-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0}
Q6_INDEX = 5
YOKO3_IP = "192.168.1.73"
YOKO4_IP = "192.168.1.77"
MAX_CURRENT_A = 0.010
RAMP_RATE_A_PER_S = 0.0005


def projected_response(i_values, q_values):
    """Return detrended strongest-axis response and a robust local-noise score."""
    x_values = np.linspace(-1.0, 1.0, i_values.size)
    i_residual = i_values - np.polyval(np.polyfit(x_values, i_values, 1), x_values)
    q_residual = q_values - np.polyval(np.polyfit(x_values, q_values, 1), x_values)
    samples = np.column_stack((i_residual, q_residual))
    _, _, axes = np.linalg.svd(samples, full_matrices=False)
    projected = samples @ axes[0]
    window = min(21, projected.size if projected.size % 2 else projected.size - 1)
    window = max(window, 5)
    smooth = savgol_filter(projected, window, 2)
    noise = np.median(np.abs(projected - smooth)) / 0.6744897501960817
    if not np.isfinite(noise) or noise <= 0:
        noise = max(float(np.std(projected - smooth)), 1e-12)
    normalized = projected / noise
    normalized_smooth = smooth / noise
    return normalized, normalized_smooth, float(np.max(np.abs(normalized_smooth)))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-ma", type=float, required=True)
    parser.add_argument("--qfreq-mhz", type=float, required=True)
    parser.add_argument("--half-span-mhz", type=float, default=2.0)
    parser.add_argument("--steps", type=int, default=401)
    parser.add_argument("--reps", type=int, default=1200)
    parser.add_argument("--qspec-gain", type=float, default=0.010)
    parser.add_argument("--qspec-length-us", type=float, default=40.0)
    parser.add_argument(
        "--readout-row",
        type=int,
        action="append",
        choices=range(1, 7),
        help="Limit acquisition to one or more one-based resonator rows; default is M1-M6",
    )
    parser.add_argument(
        "--m6-readout-mhz",
        type=float,
        help="Override the M6 readout frequency after its current-dependent res spec",
    )
    return parser.parse_args()


def verify_inactive_source():
    with YokogawaGS200(YOKO3_IP, max_current=MAX_CURRENT_A) as source:
        if source.get_mode() != "CURR" or abs(source.get_current()) > 1e-7 or source.get_output():
            raise RuntimeError("Inactive Yoko3 must be in current mode at 0 A with output off")


def main():
    args = parse_args()
    if args.current_ma not in CURRENT_GRID_MA:
        raise ValueError("Current must be one of the nine deck grid values from -10 to +10 mA")
    if not 0.1 <= args.half_span_mhz <= 50.0:
        raise ValueError("Half span must be between 0.1 and 50.0 MHz")
    if not 101 <= args.steps <= 2001:
        raise ValueError("Steps must be between 101 and 2001")
    if not 100 <= args.reps <= 4000:
        raise ValueError("Reps must be between 100 and 4000")

    verify_inactive_source()
    current_tag = f"{args.current_ma:+05.1f}mA"
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = os.path.join(DATA_ROOT, f"cross_ro_q6_{current_tag}", stamp)
    study_data = os.path.join(root, "study_data")
    documentation = os.path.join(root, "documentation")
    optimization = os.path.join(root, "optimization")
    for folder in (study_data, documentation, optimization):
        os.makedirs(folder, exist_ok=True)

    with YokogawaGS200(YOKO4_IP, max_current=MAX_CURRENT_A) as source:
        source.set_mode("current")
        source.set_range(MAX_CURRENT_A)
        if abs(source.get_current()) > 1e-7:
            source.set_output(True)
            source.ramp_current(0.0, sweeprate=RAMP_RATE_A_PER_S)
        source.set_current(0.0)
        source.set_output(True)
        try:
            source.ramp_current(args.current_ma * 1e-3, sweeprate=RAMP_RATE_A_PER_S)
            measured = source.get_current()
            if abs(measured - args.current_ma * 1e-3) > 1e-7:
                raise RuntimeError("Yoko4 setpoint verification failed")
            print(f"Yoko4 / physical Q6 verified at {args.current_ma:+.1f} mA")

            experiment = QICK_experiment(
                root,
                DAC_attenuator1=10,
                DAC_attenuator2=15,
                qubit_DAC_attenuator1=5,
                qubit_DAC_attenuator2=4,
                ADC_attenuator=17,
                fridge="QUIET",
            )
            centers = list(experiment.readout_cfg["res_freq_ge"])
            gains = list(experiment.readout_cfg["res_gain_ge"])
            lengths = list(experiment.readout_cfg.get("res_length_ge", [5.0] * 6))
            offsets = list(experiment.readout_cfg.get("res_freq_offset_ge", [0.0] * 6))
            readout_frequencies = [center + offset for center, offset in zip(centers, offsets)]
            if args.m6_readout_mhz is not None:
                readout_frequencies[Q6_INDEX] = args.m6_readout_mhz

            qspec_cfg = expt_cfg["qubit_spec_ge"]
            qspec_cfg.update(
                {
                    "reps": args.reps,
                    "rounds": 1,
                    "start": args.qfreq_mhz - args.half_span_mhz,
                    "stop": args.qfreq_mhz + args.half_span_mhz,
                    "steps": args.steps,
                    "relax_delay": 100,
                    "list_of_all_qubits": [Q6_INDEX],
                }
            )
            experiment.qubit_cfg["qubit_gain_ge"][Q6_INDEX] = args.qspec_gain
            experiment.qubit_cfg["qubit_length_ge"] = args.qspec_length_us
            experiment.qubit_cfg["qubit_freq_ge"][Q6_INDEX] = args.qfreq_mhz

            selected_indices = (
                [row - 1 for row in args.readout_row]
                if args.readout_row
                else list(range(6))
            )
            results = []
            for readout_index in selected_indices:
                experiment.readout_cfg["res_freq_ge"] = readout_frequencies[readout_index]
                experiment.readout_cfg["res_gain_ge"] = gains[readout_index]
                experiment.readout_cfg["res_gain_ef"] = gains[readout_index]
                experiment.readout_cfg["res_length"] = lengths[readout_index]
                q_state = all_qubit_state(experiment, 6)
                sweep_cfg = add_qubit_experiment(expt_cfg, "qubit_spec_ge", Q6_INDEX)
                config = {**q_state[f"Q{Q6_INDEX}"], **sweep_cfg}
                print(
                    f"Readout M{readout_index + 1}: {readout_frequencies[readout_index]:.5f} MHz, "
                    f"gain {gains[readout_index]:.4f}, length {lengths[readout_index]:.2f} us"
                )
                program = PulseProbeSpectroscopyProgram(
                    experiment.soccfg,
                    reps=args.reps,
                    final_delay=config["relax_delay"],
                    cfg=config,
                )
                iq_list = program.acquire(experiment.soc, rounds=1, progress=True)
                iq = iq_list[0][0].T
                i_values = np.asarray(iq[0], dtype=float)
                q_values = np.asarray(iq[1], dtype=float)
                frequencies = np.asarray(
                    program.get_pulse_param("qubit_pulse", "freq", as_array=True),
                    dtype=float,
                )
                projected, smooth, score = projected_response(i_values, q_values)
                results.append(
                    {
                        "readout_index": readout_index,
                        "frequencies": frequencies,
                        "i": i_values,
                        "q": q_values,
                        "projected": projected,
                        "smooth": smooth,
                        "score": score,
                        "config": config,
                    }
                )
                print(f"M{readout_index + 1} maximum smoothed response: {score:.2f} local sigma")

            matrix = np.vstack([result["smooth"] for result in results])
            limit = max(3.0, float(np.nanpercentile(np.abs(matrix), 99)))
            fig, axis = plt.subplots(figsize=(12, 6))
            image = axis.imshow(
                matrix,
                aspect="auto",
                origin="lower",
                extent=[
                    frequencies[0], frequencies[-1], 0.5,
                    len(selected_indices) + 0.5,
                ],
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
            )
            axis.set_xlabel("Q6 drive frequency (MHz)")
            axis.set_ylabel("Readout resonator row")
            axis.set_yticks(
                range(1, len(selected_indices) + 1),
                [f"M{index + 1}" for index in selected_indices],
            )
            axis.set_title(
                f"Q6 cross-readout qspec at {args.current_ma:+.1f} mA; "
                f"gain {args.qspec_gain:g}, {args.reps} reps"
            )
            fig.colorbar(image, ax=axis, label="Detrended projected response / local noise")
            fig.tight_layout()
            plot_path = os.path.join(documentation, "q6_cross_readout_qspec.png")
            fig.savefig(plot_path, dpi=200, bbox_inches="tight")
            plt.close(fig)

            saved_config = {
                "physical_qubit": "Q6",
                "drive_index": Q6_INDEX,
                "yoko": "Yoko4",
                "current_mA": args.current_ma,
                "qfreq_center_MHz": args.qfreq_mhz,
                "half_span_MHz": args.half_span_mhz,
                "steps": args.steps,
                "reps": args.reps,
                "qspec_gain": args.qspec_gain,
                "qspec_length_us": args.qspec_length_us,
                "selected_readout_rows": [index + 1 for index in selected_indices],
                "readout_frequencies_MHz": readout_frequencies,
                "readout_gains": gains,
                "readout_lengths_us": lengths,
                "res_DAC_attenuator_1_dB": 10,
                "res_DAC_attenuator_2_dB": 15,
                "qubit_DAC_attenuator_1_dB": 5,
                "qubit_DAC_attenuator_2_dB": 4,
                "ADC_attenuator_dB": 17,
            }
            data = {}
            for result in results:
                row = result["readout_index"]
                data[row] = {
                    "Dates": np.asarray([time.time()]),
                    "Round Num": np.asarray([1]),
                    "Batch Num": np.asarray([0]),
                    "Frequencies": result["frequencies"],
                    "I": result["i"],
                    "Q": result["q"],
                    "Projected Response": result["projected"],
                    "Smoothed Response": result["smooth"],
                    "Maximum Local Sigma": np.asarray([result["score"]]),
                    "Exp Config": json.dumps(saved_config, sort_keys=True),
                    "Syst Config": json.dumps(
                        {**saved_config, "program_config": str(result["config"])},
                        sort_keys=True,
                    ),
                }
            saver = Data_H5(study_data, data, batch_num=0, save_r=1)
            saver.save_to_h5("cross_readout_qspec", save_dataset_clean=True)
            print(f"Plot: {plot_path}")
            print(f"Output: {root}")
        finally:
            print("Returning Yoko4 to 0 A and disabling output")
            source.ramp_current(0.0, sweeprate=RAMP_RATE_A_PER_S)
            source.set_current(0.0)
            source.set_output(False)
            if abs(source.get_current()) > 1e-7 or source.get_output():
                raise RuntimeError("Yoko4 failed final zero/output-off verification")


if __name__ == "__main__":
    main()
