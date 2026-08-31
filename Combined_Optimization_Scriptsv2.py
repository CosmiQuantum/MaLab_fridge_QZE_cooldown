"""Alternating PUCQ4 single-shot readout optimization."""
import copy
import datetime
import os

import h5py
import matplotlib.pyplot as plt
import numpy as np

from expt_config import FRIDGE, list_of_all_qubits, tot_num_of_qubits
from section_005_single_shot_ge import GainFrequencySweep, SingleShot
from system_config import QICK_experiment

OPTIMIZATION_STAGE = "gain_frequency"  # "length" or "gain_frequency"
QUBITS = [0, 1, 2, 4, 5]  # M4/index 3 has no assigned physical qubit
READOUT_LENGTHS = [10.0, 5.0, 12.0, 10.0, 10.0, 3.0]  # us; third pass
READOUT_GAINS = [0.48, 0.56, 0.41, 0.30, 0.33, 0.60]
READOUT_OFFSETS = [-0.52, -0.40, -0.08, 0.40, 0.44, -0.04]  # MHz
LENGTH_SWEEP = np.arange(1.0, 13.0, 1.0)
N_LOOPS = 2
GAIN_HALF_SPAN = 0.12
FREQ_HALF_SPAN = 0.24
GAIN_STEPS = 7
FREQ_STEPS = 5
MAX_READOUT_GAIN = 0.95
UNMASK = True

if os.environ.get("PUCQ4_OPT_STAGE"):
    OPTIMIZATION_STAGE = os.environ["PUCQ4_OPT_STAGE"]
if os.environ.get("PUCQ4_OPT_QUBITS"):
    QUBITS = [int(value) for value in os.environ["PUCQ4_OPT_QUBITS"].split(",")]
if os.environ.get("PUCQ4_OPT_GAIN_HALF_SPAN"):
    GAIN_HALF_SPAN = float(os.environ["PUCQ4_OPT_GAIN_HALF_SPAN"])
if os.environ.get("PUCQ4_OPT_FREQ_HALF_SPAN_MHZ"):
    FREQ_HALF_SPAN = float(os.environ["PUCQ4_OPT_FREQ_HALF_SPAN_MHZ"])
if os.environ.get("PUCQ4_OPT_GAIN_STEPS"):
    GAIN_STEPS = int(os.environ["PUCQ4_OPT_GAIN_STEPS"])
if os.environ.get("PUCQ4_OPT_FREQ_STEPS"):
    FREQ_STEPS = int(os.environ["PUCQ4_OPT_FREQ_STEPS"])
if os.environ.get("PUCQ4_OPT_MAX_GAIN"):
    MAX_READOUT_GAIN = float(os.environ["PUCQ4_OPT_MAX_GAIN"])


def output_folders():
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    root = os.path.join(
        "M:/_Data/20250822 - Olivia", "pucq4_run_started_Aug_3", "PUCQ4",
        "pucq4_first_light",
        os.environ.get("PUCQ4_OPT_ROOT_TAG", f"ssf_{OPTIMIZATION_STAGE}_optimization_0mA"),
        stamp,
    )
    data = os.path.join(root, "study_data", "Data_h5")
    plots = os.path.join(root, "documentation")
    os.makedirs(data, exist_ok=True)
    os.makedirs(plots, exist_ok=True)
    return root, data, plots


def configured_experiment(root, qubit):
    experiment = QICK_experiment(
        root, DAC_attenuator1=10, DAC_attenuator2=15,
        qubit_DAC_attenuator1=5, qubit_DAC_attenuator2=4,
        ADC_attenuator=17, fridge=FRIDGE,
    )
    base_frequency = float(os.environ.get(
        "PUCQ4_OPT_RES_BASE_MHZ", experiment.readout_cfg["res_freq_ge"][qubit]
    ))
    if os.environ.get("PUCQ4_OPT_READOUT_LENGTH_US"):
        READOUT_LENGTHS[qubit] = float(os.environ["PUCQ4_OPT_READOUT_LENGTH_US"])
    if os.environ.get("PUCQ4_OPT_READOUT_GAIN"):
        READOUT_GAINS[qubit] = float(os.environ["PUCQ4_OPT_READOUT_GAIN"])
    if os.environ.get("PUCQ4_OPT_READOUT_OFFSET_MHZ"):
        READOUT_OFFSETS[qubit] = float(os.environ["PUCQ4_OPT_READOUT_OFFSET_MHZ"])
    experiment.readout_cfg["res_freq_ge"] = base_frequency + READOUT_OFFSETS[qubit]
    experiment.readout_cfg["res_gain_ge"] = READOUT_GAINS[qubit]
    experiment.readout_cfg["res_gain_ef"] = READOUT_GAINS[qubit]
    experiment.readout_cfg["res_length"] = READOUT_LENGTHS[qubit]
    for key in ("qubit_freq_ge", "qubit_gain_ge", "sigma", "pi_amp"):
        experiment.qubit_cfg[key] = experiment.qubit_cfg[key][qubit]
    if os.environ.get("PUCQ4_OPT_QFREQ_MHZ"):
        experiment.qubit_cfg["qubit_freq_ge"] = float(os.environ["PUCQ4_OPT_QFREQ_MHZ"])
    if os.environ.get("PUCQ4_OPT_SIGMA_US"):
        experiment.qubit_cfg["sigma"] = float(os.environ["PUCQ4_OPT_SIGMA_US"])
    if os.environ.get("PUCQ4_OPT_PI_AMP"):
        experiment.qubit_cfg["pi_amp"] = float(os.environ["PUCQ4_OPT_PI_AMP"])
    return experiment, base_frequency


def shortest_plateau(lengths, fidelities):
    target = float(np.nanmax(fidelities)) - 0.02
    return float(next(length_ for length_, fid in zip(lengths, fidelities) if fid >= target))


def run_length(root, data_folder, plot_folder, qubit):
    tuned, _ = configured_experiment(root, qubit)
    all_fidelities = []
    for length in LENGTH_SWEEP:
        fidelities = []
        for loop in range(N_LOOPS):
            experiment = copy.deepcopy(tuned)
            experiment.readout_cfg["res_length"] = float(length)
            ss = SingleShot(qubit, tot_num_of_qubits, plot_folder, loop, False,
                            experiment, unmasking_resgain=UNMASK)
            fidelity, _, _, _, _ = ss.run()
            fidelities.append(float(fidelity))
        all_fidelities.append(fidelities)
    averages = np.mean(all_fidelities, axis=1)
    deviations = np.std(all_fidelities, axis=1)
    chosen = shortest_plateau(LENGTH_SWEEP, averages)
    with h5py.File(os.path.join(data_folder, f"readout_length_Q{qubit + 1}.h5"), "w") as handle:
        handle.create_dataset("length_us", data=LENGTH_SWEEP)
        handle.create_dataset("fidelity", data=all_fidelities)
        handle.create_dataset("average_fidelity", data=averages)
        handle.create_dataset("std_fidelity", data=deviations)
        handle.attrs["chosen_length_us"] = chosen
    plt.figure()
    plt.errorbar(LENGTH_SWEEP, averages, yerr=deviations, fmt="-o")
    plt.axvline(chosen, color="red", linestyle="--")
    plt.xlabel("Readout length (us)")
    plt.ylabel("Single-shot fidelity")
    plt.title(f"Q{qubit + 1} readout-length optimization")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"readout_length_Q{qubit + 1}.png"), dpi=300)
    plt.close()
    print(f"Q{qubit + 1}: shortest plateau {chosen:.1f} us, max fidelity {max(averages):.4f}")


def run_gain_frequency(root, data_folder, plot_folder, qubit):
    tuned, base_frequency = configured_experiment(root, qubit)
    center_frequency = base_frequency + READOUT_OFFSETS[qubit]
    gain_range = [max(0.01, min(MAX_READOUT_GAIN, READOUT_GAINS[qubit] - GAIN_HALF_SPAN)),
                  min(MAX_READOUT_GAIN, READOUT_GAINS[qubit] + GAIN_HALF_SPAN)]
    frequency_range = [center_frequency - FREQ_HALF_SPAN,
                       center_frequency + FREQ_HALF_SPAN]
    sweep = GainFrequencySweep(
        qubit, tot_num_of_qubits, list_of_all_qubits, tuned,
        optimal_lengths=list(READOUT_LENGTHS), output_folder=data_folder,
        unmasking_resgain=UNMASK,
    )
    results = np.asarray(sweep.run_sweep(frequency_range, gain_range,
                                         FREQ_STEPS, GAIN_STEPS))
    frequencies = np.linspace(*frequency_range, FREQ_STEPS)
    gains = np.linspace(*gain_range, GAIN_STEPS)
    best_index = np.unravel_index(np.nanargmax(results), results.shape)
    best_frequency = float(frequencies[best_index[0]])
    best_gain = float(gains[best_index[1]])
    with h5py.File(os.path.join(data_folder, f"gain_frequency_Q{qubit + 1}.h5"), "w") as handle:
        handle.create_dataset("fidelity", data=results)
        handle.create_dataset("frequency_mhz", data=frequencies)
        handle.create_dataset("gain", data=gains)
        handle.attrs["best_frequency_mhz"] = best_frequency
        handle.attrs["best_offset_mhz"] = best_frequency - base_frequency
        handle.attrs["best_gain"] = best_gain
        handle.attrs["best_fidelity"] = float(results[best_index])
    plt.figure()
    plt.imshow(results, aspect="auto", origin="lower",
               extent=[gains[0], gains[-1], frequencies[0] - base_frequency,
                       frequencies[-1] - base_frequency])
    plt.colorbar(label="Single-shot fidelity")
    plt.scatter([best_gain], [best_frequency - base_frequency], c="red", marker="x")
    plt.xlabel("Readout gain")
    plt.ylabel("Readout offset (MHz)")
    plt.title(f"Q{qubit + 1} gain/frequency optimization")
    plt.tight_layout()
    plt.savefig(os.path.join(plot_folder, f"gain_frequency_Q{qubit + 1}.png"), dpi=300)
    plt.close()
    print(f"Q{qubit + 1}: gain {best_gain:.4f}, offset {best_frequency - base_frequency:+.4f} MHz, fidelity {results[best_index]:.4f}")


def main():
    root, data_folder, plot_folder = output_folders()
    print(f"Saving under {root}")
    for qubit in QUBITS:
        if OPTIMIZATION_STAGE == "length":
            run_length(root, data_folder, plot_folder, qubit)
        elif OPTIMIZATION_STAGE == "gain_frequency":
            run_gain_frequency(root, data_folder, plot_folder, qubit)
        else:
            raise ValueError(f"Unknown OPTIMIZATION_STAGE: {OPTIMIZATION_STAGE}")


if __name__ == "__main__":
    main()
