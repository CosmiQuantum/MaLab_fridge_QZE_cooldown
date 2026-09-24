"""Bounded M4 diagnostics using the established spectroscopy programs.

Importing this module does not connect to hardware. ``--execute`` runs the
explicit JSON plan, checks zero-current sources, and saves every trace through
pucq4_config/Data_H5 before continuing. No device calibration is promoted here.
"""

import argparse
import contextlib
import copy
import datetime
import io
import json
from pathlib import Path
import time

import h5py
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter

from build_state import qubit_state
from build_task import add_qubit_experiment
from expt_config import expt_cfg, FRIDGE
from malab_yokogawa_gs200 import YokogawaGS200
import pucq4_config as P
from section_002_res_spec_ge_mux import SingleToneSpectroscopyProgram
from section_004_qubit_spec_ge import PulseProbeSpectroscopyProgram
from system_config import QICK_experiment


STUDY = "pucq4_first_light"
ATTENUATORS = dict(DAC_attenuator1=10, DAC_attenuator2=15,
                   qubit_DAC_attenuator1=5, qubit_DAC_attenuator2=4,
                   ADC_attenuator=17)


def source_readbacks():
    """Read only: do not change either flux source for the fixed-row search."""
    states = []
    for label, host in (("Yoko3/Q4/M5", "192.168.1.73"),
                        ("Yoko4/Q6/M6", "192.168.1.77")):
        with YokogawaGS200(host) as source:
            state = dict(label=label, identity=source.get_id(),
                         mode=source.get_mode(), output=source.get_output(),
                         current_A=source.get_level(), range_A=source.get_range(),
                         voltage_limit_V=source.get_voltage_limit())
        print("SOURCE " + json.dumps(state), flush=True)
        states.append(state)
        if (not state["mode"].startswith("CURR") or state["output"]
                or abs(state["current_A"]) > 1e-9
                or abs(state["range_A"] - 0.01) > 1e-9):
            raise RuntimeError("Unexpected source state; no source was changed")
    return states


def validate_plan(plan):
    label = plan["sub_study"]
    if not label.startswith("m4_") or not label.replace("_", "").isalnum():
        raise ValueError("Use a short m4_ sub-study label without path components")
    runs = plan["runs"]
    if not runs or len(runs) > 40:
        raise ValueError("A bounded plan must contain 1-40 acquisitions")
    for spec in runs:
        if spec["kind"] not in ("res", "qspec") or spec["row"] not in range(6):
            raise ValueError("Unknown experiment or zero-based resonator row")
        if not spec.get("hypothesis"):
            raise ValueError("Each acquisition needs a hypothesis/control purpose")
        if not 1 <= spec["reps"] <= 16384 or not 1 <= spec.get("rounds", 1) <= 8:
            raise ValueError("Unbounded averaging")
        if not 3 <= spec["steps"] <= 20001:
            raise ValueError("Unbounded point count")
        if not 1 <= spec["readout_length_us"] <= 25:
            raise ValueError("Readout length must be within the reviewed 1-25 us")
        if not 0 < spec["readout_gain"] <= 0.95:
            raise ValueError("Readout gain outside the demonstrated envelope")
        if not 5 <= spec["relax_delay_us"] <= 4000:
            raise ValueError("Relaxation interval outside the reviewed envelope")
        if spec["kind"] == "res":
            if not 8900 <= spec["start_mhz"] < spec["stop_mhz"] <= 9080:
                raise ValueError("Resonator sweep outside PUCQ4 band")
        else:
            if not 4915.2 < spec["start_mhz"] < spec["stop_mhz"] <= 7300:
                raise ValueError("Qspec sweep outside reviewed DAC zone 2 band")
            if not 8900 <= spec["readout_mhz"] <= 9080:
                raise ValueError("Readout outside PUCQ4 band")
            if not 0 <= spec["qubit_gain"] <= 1:
                raise ValueError("Drive gain outside DAC limits")
            if not 1 <= spec["qubit_length_us"] <= 80:
                raise ValueError("Drive duration outside reviewed range")
    root = Path(P.DATA_ROOT) / P.RUN_NAME / P.DEVICE_NAME / STUDY / label
    h5_path = root / "2026-09-08_00-00-00" / "study_data" / "Data_h5" / "QSpec_ge" / (
        "2026-09-08_00-00-00_QSpec_ge_results_batch_40_Num_per_batch1.h5")
    plot_path = root / "2026-09-08_00-00-00" / "documentation" / (
        "R40_Q4_2026-09-08_00-00-00_qspec.png")
    longest = max(len(str(h5_path)), len(str(plot_path)))
    if longest >= 260:
        raise ValueError("Generated Windows artifact path is too long")
    print("PREFLIGHT longest artifact path: %d characters" % longest, flush=True)
    return longest


def selected_config(experiment, spec):
    row = spec["row"]
    # Write both keys: build_state deliberately derives scalar res_length from
    # res_length_ge. A scalar-only override would silently use the old length.
    experiment.readout_cfg["res_length_ge"] = spec["readout_length_us"]
    experiment.readout_cfg["res_length"] = spec["readout_length_us"]
    experiment.readout_cfg["res_gain_ge"] = spec["readout_gain"]
    experiment.readout_cfg["res_gain_ef"] = spec["readout_gain"]
    experiment.hw_cfg["nqz_qubit"] = 2
    config = qubit_state(experiment, row)
    if spec["kind"] == "qspec":
        expt_cfg["qubit_spec_ge"].update(
            start=spec["start_mhz"], stop=spec["stop_mhz"],
            steps=spec["steps"], reps=spec["reps"], rounds=spec.get("rounds", 1),
            relax_delay=spec["relax_delay_us"], list_of_all_qubits=[row])
        config.update(add_qubit_experiment(expt_cfg, "qubit_spec_ge", row))
        config.update(res_freq_ge=spec["readout_mhz"],
                      qubit_gain_ge=spec["qubit_gain"],
                      qubit_length_ge=spec["qubit_length_us"])
    else:
        config.update(copy.deepcopy(expt_cfg["res_spec"]))
        config.update(reps=spec["reps"], rounds=spec.get("rounds", 1),
                      steps=spec["steps"], list_of_all_qubits=[row])
    config["relax_delay"] = spec["relax_delay_us"]
    assert config["res_length"] == spec["readout_length_us"]
    assert config["res_gain_ge"] == spec["readout_gain"]
    return config


def acquire(experiment, spec, config):
    if spec["kind"] == "qspec":
        program = PulseProbeSpectroscopyProgram(
            experiment.soccfg, reps=spec["reps"],
            final_delay=config["relax_delay"], cfg=config)
        iq = np.asarray(program.acquire(experiment.soc,
                        rounds=spec.get("rounds", 1), progress=True)[0][0])
        frequencies = np.asarray(program.get_pulse_param(
            "qubit_pulse", "freq", as_array=True), dtype=float)
        return frequencies, iq[:, 0], iq[:, 1]
    frequencies = np.linspace(spec["start_mhz"], spec["stop_mhz"], spec["steps"])
    iq = []
    for index, frequency in enumerate(frequencies):
        config["res_freq_ge"] = float(frequency)
        # The established resonator program prints each point's config.
        with contextlib.redirect_stdout(io.StringIO()):
            program = SingleToneSpectroscopyProgram(
                experiment.soccfg, reps=spec["reps"],
                final_delay=config["relax_delay"], cfg=config)
        point = np.asarray(program.acquire(experiment.soc,
                           rounds=spec.get("rounds", 1), progress=False)[0][0])
        iq.append(point.reshape(-1, 2)[0])
        if index % 20 == 0 or index == len(frequencies) - 1:
            print("Res progress %d/%d" % (index + 1, len(frequencies)), flush=True)
    iq = np.asarray(iq)
    return frequencies, iq[:, 0], iq[:, 1]


def save_trace(folders, spec, config, sources, index, arrays):
    frequencies, i_values, q_values = arrays
    if (len(frequencies) != spec["steps"] or not np.all(np.diff(frequencies) > 0)
            or not all(np.isfinite(a).all() for a in arrays)):
        raise RuntimeError("Malformed or nonfinite acquisition")
    if len(i_values) != len(frequencies) or len(q_values) != len(frequencies):
        raise RuntimeError("I/Q sample count differs from realized frequency axis")
    label = "QSpec_ge" if spec["kind"] == "qspec" else "Res_ge"
    keys = ["Dates", "Round Num", "Batch Num", "Frequencies", "I", "Q",
            "Exp Config", "Syst Config"]
    data = P.create_data_dict(keys)
    row = spec["row"]
    data = {row: data[row]}
    values = [time.time(), 1, index, frequencies, i_values, q_values,
              json.dumps(spec, sort_keys=True),
              json.dumps(dict(program_config=config, sources=sources,
                              nominal_attenuators_dB=ATTENUATORS,
                              attenuation_writes=False), default=str, sort_keys=True)]
    for key, value in zip(keys, values):
        data[row][key][0] = value
    P.save_h5(folders["subStudyDataFolder"], data, label, batch_num=index)
    h5_path = next((Path(folders["subStudyDataFolder"]) / "Data_h5" / label).glob(
        "*results_batch_%d_Num_per_batch1.h5" % index))
    with h5py.File(h5_path, "r") as saved:
        for key, expected in (("Frequencies", frequencies), ("I", i_values), ("Q", q_values)):
            actual = np.asarray(saved["Q%d" % (row + 1)][key][()], dtype=float).ravel()
            if not np.array_equal(actual, expected):
                raise RuntimeError("Saved raw data do not match acquisition")
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plot_path = Path(folders["studyDocumentationFolder"]) / (
        "R%d_Q%d_%s_%s.png" % (index, row + 1, stamp, spec["kind"]))
    fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    axes[0].plot(frequencies, i_values, label="I", linewidth=0.7)
    axes[0].plot(frequencies, q_values, label="Q", linewidth=0.7)
    axes[0].set_ylabel("Measured I / Q (a.u.)")
    axes[0].legend()
    magnitude = np.hypot(i_values, q_values)
    axes[1].plot(frequencies, magnitude, linewidth=0.8)
    axes[1].set_ylabel("Measured magnitude (a.u.)")
    axes[1].set_xlabel("%s frequency (MHz)" % (
        "Qubit-drive" if spec["kind"] == "qspec" else "Readout"))
    fig.suptitle("M%d / index %d: %s\nRO gain %.3f, %.1f us; %d reps x %d rounds" % (
        row + 1, row, spec["kind"], spec["readout_gain"],
        spec["readout_length_us"], spec["reps"], spec.get("rounds", 1)))
    if spec["kind"] == "qspec":
        axes[0].set_title("Drive gain %.4f / %.1f us; physical readout %.5f MHz; uncalibrated IQ" % (
            spec["qubit_gain"], spec["qubit_length_us"], spec["readout_mhz"]), fontsize=10)
        axes[1].axvline(5575, color="gray", linestyle=":", label="VNA M4 seed")
        axes[1].set_xlim(frequencies[0], frequencies[-1])
    fig.tight_layout()
    fig.savefig(plot_path, dpi=150)
    plt.close(fig)
    if not plot_path.is_file() or not h5_path.is_file():
        raise RuntimeError("Missing required plot or HDF5")
    result = dict(index=index, spec=spec, h5=str(h5_path), plot=str(plot_path),
                  realized_start_MHz=float(frequencies[0]),
                  realized_stop_MHz=float(frequencies[-1]),
                  realized_step_MHz=float(np.median(np.diff(frequencies))),
                  acceptance="diagnostic only; not a validated qubit frequency")
    if spec["kind"] == "res":
        smooth = savgol_filter(magnitude, min(9, len(magnitude) // 2 * 2 - 1), 2)
        result["dip_MHz"] = float(frequencies[np.argmin(smooth)])
        for side, mask in (("left", frequencies < result["dip_MHz"]),
                           ("right", frequencies > result["dip_MHz"])):
            slope = np.abs(np.gradient(smooth, frequencies))
            slope[~mask] = -np.inf
            result[side + "_slope_MHz"] = float(frequencies[np.argmax(slope)])
    print("SAVED " + json.dumps(result), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8-sig"))
    longest = validate_plan(plan)
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print("Plan validated offline; hardware was not contacted.")
        return
    sources = source_readbacks()
    run_flags = {"res_spec": any(s["kind"] == "res" for s in plan["runs"]),
                 "q_spec": any(s["kind"] == "qspec" for s in plan["runs"])}
    folders = P.setup_data_folders(STUDY, plan["sub_study"], json.dumps(plan, indent=2))
    manifest_path = Path(folders["studyDocumentationFolder"]) / "manifest.json"
    manifest = dict(plan=plan, source_readbacks=sources, run_flags=run_flags,
                    longest_artifact_path=longest, results=[], status="running")
    def checkpoint():
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    checkpoint()
    experiment = None
    try:
        experiment = QICK_experiment(folders["optimizationFolder"], fridge=FRIDGE,
                                     **ATTENUATORS)
        for index, spec in enumerate(plan["runs"], 1):
            print("START %d/%d %s" % (index, len(plan["runs"]), json.dumps(spec)), flush=True)
            config = selected_config(experiment, spec)
            manifest["active_index"] = index
            checkpoint()
            arrays = acquire(experiment, spec, config)
            result = save_trace(folders, spec, config, sources, index, arrays)
            manifest["results"].append(result)
            checkpoint()
        manifest["status"] = "complete"
    except BaseException as error:
        manifest["status"] = "failed"
        manifest["error"] = repr(error)
        raise
    finally:
        # Every reused program emits finite pulses. Never enable continuous RF
        # or write a flux-source setting, including during exception cleanup.
        manifest["final_source_readbacks"] = source_readbacks()
        manifest["finished"] = datetime.datetime.now().isoformat()
        checkpoint()
        if experiment is not None:
            experiment.soc._pyroRelease()
    print("MANIFEST " + str(manifest_path), flush=True)


if __name__ == "__main__":
    main()
