"""Bounded Q6 bias spectroscopy using the existing QICK programs and HDF5 writer.

No hardware is contacted on import. A JSON plan selects each bias and frequency
window explicitly. Physical Q5 is not assigned to a resonator without evidence.
The shared drive line addresses frequency, not an independent spatial port.
"""

import argparse
import copy
import hashlib
import json
import logging
import math
import os
import time
import traceback
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("M:/_Data/20250822 - Olivia/pucq4_run_started_Oct_7/PUCQ4/pucq4_first_light")
YOKOS = ("192.168.1.73", "192.168.1.77")
LIMIT_A = 0.010
RAMP_A_S = 0.0005
TOLERANCE_A = 1e-7


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


def validate_plan(plan):
    if not plan.get("authorization") or not plan.get("calibration_evidence"):
        raise ValueError("Plan needs authorization and calibration provenance")
    if not str(plan["label"]).replace("_", "").isalnum() or len(plan["label"]) > 18:
        raise ValueError("Use a short sub-study label")
    if plan.get("anchor_mA") != -5.0 or plan.get("settle_seconds") != 0:
        raise ValueError("This protocol uses the measured -5 mA anchor and established zero extra settling")
    if not 60 <= plan["deadline_seconds"] <= 43200:
        raise ValueError("Deadline out of bounds")
    if not 1 <= len(plan["points"]) <= 121:
        raise ValueError("Bias point count out of bounds")
    for point in plan["points"]:
        current = point["current_mA"]
        if not math.isfinite(current) or not -5.8 <= current <= -4.8:
            raise ValueError("Bias outside the localized fixed-qubit survey envelope")
        if not point["traces"]:
            raise ValueError("Each bias needs an explicit trace")
        for trace in point["traces"]:
            if trace["row"] not in (1, 2, 3, 4, 6):
                raise ValueError("Only Q6 and the four unassigned fixed rows are allowed")
            lo, hi = trace["start_MHz"], trace["stop_MHz"]
            if not 5300 <= lo < hi <= 5660 or hi - lo > 250:
                raise ValueError("Frequency outside measured fixed-qubit neighborhood")
            if not 101 <= trace["steps"] <= 12501 or not 500 <= trace["reps"] <= 64000:
                raise ValueError("Unbounded point count or averaging")
            if not 0 < trace["gain"] <= 0.03 or trace["length_us"] != 20:
                raise ValueError("Pulse exceeds prior spectroscopy envelope")
            if trace.get("relax_us", 100) != 100:
                raise ValueError("Retain established 100 us qspec relaxation")
    longest = ROOT / plan["label"] / "2099-12-31_23-59-59" / "study_data/Data_h5/qspec_ge" / "2099-12-31_23-59-59_qspec_ge_results_batch_9999_Num_per_batch1.h5"
    if len(str(longest)) >= 260:
        raise ValueError("Windows artifact path exceeds 259 characters")
    return str(longest)


def instrument_state(source):
    return dict(address=source.address, identity=source.get_id(),
                mode=source.get_mode(), range_A=source.get_range(),
                current_A=source.get_current(), output=source.get_output(),
                voltage_limit_V=source.get_voltage_limit(),
                current_readback_kind=":SOURCE:LEVEL? setting, not an independent ammeter")


def require_parked(state):
    if (state["mode"] != "CURR" or abs(state["current_A"]) > TOLERANCE_A
            or state["output"] or abs(state["range_A"] - LIMIT_A) > 1e-9):
        raise RuntimeError("Expected current mode, 10 mA range, zero current and output off: %s" % state)


def park(source):
    if source.get_mode() != "CURR" or abs(source.get_range() - LIMIT_A) > 1e-9:
        raise RuntimeError("Unexpected source mode/range; refusing blind cleanup")
    source.ramp_current(0.0, sweeprate=RAMP_A_S)
    source.set_output(False)
    state = instrument_state(source)
    require_parked(state)
    return state


class Acquisition:
    def __init__(self, plan, run):
        from system_config import QICK_experiment
        from build_state import all_qubit_state
        self.plan, self.run = plan, run
        self.documentation = run / "documentation"
        self.batch = 0
        self.started = time.monotonic()
        self.manifest = dict(plan=plan, run=str(run), status="preflight", records=[], readout_updates=[])
        self.experiment = QICK_experiment(str(run), DAC_attenuator1=10,
                DAC_attenuator2=15, qubit_DAC_attenuator1=5,
                qubit_DAC_attenuator2=4, ADC_attenuator=17,
                fridge="QUIET", yoko3_current_mA=0.0, yoko4_current_mA=-5.0)
        self.states = all_qubit_state(self.experiment, 6)
        self.initial_states = copy.deepcopy(self.states)
        self.manifest["soccfg"] = str(self.experiment.soccfg)
        self.manifest["initial_program_states"] = self.initial_states
        self.manifest["bias_metadata_note"] = "System calibration anchor is -5 mA; actual bias is stored per acquisition and never represented as an interpolated calibration."
        self.manifest["attenuation_note"] = "Constructor arguments 10/15, 5/4, 17 dB retained; RF-board setter calls are disabled in system_config.py; these are nominal, not fresh attenuation readbacks."
        if self.states["Q5"]["qubit_ch"] != 1 or self.states["Q5"]["res_ch"] != 0 or self.states["Q5"]["ro_ch"] != 0:
            raise RuntimeError("Hardware channels differ from inspected configuration")
        self.flush()

    def flush(self):
        write_json(self.documentation / "manifest.json", self.manifest)

    def health(self):
        if time.monotonic() - self.started > self.plan["deadline_seconds"]:
            raise TimeoutError("Acquisition deadline")
        if (self.run / "STOP").exists():
            raise InterruptedError("Stop requested")

    def save(self, row, label, values, config, bias, acquisition_kind):
        from pucq4_config import create_data_dict, save_h5
        # Persist an acquisition that has already returned, even if a stop was
        # requested during it. The next acquisition checks health before starting.
        keys = ["Dates", "Round Num", "Batch Num", "Exp Config", "Syst Config"] + list(values)
        data = create_data_dict(keys, save_r=1, qubits=6)
        selected = data[row - 1]
        selected["Dates"][0] = time.time()
        selected["Round Num"][0] = 1
        selected["Batch Num"][0] = self.batch
        metadata = dict(config=config, acquisition_kind=acquisition_kind,
                        bias_readback=bias, calibration_anchor_mA=-5.0,
                        actual_yoko4_current_mA=bias["current_A"] * 1000,
                        readout_row_one_based=row, physical_drive="shared DAC1; frequency-selective spectroscopy",
                        physical_Q5_row="unassigned pending evidence")
        selected["Exp Config"][0] = json.dumps(metadata, default=str)
        selected["Syst Config"][0] = json.dumps(dict(hw=self.experiment.hw_cfg,
                readout=self.experiment.readout_cfg, qubit=self.experiment.qubit_cfg,
                actual_bias=bias, calibration_anchor_mA=-5.0), default=str)
        for key, value in values.items():
            selected[key][0] = value
        folder = self.run / "study_data/Data_h5" / label
        existing = set(folder.glob("*.h5"))
        save_h5(str(self.run / "study_data"), {row - 1: selected}, label, self.batch, 1)
        fresh = set(folder.glob("*.h5")) - existing
        if len(fresh) != 1:
            raise RuntimeError("Expected one newly saved HDF5")
        saved = fresh.pop()
        with h5py.File(saved, "r") as handle:
            group = handle["Q%d" % row]
            for key in keys:
                if key not in group:
                    raise RuntimeError("Missing HDF5 field " + key)
        self.batch += 1
        logging.info("Saved M%d %s %s: %s", row, bias["current_A"], acquisition_kind, saved)
        return str(saved)

    def readout(self, source, row, current):
        from section_002_res_spec_ge_mux import SingleToneSpectroscopyProgram
        from expt_config import expt_cfg
        config = copy.deepcopy(self.states["Q%d" % (row - 1)])
        sweep = expt_cfg["res_spec"]
        # Keep the calibrated absolute tone if drift is smaller than a scan bin.
        center = float(config["res_freq_ge"])
        freqs = center + sweep["start"] + sweep["step_size"] * np.arange(sweep["steps"])
        iq = []
        for freq in freqs:
            self.health()
            config["res_freq_ge"] = float(freq)
            program = SingleToneSpectroscopyProgram(self.experiment.soccfg,
                reps=sweep["reps"], final_delay=sweep["relax_delay"], cfg=config)
            iq.append(np.asarray(program.acquire(self.experiment.soc, rounds=1, progress=False)[0]).reshape(-1, 2).mean(axis=0))
        iq = np.asarray(iq)
        magnitude = np.linalg.norm(iq, axis=1)
        minimum = int(np.argmin(magnitude))
        if minimum < 3 or minimum >= len(freqs) - 3 or not np.all(np.isfinite(iq)):
            raise RuntimeError("Readout resonance absent or at scan boundary on M%d" % row)
        found = float(freqs[minimum])
        old_tone = center + float(config["res_freq_offset_ge"])
        new_tone = found + float(config["res_freq_offset_ge"])
        # Preserve the previous tone for sub-bin fluctuations; otherwise track.
        changed = abs(found - center) > sweep["step_size"] * 1.1
        if not changed:
            new_tone = old_tone
        state = self.states["Q%d" % (row - 1)]
        state["res_freq_ge"] = found if changed else center
        bias = instrument_state(source)
        config.update(reps=sweep["reps"], rounds=1, relax_delay=sweep["relax_delay"])
        path = self.save(row, "res_ge", dict(freq_pts=freqs, I=iq[:, 0], Q=iq[:, 1],
                        Amps=magnitude, **{"Found Freqs": np.array([found])}), config, bias, "readout tracking")
        fig, axis = plt.subplots(figsize=(8, 4))
        axis.plot(freqs, magnitude)
        axis.axvline(new_tone, color="C1", label="selected readout")
        axis.set(xlabel="Readout frequency (MHz)", ylabel="Magnitude (ADC units)",
                 title="M%d resonator; Yoko4 %.6f mA" % (row, current))
        axis.legend(); fig.tight_layout()
        plot = self.documentation / ("R_1_Q_%d_%04d_res_spec.png" % (row, self.batch))
        fig.savefig(plot, dpi=150); plt.close(fig)
        self.manifest["readout_updates"].append(dict(row=row, current_mA=current,
                old_tone_MHz=old_tone, new_tone_MHz=new_tone, found_MHz=found,
                gain=state["res_gain_ge"], length_us=state["res_length"], h5=path, plot=str(plot)))
        logging.info("Readout M%d at %.6f mA: %.6f -> %.6f MHz", row, current, old_tone, new_tone)
        self.flush()
        return new_tone

    def qspec(self, source, row, trace, tone, current):
        from section_004_qubit_spec_ge import PulseProbeSpectroscopyProgram
        from build_task import add_qubit_experiment
        from expt_config import expt_cfg
        # Reuse the repository sweep builder and Q6 qubit state; select the
        # actual resonator state explicitly so its integration length survives.
        config = copy.deepcopy(self.states["Q5"])
        ro = self.states["Q%d" % (row - 1)]
        for key in ("res_gain_ge", "res_length", "res_length_ge", "res_phase", "res_freq_offset_ge", "trig_time", "ro_phase"):
            config[key] = ro[key]
        config["res_freq_ge"] = tone
        registry = copy.deepcopy(expt_cfg)
        registry["qubit_spec_ge"].update(start=trace["start_MHz"], stop=trace["stop_MHz"],
                    steps=trace["steps"], reps=trace["reps"], rounds=1,
                    relax_delay=100, list_of_all_qubits=[5])
        config.update(add_qubit_experiment(registry, "qubit_spec_ge", 5))
        config.update(qubit_gain_ge=trace["gain"], qubit_length_ge=trace["length_us"])
        assert float(config["res_length"]) == float(ro["res_length_ge"])
        record = dict(row=row, current_mA=current, requested=trace, readout_MHz=tone,
                      status="acquiring", traces={})
        self.manifest["records"].append(record); self.flush()
        measured = {}
        for kind, gain in (("drive_off", 0.0), ("drive_on", trace["gain"])):
            self.health()
            config["qubit_gain_ge"] = gain
            before = instrument_state(source)
            if abs(before["current_A"] - current / 1000) > TOLERANCE_A or not before["output"]:
                raise RuntimeError("Bias readback failed before qspec")
            time.sleep(config["relax_delay"] * 1e-6)
            program = PulseProbeSpectroscopyProgram(self.experiment.soccfg,
                  reps=trace["reps"], final_delay=config["relax_delay"], cfg=config)
            print("ACQUIRE", current, "M%d" % row, kind, trace, flush=True)
            logging.info("Acquire %.6f mA M%d %s: %s", current, row, kind, trace)
            iq = np.asarray(program.acquire(self.experiment.soc, rounds=1, progress=True)[0][0]).reshape(-1, 2)
            frequencies = np.asarray(program.get_pulse_param("qubit_pulse", "freq", as_array=True), dtype=float)
            raw = np.asarray(program.get_raw()[0])
            after = instrument_state(source)
            if abs(after["current_A"] - current / 1000) > TOLERANCE_A or not after["output"]:
                raise RuntimeError("Bias changed during qspec")
            if iq.shape != (trace["steps"], 2) or not np.all(np.isfinite(iq)):
                raise RuntimeError("Invalid IQ data")
            path = self.save(row, "qspec_ge", dict(Frequencies=frequencies, I=iq[:, 0],
                    Q=iq[:, 1], **{"Raw IQ": raw, "Current mA": np.array([current]),
                    "Drive Gain": np.array([gain])}), config, after, kind)
            measured[kind] = iq
            record["traces"][kind] = path; self.flush()
        delta = measured["drive_on"] - measured["drive_off"]
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        axes[0].plot(frequencies, delta[:, 0], label="delta I")
        axes[0].plot(frequencies, delta[:, 1], label="delta Q")
        axes[0].legend(); axes[0].set_ylabel("Drive on - off (ADC units)")
        axes[1].plot(frequencies, np.linalg.norm(delta, axis=1))
        axes[1].set(xlabel="Spectroscopy frequency (MHz)", ylabel="Magnitude of delta IQ")
        axes[0].set_title("M%d readout; Yoko4 %.6f mA; shared drive gain %.5g" % (row, current, trace["gain"]))
        fig.tight_layout()
        plot = self.documentation / ("R_1_Q_%d_%04d_qspec.png" % (row, self.batch))
        fig.savefig(plot, dpi=150); plt.close(fig)
        if not plot.is_file():
            raise RuntimeError("Missing qspec plot")
        record.update(status="complete", plot=str(plot), bias_after=after)
        self.flush()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("plan")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    longest = validate_plan(plan)
    print("Longest artifact path", len(longest), longest, flush=True)
    if args.validate_only:
        return
    from malab_yokogawa_gs200 import YokogawaGS200
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run = ROOT / plan["label"] / stamp
    run.mkdir(parents=True, exist_ok=False)
    for name in ("documentation", "study_data", "optimization"):
        (run / name).mkdir()
    logging.basicConfig(filename=str(run / "documentation/acquisition.log"),
                        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("Run %s; plan %s", run, plan)
    (run / "documentation/sub_study_notes.txt").write_text(plan["authorization"], encoding="utf-8")
    (run / "documentation/plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")
    source_files = [Path(__file__), Path("system_config.py"), Path("expt_config.py"),
                    Path("section_004_qubit_spec_ge.py"), Path("section_002_res_spec_ge_mux.py"),
                    Path("malab_yokogawa_gs200.py"), Path("build_state.py"), Path("build_task.py")]
    hashes = {}
    for path in source_files:
        blob = path.read_bytes()
        (run / "documentation" / path.name).write_bytes(blob)
        hashes[path.name] = hashlib.sha256(blob).hexdigest()
    write_json(run / "documentation/source_hashes.json", hashes)
    write_json(Path(args.plan).with_suffix(".runtime.json"), dict(pid=os.getpid(), run=str(run), status="starting"))
    print("RUN", run, flush=True)
    acquisition = None
    with YokogawaGS200(YOKOS[0], max_current=LIMIT_A) as inactive, YokogawaGS200(YOKOS[1], max_current=LIMIT_A) as source:
        initial = [instrument_state(inactive), instrument_state(source)]
        write_json(run / "documentation/initial_sources.json", initial)
        for state in initial:
            require_parked(state)
        try:
            acquisition = Acquisition(plan, run)
            acquisition.manifest["initial_sources"] = initial
            acquisition.manifest["status"] = "acquiring"
            acquisition.flush()
            source.set_output(True)
            for point in plan["points"]:
                acquisition.health()
                require_parked(instrument_state(inactive))
                current = point["current_mA"]
                source.ramp_current(current / 1000, sweeprate=RAMP_A_S)
                if abs(source.get_current() - current / 1000) > TOLERANCE_A:
                    raise RuntimeError("Ramp endpoint mismatch")
                # No extra DC hold was established for Yoko4 in this range.
                # Readout tracking is performed before each spectroscopy row.
                tones = {}
                for trace in point["traces"]:
                    row = trace["row"]
                    if row not in tones:
                        tones[row] = acquisition.readout(source, row, current)
                    acquisition.qspec(source, row, trace, tones[row], current)
            acquisition.manifest["status"] = "complete"
        except BaseException as error:
            logging.exception("Acquisition failed")
            if acquisition is not None:
                acquisition.manifest.update(status="failed", error=repr(error), traceback=traceback.format_exc())
            raise
        finally:
            final = [instrument_state(inactive), park(source)]
            require_parked(final[0])
            write_json(run / "documentation/final_sources.json", final)
            if acquisition is not None:
                acquisition.manifest["final_sources"] = final
                acquisition.flush()
            write_json(Path(args.plan).with_suffix(".runtime.json"), dict(pid=os.getpid(), run=str(run),
                status=acquisition.manifest["status"] if acquisition else "failed", final_sources=final))
            print("FINAL", final, flush=True)
            logging.info("Final source states: %s", final)


if __name__ == "__main__":
    main()
