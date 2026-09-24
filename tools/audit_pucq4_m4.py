"""Inspect saved M4 round-robin data without importing acquisition code.

Reports raw-array integrity, actual configuration, fit quality and independent
spectroscopy/coherence fits. It never changes raw HDF5 or operates hardware.
"""

import argparse
import ast
import json
from pathlib import Path
import re
import sys

import h5py
import numpy as np
from scipy.optimize import curve_fit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_qspec_iq_series import decode_numeric, decode_text, joint_fit


ROOT = Path(r"M:\_Data\20250822 - Olivia\pucq4_run_started_Aug_3\PUCQ4\pucq4_first_light")
CONFIG_KEYS = (
    "qubit_freq_ge", "qubit_gain_ge", "qubit_length_ge", "sigma", "pi_amp",
    "res_freq_ge", "res_freq_offset_ge", "res_gain_ge", "res_length_ge",
    "res_length", "reps", "rounds", "steps", "relax_delay", "ramsey_freq",
    "start", "stop", "trig_time", "res_phase", "ro_phase",
    "fit_frequency_guess_MHz", "fitted_fringe_frequency_MHz",
    "unseeded_T2R_us", "unseeded_T2R_error_us",
)
FIELDS = {
    "qspec_ge": ("Frequencies", "I", "Q", "Population", "I Fit"),
    "rabi_ge": ("Gains", "I", "Q", "Population", "Fit"),
    "t1_ge": ("Delay Times", "I", "Q", "Population", "Fit", "T1", "Errors"),
    "t2_ge": ("Delay Times", "I", "Q", "Population", "Fit", "T2", "Errors"),
    "t2e_ge": ("Delay Times", "I", "Q", "Population", "Fit", "T2E", "Errors"),
    "ss_ge": ("Fidelity", "I_g", "Q_g", "I_e", "Q_e"),
    "res_ge": ("freq_pts", "freq_center", "Amps", "Found Freqs"),
}


def requested_state_calibration(runtime, environment, qubit_index, kind):
    """Follow the normal RR order for its shared Readout_Optimization settings."""
    target = kind if kind in ("qspec_ge", "rabi_ge", "ss_ge") else "coherence"
    prefixes = dict(qspec_ge="QSPEC", rabi_ge="RABI", ss_ge="", coherence="COHERENCE")
    if runtime is None:
        base = "PUCQ4_" + prefixes[target] + "_SSF" if prefixes[target] else "PUCQ4_SSF"
        return dict(steps=environment.get(base + "_SHOTS"),
                    relax_delay=environment.get(base + "_RELAX_DELAY_US"),
                    sources={})
    defaults = runtime["profiles"][str(qubit_index)]["Readout_Optimization"]
    settings = dict(steps=defaults["steps"], relax_delay=defaults["relax_delay"],
                    sources=dict(steps="row profile", relax_delay="row profile"))
    flags = runtime["run_flags"]
    phases = (("qspec_ge", flags.get("q_spec", False)),
              ("rabi_ge", flags.get("rabi", False)),
              ("ss_ge", flags.get("ss", False)),
              ("coherence", any(flags.get(f, False) for f in ("t1", "t2r", "t2e"))))
    for phase, enabled in phases:
        if enabled:
            base = "PUCQ4_" + prefixes[phase] + "_SSF" if prefixes[phase] else "PUCQ4_SSF"
            for key, suffix in (("steps", "_SHOTS"), ("relax_delay", "_RELAX_DELAY_US")):
                if environment.get(base + suffix) is not None:
                    settings[key] = environment[base + suffix]
                    settings["sources"][key] = base + suffix
        if phase == target:
            return settings
    raise ValueError("Unknown calibration stage: " + kind)


def r_squared(observed, fitted):
    return float(1 - np.sum((observed - fitted) ** 2)
                 / np.sum((observed - np.mean(observed)) ** 2))


def independent_coherence(t, population, echo_or_ramsey, frequency_hint):
    span = float(np.ptp(t))
    step = float(np.median(np.diff(t)))
    baseline = float(np.median(population[-max(10, len(t) // 5):]))
    if echo_or_ramsey:
        def model(x, tau, frequency, cosine, sine, decay_offset, offset):
            return offset + np.exp(-x / tau) * (
                cosine * np.cos(2 * np.pi * frequency * x)
                + sine * np.sin(2 * np.pi * frequency * x) + decay_offset)
        spectrum = np.abs(np.fft.rfft(population - np.mean(population)))
        fft_freq = np.fft.rfftfreq(len(t), d=step)
        candidates = list(fft_freq[np.argsort(spectrum[1:])[-3:] + 1])
        candidates.append(float(frequency_hint))
        lower = [step / 5, 0.001, -2, -2, -2, -1]
        upper = [span * 5, 0.49 / step, 2, 2, 2, 2]
        seeds = [[tau, freq, float(population[0] - baseline), 0, 0, baseline]
                 for tau in (span / 10, span / 3, span)
                 for freq in candidates if lower[1] < freq < upper[1]]
    else:
        def model(x, tau, amplitude, offset):
            return offset + amplitude * np.exp(-x / tau)
        lower, upper = [step / 5, -2, -1], [span * 5, 2, 2]
        seeds = [[tau, float(population[0] - baseline), baseline]
                 for tau in (span / 10, span / 3, span)]
    fits = []
    for seed in seeds:
        try:
            pars, cov = curve_fit(model, t, population, p0=seed,
                                  bounds=(lower, upper), maxfev=15000)
            fits.append((float(np.sum((population - model(t, *pars)) ** 2)), pars, cov))
        except (ValueError, RuntimeError):
            continue
    if not fits:
        return {"accepted": False, "reason": "Independent bounded fit failed"}
    _, pars, cov = min(fits, key=lambda item: item[0])
    error = float(np.sqrt(cov[0, 0]))
    fit_r2 = r_squared(population, model(t, *pars))
    report = dict(tau_us=float(pars[0]), tau_error_us=error, R2=fit_r2,
                  recorded_span_us=span, step_us=step,
                  accepted=bool(fit_r2 > 0.85 and step < pars[0] < span
                                and np.isfinite(error) and error < pars[0] * 0.2))
    if echo_or_ramsey:
        report.update(fringe_frequency_MHz=float(pars[1]),
                      points_per_fringe=float(1 / (step * pars[1])))
        report["accepted"] &= report["points_per_fringe"] >= 4
    return report


def audit(run, qubit_index=3, guard_name="m4_guard.json"):
    assert 0 <= qubit_index < 6
    guard_path = run / "documentation" / guard_name
    guard = json.loads(guard_path.read_text())
    assert guard["status"] == "complete", "Acquisition did not complete"
    for state in guard["before"] + guard["after"]:
        assert state["mode"] == "CURR" and state["current_A"] == 0 and not state["output"]
    if guard_name == "validation_guard.json":
        runtime = json.loads((run / "documentation/acquisition_profile.json").read_text())
        assert runtime["rounds"] == 1, "Held-bias diagnostics need separate analysis; final calibration evidence must be one pass"
        assert qubit_index in runtime["qubits"]
        bias = json.loads((run / "documentation/yoko_bias.json").read_text())
        final = json.loads((run / "documentation/yoko_final.json").read_text())
        for state, target in zip(bias, (runtime["yoko3_current_mA"], runtime["yoko4_current_mA"])):
            assert state["mode"] == "CURR" and abs(state["range_A"] - .01) < 1e-9
            assert abs(state["current_A"] - target * 1e-3) < 1e-7
            assert state["output"] == bool(target)
        assert all(s["current_A"] == 0 and not s["output"] for s in final)
    else:
        runtime = None
        assert str(qubit_index) == guard["plan"]["environment"]["PUCQ4_QS"]
    report = dict(run=str(run), qubit_index=qubit_index,
                  resonator_row="M%d" % (qubit_index + 1),
                  physical_number={4: "Q4", 5: "Q6"}.get(qubit_index, "unknown"),
                  experiments={}, final_sources=guard["after"])
    if runtime is not None:
        report.update(yoko3_current_mA=runtime["yoko3_current_mA"],
                      yoko4_current_mA=runtime["yoko4_current_mA"], bias_sources=bias)
    for kind, fields in FIELDS.items():
        files = list((run / "study_data/Data_h5" / kind).glob("*.h5"))
        if not files:
            continue
        assert len(files) == 1
        with h5py.File(files[0], "r") as handle:
            group = handle["Q%d" % (qubit_index + 1)]
            assert all(key in group for key in ("Dates", "Round Num", "Batch Num", "Exp Config", "Syst Config"))
            config_text = decode_text(group["Syst Config"])
            config = {}
            for key in CONFIG_KEYS:
                match = re.search(r"['\"]" + key + r"['\"]:\s*([-+0-9.eE]+)", config_text)
                if match:
                    config[key] = float(match[1])
            fit_start = re.search(r"['\"]fit_starting_values['\"]:\s*(\{[^{}]*\})", config_text)
            if fit_start:
                starting_values = ast.literal_eval(fit_start.group(1))
                assert set(starting_values) <= {"f", "phase", "amp", "initial_offset", "final_offset"}
                assert all(np.isfinite(value) for value in starting_values.values())
                config["fit_starting_values"] = starting_values
            assert all(key in group for key in fields), (kind, "Missing required raw field")
            data = {key: decode_numeric(group[key]) for key in fields}
            if kind == "t2_ge" and "fit_frequency_guess_MHz" in config:
                assert "Fit Unseeded" in group, "Configured fit must retain its automatic fit"
                data["Fit Unseeded"] = decode_numeric(group["Fit Unseeded"])
                assert data["Fit Unseeded"].shape == data["Fit"].shape
                assert np.all(np.isfinite(data["Fit Unseeded"]))
            calibration_fields = ("ss_I_g", "ss_Q_g", "ss_I_e", "ss_Q_e")
            has_calibration = all(key in group for key in calibration_fields)
            if "Population" in fields and kind != "qspec_ge":
                assert all(key in group for key in calibration_fields)
            if "Population" in fields and has_calibration:
                data.update({key: decode_numeric(group[key]) for key in calibration_fields})
            assert all(value.size and np.isfinite(value).all() for value in data.values())
        if "res_length_ge" in config:
            assert config["res_length"] == config["res_length_ge"]
        entry = dict(h5=str(files[0]), config=config, all_raw_finite=True)
        axis_key = {"qspec_ge": "Frequencies", "rabi_ge": "Gains",
                    "t1_ge": "Delay Times", "t2_ge": "Delay Times",
                    "t2e_ge": "Delay Times", "res_ge": "freq_pts"}.get(kind)
        if axis_key:
            axis = data[axis_key]
            if kind == "res_ge":
                assert len(data["freq_center"]) == 1
                axis = axis + data["freq_center"][0]
            entry["actual_axis"] = dict(name=axis_key, start=float(axis[0]),
                                        stop=float(axis[-1]), points=len(axis),
                                        step=float(np.median(np.diff(axis))))
        if "Population" in fields and has_calibration:
            counts = {key: len(data[key]) for key in calibration_fields}
            assert len(set(counts.values())) == 1
            requested = requested_state_calibration(
                runtime, guard["plan"]["environment"], qubit_index, kind)
            expected = requested["steps"]
            ground = np.mean(data["ss_I_g"] + 1j * data["ss_Q_g"])
            excited = np.mean(data["ss_I_e"] + 1j * data["ss_Q_e"])
            assert abs(excited - ground) > 0
            reconstructed = np.real(((data["I"] + 1j * data["Q"]) - ground)
                                    * np.conj(excited - ground)) / abs(excited - ground) ** 2
            assert np.allclose(reconstructed, data["Population"], atol=1e-5, rtol=1e-5)
            entry["state_calibration"] = dict(shots_per_state=counts["ss_I_g"],
                                               requested_shots_per_state=int(expected) if expected else None,
                                               matches_requested_shots=expected is None or counts["ss_I_g"] == int(expected),
                                               population_reconstruction_verified=True,
                                               centroid_separation=float(abs(excited-ground)),
                                               requested_settings_sources=requested["sources"])
            if requested["relax_delay"] is not None:
                entry["state_calibration"]["relax_delay_us"] = float(requested["relax_delay"])
        elif "Population" in fields:
            entry["state_calibration"] = dict(population_reconstruction_verified=False,
                                               reason="Historical qspec schema did not save calibration shots")
        if kind == "qspec_ge":
            x, pop = data["Frequencies"], data["Population"]
            assert np.all(np.diff(x) > 0)
            def lorentz(f, center, width, amplitude, offset):
                return offset + amplitude / (1 + ((f - center) / (width / 2)) ** 2)
            seed, span = float(np.mean([x[0], x[-1]])), float(np.ptp(x))
            pars, cov = curve_fit(lorentz, x, pop,
                                  p0=[x[np.argmax(pop)], 0.2, float(np.ptp(pop)), float(np.median(pop))],
                                  bounds=([x[0], np.median(np.diff(x)), 0, -1],
                                          [x[-1], span * 2, 2, 1]), maxfev=20000)
            raw_fit = joint_fit(x, data["I"], data["Q"], float(pars[0]), span / 2)
            entry.update(center_MHz=float(pars[0]), fwhm_MHz=float(pars[1]),
                         fwhm_error_MHz=float(np.sqrt(cov[1, 1])),
                         peak_population=float(pars[2] + pars[3]), baseline=float(pars[3]),
                         center_error_MHz=float(np.sqrt(cov[0, 0])),
                         R2=r_squared(pop, lorentz(x, *pars)),
                         RR_R2=r_squared(pop, data["I Fit"]),
                         raw_IQ={key: value for key, value in raw_fit.items() if np.isscalar(value)})
        elif kind == "rabi_ge":
            entry.update(R2=r_squared(data["Population"], data["Fit"]))
        elif kind in ("t1_ge", "t2_ge", "t2e_ge"):
            key = {"t1_ge": "T1", "t2_ge": "T2", "t2e_ge": "T2E"}[kind]
            entry.update(tau_us=float(data[key][0]), error_us=float(data["Errors"][0]),
                         R2=r_squared(data["Population"], data["Fit"]),
                         independent=independent_coherence(data["Delay Times"], data["Population"],
                                                           kind != "t1_ge", config.get("ramsey_freq", .5)))
            if "Fit Unseeded" in data:
                entry["fit_initialization"] = dict(
                    frequency_guess_MHz=config["fit_frequency_guess_MHz"],
                    starting_values=config.get("fit_starting_values", {"f": config["fit_frequency_guess_MHz"]}),
                    fitted_frequency_MHz=config["fitted_fringe_frequency_MHz"],
                    unseeded_tau_us=config["unseeded_T2R_us"],
                    unseeded_error_us=config["unseeded_T2R_error_us"],
                    unseeded_R2=r_squared(data["Population"], data["Fit Unseeded"]),
                    unseeded_fit_preserved=True,
                )
            if kind != "t1_ge":
                residual = data["Population"] - data["Fit"]
                spectrum = np.abs(np.fft.rfft(residual - residual.mean())) ** 2
                total_power = float(spectrum[1:].sum())
                entry["residual_peak_power_fraction"] = float(spectrum[1:].max() / total_power) if total_power else 0.0
                entry["residual_rms_per_fit_span"] = float(np.std(residual) / np.ptp(data["Fit"]))
        elif kind == "ss_ge":
            assert all(len(data[key]) == len(data["I_g"])
                       for key in ("Q_g", "I_e", "Q_e"))
            assert len(data["I_g"]) == int(config["steps"])
            entry.update(contrast=float(data["Fidelity"][0]), shots_per_state=len(data["I_g"]))
        elif kind == "res_ge":
            found = float(data["Found Freqs"][0])
            amplitudes = data["Amps"]
            assert len(amplitudes) == len(axis)
            index = int(np.argmin(np.abs(axis - found)))
            edge_count = max(3, len(axis) // 10)
            background = float(np.median(np.r_[amplitudes[:edge_count], amplitudes[-edge_count:]]))
            second_difference = np.diff(amplitudes, 2)
            noise = float(np.median(np.abs(second_difference - np.median(second_difference))) / .67449 / np.sqrt(6))
            contrast = abs(float(amplitudes[index]) - background)
            entry.update(found_frequency_MHz=found,
                         edge_distance_points=min(index, len(axis) - 1 - index),
                         resonance_contrast_to_noise=contrast / max(noise, np.finfo(float).eps))
        report["experiments"][kind] = entry
    flag_datasets = {"res_spec": "res_ge", "q_spec": "qspec_ge", "rabi": "rabi_ge",
                     "ss": "ss_ge", "t1": "t1_ge", "t2r": "t2_ge", "t2e": "t2e_ge"}
    enabled_flags = ([flag for flag, enabled in runtime["run_flags"].items() if enabled]
                     if runtime is not None else guard["plan"]["environment"]["PUCQ4_RUN_FLAGS"].split(","))
    required = {flag_datasets[flag] for flag in enabled_flags
                if flag in flag_datasets}
    assert required <= set(report["experiments"]), "Requested datasets did not persist"
    report["plots"] = [str(path) for path in (run / "documentation").rglob("*.png")
                       if "_Q_%d_" % (qubit_index + 1) in path.name]
    assert report["plots"], "No plots persisted"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sub_study")
    parser.add_argument("--qubit-index", type=int, default=3)
    parser.add_argument("--guard-name", default="m4_guard.json")
    parser.add_argument("--report-name", default="m4_audit.json")
    args = parser.parse_args()
    root = ROOT / args.sub_study
    run = max(root.glob("*/documentation/" + args.guard_name), key=lambda path: path.stat().st_mtime).parents[1]
    report = audit(run, args.qubit_index, args.guard_name)
    (run / "documentation" / args.report_name).write_text(json.dumps(report, indent=2))
    compact = {kind: {key: value for key, value in record.items() if key not in ("h5", "config")}
               for kind, record in report["experiments"].items()}
    print(json.dumps(dict(run=str(run), experiments=compact), indent=2))


if __name__ == "__main__":
    main()
