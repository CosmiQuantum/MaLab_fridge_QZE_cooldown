"""Compare new canonical RR traces with the September 8 PowerPoint evidence.

Offline only. Reads HDF5 and guard/configuration records; preserves raw data.
"""
import argparse
import json
from pathlib import Path
import sys

import h5py
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.audit_pucq4_m4 import ROOT, audit
from tools.analyze_qspec_iq_series import decode_numeric

KINDS = ("res_ge", "qspec_ge", "rabi_ge", "ss_ge", "t1_ge", "t2_ge", "t2e_ge")
COHERENCE = {"t1_ge": ("t1", "T1"), "t2_ge": ("t2r", "T2"), "t2e_ge": ("t2e", "T2E")}
FOLDERS = dict(res_ge="res_spec_ge_plots", qspec_ge="qubit_spec_ge_plots",
               rabi_ge="power_rabi_ge_plots", ss_ge="ss_repeat_meas_ge", t1_ge="T1_ge",
               t2_ge="Ramsey_ge_1", t2e_ge="SpinEcho_ge")
ARTIFACTS = ROOT / "PUCQ4_round_robin_validation_20260908_artifacts"


def reference_for(q, current):
    zero = json.loads((ROOT / "PUCQ4_chip_summary_20260908_artifacts/summary_provenance.json").read_text())
    if q < 4 or current == 0:
        row = zero["records"][str(q + 1)]
        ref = dict(qfreq_MHz=row["rabi_ge"]["config"]["qubit_freq_ge"],
                   sigma_us=row["rabi_ge"]["config"]["sigma"],
                   pi_gain=row["t1_ge"]["config"]["pi_amp"],
                   res_MHz=row["res_ge"]["values"]["Found Freqs"][0],
                   ssf_contrast=row.get("ss_ge", {}).get("values", {}).get("Fidelity", [None])[0],
                   source="PUCQ4_chip_summary_20260908.pptx")
        for kind, (_, field) in COHERENCE.items():
            ref[kind] = dict(tau_us=row[kind]["values"][field][0],
                             error_us=row[kind]["values"]["Errors"][0], h5=row[kind]["h5"])
        ref["qspec"] = dict(zero["zero_current_linewidths"][str(q + 1)],
                            resolved=True, h5=row["qspec_ge"]["h5"])
    else:
        data = json.loads((ROOT / "yoko_flux_campaign/ramsey_sigma_coherence_manifest.json").read_text())
        row = data["devices"]["q4" if q == 4 else "q6"]["map"]["points"]["%+.1f" % current]
        selected = row["sigma_calibration"]["selected"]
        ref = dict(qfreq_MHz=selected["qfreq_MHz"], sigma_us=selected["sigma_us"],
                   pi_gain=selected["pi_gain"], res_MHz=selected["res_base_MHz"],
                   ssf_contrast=selected.get("readout", {}).get("fidelity"),
                   source="ramsey_sigma_coherence_manifest.json")
        for kind, (flag, _) in COHERENCE.items():
            item = row["coherence"]["experiments"][flag]
            ref[kind] = dict(tau_us=item["fit_value_us"], error_us=item["fit_error_us"], h5=item["h5"])
        title = "Yoko %d / physical Q%d / M%d | selected pulse settings at every current" % (
            3 if q == 4 else 4, 4 if q == 4 else 6, q + 1)
        ref["qspec"] = zero["yoko_linewidth_table_provenance"][title]["%+.1f" % current]
    return ref


def analyze(run, qubits, audit_suffix=""):
    assert not audit_suffix or audit_suffix.replace("_", "").isalnum()
    runtime = json.loads((Path(run) / "documentation/acquisition_profile.json").read_text())
    reports = []
    for q in qubits:
        result = audit(run, q, "validation_guard.json")
        current = result["yoko3_current_mA"] if q == 4 else result["yoko4_current_mA"] if q == 5 else 0
        ref = reference_for(q, current)
        result.update(reference=ref, current_mA=current, issues=[], comparisons={},
                      yoko_settle_seconds=runtime.get(
                          "yoko_settle_seconds", 600 if result["yoko3_current_mA"] == -10 else 0))
        entries, issues = result["experiments"], result["issues"]
        for kind, entry in entries.items():
            matching = [p for p in result["plots"] if FOLDERS.get(kind) in Path(p).parts]
            if matching:
                entry["plot"] = max(matching, key=lambda p: Path(p).stat().st_mtime)
            else:
                issues.append(kind + ": normal plot missing")
            if "state_calibration" in entry:
                cal = entry["state_calibration"]
                if not cal["population_reconstruction_verified"] or not cal["matches_requested_shots"]:
                    issues.append(kind + ": state calibration mismatch")
        if "res_ge" in entries:
            resonator = entries["res_ge"]
            if resonator["edge_distance_points"] < 3 or resonator["resonance_contrast_to_noise"] < 6:
                issues.append("res spec: resonance is unresolved or at a scan boundary")
        if "qspec_ge" in entries:
            spec = entries["qspec_ge"]
            spacing = spec["actual_axis"]["step"]
            spec["samples_per_FWHM"] = spec["fwhm_MHz"] / spacing
            if spec["R2"] < .85 or spec["fwhm_error_MHz"] > .2 * spec["fwhm_MHz"]:
                issues.append("qspec: line fit requires review")
            if spec["samples_per_FWHM"] < 4:
                issues.append("qspec: underresolved linewidth")
            raw_iq = spec["raw_IQ"]
            if raw_iq["vector_snr"] < 6 or raw_iq["residual_rms_sigma"] > 2:
                issues.append("qspec: raw I/Q line or residuals require review")
            if abs(spec["center_MHz"] - raw_iq["center_MHz"]) > max(
                    .1 * spec["fwhm_MHz"], 3 * spec["center_error_MHz"]):
                issues.append("qspec: raw I/Q and population centers disagree")
            if min(spec["center_MHz"] - spec["actual_axis"]["start"],
                   spec["actual_axis"]["stop"] - spec["center_MHz"]) < spec["fwhm_MHz"]:
                issues.append("qspec: insufficient baseline around the fitted line")
            prior_spec = ref["qspec"]
            # Compare spectroscopy with spectroscopy. The previous pulse
            # frequency may instead come from a later Ramsey correction.
            comparison = dict(previous_pulse_frequency_MHz=ref["qfreq_MHz"],
                              new_center_MHz=spec["center_MHz"],
                              delta_from_pulse_frequency_MHz=spec["center_MHz"] - ref["qfreq_MHz"],
                              previous_qspec_resolved=prior_spec["resolved"])
            if prior_spec["resolved"]:
                delta = spec["center_MHz"] - prior_spec["center_MHz"]
                limit = max(.1, prior_spec["fwhm_MHz"], spec["fwhm_MHz"])
                comparison.update(previous_qspec_center_MHz=prior_spec["center_MHz"],
                                  delta_from_qspec_MHz=delta, tolerance_MHz=limit,
                                  roughly_matches=abs(delta) <= limit)
                if abs(delta) > limit:
                    issues.append("qspec: frequency shift exceeds reference linewidth")
            result["comparisons"]["qspec_ge"] = comparison
        if "rabi_ge" in entries:
            rabi = entries["rabi_ge"]
            with h5py.File(rabi["h5"], "r") as handle:
                g = handle["Q%d" % (q + 1)]
                gains, fitted = decode_numeric(g["Gains"]), decode_numeric(g["Fit"])
                rabi["pi_gain_from_curve"] = float(gains[np.argmax(fitted)])
            propagated = next((entries[k]["config"]["pi_amp"] for k in ("ss_ge", "t1_ge", "t2_ge", "t2e_ge") if k in entries), None)
            rabi["measured_pi_gain"] = propagated if propagated is not None else rabi["pi_gain_from_curve"]
            if not .6 <= rabi["measured_pi_gain"] <= .7 or rabi["R2"] < .9:
                issues.append("Rabi: pi amplitude or contrast requires review")
            if abs(rabi["measured_pi_gain"] - rabi["pi_gain_from_curve"]) > rabi["actual_axis"]["step"]:
                issues.append("Rabi: propagated pi amplitude differs from the saved curve")
            for kind in ("ss_ge", "t1_ge", "t2_ge", "t2e_ge"):
                if kind not in entries:
                    continue
                config = entries[kind]["config"]
                if not np.isclose(config["pi_amp"], rabi["measured_pi_gain"], rtol=1e-8):
                    issues.append(kind + ": measured pi amplitude was not propagated")
                if abs(config["qubit_freq_ge"] - rabi["config"]["qubit_freq_ge"]) > 1e-6:
                    issues.append(kind + ": frequency differs from the Rabi calibration")
        if "ss_ge" in entries and ref.get("ssf_contrast") is not None:
            previous = ref["ssf_contrast"]
            observed = entries["ss_ge"]["contrast"]
            if previous - observed > max(.03, .2 * previous):
                issues.append("SSF: contrast fell more than the comparison tolerance")
        for kind in COHERENCE:
            if kind not in entries:
                continue
            entry, prior = entries[kind], ref[kind]
            change = (entry["tau_us"] / prior["tau_us"] - 1) * 100
            limit = max(.3 * prior["tau_us"], 3 * np.hypot(entry["error_us"], prior["error_us"]))
            match = abs(entry["tau_us"] - prior["tau_us"]) <= limit
            result["comparisons"][kind] = dict(previous_us=prior["tau_us"], new_us=entry["tau_us"],
                                               change_percent=change, roughly_matches=bool(match))
            if not match:
                issues.append(kind + ": coherence outside comparison band")
            if entry["R2"] < .85 or not entry["independent"]["accepted"]:
                issues.append(kind + ": independent fit requires review")
            if not 0 < entry["error_us"] < .2 * entry["tau_us"]:
                issues.append(kind + ": saved fit uncertainty exceeds acceptance")
            if entry.get("residual_peak_power_fraction", 0) > .25 and entry.get("residual_rms_per_fit_span", 0) > .03:
                issues.append(kind + ": structured oscillations remain in fit residuals")
            if "tau_us" in entry["independent"] and abs(entry["tau_us"] - entry["independent"]["tau_us"]) > max(
                    3 * np.hypot(entry["error_us"], entry["independent"]["tau_error_us"]), .1 * entry["tau_us"]):
                issues.append(kind + ": saved and independent fits disagree")
        result["complete_chain"] = set(KINDS) <= set(entries)
        result["status"] = "pass" if not issues else "review"
        target = run / "documentation" / ("validation_M%d%s.json" % (q + 1, audit_suffix))
        result["audit_path"] = str(target)
        target.write_text(json.dumps(result, indent=2))
        reports.append(result)
        print(json.dumps(dict(row="M%d" % (q + 1), current_mA=current,
                              status=result["status"], issues=issues,
                              qspec={k: entries.get("qspec_ge", {}).get(k) for k in ("center_MHz", "fwhm_MHz", "R2")},
                              pi_gain=entries.get("rabi_ge", {}).get("measured_pi_gain"),
                              coherence=result["comparisons"])), flush=True)
    return reports


def record_attempt(reports):
    """Keep every attempt; automatically select only rows that pass all gates."""
    path = ARTIFACTS / "validation_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8-sig")) if path.exists() else dict(
        task_status="running", zero={}, bias={"4": {}, "5": {}}, controls=[], attempts=[])
    for result in reports:
        q, current = result["qubit_index"], result["current_mA"]
        source = result.get("audit_path", str(Path(result["run"]) / "documentation" / ("validation_M%d.json" % (q + 1))))
        item = dict(audit=source, qubit_index=q, current_mA=current,
                    yoko3_current_mA=result["yoko3_current_mA"],
                    yoko4_current_mA=result["yoko4_current_mA"], status=result["status"],
                    complete_chain=result["complete_chain"])
        if not any(old["audit"] == source for old in manifest["attempts"]):
            manifest["attempts"].append(item)
        zero = result["yoko3_current_mA"] == result["yoko4_current_mA"] == 0
        is_bias_row = q in (4, 5) and current and (
            result["yoko4_current_mA"] if q == 4 else result["yoko3_current_mA"]) == 0
        if result["issues"] and (zero or is_bias_row):
            selected = manifest["zero"] if zero else manifest["bias"][str(q)]
            key = str(q) if zero else "%+.1f" % current
            old = selected.get(key)
            previous = json.loads(Path(old).read_text(encoding="utf-8-sig")) if old else None
            new_findings = set(result["issues"]) - set(previous.get("original_issues", [])) if previous else set()
            if previous and previous["run"] == result["run"] and new_findings:
                # A stricter review of the same raw pass supersedes its old approval.
                # Preserve the earlier audit/attempt and any accepted different run.
                del selected[key]
        if not zero and not is_bias_row and result["complete_chain"]:
            # Keep failed endpoint controls visible as well as passing ones.
            if not any(old["audit"] == source for old in manifest["controls"]):
                manifest["controls"].append(item)
        if result["status"] == "pass" and result["complete_chain"]:
            if zero:
                manifest["zero"][str(q)] = source
            elif is_bias_row:
                manifest["bias"][str(q)]["%+.1f" % current] = source
    ARTIFACTS.mkdir(exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2))
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--qubits", default="0,1,2,3,4,5")
    parser.add_argument("--audit-suffix", default="", help="Preserve an earlier audit when adding a later review")
    args = parser.parse_args()
    record_attempt(analyze(args.run, [int(q) for q in args.qubits.split(",")], args.audit_suffix))
