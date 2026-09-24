"""Assess a predeclared fixed-profile map and retain the existing coupling gates."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from check_pucq4_profile_pilot import assess_blocks, fit_profile
from diagnose_pucq4_profile_sensitivity import load_blocks
from analyze_pucq4_crossing import coupled_fit, coupled_plot, line_candidates, recorded_current_mA, read_record


def assess_record(record, expected, family):
    if record["status"] != "complete" or record["row"] != expected["trace"]["row"]:
        raise ValueError("Incomplete/unexpected map record")
    for key, value in expected["trace"].items():
        if record["requested"].get(key) != value:
            raise ValueError("Record differs from frozen plan: " + key)
    f, delta, blocks, norm = load_blocks(record)
    _, _, saved = read_record(record)
    current = recorded_current_mA(saved)
    if abs(current - expected["current_mA"]) > 1e-4:
        raise ValueError("Unexpected recorded bias")
    prediction = expected["prediction"]
    result = assess_blocks(f, blocks["drive_on"], blocks["drive_off"], prediction, family=family)
    result.update(current_mA=current, branch=expected["branch"], traces=record["traces"],
                  normalization=norm, fit=fit_profile(f, delta, prediction),
                  original_metrics=line_candidates(f, delta)[0])
    result["window_pass"] = bool(result["fixed_profile_controls_pass"] and result["fit"]["parameter_quality_pass"])
    return result


def pair_results(results):
    pairs = []
    for low, high in zip(results[::2], results[1::2]):
        if low["branch"] != "lower" or high["branch"] != "upper" or low["current_mA"] != high["current_mA"]:
            raise ValueError("Invalid sequential pair ordering or mismatched actual bias")
        peaks = []
        for item in (low, high):
            peak = dict(item["fit"])
            peak.update(accepted=item["window_pass"], fit_method="prospective frozen-profile detection; conditional complex-IQ fit")
            peaks.append(peak)
        separation = peaks[1]["center_MHz"] - peaks[0]["center_MHz"]
        resolved = bool(separation > max(p["FWHM_MHz"] for p in peaks)
                        and separation > 3 * np.hypot(*(p["center_error_MHz"] for p in peaks)))
        pairs.append(dict(current_mA=low["current_mA"], metrics=dict(candidates=peaks),
                          control_review_required=not resolved,
                          traces=[low["traces"], high["traces"]],
                          qualification="Separate prospective method; original detector results retained"))
    return pairs


def self_test():
    from analyze_pucq4_crossing import mode_frequencies
    currents = np.linspace(-5.70, -5.60, 7)
    f = mode_frequencies([5438, 235, 5438, 6], currents + 5.65)
    pairs = [dict(current_mA=float(c), metrics=dict(candidates=[
        dict(center_MHz=float(v), center_error_MHz=.01, FWHM_MHz=.2, accepted=True) for v in row]))
        for c, row in zip(currents, f)]
    result = coupled_fit(pairs)
    assert result["status"] == "resolved" and abs(result["J_Hz"] - 6e6) < 1
    assert coupled_fit(pairs[:6])["status"] == "unresolved"
    assert coupled_fit([pairs[0]] * 7)["status"] == "unresolved"
    pairs[3]["control_review_required"] = True
    assert coupled_fit(pairs)["status"] == "unresolved"
    print("PASS: known J recovery; incomplete, repeated-bias and failed-control maps rejected")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    if manifest["status"] != "complete" or len(manifest["records"]) != len(spec["windows"]):
        raise ValueError("Require the complete predeclared map")
    if len(spec["windows"]) != 16:
        raise ValueError("Require seven unique paired biases plus the repeated first bias")
    results = [assess_record(record, expected, 16) for record, expected in zip(manifest["records"], spec["windows"])]
    all_pairs = pair_results(results)
    pairs = all_pairs[:7]
    if len(set(p["current_mA"] for p in pairs)) != 7:
        raise ValueError("Map does not contain seven unique recorded biases")
    every_window_passes = all(r["window_pass"] for r in results)
    if all_pairs[-1]["current_mA"] != all_pairs[0]["current_mA"]:
        raise ValueError("Final stability control must repeat the first recorded bias")
    repeat_checks = []
    for first, last in zip(pairs[0]["metrics"]["candidates"], all_pairs[-1]["metrics"]["candidates"]):
        difference = abs(first["center_MHz"] - last["center_MHz"])
        limit = max(.05, 3 * np.hypot(first["center_error_MHz"], last["center_error_MHz"]))
        repeat_checks.append(dict(difference_MHz=difference, limit_MHz=float(limit), passed=bool(difference <= limit)))
    stable = all(item["passed"] for item in repeat_checks)
    fit = coupled_fit(pairs) if every_window_passes and stable else dict(status="unresolved", reason="A predeclared window or temporal stability control failed")
    args.output.mkdir(parents=True, exist_ok=False)
    coupled_plot("fixed_profile_M2", fit, args.output)
    output = dict(windows=results, pairs=pairs, repeat_checks=repeat_checks, coupled_fit=fit,
                  all_windows_pass=every_window_passes, source_manifest=str(args.manifest),
                  spec_sha256=hashlib.sha256(args.spec.read_bytes()).hexdigest(),
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  physical_identity="Intended PUCQ4 design assignment Q5/M2; hardware indexing uncertainty retained")
    (args.output / "map_assessment.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(dict(all_windows_pass=every_window_passes, coupled_status=fit["status"], output=str(args.output))))


if __name__ == "__main__":
    main()
