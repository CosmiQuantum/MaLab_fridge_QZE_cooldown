"""Prospective unknown-center profile map with disjoint discovery/validation.

No hardware imports or actions. Preserve the failed fixed-center map and legacy
detector metrics. The model locates frequency windows, never detection centers.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from check_pucq4_profile_pilot import assess_blocks, fit_profile
from check_pucq4_profile_map import pair_results
from diagnose_pucq4_profile_sensitivity import load_blocks, select_profile, bank
from analyze_pucq4_crossing import coupled_fit, coupled_plot, line_candidates, read_record, recorded_current_mA


def discover_and_validate(f, on, off, family=18, templates=None):
    if on.shape != off.shape or on.shape != (128, len(f), 2):
        raise ValueError("Require 128 equal ON and OFF blocks, giving 64 validation blocks")
    discovery = (on[::2] - off[::2]).mean(axis=0)
    _, prediction = select_profile(f, discovery, templates)
    result = assess_blocks(f, on[1::2], off[1::2], prediction, family=family)
    result["discovery_selection"] = prediction
    result["discovery_blocks"] = list(range(0, 128, 2))
    result["validation_blocks"] = list(range(1, 128, 2))
    return result


def assess_record(record, expected, family=18):
    if record["status"] != "complete":
        raise ValueError("Require complete ON/OFF acquisition")
    for key, value in expected["trace"].items():
        if record["requested"].get(key) != value:
            raise ValueError("Trace differs from frozen protocol: " + key)
    f, delta, blocks, norm = load_blocks(record, block_count=128)
    _, _, saved = read_record(record)
    current = recorded_current_mA(saved)
    if abs(current - expected["current_mA"]) > 1e-4 or record["row"] != 2:
        raise ValueError("Unexpected readout or actual current")
    result = discover_and_validate(f, blocks["drive_on"], blocks["drive_off"], family)
    prediction = result["discovery_selection"]
    original = line_candidates(f, delta)[0]
    fitted = (fit_profile(f, delta, prediction) if .06 <= prediction["FWHM_MHz"] <= .8
              else dict(parameter_quality_pass=False, reason="Discovery width outside predeclared fit bounds"))
    result.update(current_mA=current, branch=expected["branch"], traces=record["traces"],
                  normalization=norm, original_metrics=original,
                  fit=fitted)
    ambiguous = sum(c["accepted"] for c in original["candidates"]) > 1
    result["window_pass"] = bool(result["fixed_profile_controls_pass"]
                                  and result["fit"]["parameter_quality_pass"] and not ambiguous)
    result["multiple_legacy_lines"] = ambiguous
    result["method"] = "Discovery selects center/width/IQ direction; disjoint 64-block validation t>=5 and family18 controls"
    return result


def repeat_check(first, repeated, label):
    if first["current_mA"] != repeated["current_mA"]:
        raise ValueError("Repeat has a different recorded current")
    checks = []
    for a, b in zip(first["metrics"]["candidates"], repeated["metrics"]["candidates"]):
        difference = abs(a["center_MHz"] - b["center_MHz"])
        limit = min(.1, max(.05, 3 * np.hypot(a["center_error_MHz"], b["center_error_MHz"])))
        checks.append(dict(control=label, difference_MHz=difference, limit_MHz=float(limit),
                           passed=bool(difference <= limit)))
    return checks


def self_test():
    rng = np.random.default_rng(15092027)
    f = np.linspace(5430.7, 5432.7, 101)
    templates = bank(f)
    counts = {}
    for scenario in ("white_null", "frequency_correlated_null", "shifted_line", "discovery_only_spur", "strong_drift"):
        passes, flagged = 0, 0
        for _ in range(80):
            on, off = rng.normal(size=(2, 128, len(f), 2))
            center = 5431.7 + rng.uniform(-.35, .35)
            width = rng.uniform(.12, .5)
            shape = 1 / (1 + (2 * (f - center) / width) ** 2)
            direction = np.array([.8, -.6])
            if scenario == "frequency_correlated_null":
                on += rng.normal(size=(128, 1, 2)) * shape[None, :, None] * 3
                off += rng.normal(size=(128, 1, 2)) * shape[None, :, None] * 3
            elif scenario == "shifted_line":
                on += .9 * shape[None, :, None] * direction
            elif scenario == "discovery_only_spur":
                on[::2] += 3 * shape[None, :, None] * direction
            elif scenario == "strong_drift":
                on += np.linspace(0, 5, 128)[:, None, None] * shape[None, :, None] * direction
            result = discover_and_validate(f, on, off, templates=templates)
            passes += result["fixed_profile_controls_pass"]
            flagged += result["signal"]["dependence_flag"]
        counts[scenario] = dict(trials=80, controls_pass=int(passes), dependence_flags=int(flagged))
    passed = bool(counts["white_null"]["controls_pass"] <= 2
                  and counts["frequency_correlated_null"]["controls_pass"] <= 2
                  and counts["shifted_line"]["controls_pass"] >= 65
                  and counts["discovery_only_spur"]["controls_pass"] <= 2
                  and counts["strong_drift"]["controls_pass"] <= 2)
    print(json.dumps(dict(passed=passed, scenarios=counts)), flush=True)
    if not passed:
        raise AssertionError("Discovery-map synthetic validation failed")


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
    if manifest["status"] != "complete" or len(manifest["records"]) != 18 or len(spec["windows"]) != 18:
        raise ValueError("Require the complete 18-window planned map")
    results = [assess_record(record, expected) for record, expected in zip(manifest["records"], spec["windows"])]
    all_pairs = pair_results(results)
    for pair in all_pairs:
        for candidate in pair["metrics"]["candidates"]:
            candidate["fit_method"] = "Disjoint discovery/validation detection; conditional complex-IQ fit"
    # Visit2 is the shifted-grid control; visit9 repeats the first at the end.
    pairs = [all_pairs[0]] + all_pairs[2:8]
    if len(set(p["current_mA"] for p in pairs)) != 7:
        raise ValueError("Require seven distinct paired biases")
    repeat_checks = repeat_check(all_pairs[0], all_pairs[1], "early shifted frequency grid")
    repeat_checks += repeat_check(all_pairs[0], all_pairs[-1], "final temporal repeat")
    controls = all(r["window_pass"] for r in results) and all(c["passed"] for c in repeat_checks)
    fit = coupled_fit(pairs) if controls else dict(status="unresolved", reason="A window, grid control or temporal repeat failed")
    args.output.mkdir(parents=True, exist_ok=False)
    coupled_plot("discovery_profile_M2", fit, args.output)
    output = dict(windows=results, pairs=pairs, repeat_checks=repeat_checks, all_controls_pass=controls,
                  coupled_fit=fit, source_manifest=str(args.manifest),
                  spec_sha256=hashlib.sha256(args.spec.read_bytes()).hexdigest(),
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  limitation="Conditional block-noise assumptions and fit errors; intended Q5/M2 design assignment is not independent experimental identity")
    (args.output / "map_assessment.json").write_text(json.dumps(output, indent=2), encoding="utf-8")
    print(json.dumps(dict(all_controls_pass=controls, coupled_status=fit["status"], output=str(args.output))))


if __name__ == "__main__":
    main()
