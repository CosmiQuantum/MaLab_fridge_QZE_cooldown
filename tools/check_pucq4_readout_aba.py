"""Offline assessment of the predeclared dual-readout A-B-A diagnostic."""

import argparse
import json
from pathlib import Path

import numpy as np

from analyze_pucq4_crossing import line_candidates, read_record, recorded_current_mA
from check_pucq4_discovery_map import discover_and_validate
from check_pucq4_profile_pilot import fit_profile
from diagnose_pucq4_profile_sensitivity import load_blocks


def assess_record(record, expected):
    if record["status"] != "complete":
        raise ValueError("Require a saved complete ON/OFF window")
    if record["requested"] != expected["trace"] or record["row"] != expected["trace"]["row"]:
        raise ValueError("Requested trace differs from frozen specification")
    f, delta, blocks, norm = load_blocks(record, block_count=128)
    _, _, saved = read_record(record)
    current = recorded_current_mA(saved)
    if abs(current - expected["current_mA"]) > 1e-4:
        raise ValueError("Wrong recorded bias")
    result = discover_and_validate(f, blocks["drive_on"], blocks["drive_off"], family=6)
    prediction = result["discovery_selection"]
    fit = (fit_profile(f, delta, prediction) if .06 <= prediction["FWHM_MHz"] <= .8
           else dict(parameter_quality_pass=False, reason="Unsupported discovery width"))
    legacy = line_candidates(f, delta)[0]
    valid_reference = bool(result["reference_p_two_sided"] > .01
                           and not result["reference"]["dependence_flag"])
    result.update(row=record["row"], current_mA=current, fit=fit,
                  reference_valid=valid_reference, legacy_metrics=legacy,
                  normalization=norm, traces=record["traces"],
                  window_pass=bool(result["fixed_profile_controls_pass"]
                                   and fit["parameter_quality_pass"]
                                   and sum(c["accepted"] for c in legacy["candidates"]) <= 1))
    return result


def center_check(first, second, label):
    if not first["window_pass"] or not second["window_pass"]:
        return dict(label=label, passed=False, reason="Both lines must independently pass")
    a, b = first["fit"], second["fit"]
    difference = abs(a["center_MHz"] - b["center_MHz"])
    limit = min(.1, max(.05, 3 * np.hypot(a["center_error_MHz"], b["center_error_MHz"])))
    return dict(label=label, difference_MHz=difference, limit_MHz=float(limit), passed=bool(difference <= limit))


def assess_complete(results):
    if len(results) != 6:
        return dict(status="incomplete", lower_protocol_ready=False)
    if [r["row"] for r in results] != [6, 2, 2, 6, 6, 2]:
        raise ValueError("Unexpected row order")
    a1m6, a1m2, bm2, bm6, a2m6, a2m2 = results
    checks = [center_check(a1m6, a2m6, "M6 A return"),
              center_check(a1m2, a2m2, "M2 A return"),
              center_check(a1m6, a1m2, "A1 cross-readout center"),
              center_check(a2m6, a2m2, "A2 cross-readout center")]
    ready = (all(r["reference_valid"] for r in results)
             and all(r["window_pass"] for r in (a1m6, a1m2, bm6, a2m6, a2m2))
             and all(c["passed"] for c in checks))
    return dict(status="lower_protocol_supported" if ready else "unresolved",
                lower_protocol_ready=bool(ready), center_checks=checks,
                M2_B_detected=bm2["window_pass"], M6_B_detected=bm6["window_pass"],
                full_map_authorized_by_result=False,
                limitation="Upper branch and repeated B validation still required. Sequential readouts do not prove a microscopic mechanism or Q5 identity.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--spec", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    if len(manifest["records"]) != 6 or len(spec["windows"]) != 6:
        raise ValueError("Require the full predeclared six-record diagnostic")
    results = [assess_record(r, e) for r, e in zip(manifest["records"], spec["windows"])]
    output = dict(assessment=assess_complete(results), windows=results)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(output["assessment"]))


if __name__ == "__main__":
    main()
