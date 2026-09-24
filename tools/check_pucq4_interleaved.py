"""Read-only analysis of predeclared short-block spectroscopy groups.

All acquisition blocks are selected before measurement, with equal counts and
explicit provenance. This never pools failed historical visits.
"""
import json
from pathlib import Path

import numpy as np

from analyze_pucq4_crossing import line_candidates, read_record, recorded_current_mA
from check_pucq4_discovery_map import discover_and_validate
from check_pucq4_profile_pilot import fit_profile
from check_pucq4_readout_aba import center_check
from diagnose_pucq4_profile_sensitivity import load_blocks


def combine(records, expected):
    if len(records) != 32 or len(expected) != 32:
        raise ValueError("Require all32 predeclared equal-count blocks")
    on, off, provenance = [], [], []
    first_f, row, current, tone = None, None, None, None
    for record, target in zip(records, expected):
        if record["status"] != "complete" or record["requested"] != target["trace"]:
            raise ValueError("Missing block or changed request")
        if record["requested"]["reps"] != 500:
            raise ValueError("Require500 repetitions per acquisition block")
        f, delta, blocks, norm = load_blocks(record, block_count=4)
        _, _, saved = read_record(record)
        actual = recorded_current_mA(saved)
        if abs(actual-target["current_mA"]) > 1e-4:
            raise ValueError("Unexpected bias")
        if first_f is None:
            first_f, row, current, tone = f, record["row"], actual, record["readout_MHz"]
        if row != record["row"] or actual != current or record["readout_MHz"] != tone:
            raise ValueError("Mixed bias, readout, or readout tone")
        np.testing.assert_array_equal(f, first_f)
        on.append(blocks["drive_on"]); off.append(blocks["drive_off"])
        provenance.append(dict(traces=record["traces"], normalization=norm, requested=record["requested"]))
    return first_f, np.concatenate(on), np.concatenate(off), dict(row=row,current_mA=current,readout_MHz=tone,provenance=provenance)


def assess_group(records, expected, family=2):
    f,on,off,metadata=combine(records,expected)
    result=discover_and_validate(f,on,off,family=family)
    prediction=result["discovery_selection"]
    delta=(on-off).mean(axis=0)
    fit=(fit_profile(f,delta,prediction) if .06 <= prediction["FWHM_MHz"] <= .8
         else dict(parameter_quality_pass=False,reason="Unsupported discovery width"))
    legacy=line_candidates(f,delta)[0]
    result.update(metadata)
    result.update(fit=fit,legacy_metrics=legacy,reference_valid=bool(result["reference_p_two_sided"]>.01 and not result["reference"]["dependence_flag"]),
        window_pass=bool(result["fixed_profile_controls_pass"] and fit["parameter_quality_pass"] and sum(p["accepted"] for p in legacy["candidates"])<=1))
    halves=[]
    if "center_MHz" in fit:
        for indices in (slice(0,64),slice(64,128)):
            half=fit_profile(f,(on[indices]-off[indices]).mean(axis=0),fit)
            halves.append(dict(fit=half,window_pass=half["parameter_quality_pass"]))
    result["temporal_center_check"]=(center_check(halves[0],halves[1],"Early/late acquisition halves")
                                      if len(halves)==2 else dict(passed=False,reason="No valid full fit"))
    result["temporal_half_fits"]=halves
    return result


def assess_manifest(manifest, spec):
    expected=spec["windows"]
    if len(manifest["records"])!=64 or len(expected)!=64:
        raise ValueError("Require all64 short blocks")
    for record,target in zip(manifest["records"],expected):
        if record["requested"]!=target["trace"]:raise ValueError("Acquisition ordering changed")
    groups=[]
    for row in (6,2):
        positions=[i for i,e in enumerate(expected) if e["trace"]["row"]==row]
        groups.append(assess_group([manifest["records"][i] for i in positions],[expected[i] for i in positions]))
    comparison=center_check(groups[0],groups[1],"Interleaved cross-readout center")
    passed=all(g["window_pass"] and g["temporal_center_check"]["passed"] for g in groups) and comparison["passed"]
    return dict(groups=groups,comparison=comparison,passed=bool(passed),
        limitation="Prospective counterbalanced short-block comparison. Not simultaneous readout; no Q5 identity or microscopic mechanism is established.")
