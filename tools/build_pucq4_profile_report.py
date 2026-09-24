"""Prepend validated profile-pilot and design evidence to the preserved report."""

import argparse
import json
from pathlib import Path

from pptx import Presentation

from build_pucq4_crossing_ppt import text_box, data_table, fitted_image


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--pilot-folder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing reports")
    folder = args.pilot_folder
    result = json.loads((folder / "pilot_assessment.json").read_text(encoding="utf-8"))
    guard = json.loads((folder / "pilot.guard.json").read_text(encoding="utf-8"))
    if guard["status"] != "complete" or guard["exit_code"] != 0:
        raise ValueError("Require a terminal successful pilot guard")
    if any(s["output"] or s["current_A"] != 0 or s["mode"] != "CURR" for s in guard["final_sources"]):
        raise ValueError("Require verified parked sources")
    manifest = json.loads(Path(result["source_manifest"]).read_text(encoding="utf-8"))
    diagnostics = json.loads((folder.parent / "q5q6_profile_debug_0915_data/diagnostics.json").read_text(encoding="utf-8"))
    validation = json.loads((folder.parent / "q5q6_profile_debug_0915_validation/validation.json").read_text(encoding="utf-8"))
    design = json.loads((folder / "design_review/design_evidence.json").read_text(encoding="utf-8"))
    if not diagnostics["complete"] or not validation["passed"]:
        raise ValueError("Missing validated method diagnostics")
    prs = Presentation(args.source)
    prior = len(prs.slides)
    additions = []
    def slide(title):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        additions.append(s)
        text_box(s, title, .5, .28, 12.3, .7, 27)
        text_box(s, "PUCQ4 | September 15 profile pilot and design-source update | %d" % len(additions),
                 .5, 7.05, 12.3, .25, 10)
        return s
    s = slide("Latest result: prospective profile pilot and PUCQ4 design")
    status = "All four fresh windows and both center-repeat checks pass." if result["pilot_pass"] else "The full prospective pilot does not pass; failed controls are retained."
    text_box(s, status + "\n\nThe separate fixed-profile method tests a predicted signed response across the full line. Its result does not alter the earlier peak-height rejections."
             "\n\nThe recovered chip-specific design assigns Q5 to the nominal M2 resonator. Independent experimental indexing uncertainty remains."
             "\n\nMinimum splitting and J still require a bracketed seven-bias map with passing model and stability checks.", .6, 1.2, 12, 4.9, 21)
    text_box(s, "The preserved %d-slide report follows this update with its original numbering and historical conclusions." % prior,
             .6, 6.35, 12, .5, 14)
    s.notes_slide.notes_text_frame.text = json.dumps(dict(pilot=result, design_source=design["source"]), indent=2)
    s = slide("Why peak height and profile evidence give different results")
    text_box(s, "The original SNR is fitted peak height / point-to-point noise. A broad line contributes information across many frequency samples."
             "\n\nRetrospective method: 64 chronological blocks per state; alternating blocks select and test a line profile. Discovery and validation use disjoint ON and OFF blocks."
             "\n\nFresh pilot: center, width and signed IQ direction were frozen before acquisition. Every window must pass t >= 5, corrected sign p < 0.01, reference, dependence, line-shape and repeat controls."
             "\n\nThe sign reference assumes symmetric independent block errors. Repeatable sequential-acquisition artifacts remain a limitation; bias-dependent branch tracking is still required.",
             .6, 1.15, 12, 5.65, 20)
    s.notes_slide.notes_text_frame.text = "Method references: https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.correlate.html\nhttps://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.permutation_test.html\n" + str(folder / "frozen_spec.json")
    s = slide("Retrospective controls: all predefined windows retained")
    rows = []
    for index, r in enumerate(diagnostics["records"]):
        original = r["original_metrics"]["candidates"]
        original_status = "accepted" if any(c["accepted"] for c in original) else "unaccepted"
        rows.append([Path(r["source_manifest"]).parents[2].name, "M%d" % r["row"],
                     "lower" if index % 2 == 0 else "upper", "%.2f" % r["validation"]["t"],
                     "%.3f" % r["off_reference_p_two_sided"], original_status])
    data_table(s, ["Run", "Readout", "Window", "Validation t", "OFF p", "Original detector"], rows,
               [2.2, 1.0, 1.2, 1.8, 1.6, 4.5])
    text_box(s, "320 synthetic checks: 0/160 white/correlated nulls qualified; 65/80 planted signals qualified; all 80 drift cases flagged. Finite validation, not a hardware false-positive guarantee.",
             .6, 6.45, 12, .45, 12)
    s.notes_slide.notes_text_frame.text = json.dumps(dict(diagnostics=diagnostics, synthetic=validation), indent=2)
    s = slide("Fresh central pilot: two separately acquired visits")
    rows = []
    for r in result["records"]:
        fit = r["fit"]
        rows.append([r["repeat"], r["branch"], "%.6f +/- %.6f" % (fit["center_MHz"], fit["center_error_MHz"]),
                     "%.4f" % fit["FWHM_MHz"], "%.2f" % r["signal"]["t"], "pass" if r["window_pass"] else "fail"])
    data_table(s, ["Visit", "Window", "Center +/- conditional SE (MHz)", "FWHM MHz", "Profile t", "Controls"],
               rows, [.7, 1.0, 4.5, 2.0, 1.5, 2.6])
    text_box(s, "M2 at Yoko4 -5.6566 mA; 16000 repetitions per ON/OFF window; gain 0.03; pulse 20 us; relaxation 100 us."
             "\nSeparate lower/upper windows, 101/131 frequency points. Shared DAC1; calibrated M2 readout and tracking retained."
             "\nBoth sources returned to 0 A / output off. Conditional errors exclude long-term drift and line-shape systematics.",
             .6, 4.55, 12, 2.0, 18)
    s.notes_slide.notes_text_frame.text = json.dumps(result, indent=2)
    for index, record in enumerate(manifest["records"]):
        assessment = result["records"][index]
        s = slide("Fresh visit %d: %s window, measured signed IQ" % (assessment["repeat"], assessment["branch"]))
        fitted_image(s, record["plot"], y=1.1, height=5.4)
        text_box(s, "Fixed-profile controls: %s | t = %.2f | OFF p = %.3f | original raw acquisition plot" %
                 ("pass" if assessment["window_pass"] else "fail", assessment["signal"]["t"], assessment["reference_p_two_sided"]),
                 .6, 6.6, 12, .35, 13)
        s.notes_slide.notes_text_frame.text = json.dumps(dict(record=record, assessment=assessment), indent=2)
    s = slide("Recovered PUCQ4 design: geometry and source provenance")
    fitted_image(s, folder / "design_review/PUCQ4_dxf.png", x=.5, y=1.1, width=6, height=5.65)
    text_box(s, "Device04V2.py explicitly declares ChipID PUCQ4. Its archived DXF matches the characterization layout."
             "\n\nSubmission note: December 5, 2025. SQUID junctions occur only at zero-based indices 3 and 5, matching physical Q4 and Q6."
             "\n\nThe drawn connections are one readout transmission line and two flux lines. This source establishes no independently wired Q5 microwave drive."
             "\n\nSources were read as data; no mask-generation code was executed.", 6.8, 1.3, 5.8, 5.3, 19)
    s.notes_slide.notes_text_frame.text = json.dumps(design, indent=2)
    s = slide("Intended design mapping and measured-index limits")
    assignment = [3, 4, 1, 5, 2, 6]
    rows = [["Q%d" % q, "%.3f" % design["assignments"]["freq_res_%d" % q], "M%d / index %d" % (row, row - 1),
             "flux verified" if q in (4, 6) else "design assignment"] for q, row in enumerate(assignment, 1)]
    data_table(s, ["Physical design label", "Design resonator (GHz)", "Nominal table match", "Evidence status"],
               rows, [2.5, 3.0, 3.1, 3.7])
    text_box(s, "The original characterization explicitly flagged uncertainty from shifted measured resonator frequencies. Design-backed Q5/M2 is distinct from independent experimental confirmation. Shared configurations remain unchanged.",
             .6, 5.95, 12, .85, 17)
    s.notes_slide.notes_text_frame.text = json.dumps(design, indent=2)
    # Place the latest evidence first, retaining every historical slide unchanged.
    ids = prs.slides._sldIdLst
    new_ids = list(ids)[prior:]
    for item in new_ids:
        ids.remove(item)
    for index, item in enumerate(new_ids):
        ids.insert(index, item)
    prs.save(args.output)
    assert len(Presentation(args.output).slides) == prior + len(additions)
    print(json.dumps(dict(output=str(args.output), slides=len(prs.slides), new_slides=len(additions), pilot_pass=result["pilot_pass"])))


if __name__ == "__main__":
    main()
