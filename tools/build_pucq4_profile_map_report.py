"""Prepend terminal map evidence to the preserved pilot and historical report."""
import argparse
import json
from pathlib import Path
from pptx import Presentation
from build_pucq4_crossing_ppt import text_box, data_table, fitted_image


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--fixed-folder", type=Path, required=True)
    parser.add_argument("--discovery-folder", type=Path, required=True)
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Preserve existing report")
    audit = load(args.audit)
    if audit["status"] != "pass":
        raise ValueError("Require artifact audit")
    runs = []
    for label, folder in (("Fixed-center map", args.fixed_folder), ("Discovery/validation map", args.discovery_folder)):
        guard = load(folder / "map.guard.json")
        runtime = load(folder / "map.runtime.json")
        if guard["status"] not in ("complete", "failed"):
            raise ValueError("Require terminal guarded runs")
        states = guard.get("final_sources", [])
        if len(states) != 2 or any(s["mode"] != "CURR" or s["current_A"] != 0 or s["output"] for s in states):
            raise ValueError("Require both verified parked sources")
        manifest = load(Path(runtime["run"]) / "documentation/manifest.json")
        results = [load(path) for path in sorted(folder.glob("window_*.json"))]
        complete = [record for record in manifest["records"] if record["status"] == "complete"]
        if len(results) != len(complete):
            raise ValueError("Require assessment for every completed window")
        runs.append(dict(label=label, folder=folder, guard=guard, manifest=manifest, results=results, complete=complete))
    final_path = args.discovery_folder / "final_analysis/map_assessment.json"
    final = load(final_path) if final_path.exists() else {"coupled_fit": {"status": "unresolved", "reason": "The predeclared full map did not complete"}}
    fit = final["coupled_fit"]
    prs = Presentation(args.source)
    prior = len(prs.slides)
    def slide(title):
        s = prs.slides.add_slide(prs.slide_layouts[6])
        text_box(s, title, .5, .28, 12.3, .7, 27)
        text_box(s, "PUCQ4 | September 15 terminal map review | %d" % (len(prs.slides)-prior), .5, 7.05, 12.3, .25, 10)
        return s
    s = slide("Latest Q5/Q6 evidence and remaining limits")
    conclusion = ("The seven-bias coupled-mode fit passes the predeclared map and repeat controls."
                  if fit["status"] == "resolved" else "The predeclared map does not establish an accepted minimum splitting or coupling J.")
    text_box(s, conclusion + "\n\nThe earlier fresh pilot passed all four predicted-profile windows and both center repeats. The fixed-center map then failed; that failure remains recorded."
             "\n\nThe separate discovery/validation protocol learns each unknown line from half the data and tests it on the other half. Every completed window and partial acquisition is retained."
             "\n\nThe recovered PUCQ4 design supports the intended Q5/M2 assignment. The original author's August 4 slides still label M2 as Q5?; independent experimental identity and a separate Q5 microwave port remain unverified.", .6, 1.15, 12, 5.45, 20)
    s.notes_slide.notes_text_frame.text = json.dumps(final, indent=2)
    s = slide("Why the fixed-center map stopped")
    failed = runs[0]["results"][0]
    text_box(s, "The first lower window at Yoko4 -5.655 mA returned profile t = %.3f, below the frozen t >= 5 threshold. The run stopped at the next acquisition boundary."
             "\n\nIts conditional fitted center was %.6f MHz, about 0.131 MHz above the frozen prediction. A predicted-center test is too restrictive for this unknown-frequency map."
             "\n\nRetrospective discovery on half the data improved the held-out statistic to 4.746, which still fails the prospective threshold. This diagnostic does not promote the failed record."
             "\n\nThe upper OFF trace already underway was saved. It is a partial reference, not a completed upper-line measurement. Both sources returned to zero with outputs off."
             % (failed["signal"]["t"], failed["fit"]["center_MHz"]), .6, 1.2, 12, 5.5, 20)
    s.notes_slide.notes_text_frame.text = json.dumps(dict(assessment=failed, guard=runs[0]["guard"]), indent=2)
    s = slide("Prospective unknown-center protocol and controls")
    text_box(s, "Each ON/OFF state: 32000 repetitions, 101 frequency points, gain 0.03, 20 us pulse, 100 us relaxation; M2 readout with existing resonator tracking."
             "\n\n128 chronological blocks: alternating 64 blocks choose center, width and signed IQ direction; the other 64 blocks test that profile. The validation half retains 16000 repetitions."
             "\n\nEvery window requires t >= 5, family-18 corrected sign p < 0.01, OFF p > 0.01, passing dependence and line-shape checks. The original detector and its results remain unchanged."
             "\n\nNine paired visits plan seven unique biases, an early +0.25 MHz grid repeat, and a final temporal repeat. Centers must agree within min(0.1, max(0.05, 3 combined SE)) MHz."
             "\n\n400 synthetic trials: 0/160 white/correlated nulls, 0/80 discovery-only artifacts and 0/80 drift cases accepted; 78/80 shifted signals accepted. Conditional noise assumptions remain a limitation.", .6, 1.1, 12, 5.85, 18)
    s.notes_slide.notes_text_frame.text = json.dumps(load(args.discovery_folder / "frozen_spec.json"), indent=2)
    for run in runs:
        for start in range(0, len(run["results"]), 8):
            s = slide(run["label"] + ": all completed windows")
            rows = []
            for index, r in enumerate(run["results"][start:start+8], start):
                local = r["fit"]
                center = "%.6f" % local["center_MHz"] if "center_MHz" in local else "fit rejected"
                rows.append([index+1, "%.6f" % r["current_mA"], r["branch"], center,
                             "%.3f" % r["signal"]["t"], "%.4f" % r["reference_p_two_sided"], "pass" if r["window_pass"] else "FAIL"])
            data_table(s, ["#", "Y4 mA", "Branch", "Center MHz", "Profile t*", "OFF p", "Controls"], rows,
                       [.8, 1.6, 1.2, 2.4, 1.7, 1.7, 2.9])
            text_box(s, "*Fixed-center map uses its full frozen-profile sample; discovery map uses disjoint validation data. Conditional fit centers do not imply acceptance.", .6, 6.45, 12, .45, 12)
            s.notes_slide.notes_text_frame.text = json.dumps(run["results"][start:start+8], indent=2)
        for index, (record, r) in enumerate(zip(run["complete"], run["results"])):
            s = slide("%s: window %d, %s" % (run["label"], index+1, r["branch"]))
            fitted_image(s, record["plot"], y=1.1, height=5.4)
            text_box(s, "Y4 %.6f mA | t %.3f | OFF p %.4f | controls %s | saved acquisition plot" %
                     (r["current_mA"], r["signal"]["t"], r["reference_p_two_sided"], "pass" if r["window_pass"] else "FAIL"), .6, 6.6, 12, .35, 13)
            s.notes_slide.notes_text_frame.text = json.dumps(dict(record=record, assessment=r), indent=2)
    review_path = args.discovery_folder / "readout_review.json"
    if review_path.exists():
        s = slide("Readout check: no tracked-setting change explains the failure")
        fitted_image(s, args.discovery_folder / "readout_review.png", y=1.1, height=4.4)
        text_box(s, "All four saved M2 checks selected 8951.210 MHz, gain 0.56 and 5 us readout; the coarse minimum remained 8951.610 MHz."
                 "\nGround-state tracking does not measure excited-state discrimination or identify the cause of weaker spectroscopy.", .6, 5.7, 12, 1.0, 16)
        s.notes_slide.notes_text_frame.text = json.dumps(load(review_path), indent=2)
    s = slide("Terminal state, artifact audit and scientific acceptance")
    lines = ["%s: %s; completed windows: %d; both sources verified 0 A / output off." % (r["label"], r["guard"]["status"], len(r["complete"])) for r in runs]
    lines += ["Artifact audit: all referenced raw HDF5, plots, source hashes and terminal state checks pass. Partial references remain explicitly marked.",
              "Coupled-model status: " + fit["status"] + ". " + fit.get("reason", "See the saved full model assessment."),
              "Physical identity remains qualified: intended design Q5/M2 is not an independent experimental assignment. No shared device configuration changed."]
    text_box(s, "\n\n".join(lines), .6, 1.2, 12, 5.5, 20)
    s.notes_slide.notes_text_frame.text = json.dumps(dict(audit=audit, fit=fit), indent=2)
    if fit["status"] == "resolved":
        plot = args.discovery_folder / "final_analysis/discovery_profile_M2_coupled_fit.png"
        if not plot.exists():
            raise ValueError("Resolved model needs its plot; check coupled_plot output name")
        s = slide("Seven-bias coupled-mode fit with passing repeat controls")
        fitted_image(s, plot, y=1.1, height=5.65)
        s.notes_slide.notes_text_frame.text = json.dumps(fit, indent=2)
    ids = prs.slides._sldIdLst
    additions = list(ids)[prior:]
    for item in additions:
        ids.remove(item)
    for index, item in enumerate(additions):
        ids.insert(index, item)
    prs.save(args.output)
    assert len(Presentation(args.output).slides) == len(prs.slides)
    print(json.dumps(dict(output=str(args.output), slides=len(prs.slides), new_slides=len(additions), coupled_status=fit["status"])))


if __name__ == "__main__":
    main()
