"""Build an evidence-only PowerPoint for the physical Q5-Q6 crossing task."""

import argparse
import json
import math
import os
from pathlib import Path

from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.util import Inches, Pt


def text_box(slide, text, x, y, w, h, size=18, color=(32, 42, 56)):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(w), Inches(h))
    frame = shape.text_frame
    frame.word_wrap = True
    for index, line in enumerate(text.split("\n")):
        paragraph = frame.paragraphs[0] if index == 0 else frame.add_paragraph()
        paragraph.text = line
        paragraph.font.size = Pt(size)
        paragraph.font.color.rgb = RGBColor(*color)
        paragraph.space_after = Pt(8)
    return shape


def fitted_image(slide, path, x=.5, y=1.15, width=12.3, height=5.5):
    with Image.open(path) as picture:
        pw, ph = picture.size
    scale = min(width / pw, height / ph)
    w, h = pw * scale, ph * scale
    slide.shapes.add_picture(str(path), Inches(x + (width - w) / 2),
                            Inches(y + (height - h) / 2), width=Inches(w))


def data_table(slide, headers, rows, widths):
    table = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(.5),
                Inches(1.2), Inches(12.3), Inches(min(5.1, .62 * (len(rows) + 1)))).table
    for column, width in zip(table.columns, widths):
        column.width = Inches(width)
    for row_index, values in enumerate([headers] + rows):
        for col_index, value in enumerate(values):
            cell = table.cell(row_index, col_index)
            cell.text = str(value)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(*( (26, 93, 118) if row_index == 0 else (239, 243, 246)))
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(14)
                paragraph.font.color.rgb = RGBColor(*( (255, 255, 255) if row_index == 0 else (32, 42, 56)))


def build(manifest_paths, analysis_path, destination, conclusion, drift_path=None, findings_path=None):
    analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    manifests = [json.loads(path.read_text(encoding="utf-8")) for path in manifest_paths]
    presentation = Presentation()
    presentation.slide_width, presentation.slide_height = Inches(13.333), Inches(7.5)
    def slide(title):
        result = presentation.slides.add_slide(presentation.slide_layouts[6])
        text_box(result, title, .5, .28, 12.3, .7, 27)
        text_box(result, "PUCQ4 | physical Q5-Q6 spectroscopy | analysis %s UTC | %d" %
                 (analysis.get("created_UTC", "undated")[:10], len(presentation.slides)),
                 .5, 7.05, 12.3, .25, 10)
        return result
    first = slide("Physical Q5-Q6 avoided-crossing measurement")
    text_box(first, conclusion, .6, 1.3, 12, 2.2, 23)
    text_box(first, "Requested pair: physical Q5 and Q6 in the Downloads characterization deck.\nQ6 = M6 / Python index 5 / Yoko4 / DC A5.\nPhysical Q5 requires independent mapping evidence; displayed Q5/M5 is physical Q4. A shared response through M2 alone cannot assign Q5.\nAn avoided crossing indicates hybridization; it does not demonstrate entangled-state preparation.", .6, 3.6, 12, 2.8, 19)
    if findings_path is not None:
        paragraphs = [p.strip() for p in findings_path.read_text(encoding="utf-8").split("\n\n") if p.strip()]
        for start in range(0, len(paragraphs), 2):
            findings = slide("Measured findings and interpretation (%d/%d)" %
                             (start // 2 + 1, (len(paragraphs) + 1) // 2))
            text_box(findings, "\n\n".join(paragraphs[start:start + 2]), .6, 1.2, 12, 5.6, 20)
            findings.notes_slide.notes_text_frame.text = str(findings_path) + "\nAnalysis: " + str(analysis_path)
    if analysis.get("current_axis"):
        axis_slide = slide("Current-axis provenance")
        text_box(axis_slide, analysis["current_axis"] +
                 "\n\nExample: requested -5.656667 mA; recorded -5.656600 mA."
                 " The 0.000067 mA difference corresponds to about 15 kHz at 235 MHz/mA."
                 "\n\nSpectroscopy analysis tables, maps and branch fits use the recorded setting."
                 " Original acquisition plots and raw files are preserved."
                 "\n\nDrive-on and drive-off settings must agree; frequency axes use the saved QICK grid."
                 " Reported fit errors are conditional statistical errors, not full calibration uncertainties.",
                 .6, 1.2, 12, 5.6, 21)
    reference = Path(os.environ["TEMP"]) / "pucq4_cross_20260909/reference_preview/source_slide1.png"
    if reference.is_file():
        original = slide("Physical labels in the original characterization deck")
        fitted_image(original, reference, height=5.8)
        original.notes_slide.notes_text_frame.text = "Read-only native export from Downloads/PUCQ4 Initial Characterization (1).pptx. SHA256 626B42201DD6C2E02417806AAE34EF94122196F2D4E88D2DAC59B9492B44EC71. Original deck is unchanged; frequencies on this source slide are historical."
    setup = slide("Hardware, bias and measurement provenance")
    text_box(setup, "Shared qubit-drive DAC 1; resonator DAC 0; ADC 0. One readout tone at a time.\nNo independent spatial Q5/Q6 drive port is documented here; spectroscopy addresses frequencies on the shared line.\nYoko4 bounded by -10 to +10 mA; local task range -5.8 to -4.8 mA; normal ramp 0.5 mA/s.\nYoko3 held at 0 A/output off; Yoko4 returned to 0 A/output off after each guarded plan.\nEstablished Yoko4 protocol: no extra DC hold in this region; resonator tracking before spectroscopy; 100 us inter-shot relaxation.\nDrive-off references use the same axis, timing and averaging. Raw I/Q, source-level readback, pulse/readout settings and source hashes are saved in canonical HDF5 and documentation.\nNominal attenuation arguments: readout 10/15 dB, qubit 5/4 dB, ADC 17 dB. Setters disabled in shared config; actual attenuations were not independently read back.", .6, 1.15, 12, 5.7, 18)
    calibration = slide("Calibration used to locate the survey")
    text_box(calibration, "Q6 stored measured frequencies:\n-7.5 mA: 4839.206595 MHz; -5 mA: 5588.103711 MHz; -2.5 mA: 6103.731048 MHz.\nThe first two retain September 7/8 provenance; -2.5 mA was revalidated September 9.\nFixed measured-row candidates at zero Yoko current:\nM1: 5517.691628 MHz; M2: 5438.710887 MHz; M3: 5557.335688 MHz; M4: 5594.541903 MHz.\nThese frequencies guide localization. A resonance bias is determined by the smallest resolved branch separation, not a crossing with a seed frequency.\nSource: system_config.py PUCQ4_FLUX_PULSE_CALIBRATION and fixed-row config; exact launch source copies accompany each run.", .6, 1.2, 12, 5.7, 20)
    interpretation = slide("A shared peak is not yet a pair of eigenbranches")
    text_box(interpretation, "Shared-drive symmetry can suppress a branch. Higher power can also reveal a two-photon transition to the doubly excited state.\n\nThese are possible explanations to test here, not identified mechanisms in PUCQ4.\n\nThe experiment compares bias dependence, both readouts and drive power before accepting branch identities or extracting a minimum splitting. A shared single peak is insufficient.", .6, 1.25, 12, 4.6, 23)
    citation = text_box(interpretation, "Reference: Filipp et al., Phys. Rev. A 83, 063827 (2011), Secs. V-VI.\nPreprint: arxiv.org/abs/1011.3732", .6, 6.05, 12, .8, 16)
    citation.text_frame.paragraphs[0].runs[0].hyperlink.address = "https://doi.org/10.1103/PhysRevA.83.063827"
    citation.text_frame.paragraphs[1].runs[0].hyperlink.address = "https://arxiv.org/abs/1011.3732"
    if analysis.get("line_centers_plot"):
        item = slide("Measured line centers")
        fitted_image(item, analysis["line_centers_plot"])
        item.notes_slide.notes_text_frame.text = "Individual line fits do not alone establish coupling. " + str(analysis_path)
    measured = []
    for point in analysis["points"]:
        peaks = [p for p in point["metrics"]["candidates"] if p["accepted"]]
        for peak in peaks or [None]:
            series = Path(point["source_manifest"]).parents[2].name
            measured.append(["M%d / %s\n%.6f mA" % (point["row"], series, point["current_mA"]),
                "%g\n%d avg" % (point["gain"], point["requested"]["reps"]),
                "%.6f +/- %.6f" % (peak["center_MHz"], peak["center_error_MHz"]) if peak else "No accepted line",
                "%.4f +/- %.4f" % (peak["FWHM_MHz"], peak["FWHM_error_MHz"]) if peak else "Unresolved",
                "%.1f" % peak["snr"] if peak else "-",
                "Reference review" if point.get("control_review_required") else
                    "%d accepted line%s" % (len(peaks), "s" if len(peaks) != 1 else "")])
    for start in range(0, len(measured), 8):
        item = slide("Line fits: centers and linewidths in MHz")
        data_table(item, ["Readout / bias", "Gain / averages", "Center +/- SE (MHz)",
                         "FWHM +/- SE (MHz)", "SNR", "Trace result"],
                   measured[start:start + 8], [1.8, 1.4, 3.3, 2.6, .7, 2.5])
        text_box(item, "Current is source-setting readback. Conditional fit errors exclude temporal drift, current accuracy and line-shape systematics.", .5, 6.5, 12.3, .4, 12)
        item.notes_slide.notes_text_frame.text = str(analysis_path)
    pair_rows, pair_notes, seen_pairs = [], [], set()
    for review in analysis.get("paired_readout_reviews", []) + analysis.get("segmented_readout_reviews", []):
        for pair in review["accepted_pairs"]:
            peaks = sorted([c for c in pair["metrics"]["candidates"] if c["accepted"]],
                           key=lambda c: c["center_MHz"])
            source_label = review.get("source_label", review["series"].split("_M6_")[0])
            signature = (source_label, pair["current_mA"], pair["gain"],
                         tuple(c["center_MHz"] for c in peaks))
            if signature in seen_pairs:
                continue
            seen_pairs.add(signature)
            variance = sum(c["center_error_MHz"] ** 2 for c in peaks)
            covariance = peaks[0].get("joint_center_covariance_MHz2")
            if (peaks[0].get("fit_method") == "joint two-line complex-IQ"
                    and covariance is not None
                    and covariance == peaks[1].get("joint_center_covariance_MHz2")):
                variance = covariance[0][0] + covariance[1][1] - 2 * covariance[0][1]
            selected_rows = ({c["readout_row"] for c in peaks}
                             if all("readout_row" in c for c in peaks) else set(pair["readout_rows"]))
            rows_label = "/".join(str(row) for row in sorted(selected_rows))
            pair_rows.append(["%s\nM%s g%g" % (source_label, rows_label, pair["gain"]),
                "%.6f" % pair["current_mA"],
                "%.6f +/- %.6f" % (peaks[0]["center_MHz"], peaks[0]["center_error_MHz"]),
                "%.6f +/- %.6f" % (peaks[1]["center_MHz"], peaks[1]["center_error_MHz"]),
                "%.6f +/- %.6f" % (peaks[1]["center_MHz"] - peaks[0]["center_MHz"],
                                    math.sqrt(max(0, variance)))])
            pair_notes.append(dict(series=review["series"], pair=pair))
    def first_pair_time(pair):
        times = pair.get("acquired_unix", 0)
        return min(times) if isinstance(times, list) else times

    ordered_pairs = sorted(zip(pair_rows, pair_notes), key=lambda item: (
        item[1]["pair"]["current_mA"], item[1]["pair"]["gain"],
        first_pair_time(item[1]["pair"]), item[1]["series"]))
    pair_rows = [item[0] for item in ordered_pairs]
    pair_notes = [item[1] for item in ordered_pairs]
    for start in range(0, len(pair_rows), 6):
        item = slide("Resolved line pairs at matched drive settings")
        data_table(item, ["Run / RO / gain", "Y4 (mA)", "Lower +/- SE (MHz)",
                         "Upper +/- SE (MHz)", "Separation +/- SE (MHz)"],
                   pair_rows[start:start + 6], [1.9, 1.1, 3.1, 3.1, 3.1])
        text_box(item, "Sequential acquisitions; conditional errors exclude temporal/source systematics. Windows may be disjoint. Separation is measured at the stated bias; minimum splitting and J require an accepted bias-dependent fit.", .5, 6.25, 12.3, .65, 13)
        item.notes_slide.notes_text_frame.text = json.dumps(pair_notes[start:start + 6], indent=2)
    for path in analysis.get("maps", []):
        item = slide(Path(path).stem.replace("_", " "))
        fitted_image(item, path, height=5.2)
        text_box(item, "White regions were not measured. Colored strips show acquired frequency bins at each bias; no interpolation between scans.", .5, 6.5, 12.3, .4, 12)
        item.notes_slide.notes_text_frame.text = "Derived from paired canonical drive-on/off IQ. " + str(analysis_path)
    for path in analysis.get("power_comparisons", []):
        item = slide("Same-bias drive-power comparison")
        fitted_image(item, path)
        item.notes_slide.notes_text_frame.text = "Signed I and Q in measured ADC units, without amplitude normalization. " + str(analysis_path)
    for review in analysis.get("segmented_readout_reviews", []):
        for pair in review["accepted_pairs"]:
            if pair.get("measured_windows_plot"):
                item = slide("Resolved pair in separate measured windows")
                fitted_image(item, pair["measured_windows_plot"], height=5.5)
                item.notes_slide.notes_text_frame.text = json.dumps(pair, indent=2)
    for control in analysis.get("bias_reversal_checks", []):
        if control.get("plot"):
            item = slide("Bias response versus temporal drift")
            fitted_image(item, control["plot"], height=5.1)
            text_box(item, "Review: %s. This tests line-center stability; it does not measure a splitting." % control["status"].replace("_", " "), .5, 6.4, 12.3, .55, 14)
            item.notes_slide.notes_text_frame.text = json.dumps(control, indent=2)
    if drift_path is not None:
        drift = json.loads(drift_path.read_text(encoding="utf-8"))
        for start in range(0, len(drift["records"]), 2):
            item = slide("Line-center changes within a held source setting")
            subset = drift["records"][start:start + 2]
            for index, record in enumerate(subset):
                fitted_image(item, record["plot"], y=1.1 + index * 2.65, height=2.45)
            text_box(item, "Raw repetitions are split chronologically. Error bars are conditional fit errors with a shared drive-off reference. Changes within a trace limit interpretation of whole-trace centers.", .5, 6.4, 12.3, .6, 13)
            item.notes_slide.notes_text_frame.text = json.dumps(dict(analysis=str(drift_path), records=subset), indent=2)
    updates = [u for m in manifests for u in m["readout_updates"]]
    if updates:
        item = slide("Readout tracking during the spectroscopy survey")
        rows = []
        for row in sorted(set(u["row"] for u in updates)):
            subset = [u for u in updates if u["row"] == row]
            rows.append(["M%d" % row, "%g to %g" % (min(u["current_mA"] for u in subset), max(u["current_mA"] for u in subset)),
                "%.5f to %.5f" % (min(u["new_tone_MHz"] for u in subset), max(u["new_tone_MHz"] for u in subset)),
                ", ".join(str(g) for g in sorted(set(u["gain"] for u in subset))),
                ", ".join(str(v) for v in sorted(set(u["length_us"] for u in subset))),
                str(sum(abs(u["new_tone_MHz"] - u["old_tone_MHz"]) > 1e-8 for u in subset))])
        data_table(item, ["Readout", "Requested bias (mA)", "Selected RO tone (MHz)", "DAC gain", "Length (us)", "Tone changes"],
                   rows, [1, 2.2, 3.8, 1.6, 1.5, 2.2])
        text_box(item, "Resonator scans: 120 points, 50 kHz steps, 600 averages, one round.\nSub-bin fluctuations retain the calibrated tone. Raw tracking scans and every selected setting are saved.", .6, 4.5, 12, 1.7, 20)
        item.notes_slide.notes_text_frame.text = json.dumps(updates, indent=2)
    for manifest in manifests:
        for record in manifest["records"]:
            if record["status"] != "complete":
                continue
            detail_point = next((p for p in analysis["points"] if p["traces"] == record["traces"]), {})
            item = slide("M%d readout at %.6f mA (source setting)" %
                         (record["row"], detail_point.get("current_mA", record["current_mA"])))
            fitted_image(item, record["plot"], height=5.15)
            request = record["requested"]
            text_box(item, "%g-%g MHz | %d points | %d averages/state | drive gain %g | pulse %g us | RO %.5f MHz" %
                (request["start_MHz"], request["stop_MHz"], request["steps"], request["reps"], request["gain"], request["length_us"], record["readout_MHz"]), .5, 6.45, 12.3, .45, 12)
            item.notes_slide.notes_text_frame.text = json.dumps(record, indent=2)
            detail = detail_point.get("detail_plot")
            if detail:
                item = slide(("Sampling-limited candidate: M%d" if detail_point.get("detail_status") == "sampling limited candidate"
                              else "Detail of the measured narrow line: M%d") % record["row"])
                fitted_image(item, detail, height=5.5)
                item.notes_slide.notes_text_frame.text = "Detail of the same saved scan; full frequency coverage is on the preceding slide. " + json.dumps(record, indent=2)
    unresolved = {key: fit for key, fit in analysis["coupled_fits"].items() if fit["status"] == "unresolved"}
    if unresolved:
        item = slide("Branch resolution and limits of the measurement")
        counts = [len([c for c in p["metrics"]["candidates"] if c["accepted"]]) for p in analysis["points"]]
        wording = ("Completed drive-on/off spectra: %d\nOne accepted line per window: %d; two or more: %d; none: %d.\n"
            "Pairs assembled from matched disjoint windows are listed separately.\n\n"
            "The unresolved series do not support a coupling estimate or an avoided-crossing resonance bias.\n"
            "A frequency match alone does not locate the minimum splitting. A missing branch can reflect linewidth or readout visibility; it does not establish zero coupling.\n"
            "Physical Q5 remains unassigned unless the measured coupling and device mapping identify it.") % (
                len(counts), counts.count(1), sum(c >= 2 for c in counts), counts.count(0))
        text_box(item, wording, .6, 1.2, 12, 4.9, 21)
        text_box(item, "Model when branches are resolved: f+/- = (ft + ff)/2 +/- sqrt(((ft - ff)/2)^2 + J^2). Minimum splitting = 2|J|.", .6, 6.1, 12, .8, 17)
        item.notes_slide.notes_text_frame.text = json.dumps(unresolved, indent=2)
    for key, fit in analysis["coupled_fits"].items():
        if fit["status"] == "unresolved":
            continue
        item = slide("Coupled-mode analysis: " + key)
        if fit["status"] == "resolved":
            wording = "Resonance bias: %.8f +/- %.8f mA\nMinimum splitting: %.6f +/- %.6f MHz\nExchange |J|: %.6f +/- %.6f MHz\nUncertainties are conditional standard errors.\nFixed-frequency slope test: delta AIC %.3f (positive favors a slope)." % (
                fit["resonance_bias_mA"], fit["resonance_bias_standard_error_mA"],
                fit["minimum_splitting_Hz"] / 1e6, fit["minimum_splitting_standard_error_Hz"] / 1e6,
                fit["J_Hz"] / 1e6, fit["J_standard_error_Hz"] / 1e6,
                fit["fixed_frequency_test"]["delta_AIC_in_favor_of_linear_fixed"])
        else:
            wording = "No accepted coupling estimate.\n" + fit.get("reason", "A candidate fit needs model-quality or fixed-frequency-drift review.")
        text_box(item, wording, .6, 1.3, 12, 4.5, 23)
        text_box(item, "f+/- = (ft + ff)/2 +/- sqrt(((ft - ff)/2)^2 + J^2)\nMinimum splitting = 2|J|. Nearby lines are fit jointly and must beat the single-line model, with resolved linewidth and sufficient SNR.", .6, 5.7, 12, 1.1, 18)
        item.notes_slide.notes_text_frame.text = json.dumps(fit, indent=2)
        if fit.get("plot"):
            item = slide("Measured branches and accepted coupled-mode fit")
            fitted_image(item, fit["plot"], height=5.2)
            text_box(item, fit.get("readout_note", fit["uncertainty_note"]), .5, 6.45, 12.3, .55, 12)
            item.notes_slide.notes_text_frame.text = json.dumps(fit, indent=2)
    final = slide("Saved evidence and recorded cleanup")
    states = [m.get("final_sources") for m in manifests]
    parked = bool(all(ss and all(not s["output"] and abs(s["current_A"]) <= 1e-7
                  and s["mode"] == "CURR" and abs(s["range_A"] - .01) < 1e-9 for s in ss) for ss in states))
    text_box(final, ("Each included plan ended with Yoko3 and Yoko4 verified at 0 A/output off." if parked else "Final source cleanup is pending or missing for at least one included plan.") +
            "\nThese are recorded end-of-run states, not live instrument readbacks." +
            "\n\nRaw files: <run>/study_data/Data_h5/{qspec_ge,res_ge}/\nPlots, plans, instrument settings and launch sources: <run>/documentation/\nThis report's slide notes retain exact per-trace raw paths and settings.\n\nIncluded runs: " + ", ".join(m["plan"]["label"] for m in manifests), .6, 1.2, 12, 5.6, 18)
    final.notes_slide.notes_text_frame.text = json.dumps(dict(runs=[m["run"] for m in manifests], final_sources=states), indent=2)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(destination.stem + ".staging.pptx")
    presentation.save(temporary)
    reopened = Presentation(temporary)
    if len(reopened.slides) != len(presentation.slides):
        raise RuntimeError("PowerPoint reopen count mismatch")
    temporary.replace(destination)
    print(json.dumps(dict(pptx=str(destination), slides=len(presentation.slides), sources_parked=parked)))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="+")
    parser.add_argument("--analysis", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--conclusion", required=True)
    parser.add_argument("--drift-analysis", type=Path)
    parser.add_argument("--findings", type=Path)
    args = parser.parse_args()
    build(args.manifest, args.analysis, args.output, args.conclusion, args.drift_analysis, args.findings)


if __name__ == "__main__":
    main()
