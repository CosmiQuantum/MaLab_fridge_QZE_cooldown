"""Build a six-row PUCQ4 summary from validated canonical measurement artifacts.

This reporting utility does not operate hardware. Existing measurement images
are embedded unchanged, and newly drawn Rabi maps use the saved populations.
"""

import argparse
import copy
import datetime
import hashlib
import json
from pathlib import Path
import shutil
import sys
import zipfile

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.analyze_qspec_iq_series import decode_numeric, decode_text
from system_config import get_pucq4_flux_calibration


ROOT = Path(r"M:\_Data\20250822 - Olivia\pucq4_run_started_Oct_7\PUCQ4\pucq4_first_light")
FLUX = ROOT / "yoko_flux_campaign"
EXPERIMENTS = {
    "res_ge": ("Resonator spectroscopy", "res_spec_ge_plots"),
    "qspec_ge": ("Qubit spectroscopy", "qubit_spec_ge_plots"),
    "rabi_ge": ("Amplitude Rabi", "power_rabi_ge_plots"),
    "t1_ge": ("T1 energy relaxation", "T1_ge"),
    "t2_ge": ("T2R Ramsey", "Ramsey_ge_1"),
    "t2e_ge": ("T2E spin echo", "SpinEcho_ge"),
}
# These six source figures reserve an otherwise empty multipanel canvas.
# PowerPoint crops the viewport; the embedded image and raw data stay intact.
RESONATOR_CROP = (0.0, .12, .57, .40)  # left, top, right, bottom fractions


def textbox(slide, text, x, y, width, height, size=14, bold=False):
    box = slide.shapes.add_textbox(Inches(x), Inches(y), Inches(width), Inches(height))
    frame = box.text_frame
    frame.word_wrap = True
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.margin_top = frame.margin_bottom = 0
    for i, line in enumerate(str(text).split("\n")):
        p = frame.paragraphs[0] if i == 0 else frame.add_paragraph()
        p.text = line
        p.font.name = "Arial"
        p.font.size = Pt(size)
        p.font.bold = bold
        p.font.color.rgb = RGBColor(25, 43, 62)
    return box


def slide_base(prs, title, footer="", provenance=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    slide.background.fill.solid()
    slide.background.fill.fore_color.rgb = RGBColor(248, 250, 252)
    textbox(slide, title, .28, .18, 12.75, .55, 24 if len(title) < 66 else 21, True)
    textbox(slide, footer, .3, 7.05, 12.7, .34, 8)
    if provenance:
        slide.notes_slide.notes_text_frame.text = json.dumps(provenance, indent=2)
    return slide


def picture(slide, path, x, y, width, height, crop=None):
    with Image.open(path) as image:
        ratio = image.width / image.height
    if crop:
        left, top, right, bottom = crop
        ratio *= (1 - left - right) / (1 - top - bottom)
    actual_width = min(width, height * ratio)
    actual_height = actual_width / ratio
    shape = slide.shapes.add_picture(str(path), Inches(x + (width - actual_width) / 2),
                                    Inches(y + (height - actual_height) / 2),
                                    width=Inches(actual_width), height=Inches(actual_height))
    if crop:
        shape.crop_left, shape.crop_top, shape.crop_right, shape.crop_bottom = crop
    return shape


def table(slide, headers, rows, y=.98, height=5.4, font_size=11):
    shape = slide.shapes.add_table(len(rows) + 1, len(headers), Inches(.3), Inches(y),
                                    Inches(12.7), Inches(height))
    for ri, values in enumerate([headers] + rows):
        for ci, value in enumerate(values):
            cell = shape.table.cell(ri, ci)
            cell.text = str(value)
            cell.margin_left = cell.margin_right = Inches(.045)
            cell.margin_top = cell.margin_bottom = Inches(.035)
            cell.fill.solid()
            cell.fill.fore_color.rgb = RGBColor(218, 230, 240) if ri == 0 else RGBColor(255, 255, 255)
            for p in cell.text_frame.paragraphs:
                p.font.name = "Arial"
                p.font.size = Pt(font_size)
                p.font.bold = ri == 0
                p.font.color.rgb = RGBColor(25, 43, 62)
                p.alignment = PP_ALIGN.CENTER
    return shape


def gallery(prs, kind, records):
    footer = "Unchanged round-robin traces; individual acquisition dates and HDF5/configuration sources are in slide notes."
    if kind == "res_ge":
        footer += " Viewports omit unused panels and the common header; all six scans used 600 x 1 averages."
    if kind == "qspec_ge":
        footer += " M5/M6 qspec predates their final Ramsey frequency correction."
    elif kind == "t2e_ge":
        footer += " The legacy 'T2R' text inside echo plot titles is a label typo; these are T2E/SpinEcho data."
    slide = slide_base(prs, EXPERIMENTS[kind][0] + " | all six resonator rows", footer,
                       {row: entries[kind] for row, entries in records.items()})
    for index, row in enumerate(("1", "2", "3", "4", "5", "6")):
        x, y = .23 + (index % 3) * 4.32, .83 + (index // 3) * 3.02
        item = records[row][kind]
        identity = "physical Q4" if row == "5" else "physical Q6" if row == "6" else "physical ID unknown"
        textbox(slide, "M%s / index %d | %s | %s" % (row, int(row) - 1, identity, item["date"]),
                x, y, 4.25, .25, 9, True)
        picture(slide, item["plot"], x, y + .3, 4.25, 2.66,
                crop=RESONATOR_CROP if kind == "res_ge" else None)
    return slide


def plot_slide(prs, title, paths, footer, provenance, crop=None):
    slide = slide_base(prs, title, footer, provenance)
    width = 12.7 / len(paths)
    for index, path in enumerate(paths):
        picture(slide, path, .3 + index * width, .87, width, 6.0, crop=crop)
    return slide


def m4_records(run):
    report = json.loads((run / "documentation/m4_audit.json").read_text())
    entries = report["experiments"]
    assert set(EXPERIMENTS) <= set(entries), "M4 needs all six final experiment types"
    records = {}
    for kind, (_, folder) in EXPERIMENTS.items():
        entry = entries[kind]
        paths = list((run / "documentation" / folder).glob("*Q_4_*.png"))
        assert paths, (kind, folder)
        record = dict(h5=entry["h5"], config=entry["config"],
                      plot=str(max(paths, key=lambda p: p.stat().st_mtime)),
                      values={}, date=run.name[:10], run=str(run))
        if kind in ("t1_ge", "t2_ge", "t2e_ge"):
            name = {"t1_ge": "T1", "t2_ge": "T2", "t2e_ge": "T2E"}[kind]
            record["values"] = {name: [entry["tau_us"]], "Errors": [entry["error_us"]]}
            assert entry["independent"]["accepted"], kind + " independent fit not accepted"
            assert entry["R2"] > .85, kind + " saved round-robin fit not accepted"
            assert abs(entry["tau_us"] - entry["independent"]["tau_us"]) < max(
                entry["error_us"], .01 * entry["tau_us"]), kind + " fit disagreement"
        elif kind == "res_ge":
            record["values"]["Found Freqs"] = [entry["found_frequency_MHz"]]
        records[kind] = record
    assert entries["qspec_ge"]["R2"] > .85
    assert entries["rabi_ge"]["R2"] > .95
    for kind in ("qspec_ge", "rabi_ge", "t1_ge", "t2_ge", "t2e_ge"):
        calibration = entries[kind]["state_calibration"]
        assert calibration["population_reconstruction_verified"]
        assert calibration["matches_requested_shots"]
    return records, report


def summary_rows(records):
    rows = []
    for row in map(str, range(1, 7)):
        item = records[row]
        physical = "Q4" if row == "5" else "Q6" if row == "6" else "unknown"
        freq = item["rabi_ge"]["config"]["qubit_freq_ge"]
        fres = item["res_ge"]["values"]["Found Freqs"][0]
        coherence = []
        for kind, key in (("t1_ge", "T1"), ("t2_ge", "T2"), ("t2e_ge", "T2E")):
            values = item[kind]["values"]
            coherence.append("%.2f +/- %.2f" % (values[key][0], values["Errors"][0]))
        rows.append([physical + " / " + str(int(row) - 1), "M" + row,
                     "%.3f" % fres, "%.5f" % freq] + coherence)
    return rows


def enrich_records(records):
    """Keep actual axes and complete saved configs alongside every source image."""
    axes = {"res_ge": "freq_pts", "qspec_ge": "Frequencies", "rabi_ge": "Gains",
            "t1_ge": "Delay Times", "t2_ge": "Delay Times", "t2e_ge": "Delay Times"}
    for row, experiments in records.items():
        for kind, record in experiments.items():
            with h5py.File(record["h5"], "r") as handle:
                group = handle["Q" + row]
                record["saved_system_configuration"] = decode_text(group["Syst Config"])
                record["saved_experiment_configuration"] = decode_text(group["Exp Config"])
                if kind in axes:
                    axis = decode_numeric(group[axes[kind]])
                    if kind == "res_ge":
                        center = decode_numeric(group["freq_center"])
                        assert len(center) == 1
                        axis = axis + center[0]
                    assert np.isfinite(axis).all() and np.all(np.diff(axis) > 0)
                    record["actual_axis"] = dict(start=float(axis[0]), stop=float(axis[-1]),
                                                  step=float(np.median(np.diff(axis))), points=len(axis))
                if "ss_I_g" in group:
                    shots = decode_numeric(group["ss_I_g"])
                    record["calibration_shots_per_state"] = len(shots) if np.isfinite(shots).all() else None
                else:
                    record["calibration_shots_per_state"] = None


def sampling_slides(prs, records):
    spectrum_rows, delay_rows, calibration_rows = [], [], []
    for row in map(str, range(1, 7)):
        experiments = records[row]
        res, spec = experiments["res_ge"], experiments["qspec_ge"]
        rcfg, qcfg = res["config"], spec["config"]
        spectrum_rows.append(["M" + row,
                              "%.3f-%.3f" % (res["actual_axis"]["start"], res["actual_axis"]["stop"]),
                              "%.3g / %.3g" % (rcfg["res_gain_ge"], rcfg["res_length"]),
                              "%g x %g / %d" % (rcfg["reps"], rcfg["rounds"], res["actual_axis"]["points"]),
                              "%.3f-%.3f" % (spec["actual_axis"]["start"], spec["actual_axis"]["stop"]),
                              "%.3f / %.3g / %.3g" % (qcfg["res_freq_ge"], qcfg["res_gain_ge"], qcfg["res_length"])])
        cells = ["M" + row]
        for kind in ("rabi_ge", "t1_ge", "t2_ge", "t2e_ge"):
            axis, cfg = experiments[kind]["actual_axis"], experiments[kind]["config"]
            cell = "%.4g-%.4g / %.4g" % (axis["start"], axis["stop"], axis["step"])
            if kind in ("t2_ge", "t2e_ge"):
                cell += " / %.3g MHz" % cfg["ramsey_freq"]
            cells.append(cell)
        delay_rows.append(cells)
        calibration_rows.append(["M" + row] + [
            str(experiments[k]["calibration_shots_per_state"] or "not saved")
            for k in ("qspec_ge", "rabi_ge", "t1_ge", "t2_ge", "t2e_ge")])
    slide = slide_base(prs, "Spectroscopy acquisition bounds and readout settings",
                       "Bounds are actual realized frequency axes in MHz. Resonator cells: gain / us and reps x rounds / points. Qspec RO: MHz / gain / us.", records)
    table(slide, ["Row", "Res bounds MHz", "Res gain / us", "Res averages / N", "Qspec bounds MHz", "Qspec RO MHz / gain / us"],
          spectrum_rows, height=4.6, font_size=9)
    slide = slide_base(prs, "Rabi and coherence: actual sampling and phase detuning",
                       "Each cell: realized start-stop / median step. Rabi axes are DAC gain; all delay axes are us. T2 columns also give programmed phase detuning.", records)
    table(slide, ["Row", "Rabi gain span / step", "T1 delay span / step", "T2R delay / step / MHz", "T2E delay / step / MHz"],
          delay_rows, height=4.6, font_size=10)
    slide = slide_base(prs, "Population calibration shots preserved with each trace",
                       "Shots per prepared state from the canonical HDF5. Historical qspec files did not preserve the calibration arrays; no missing count is inferred.", records)
    table(slide, ["Row", "Qspec", "Rabi", "T1", "T2R", "T2E"], calibration_rows, height=4.6, font_size=12)


SUMMARY_HEADERS = ["Physical / index", "Res. row", "fres (MHz)", "f01 (MHz)",
                   "T1 (us)", "T2R (us)", "T2E (us)"]


def summary_slide(prs, records):
    slide = slide_base(prs, "PUCQ4 | six-qubit results at zero Yoko current",
                       "Python indices are zero-based; M1-M6 are resonator rows. Errors are statistical fit errors. Acquisition dates differ.", records)
    table(slide, SUMMARY_HEADERS, summary_rows(records), height=4.4, font_size=11)
    textbox(slide, "M1-M3: August 28. M4: September 8. M5/Q4 and M6/Q6 pulse/coherence: September 7.\n"
            "M5/M6 f01 uses the final Ramsey correction; their res/qspec traces retain the original acquisition frequencies.\n"
            "M4 has no documented physical-number assignment or flux line. Both known Yokogawas were at 0 A.",
            .35, 5.65, 12.6, .95, 12)
    return slide


def zero_config_slides(prs, records, m4_report):
    headers = ["Row / index", "f01 MHz", "qspec gain", "qspec us", "sigma ns", "pi gain", "RO MHz", "RO gain", "RO us"]
    rows = []
    timing_rows = []
    for row in map(str, range(1, 7)):
        item = records[row]
        pulse, spec = item["t1_ge"]["config"], item["qspec_ge"]["config"]
        rows.append(["M%s / %d" % (row, int(row) - 1), "%.5f" % pulse["qubit_freq_ge"],
                     "%.5g" % spec["qubit_gain_ge"], "%.3g" % spec["qubit_length_ge"],
                     "%.3f" % (1000 * pulse["sigma"]), "%.6f" % pulse["pi_amp"],
                     "%.3f" % pulse["res_freq_ge"], "%.3f" % pulse["res_gain_ge"], "%.3g" % pulse["res_length"]])
        cells = ["M" + row]
        for kind in ("qspec_ge", "rabi_ge", "t1_ge", "t2_ge", "t2e_ge"):
            cfg = item[kind]["config"]
            cells.append("%g x %g / %g" % (cfg["reps"], cfg["rounds"], cfg["steps"]))
        cells.append("%g / %g" % (pulse["relax_delay"], item["t2_ge"]["config"]["ramsey_freq"]))
        timing_rows.append(cells)
    slide = slide_base(prs, "Zero-current operating settings used by the saved traces",
                       "DAC gains are dimensionless. Gaussian total length = 4 sigma. Qspec and pulse/coherence runs can have different dates.", records)
    table(slide, headers, rows, height=4.5, font_size=10)
    textbox(slide, "Readout offsets are relative to the selected resonator base; the RO MHz column is the actual physical tone.\n"
            "M4 final spectroscopy: FWHM %.4f MHz, peak population %.3f, R-squared %.3f."
            % (m4_report["experiments"]["qspec_ge"]["fwhm_MHz"],
               m4_report["experiments"]["qspec_ge"]["peak_population"],
               m4_report["experiments"]["qspec_ge"]["R2"]), .35, 5.75, 12.6, .8, 12)
    slide = slide_base(prs, "Averaging and sampling for the zero-current traces",
                       "Each experiment cell: reps x rounds / frequency, gain or delay points. Full bounds and configs are in the accompanying JSON and notes.", records)
    table(slide, ["Row", "Qspec", "Rabi", "T1", "T2R", "T2E", "Relax us / T2 MHz"], timing_rows,
          height=4.5, font_size=11)


def final_sigma_rabi_map(device, points, output):
    currents = sorted(float(key) for key in points)
    populations, gains, sources = [], [], []
    for current in currents:
        selected = points["%+.1f" % current]["sigma_calibration"]["selected"]
        group_name = "Q5" if device == "q4" else "Q6"
        with h5py.File(selected["h5"], "r") as handle:
            gain = decode_numeric(handle[group_name]["Gains"])
            population = decode_numeric(handle[group_name]["Population"])
        assert np.isfinite(population).all() and len(gain) == len(population)
        assert selected["fit_quality"]["r_squared"] > .9
        gains.append(gain)
        populations.append(population)
        sources.append(selected)
    assert all(np.allclose(gain, gains[0], rtol=0, atol=1e-7) for gain in gains)
    x = np.asarray(currents)
    gain = gains[0]
    fig, ax = plt.subplots(figsize=(12, 6.4), constrained_layout=True)
    mesh = ax.pcolormesh(x, gain, np.asarray(populations).T, shading="nearest", cmap="viridis", vmin=-.3, vmax=1.5)
    ax.set_xlabel("Yokogawa current (mA)")
    ax.set_ylabel("Actual qubit drive gain (DAC units)")
    ax.set_xticks(x)
    ax.set_title(("Yoko 3 / physical Q4 / M5" if device == "q4" else "Yoko 4 / physical Q6 / M6")
                 + " | Rabi at the selected sigma for each current")
    fig.colorbar(mesh, ax=ax, label="SSF-calibrated population")
    path = output / (device + "_final_sigma_rabi_map.png")
    fig.savefig(path, dpi=220)
    plt.close(fig)
    return path, sources


def flux_slides(prs, output):
    manifest = json.loads((FLUX / "ramsey_sigma_coherence_manifest.json").read_text())
    wide_manifest = json.loads((FLUX / "single_sweep_qspec_manifest.json").read_text())
    provenance = {"manifest": str(FLUX / "ramsey_sigma_coherence_manifest.json"), "devices": {}}
    for device, index, label in (("q4", 4, "Yoko 3 / physical Q4 / M5"),
                                 ("q6", 5, "Yoko 4 / physical Q6 / M6")):
        points = manifest["devices"][device]["map"]["points"]
        assert len(points) == 9 and all(p["status"] == "complete" for p in points.values())
        rabi_path, rabi_sources = final_sigma_rabi_map(device, points, output)
        provenance["devices"][device] = dict(points=points, rabi_sources=rabi_sources)
        pulse_rows, readout_rows, coherence_rows = [], [], []
        for current in sorted(float(key) for key in points):
            point = points["%+.1f" % current]
            selected = point["sigma_calibration"]["selected"]
            readout = selected["readout"]
            calibration = get_pucq4_flux_calibration(
                current if index == 4 else 0, current if index == 5 else 0)[index]
            assert abs(calibration["qfreq_MHz"] - point["corrected_qfreq_MHz"]) < 1e-6
            assert abs(calibration["sigma_us"] - selected["sigma_us"]) < 1e-8
            pulse_rows.append(["%+.1f" % current, "%.6f" % point["corrected_qfreq_MHz"],
                               "%.3f" % (1000 * selected["sigma_us"]), "%.6f" % selected["pi_gain"],
                               "%.5g" % calibration["qspec_gain"], "%.3g" % calibration["qspec_length_us"],
                               "%d x %d / %d" % (selected["reps"], selected["rounds"], selected["gain_steps"])])
            readout_rows.append(["%+.1f" % current, "%.3f" % selected["res_base_MHz"],
                                 "%.3f" % readout["frequency_MHz"], "%.3f" % readout["offset_mhz"],
                                 "%.3f" % readout["gain"], "%.2f" % readout["length_us"],
                                 ("not recorded" if readout["fidelity"] is None
                                  else "%.2f" % (100 * readout["fidelity"]))])
            coherences = point["coherence"]["experiments"]
            coherence_rows.append(["%+.1f" % current] + [
                "%.2f +/- %.2f" % (coherences[k]["fit_value_us"], coherences[k]["fit_error_us"])
                for k in ("t1", "t2r", "t2e")] + [
                "%d x %d" % (coherences[k]["reps"], coherences[k]["rounds"])
                for k in ("t1", "t2r", "t2e")])
        slide = slide_base(prs, label + " | selected pulse settings at every current",
                           "Qspec columns are the shared narrow-tracking settings; the wide qspec map has its own recorded probe settings. Rabi: reps x rounds / points.", points)
        table(slide, ["I mA", "Corrected f01 MHz", "Sigma ns", "Pi gain", "Qspec gain", "Qspec us", "Rabi averages / points"], pulse_rows, font_size=10)
        slide = slide_base(prs, label + " | optimized readout settings",
                           "RO frequency = resonator base + offset. SSF is the repository histogram contrast metric; it is not assignment accuracy.", points)
        table(slide, ["I mA", "Base MHz", "RO MHz", "Offset MHz", "RO gain", "RO us", "SSF %"], readout_rows, font_size=11)
        slide = slide_base(prs, label + " | measured coherence and averaging",
                           "Times and statistical fit errors are in us. Normal calibrated-population round-robin traces at the selected sigma and corrected f01.", points)
        table(slide, ["I mA", "T1 us", "T2R us", "T2E us", "T1 averages", "T2R averages", "T2E averages"], coherence_rows, font_size=10)
        delay_rows = []
        for key in sorted(points, key=float):
            measurements = points[key]["coherence"]["experiments"]
            delay_rows.append([key] + [
                "%.3f-%.3f / %.4f" % (measurements[k]["actual_delay_start_us"],
                                      measurements[k]["actual_delay_stop_us"],
                                      measurements[k]["actual_delay_median_step_us"])
                for k in ("t1", "t2r", "t2e")] + [
                " / ".join(str(measurements[k]["ssf_shots_per_state"]) for k in ("t1", "t2r", "t2e"))])
        slide = slide_base(prs, label + " | coherence sampling and state calibration",
                           "Actual saved delay start-stop / median step, in us. Both T2 phase detunings: 0.5 MHz. Rabi/coherence relaxation: 250 us. No current interpolation.", points)
        table(slide, ["I mA", "T1 delay / step us", "T2R delay / step us", "T2E delay / step us", "SSF shots per state: T1 / R / E"],
              delay_rows, font_size=9)
        wide = wide_manifest["devices"][device]["map"]
        wide_rows = []
        for key in sorted(wide["points"], key=float):
            point = wide["points"][key]
            assert point["status"] == "complete"
            assert Path(point["h5"]).is_file()
            # Two historical Q6 rows have SNR just below the controller's 4
            # threshold. Their independent review is recorded in PUCQ4_STATE;
            # preserve that distinction rather than treating a status as a fit.
            marked_key = key + (" *" if not point["peak_visibility_pass"] else "")
            wide_rows.append([marked_key, "%.4g" % point["qspec_gain"], "%.3g" % point["qspec_length_us"],
                              "%.3f" % point["readout"]["frequency_MHz"],
                              "%.2f / %.3f" % (point["readout"]["length_us"], point["readout"]["gain"]),
                              "%d / %d" % (point["reps"], point["steps"]),
                              "%.3f-%.3f" % (point["actual_frequency_start_MHz"], point["actual_frequency_stop_MHz"])])
        slide = slide_base(prs, label + " | acquisition settings for the wide qspec map",
                           "Actual continuous-sweep settings. * Q6 -10/-2.5 mA: local SNR 3.76/3.78, below threshold 4; independently reviewed lines, with weak state separation.", wide)
        table(slide, ["I mA", "Probe gain", "Probe us", "RO MHz", "RO us / gain", "Reps / points", "Actual bounds MHz"],
              wide_rows, font_size=9)
        map_specs = [
            ("Resonator spectroscopy", FLUX / "aggregate_2d" / (device + "_res_spec_2d.png"),
             "Measured resonator response on the original current/frequency grid; source: aggregate_2d/flux_2d_provenance.json."),
            ("Qubit spectroscopy", FLUX / "single_sweep_qspec_2d" / (device + "_single_sweep_qspec_population_contrast_2d.png"),
             "Existing display: 20-MHz baseline subtraction; 1500 frequency bins; line maxima retained over 3 pixels; gamma 0.55; fixed 0-0.35 contrast scale. Raw HDF5 unchanged."),
            ("Rabi at selected sigma", rabi_path,
             "Saved signed SSF-calibrated populations at each current's selected sigma and corrected frequency; no row normalization or smoothing."),
        ]
        for kind, title in (("t1", "T1"), ("t2r", "T2R Ramsey"), ("t2e", "T2E spin echo")):
            map_specs.append((title, FLUX / "ramsey_sigma_coherence_2d" / (device + "_" + kind + "_population_2d.png"),
                              "Saved calibrated populations on actual delay axes. Existing map uses a three-point display mean and fixed color limits; fitted times use saved traces."))
        for title, path, footer in map_specs:
            assert path.is_file(), path
            plot_slide(prs, label + " | " + title + " map", [path], footer,
                       dict(image=str(path), current_records=points,
                            qspec_manifest=str(FLUX / "single_sweep_qspec_manifest.json"),
                            res_provenance=str(FLUX / "aggregate_2d/flux_2d_provenance.json")))
        plot_slide(prs, label + " | selected Gaussian sigma versus current",
                   [FLUX / "ramsey_sigma_coherence_2d" / (device + "_sigma_vs_current.png")],
                   "The selected sigma is current dependent; total Gaussian length is four times sigma. Use the corresponding current row in shared configuration.", points)
    return provenance


def readout_table(prs, records, m4_report):
    manifest = json.loads((FLUX / "ramsey_sigma_coherence_manifest.json").read_text())
    rows = []
    for row in map(str, range(1, 7)):
        pulse = records[row]["t1_ge"]["config"]
        if row in ("5", "6"):
            device = "q4" if row == "5" else "q6"
            selected = manifest["devices"][device]["map"]["points"]["+0.0"]["sigma_calibration"]["selected"]
            base, offset = selected["res_base_MHz"], selected["readout"]["offset_mhz"]
            fidelity = selected["readout"]["fidelity"]
        elif row == "4":
            ss = m4_report["experiments"]["ss_ge"]
            offset = pulse["res_freq_offset_ge"]
            base = pulse["res_freq_ge"] - offset
            fidelity = ss["contrast"]
        else:
            base = records[row]["res_ge"]["values"]["Found Freqs"][0]
            offset = pulse["res_freq_ge"] - base
            fidelity = records[row]["ss_ge"]["values"]["Fidelity"][0]
        rows.append(["M" + row, "%.3f" % base, "%.3f" % pulse["res_freq_ge"],
                     "%+.3f" % offset, "%.3f" % pulse["res_gain_ge"], "%.2f" % pulse["res_length"],
                     "%.2f" % (100 * fidelity)])
    slide = slide_base(prs, "Zero-current selected readout settings",
                       "Physical tone = selected base + offset. SSF is measured histogram contrast. Values come from the cited optimization/calibration runs.", records)
    table(slide, ["Row", "Base MHz", "RO MHz", "Offset MHz", "RO gain", "RO us", "SSF %"], rows,
          height=4.5, font_size=12)
    textbox(slide, "Shared non-multiplexed hardware: resonator DAC 0, qubit DAC 1, ADC 0; Nyquist zone 2.\n"
            "Nominal attenuation: resonator 10/15 dB, qubit 5/4 dB, ADC 17 dB. No attenuator changes in the M4 follow-up.",
            .35, 5.7, 12.6, .8, 12)


def m4_result_slide(prs, report, linewidth_target_MHz=.2):
    experiments = report["experiments"]
    spec, rabi, ss = [experiments[k] for k in ("qspec_ge", "rabi_ge", "ss_ge")]
    pulse = experiments["t1_ge"]["config"]
    rows = [["f01 / spectroscopy", "%.6f MHz\nFWHM %.4f MHz" % (spec["center_MHz"], spec["fwhm_MHz"]),
             "Gain %.4g; %.3g us\n%d averages x %d" % (spec["config"]["qubit_gain_ge"], spec["config"]["qubit_length_ge"],
                                                        spec["config"]["reps"], spec["config"]["rounds"]),
             "R2 %.4f\nPeak population %.3f" % (spec["R2"], spec["peak_population"])],
            ["Gaussian / amplitude Rabi", "Sigma %.3f ns\nPi gain %.6f" % (1000*pulse["sigma"], pulse["pi_amp"]),
             "%d averages x %d\n%d gain points" % (rabi["config"]["reps"], rabi["config"]["rounds"], rabi["actual_axis"]["points"]),
             "R2 %.4f" % rabi["R2"]],
            ["Readout / SSF", "%.3f MHz\nOffset %+.3f MHz" % (pulse["res_freq_ge"], pulse["res_freq_offset_ge"]),
             "Gain %.3g; %.3g us\n%d shots/state" % (pulse["res_gain_ge"], pulse["res_length"], ss["shots_per_state"]),
             "SSF contrast %.2f%%" % (100*ss["contrast"])]]
    for kind, label in (("t1_ge", "T1"), ("t2_ge", "T2R Ramsey"), ("t2e_ge", "T2E echo")):
        item = experiments[kind]
        config, axis = item["config"], item["actual_axis"]
        rows.append([label, "%.4f +/- %.4f us" % (item["tau_us"], item["error_us"]),
                     "%d averages x %d; %d points\nDelay %.3f-%.3f us" %
                     (config["reps"], config["rounds"], axis["points"], axis["start"], axis["stop"]),
                     "R2 %.4f\nIndependent fit passed" % item["R2"]])
    linewidth_reference = "https://arxiv.org/pdf/2005.12667#page=39"
    index = report.get("qubit_index", 3)
    slide = slide_base(prs, "M%d / index %d | final frequency, pulse calibration and coherence" % (index + 1, index),
                       "One complete round-robin pass; statistical fit errors. Linewidth comparison: Blais et al., Rev. Mod. Phys. 93, 025005 (2021), Sec. VI.B.1; stationary weak-drive model.",
                       {"measurement": report, "linewidth_model_reference": linewidth_reference,
                        "linewidth_model_assumption": "Stationary exponential coherence, weak drive; comparison only, not a fitted intrinsic linewidth"})
    table(slide, ["Measurement", "Measured result", "Actual acquisition settings", "Quality"], rows,
          height=4.9, font_size=11)
    line = "RO offset is relative to the measured resonator minimum. Coherence: %.3g us relaxation; %d calibration shots/state; T2 phase %.3g / %.3g MHz.\n" % (
        pulse["relax_delay"], experiments["t1_ge"]["state_calibration"]["shots_per_state"],
        experiments["t2_ge"]["config"]["ramsey_freq"], experiments["t2e_ge"]["config"]["ramsey_freq"])
    if spec["fwhm_MHz"] > linewidth_target_MHz:
        line += "The %.2f MHz linewidth target was not reached. For comparison, the stationary weak-drive estimate 1/(pi T2R) is %.3f MHz." % (
            linewidth_target_MHz, 1 / (np.pi * experiments["t2_ge"]["tau_us"]))
    textbox(slide, line, .35, 6.05, 12.6, .75, 11)


def append_progress(prs, entries):
    existing_titles = {shape.text for slide in prs.slides for shape in slide.shapes
                       if shape.has_text_frame}
    for entry in entries:
        if entry["title"] in existing_titles:
            continue
        slide = slide_base(prs, entry["title"], entry.get("status", ""), entry)
        textbox(slide, entry.get("subtitle", ""), .3, .8, 12.7, .3, 11)
        images = entry.get("images", [])
        if images:
            width = 12.7 / len(images)
            for index, path in enumerate(images):
                picture(slide, path, .3 + index * width, 1.12, width, 4.42)
        lines = entry.get("lines", [entry.get("footer", "")])
        textbox(slide, "\n".join(lines), .35, 5.65 if images else 1.35,
                12.6, 1.18 if images else 5.4, 12)


def update_existing(deck, records, m4_report, progress):
    before_bytes = deck.read_bytes()
    before_hash = hashlib.sha256(before_bytes).hexdigest()
    prs = Presentation(deck)
    previous_count = len(prs.slides)
    first = prs.slides[0]
    old = next(shape for shape in first.shapes if shape.has_table)
    old._element.getparent().remove(old._element)
    table(first, SUMMARY_HEADERS, summary_rows(records), y=1.1, height=3.9, font_size=11)
    for shape in first.shapes:
        if shape.has_text_frame and "M4" in shape.text and "excluded" in shape.text:
            shape.text_frame.text = (
                "M4/index 3 now has validated spectroscopy, Rabi and coherence. Its physical number is unknown. "
                "Physical Q4 is M5/Yoko 3; physical Q6 is M6/Yoko 4. Both sources were at 0 A. "
                "M1-M3: Aug 28; M4: Sep 8; M5/M6 pulse/coherence: Sep 7. "
                "Statistical fit errors are shown in us; acquisitions are not simultaneous."
            )
            shape.left, shape.top = Inches(.35), Inches(5.55)
            shape.width, shape.height = Inches(12.6), Inches(1.05)
            shape.text_frame.word_wrap = True
            for p in shape.text_frame.paragraphs:
                p.font.size = Pt(12)
    first_note_part = str(first.notes_slide.part.partname).lstrip("/")
    first.notes_slide.notes_text_frame.text = json.dumps(
        {"updated_zero_current_table": records, "M4_final_audit": m4_report}, indent=2)
    append_progress(prs, progress)
    m4_result_slide(prs, m4_report)
    for kind in ("res_ge", "qspec_ge", "rabi_ge", "t1_ge", "t2_ge", "t2e_ge"):
        record = records["4"][kind]
        footer = "Normal one-pass round-robin output. Physical number unassigned. Actual configurations and HDF5 provenance are in slide notes."
        if kind == "t2e_ge":
            footer += " Legacy title typo: 'T2R' refers here to the T2E echo fit."
        plot_slide(prs, "M4 / index 3 | final " + EXPERIMENTS[kind][0], [record["plot"]],
                   footer, record, crop=RESONATOR_CROP if kind == "res_ge" else None)
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = deck.parent / "_PowerPoint_archive"
    archive.mkdir(exist_ok=True)
    backup = archive / (deck.stem + "_pre_m4_coh_update_" + stamp + ".pptx")
    pending = archive / (deck.stem + "_m4_coh_pending_" + stamp + ".pptx")
    assert not backup.exists() and not pending.exists()
    prs.save(pending)
    assert hashlib.sha256(deck.read_bytes()).hexdigest() == before_hash, "Deck changed during reporting"
    with zipfile.ZipFile(deck) as old_zip, zipfile.ZipFile(pending) as new_zip:
        parts = [name for name in old_zip.namelist()
                 if name.startswith(("ppt/slides/", "ppt/media/", "ppt/notesSlides/"))]
        changed = [name for name in parts if old_zip.read(name) != new_zip.read(name)]
        # Only the requested first-slide table, note, and its note relationship
        # can change; every previous evidence slide/media part must be identical.
        permitted = {"ppt/slides/slide1.xml", "ppt/slides/_rels/slide1.xml.rels",
                     first_note_part}
        assert set(changed) <= permitted, changed
    shutil.copy2(deck, backup)
    pending.replace(deck)
    return dict(deck=str(deck), backup=str(backup), previous_slides=previous_count,
                final_slides=len(prs.slides), changed_original_parts=changed,
                checked_original_parts=len(parts), first_added_slide=previous_count + 1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--m4-run", required=True, type=Path)
    parser.add_argument("--archive-sources", required=True, type=Path)
    parser.add_argument("--progress-spec", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--update-existing", type=Path)
    args = parser.parse_args()
    records = json.loads(args.archive_sources.read_text())
    records["4"], m4_report = m4_records(args.m4_run)
    assert set(records) == set(map(str, range(1, 7)))
    for row, experiments in records.items():
        for kind in EXPERIMENTS:
            assert Path(experiments[kind]["h5"]).is_file()
            assert Path(experiments[kind]["plot"]).is_file()
    enrich_records(records)
    progress = json.loads(args.progress_spec.read_text(encoding="utf-8-sig"))["slides"] if args.progress_spec else []
    assets = args.output.parent / (args.output.stem + "_artifacts")
    assets.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333333), Inches(7.5)
    summary_slide(prs, records)
    m4_result_slide(prs, m4_report)
    zero_config_slides(prs, records, m4_report)
    readout_table(prs, records, m4_report)
    sampling_slides(prs, records)
    for kind in EXPERIMENTS:
        gallery(prs, kind, records)
    summary_progress = [entry for entry in progress if entry.get("summary_include", False)]
    append_progress(prs, summary_progress)
    flux = flux_slides(prs, assets)
    report = dict(created=datetime.datetime.now().astimezone().isoformat(),
                  records=records, M4_audit=m4_report, progress=progress,
                  summary_progress_titles=[entry["title"] for entry in summary_progress],
                  flux=flux, summary_deck=str(args.output), summary_slides=len(prs.slides))
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    archive = args.output.parent / "_PowerPoint_archive"
    archive.mkdir(exist_ok=True)
    pending = archive / (args.output.stem + "_pending_" + stamp + ".pptx")
    prs.save(pending)
    assert len(Presentation(pending).slides) == len(prs.slides)
    if args.output.exists():
        backup = archive / (args.output.stem + "_before_" + stamp + ".pptx")
        assert not backup.exists()
        shutil.copy2(args.output, backup)
    pending.replace(args.output)
    if args.update_existing:
        report["existing_deck_update"] = update_existing(args.update_existing, records, m4_report, progress)
    (assets / "summary_provenance.json").write_text(json.dumps(report, indent=2))
    (assets / "chip_trace_sources.json").write_text(json.dumps(records, indent=2))
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("records", "M4_audit", "progress", "flux")}, indent=2))


if __name__ == "__main__":
    main()
