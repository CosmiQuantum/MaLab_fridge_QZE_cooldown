"""Build the reusable PUCQ4 first-light characterization deck.

This is analysis-only: it reads existing PNG artifacts and writes one PPTX.
"""

from pathlib import Path
import json
import shutil
import sys

from PIL import Image, ImageDraw, ImageFont
from pptx import Presentation
from pptx.enum.text import MSO_AUTO_SIZE, PP_ALIGN
from pptx.util import Inches, Pt


ROOT = Path(r"M:\_Data\20250822 - Olivia\pucq4_run_started_Aug_3\PUCQ4\pucq4_first_light")
FULL = ROOT / "rr_full5_optimized_0mA" / "2026-08-28_16-21-09" / "documentation"
QR = ROOT / "rr_m5_m6_qspec_rabi_followup_0mA" / "2026-08-28_16-55-52" / "documentation"
Q6SPEC = ROOT / "rr_m6_qspec_gain_followup_0mA" / "2026-08-28_17-05-21" / "documentation"
COH = ROOT / "rr_m5_m6_coherence_followup_0mA" / "2026-08-28_17-10-46" / "documentation"
LENOPT = ROOT / "ssf_length_optimization_0mA" / "2026-08-28_16-05-29" / "documentation"
GFOPT = ROOT / "ssf_gain_frequency_optimization_0mA" / "2026-08-28_16-07-41" / "documentation"
OUT = ROOT / "PUCQ4_first_light_characterization.pptx"
ORIGINAL = Path(r"C:\Users\Ma Quantum Lab\Downloads\PUCQ4 Initial Characterization (1).pptx")
FLUX = ROOT / "yoko_flux_campaign"
FLUX_2D = FLUX / "aggregate_2d"
FLUX_MANIFEST = FLUX / "flux_campaign_manifest.json"
FLUX_PROVENANCE = FLUX_2D / "flux_2d_provenance.json"
SINGLE_SWEEP_MANIFEST = FLUX / "single_sweep_qspec_manifest.json"
SINGLE_SWEEP_2D = FLUX / "single_sweep_qspec_2d"
SINGLE_RABI_MANIFEST = FLUX / "single_sweep_rabi_manifest.json"
SINGLE_RABI_2D = FLUX / "single_sweep_rabi_2d"
RAMSEY_RABI_MANIFEST = FLUX / "ramsey_corrected_rabi_manifest.json"
RAMSEY_RABI_2D = FLUX / "ramsey_corrected_rabi_2d"
SIGMA_COHERENCE_MANIFEST = FLUX / "ramsey_sigma_coherence_manifest.json"
SIGMA_COHERENCE_2D = FLUX / "ramsey_sigma_coherence_2d"
FREQUENCY_SUMMARY_PROVENANCE = (
    SIGMA_COHERENCE_2D / "frequency_axis_summary_provenance.json"
)
FREQUENCY_ONLY_SUMMARY_PROVENANCE = (
    SIGMA_COHERENCE_2D / "frequency_only_summary_provenance.json"
)
PRE_WIDE_DECK = ROOT / "PUCQ4_first_light_characterization_pre_wide_qspec_20260903.pptx"
PRE_RABI_DECK = (
    ROOT / "PUCQ4_first_light_characterization_pre_fixed_rabi_20260907.pptx"
)
PRE_RAMSEY_RABI_DECK = (
    ROOT / "PUCQ4_first_light_characterization_pre_ramsey_rabi_20260907.pptx"
)
PRE_SIGMA_COHERENCE_DECK = (
    ROOT / "PUCQ4_first_light_characterization_pre_sigma_coherence_20260907.pptx"
)
PRE_FREQUENCY_SUMMARY_DECK = (
    ROOT
    / "PUCQ4_first_light_characterization_pre_frequency_summary_20260908.pptx"
)
PRE_FREQUENCY_ONLY_SUMMARY_DECK = (
    ROOT
    / "PUCQ4_first_light_characterization_pre_frequency_only_summary_20260908.pptx"
)
SEGMENTED_DECK_BACKUP = (
    ROOT / "PUCQ4_first_light_characterization_superseded_segmented_qspec_20260904.pptx"
)

ROWS = [1, 2, 3, 5, 6]
SUMMARY = [
    ("Q1 / M1", "R1", 8919.830, 5517.694, 31.45, 16.53, 37.13),
    ("Q2 / M2", "R2", 8951.650, 5438.714, 8.77, 11.71, 15.88),
    ("Q3 / M3", "R3", 8974.970, 5557.334, 45.79, 42.85, 77.22),
    ("Q4 / M5", "R5", 9014.590, 7084.778, 7.63, 5.28, 11.26),
    ("Q6 / M6", "R6", 9059.310, 6522.540, 3.34, 1.53, 4.19),
]


def format_current_averaging(averaging_groups, current_count):
    default_statistics, default_currents = max(
        averaging_groups.items(), key=lambda item: len(item[1])
    )
    default_reps, default_rounds = default_statistics
    if len(default_currents) == current_count:
        return f"{default_reps} reps x {default_rounds} rounds at all currents"
    exceptions = []
    for (reps, rounds), group_currents in averaging_groups.items():
        if (reps, rounds) == default_statistics:
            continue
        current_text = ", ".join(
            f"{current:+.1f}" for current in group_currents
        )
        verb = "uses" if len(group_currents) == 1 else "use"
        exceptions.append(
            f"{current_text} mA {verb} {reps} reps x {rounds} rounds"
        )
    return (
        f"{default_reps} reps x {default_rounds} rounds at other currents; "
        + "; ".join(exceptions)
    )


def newest(folder, pattern):
    paths = list(folder.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No {pattern} under {folder}")
    return max(paths, key=lambda p: p.stat().st_mtime)


def add_title(slide, title, subtitle=None):
    box = slide.shapes.add_textbox(Inches(0.15), Inches(0.08), Inches(13.03), Inches(0.62))
    frame = box.text_frame
    frame.auto_size = MSO_AUTO_SIZE.NONE
    frame.word_wrap = False
    frame.margin_left = 0
    frame.margin_right = 0
    frame.margin_top = 0
    frame.margin_bottom = 0
    p = frame.paragraphs[0]
    p.text = title
    p.font.name = "Arial"
    p.font.size = Pt(19 if len(title) > 62 else (23 if len(title) > 52 else 25))
    p.font.bold = True
    p.alignment = PP_ALIGN.LEFT
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.35), Inches(7.0), Inches(12.6), Inches(0.42))
        sub_frame = sub.text_frame
        sub_frame.word_wrap = True
        sub_frame.margin_left = 0
        sub_frame.margin_right = 0
        sub_frame.margin_top = 0
        sub_frame.margin_bottom = 0
        p = sub_frame.paragraphs[0]
        p.text = subtitle
        p.font.size = Pt(8)


def add_raster_title(slide, title, file_stem, output_folder=None):
    """Use a fixed raster title for stable PowerPoint/COM rendering."""
    if output_folder is None:
        output_folder = SINGLE_SWEEP_2D
    output_folder.mkdir(parents=True, exist_ok=True)
    title_path = output_folder / f"ppt_title_{file_stem}.png"
    canvas = Image.new("RGB", (2100, 100), "white")
    draw = ImageDraw.Draw(canvas)
    font_path = Path(r"C:\Windows\Fonts\arialbd.ttf")
    font_size = 62
    while font_size >= 36:
        font = ImageFont.truetype(str(font_path), font_size)
        bounds = draw.textbbox((0, 0), title, font=font)
        if bounds[2] - bounds[0] <= 2070 and bounds[3] - bounds[1] <= 88:
            break
        font_size -= 1
    draw.text((8, 2 - bounds[1]), title, fill="black", font=font)
    canvas.save(title_path)
    slide.shapes.add_picture(
        str(title_path), Inches(0.15), Inches(0.08),
        width=Inches(13.03), height=Inches(0.62),
    )


def add_grid_slide(prs, title, paths, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, title, notes)
    positions = [(0.25, 0.72), (4.55, 0.72), (8.85, 0.72), (2.4, 3.88), (6.7, 3.88)]
    for path, (x, y) in zip(paths, positions):
        add_contained_picture(slide, path, x, y, 4.15, 3.0)
    return slide


def add_full_width_plot_slide(
        prs, title, path, notes, raster_title_stem=None,
        raster_title_folder=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    if raster_title_stem:
        add_raster_title(
            slide, title, raster_title_stem,
            output_folder=raster_title_folder,
        )
        add_title(slide, "", notes)
    else:
        add_title(slide, title, notes)
    add_contained_picture(slide, path, 0.25, 0.7, 12.83, 6.25)
    return slide


def add_two_plot_slide(prs, title, paths, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, title, notes)
    for path, x in zip(paths, (0.18, 6.75)):
        add_contained_picture(slide, path, x, 0.72, 6.40, 6.25)
    return slide


def add_contained_picture(slide, path, x, y, box_width, box_height):
    """Fit an image inside a slide box without changing its aspect ratio."""
    with Image.open(path) as image:
        pixel_width, pixel_height = image.size
    image_ratio = pixel_width / pixel_height
    box_ratio = box_width / box_height
    if image_ratio >= box_ratio:
        width = box_width
        height = width / image_ratio
    else:
        height = box_height
        width = height * image_ratio
    left = x + (box_width - width) / 2
    top = y + (box_height - height) / 2
    slide.shapes.add_picture(
        str(path), Inches(left), Inches(top),
        width=Inches(width), height=Inches(height)
    )


def add_summary(prs):
    summary = list(SUMMARY)
    if FLUX_MANIFEST.exists() and FLUX_PROVENANCE.exists():
        manifest = json.loads(FLUX_MANIFEST.read_text())
        provenance = json.loads(FLUX_PROVENANCE.read_text())
        for device_key, summary_index in (("q4", 3), ("q6", 4)):
            point = manifest["devices"][device_key].get("points", {}).get("+0.0", {})
            experiments = (provenance.get("devices", {}).get(device_key, {})
                           .get("currents", {}).get("+0.0", {}).get("experiments", {}))
            coherence = [
                experiments.get(name, {}).get("fit_value_us")
                for name in ("t1", "t2r", "t2e")
            ]
            if point.get("status") == "complete" and all(value is not None for value in coherence):
                identity, resonator, *_ = summary[summary_index]
                summary[summary_index] = (
                    identity, resonator, point["res_base_MHz"], point["qfreq_MHz"], *coherence,
                )
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "PUCQ4 first light at Yoko current = 0 A")
    rows, cols = len(summary) + 1, 7
    table = slide.shapes.add_table(rows, cols, Inches(0.35), Inches(1.1), Inches(12.6), Inches(3.7)).table
    headers = ["Physical / row", "Resonator", "fres (MHz)", "f01 (MHz)", "T1 (us)", "T2R (us)", "T2E (us)"]
    for c, text in enumerate(headers):
        table.cell(0, c).text = text
    for r, values in enumerate(summary, 1):
        for c, value in enumerate(values):
            table.cell(r, c).text = str(value) if isinstance(value, str) else f"{value:.3f}"
    for row in table.rows:
        for cell in row.cells:
            for p in cell.text_frame.paragraphs:
                p.font.size = Pt(12)
                p.alignment = PP_ALIGN.CENTER
    note = slide.shapes.add_textbox(Inches(0.5), Inches(5.2), Inches(12), Inches(1.2))
    note.text_frame.text = (
        "M4 / Python index 3 is unresolved and intentionally excluded. Physical Q4 is the flux-tunable M5 row "
        "on Yoko 3 (192.168.1.73); physical Q6/M6 is on Yoko 4 (192.168.1.77). Both were held at 0 A."
    )
    note.text_frame.paragraphs[0].font.size = Pt(18)


def newest_flux_artifact(run_name, experiment_folder, pattern="*.png"):
    run_root = ROOT / run_name
    middle = f"{experiment_folder}/" if experiment_folder else ""
    matches = list(run_root.glob(f"*/documentation/{middle}{pattern}"))
    if not matches:
        raise FileNotFoundError(f"No {middle}{pattern} under {run_root}")
    return max(matches, key=lambda path: path.stat().st_mtime)


def add_manifest_flux_tables(prs):
    manifest = json.loads(FLUX_MANIFEST.read_text())
    provenance = json.loads(FLUX_PROVENANCE.read_text()) if FLUX_PROVENANCE.exists() else {}
    metadata = {"q4": ("Yoko 3", "Q4/M5"), "q6": ("Yoko 4", "Q6/M6")}
    currents = ["-10.0", "-7.5", "-5.0", "-2.5", "+0.0", "+2.5", "+5.0", "+7.5", "+10.0"]
    for device_key, (yoko, qubit) in metadata.items():
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        add_title(slide, f"{yoko} / physical {qubit} current-sweep status table")
        headers = [
            "I (mA)", "fres", "f01", "FWHM", "qspec gain", "sigma", "pi gain",
            "RO len/gain/off", "SSF", "T1", "T2R", "T2E", "status",
        ]
        table = slide.shapes.add_table(10, len(headers), Inches(0.12), Inches(0.9),
                                       Inches(13.08), Inches(4.65)).table
        for col, value in enumerate(headers):
            table.cell(0, col).text = value
        points = manifest["devices"][device_key].get("points", {})
        surveys = manifest["devices"][device_key].get("wide_res_survey", {})
        for row, current in enumerate(currents, 1):
            point = points.get(current, {})
            survey = surveys.get(current, {})
            readout = point.get("readout", {}) or {}
            experiments = (provenance.get("devices", {}).get(device_key, {})
                           .get("currents", {}).get(current, {}).get("experiments", {}))
            coherence = [
                experiments.get(name, {}).get("fit_value_us")
                for name in ("t1", "t2r", "t2e")
            ]
            status = point.get("status", "not final")
            if status == "complete":
                qualifiers = []
                if point.get("sigma_recalibration_required"):
                    qualifiers.append("Rabi provisional")
                if readout and readout.get("fidelity", 1.0) < 0.20:
                    qualifiers.append("low-fidelity SSF")
                linewidth_window = point.get("qspec_fwhm_acceptance_MHz")
                if linewidth_window and list(linewidth_window) != [0.099, 0.21]:
                    qualifiers.append("population-optimal broadened qspec")
                if qualifiers:
                    status = "complete; " + ", ".join(qualifiers)
            values = [current, point.get("res_base_MHz", survey.get("res_freq_MHz")), point.get("qfreq_MHz"),
                      point.get("qspec_fwhm_MHz"), point.get("qspec_gain"), point.get("sigma_us"), point.get("pi_amp"),
                      None if not readout else f"{readout.get('length_us', 0):.1f}/{readout.get('gain', 0):.3f}/{readout.get('offset_mhz', 0):+.2f}",
                      readout.get("fidelity") if readout else None, *coherence,
                      status]
            for col, value in enumerate(values):
                table.cell(row, col).text = "—" if value is None else (f"{value:.5g}" if isinstance(value, float) else str(value))
        for table_row in table.rows:
            for cell in table_row.cells:
                for paragraph in cell.text_frame.paragraphs:
                    paragraph.font.size = Pt(7)
                    paragraph.alignment = PP_ALIGN.CENTER
        note = slide.shapes.add_textbox(Inches(0.4), Inches(5.8), Inches(12.5), Inches(1.0))
        note_text = (
            "MHz for frequencies/FWHM; us for sigma, T1, T2R, and T2E. Dashes exclude diagnostic or failed fits; "
            "fres is the final local resonator base used downstream, seeded by the widened ±15 MHz survey; "
            "the resonator heatmap shows the complete widened acquisitions."
        )
        if device_key == "q4":
            complete_count = sum(point.get("status") == "complete" for point in points.values())
            minus_ten = points.get("-10.0", {})
            if minus_ten.get("status") == "complete":
                note_text += (
                    f" Q4 is characterized at {complete_count}/9 currents. The expanded -10 mA sweep found "
                    f"{minus_ten['qfreq_MHz']:.4f} MHz; its Rabi sigma is explicitly provisional "
                    "when sigma_recalibration_required is true."
                )
            else:
                note_text += (
                    f" Q4 is validated at {complete_count}/9 currents; -10 mA status is "
                    f"{minus_ten.get('status', 'not final')}."
                )
        note.text_frame.text = note_text
        note.text_frame.word_wrap = True
        note.text_frame.paragraphs[0].font.size = Pt(11)


def add_final_q4_minus_ten(prs):
    """Add only the validated final -10 mA Q4 artifacts and settings."""
    manifest = json.loads(FLUX_MANIFEST.read_text())
    point = manifest["devices"]["q4"].get("points", {}).get("-10.0", {})
    if point.get("status") != "complete":
        raise RuntimeError(
            "The clean final deck requires a completed Q4/M5 -10 mA calibration"
        )
    paths = [
        newest_flux_artifact(
            "flux_yoko3_q4_-10.0mA_post_readout_calibration",
            "qubit_spec_ge_plots", "*q5.png",
        ),
        newest_flux_artifact(
            "flux_yoko3_q4_-10.0mA_post_readout_calibration",
            "power_rabi_ge_plots", "*q5.png",
        ),
        newest_flux_artifact(
            "flux_yoko3_q4_-10.0mA_coherence", "T1_ge", "*q5.png",
        ),
        newest_flux_artifact(
            "flux_yoko3_q4_-10.0mA_coherence", "Ramsey_ge_1", "*q5.png",
        ),
        newest_flux_artifact(
            "flux_yoko3_q4_-10.0mA_coherence", "SpinEcho_ge", "*q5.png",
        ),
    ]
    readout = point["readout"]
    rabi_note = (
        "Rabi is retained as provisional, as authorized; sigma retuning remains a future step."
        if point.get("sigma_recalibration_required")
        else "The fitted pi amplitude is inside the 0.6-0.7 target window."
    )
    readout_note = (
        "SSF is a low-fidelity optimum used only for averaged calibration."
        if readout["fidelity"] < 0.20
        else "SSF passed the standard 0.20 campaign floor."
    )
    add_grid_slide(
        prs,
        "Final -10 mA Q4/M5 characterization",
        paths,
        f"f01={point['qfreq_MHz']:.6f} MHz; FWHM={point['qspec_fwhm_MHz']:.6f} MHz; "
        f"sigma={point['sigma_us']:.8f} us; pi={point['pi_amp']:.6f}; "
        f"readout={readout['length_us']:.1f} us/{readout['gain']:.4f}/"
        f"{readout['offset_mhz']:+.3f} MHz; SSF={readout['fidelity']:.4f}. "
        f"{rabi_note} {readout_note}",
    )


def add_uniform_qspec_settings_table(prs):
    manifest = json.loads(FLUX_MANIFEST.read_text())
    uniform = manifest["devices"]["q4"].get("uniform_qspec_map", {})
    q4_provenance = json.loads(FLUX_PROVENANCE.read_text())["devices"]["q4"]
    provenance = q4_provenance["uniform_qspec_map"]
    points = uniform.get("points", {})
    currents = ["-10.0", "-7.5", "-5.0", "-2.5", "+0.0", "+2.5", "+5.0", "+7.5", "+10.0"]
    if any(points.get(current, {}).get("status") != "complete" for current in currents):
        raise RuntimeError("The uniform-qspec settings table requires all nine Q4 rows")
    if (not provenance.get("all_current_rows_available")
            or not provenance.get("frequency_arrays_identical")):
        raise RuntimeError(
            "The uniform-qspec settings table requires nine readable, identical-axis HDF5 rows"
        )
    rabi_rows = q4_provenance.get("currents", {})
    unverified_rabi_frequencies = [
        current for current in currents
        if not (rabi_rows.get(current, {}).get("experiments", {})
                .get("rabi", {}).get("qfreq_matches_final"))
    ]
    if unverified_rabi_frequencies:
        raise RuntimeError(
            "The Q4 deck requires the selected Rabi logs to record every final "
            f"qspec frequency; unverified currents: {unverified_rabi_frequencies}"
        )
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Q4/M5 identical-grid qspec acquisition settings")
    headers = [
        "I (mA)", "final f01", "map gain", "qspec len", "fres", "RO freq",
        "RO len", "RO gain", "RO off", "sigma", "pi", "SSF g (I,Q)", "SSF e (I,Q)",
    ]
    table = slide.shapes.add_table(10, len(headers), Inches(0.12), Inches(0.9),
                                   Inches(13.08), Inches(4.7)).table
    for column, value in enumerate(headers):
        table.cell(0, column).text = value
    for row, current in enumerate(currents, 1):
        point = points[current]
        readout = point["readout"]
        centroids = point["ssf_centroids"]
        values = [
            current,
            f"{point['qfreq_seed_MHz']:.5f}",
            f"{point['qspec_gain']:.6g}",
            f"{point['qspec_length_us']:.1f}",
            f"{point['res_base_MHz']:.3f}",
            f"{readout.get('frequency_MHz', point['res_base_MHz'] + readout['offset_mhz']):.3f}",
            f"{readout['length_us']:.1f}",
            f"{readout['gain']:.4f}",
            f"{readout['offset_mhz']:+.3f}",
            f"{point['sigma_us']:.7f}",
            f"{point['pi_amp']:.6f}",
            f"{centroids['g_I']:.4g}, {centroids['g_Q']:.4g}",
            f"{centroids['e_I']:.4g}, {centroids['e_Q']:.4g}",
        ]
        for column, value in enumerate(values):
            table.cell(row, column).text = value
    for table_row in table.rows:
        for cell in table_row.cells:
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(7)
                paragraph.alignment = PP_ALIGN.CENTER
    note = slide.shapes.add_textbox(Inches(0.35), Inches(5.72), Inches(12.65), Inches(1.35))
    actual_start = uniform.get("actual_frequency_start_MHz", uniform["frequency_start_MHz"])
    actual_stop = uniform.get("actual_frequency_stop_MHz", uniform["frequency_stop_MHz"])
    actual_step = uniform.get(
        "actual_frequency_median_step_MHz", uniform["frequency_step_MHz"]
    )
    minimum_edge_margin = min(
        min(
            float(points[current]["qfreq_seed_MHz"]) - float(actual_start),
            float(actual_stop) - float(points[current]["qfreq_seed_MHz"]),
        )
        for current in currents
    )
    note.text_frame.text = (
        f"All rows requested {uniform['frequency_start_MHz']:.0f}-{uniform['frequency_stop_MHz']:.0f} MHz / "
        f"{uniform['frequency_step_MHz']:.3f} MHz; actual QICK axis {actual_start:.6f}-{actual_stop:.6f} MHz / "
        f"{actual_step:.6f} MHz median spacing, {uniform['steps']:,} points, "
        f"{uniform['reps']} qspec averages, {uniform['ssf_shots_per_state']} SSF shots/state, "
        f"{uniform['ssf_relax_delay_us']:.0f} us SSF relaxation; every final f01 is at least "
        f"{minimum_edge_margin:.1f} MHz from an acquired edge. The final f01 values come from the accepted wide/local "
        "frequency scans used for Rabi. Each bias has one fixed SSF centroid pair for its complete frequency row; "
        "all rows use the same signed g=0/e=1 population convention and no row normalization."
    )
    note.text_frame.word_wrap = True
    note.text_frame.paragraphs[0].font.size = Pt(10)


def add_flux_2d_maps(prs):
    slide_specs = [
        ("qspec", "Local high-resolution qubit-frequency calibration windows",
         "These are the final narrow local scans, not the definitive full-range Q4 map. X = bias current, Y = actual swept qubit frequency, color = measured signal magnitude."),
        ("qspec_population", "Local high-resolution calibrated-population windows",
         "These are the final narrow local scans, not the definitive full-range Q4 map. X = bias current, Y = actual swept qubit frequency, color = SSF-calibrated population."),
        ("res_spec", "Yokogawa current × resonator-frequency maps",
         "X = bias current, Y = actual swept resonator frequency, color = measured signal magnitude. No interpolation across currents."),
        ("found_frequencies_vs_current", "Found spectroscopy centers versus bias current",
         "Separate accepted fitted qubit-transition and resonator-center curves; failed points are omitted rather than inferred."),
        ("qspec_fwhm_vs_current", "Qspec FWHM versus bias current",
         "Final measured qspec linewidths; shading marks the nominal 0.1-0.2 MHz target. Out-of-target values remain visible, including the +5 mA population-optimal power-broadened result."),
        ("rabi", "Yokogawa current × Rabi-gain maps",
         "X = current, Y = the sorted acquired DAC-gain axis, color = signed SSF population (g=0, e=1). No column is reversed to improve appearance."),
        ("sigma_pi", "Current-indexed sigma and pi-amplitude calibration",
         "Required pi window 0.6-0.7. Only completed current points are plotted; rejected diagnostic pulse candidates are omitted."),
        ("t1", "Yokogawa current × T1-delay maps",
         "Numeric round-robin delay traces only; missing coherence rows remain blank."),
        ("t2r", "Yokogawa current × T2 Ramsey-delay maps",
         "Numeric round-robin delay traces only; configured detuning and sampling are retained in each HDF5 provenance record."),
        ("t2e", "Yokogawa current × T2 echo-delay maps",
         "Numeric round-robin delay traces only; missing/rejected rows are not synthesized."),
    ]
    for stem, title, note in slide_specs:
        suffix = "_2d.png" if stem in {"qspec", "qspec_population", "res_spec", "rabi", "sigma_pi", "t1", "t2r", "t2e"} else ".png"
        paths = [FLUX_2D / f"q4_{stem}{suffix}", FLUX_2D / f"q6_{stem}{suffix}"]
        add_two_plot_slide(prs, title, paths, note)
        if stem == "qspec_population":
            for device_key, device_label in (("q4", "Yoko 3 / Q4-M5"), ("q6", "Yoko 4 / Q6-M6")):
                add_full_width_plot_slide(
                    prs,
                    f"{device_label} - complete local-window qspec magnitude curves",
                    FLUX_2D / f"{device_key}_qspec_full_curves.png",
                    "All samples from each final local calibration window are shown. These panels intentionally use local Y ranges; the separate Q4 identical-grid slides show the full 5950-7200 MHz range.",
                )
                add_full_width_plot_slide(
                    prs,
                    f"{device_label} - complete local-window population curves",
                    FLUX_2D / f"{device_key}_qspec_population_full_curves.png",
                    "SSF-calibrated population only. Missing or non-final bias points remain explicitly unavailable rather than being synthesized.",
                )
    uniform_paths = [
        FLUX_2D / "q4_uniform_qspec_signal_2d.png",
        FLUX_2D / "q4_uniform_qspec_population_2d.png",
    ]
    if all(path.exists() for path in uniform_paths):
        uniform = json.loads(FLUX_MANIFEST.read_text())["devices"]["q4"]["uniform_qspec_map"]
        actual_start = uniform.get("actual_frequency_start_MHz", uniform["frequency_start_MHz"])
        actual_stop = uniform.get("actual_frequency_stop_MHz", uniform["frequency_stop_MHz"])
        actual_step = uniform.get(
            "actual_frequency_median_step_MHz", uniform["frequency_step_MHz"]
        )
        add_uniform_qspec_settings_table(prs)
        add_two_plot_slide(
            prs,
            "Q4/M5 identical-grid qubit spectroscopy at all Yoko 3 biases",
            uniform_paths,
            f"Both maps use the identical actual {actual_start:.6f}-{actual_stop:.6f} MHz QICK axis "
            f"at every bias with {actual_step:.6f} MHz median spacing ({uniform['frequency_start_MHz']:.0f}-"
            f"{uniform['frequency_stop_MHz']:.0f} MHz / {uniform['frequency_step_MHz']:.3f} MHz requested). Each row uses "
            f"{uniform['reps']:,} qspec averages plus {uniform['ssf_shots_per_state']:,} SSF shots per state "
            f"({uniform['ssf_relax_delay_us']:.0f} us SSF relax delay), and the current-specific "
            "map gain, pulse length, and readout listed in the settings table. "
            "Left: acquired |I+iQ| magnitude with one global color scale. Right: SSF population on a fixed g=0/e=1 "
            "scale with no per-row normalization; the final local qspec centers are overlaid. The -10 mA population "
            "column is retained but low-confidence because SSF=0.124 and its Rabi calibration is provisional.",
        )
        add_full_width_plot_slide(
            prs,
            "Q4/M5 complete identical-range qspec magnitude curves",
            FLUX_2D / "q4_uniform_qspec_signal_full_curves.png",
            f"Every panel shows the entire actual {actual_start:.3f}-{actual_stop:.3f} MHz acquisition. Insets magnify +/-2 MHz around the "
            "validated local transition using samples from the same full scan.",
        )
        add_full_width_plot_slide(
            prs,
            "Q4/M5 complete identical-range calibrated-population qspec curves",
            FLUX_2D / "q4_uniform_qspec_population_full_curves.png",
            "All frequencies and all nine biases are shown. Each bias uses one SSF g/e calibration for its "
            "entire row; every row shares the same physical population definition and is not independently normalized.",
        )
        add_full_width_plot_slide(
            prs,
            "Q4/M5 full-grid frequency check against Rabi settings",
            FLUX_2D / "q4_uniform_qspec_frequency_check.png",
            "The orange markers are measured population extrema from the complete identical-grid scans. "
            "The black curve is the actual f01 parsed from the selected Rabi run at each current and has been "
            "verified against the final local fit; the lower panel shows "
            "their signed difference and labels the local full-grid population SNR.",
        )


def add_wide_lower_qspec_slides(prs):
    """Append lower-frequency population, readout, and peak-check slides."""
    if not FLUX_MANIFEST.exists() or not FLUX_PROVENANCE.exists():
        return
    manifest = json.loads(FLUX_MANIFEST.read_text())
    provenance = json.loads(FLUX_PROVENANCE.read_text())
    currents = ["-10.0", "-7.5", "-5.0", "-2.5", "+0.0", "+2.5", "+5.0", "+7.5", "+10.0"]
    labels = {
        "q4": "Yoko 3 / physical Q4 / M5",
        "q6": "Yoko 4 / physical Q6 / M6",
    }
    for device_key, label in labels.items():
        image_path = FLUX_2D / f"{device_key}_wide_lower_qspec_population_2d.png"
        local_path = (
            FLUX_2D
            / f"{device_key}_wide_lower_qspec_population_local_curves.png"
        )
        wide = manifest.get("devices", {}).get(device_key, {}).get(
            "wide_uniform_qspec_map", {}
        )
        wide_provenance = provenance.get("devices", {}).get(device_key, {}).get(
            "wide_uniform_qspec_map", {}
        )
        points = wide.get("points", {})
        if (not image_path.exists()
                or any(points.get(current, {}).get("status") != "complete"
                       for current in currents)
                or not wide_provenance.get("all_current_rows_available")
                or not wide_provenance.get("frequency_arrays_identical")):
            continue
        slide = prs.slides.add_slide(prs.slide_layouts[6])
        add_title(slide, f"{label}: lower-frequency calibrated qspec map")
        add_contained_picture(slide, image_path, 0.12, 0.72, 7.15, 6.22)

        heading = slide.shapes.add_textbox(
            Inches(7.48), Inches(0.72), Inches(5.60), Inches(0.30)
        )
        heading.text_frame.text = "Current-specific shifted resonator readout"
        heading.text_frame.paragraphs[0].font.size = Pt(12)
        heading.text_frame.paragraphs[0].font.bold = True
        headers = ["I", "f01", "fres", "RO freq", "RO len", "RO gain", "RO off"]
        table = slide.shapes.add_table(
            10, len(headers), Inches(7.45), Inches(1.05), Inches(5.72), Inches(4.55)
        ).table
        widths = [0.52, 0.77, 0.77, 0.82, 0.72, 0.78, 0.78]
        for column, width in enumerate(widths):
            table.columns[column].width = Inches(width)
        for column, value in enumerate(headers):
            table.cell(0, column).text = value
        for row, current in enumerate(currents, 1):
            point = points[current]
            readout = point["readout"]
            values = [
                current,
                f"{point['qfreq_MHz']:.2f}",
                f"{point['res_base_MHz']:.2f}",
                f"{readout['frequency_MHz']:.2f}",
                f"{readout['length_us']:.1f}",
                f"{readout['gain']:.4f}",
                f"{readout['offset_mhz']:+.2f}",
            ]
            for column, value in enumerate(values):
                table.cell(row, column).text = value
        for row_index, table_row in enumerate(table.rows):
            for cell in table_row.cells:
                cell.text_frame.margin_left = 0
                cell.text_frame.margin_right = 0
                for paragraph in cell.text_frame.paragraphs:
                    paragraph.font.size = Pt(6.8)
                    paragraph.font.bold = row_index == 0
                    paragraph.alignment = PP_ALIGN.CENTER

        actual_start = wide_provenance["actual_frequency_start_MHz"]
        actual_stop = wide_provenance["actual_frequency_stop_MHz"]
        actual_step = wide_provenance["actual_frequency_median_step_MHz"]
        note = slide.shapes.add_textbox(
            Inches(7.48), Inches(5.75), Inches(5.55), Inches(1.25)
        )
        note.text_frame.word_wrap = True
        note.text_frame.text = (
            f"Requested {wide['frequency_start_MHz']:.0f}-{wide['frequency_stop_MHz']:.0f} MHz; "
            f"actual stitched axis {actual_start:.3f}-{actual_stop:.3f} MHz with "
            f"{actual_step:.6f} MHz median spacing and {wide['reps']} averages/band. "
            f"Each current and each of {len(wide['segments_MHz'])} overlapping bands uses its own "
            f"built-in {wide['ssf_shots_per_state']}-shot/state SSF calibration. Population is the "
            "saved signed g=0/e=1 projection; no row normalization. f01, fres, and RO frequency are "
            "MHz; RO length is us."
        )
        note.text_frame.paragraphs[0].font.size = Pt(8)
        if local_path.exists():
            add_full_width_plot_slide(
                prs,
                f"{label}: measured wide-qspec peak verification",
                local_path,
                "Each panel shows the actual swept samples within +/-5 MHz of the current-specific "
                "f01. Light blue is raw saved population, dark blue is a five-sample mean, the dashed "
                "line is the validated drive frequency used for the row, and red marks a bounded local "
                "positive-Lorentzian center. Broad or noisy cases remain visibly broad; panel scales are "
                "local and no population normalization was applied.",
            )


def add_single_sweep_qspec_slides(prs):
    """Append continuous one-sweep qspec maps and their readout provenance."""
    if not SINGLE_SWEEP_MANIFEST.exists():
        return
    manifest = json.loads(SINGLE_SWEEP_MANIFEST.read_text(encoding="utf-8"))
    currents = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]
    labels = {
        "q4": "Yoko 3 / physical Q4 / M5",
        "q6": "Yoko 4 / physical Q6 / M6",
    }
    for device_key, label in labels.items():
        title_label = label.replace("physical ", "")
        population_path = (
            SINGLE_SWEEP_2D
            / f"{device_key}_single_sweep_qspec_population_contrast_2d.png"
        )
        magnitude_path = (
            SINGLE_SWEEP_2D
            / f"{device_key}_single_sweep_qspec_signal_contrast_2d.png"
        )
        local_path = (
            SINGLE_SWEEP_2D
            / f"{device_key}_single_sweep_qspec_local_curves.png"
        )
        map_data = (
            manifest.get("devices", {}).get(device_key, {}).get("map", {})
        )
        points = map_data.get("points", {})
        if (not all(path.exists() for path in (
                population_path, magnitude_path, local_path))
                or any(points.get(f"{current:+.1f}", {}).get("status") != "complete"
                       for current in currents)):
            continue

        slide = prs.slides.add_slide(prs.slide_layouts[6])
        add_raster_title(
            slide,
            f"{title_label}: wide qspec population-contrast map",
            f"{device_key}_population",
        )
        add_contained_picture(slide, population_path, 0.10, 0.70, 7.20, 6.24)
        heading = slide.shapes.add_textbox(
            Inches(7.48), Inches(0.72), Inches(5.60), Inches(0.30)
        )
        heading.text_frame.text = "Drive and shifted readout used for each sweep"
        heading.text_frame.paragraphs[0].font.size = Pt(12)
        heading.text_frame.paragraphs[0].font.bold = True
        headers = [
            "I mA", "qG", "q us", "seed", "peak", "fres", "RO", "r us",
            "rG", "off",
        ]
        table = slide.shapes.add_table(
            10, len(headers), Inches(7.42), Inches(1.05), Inches(5.80), Inches(4.58)
        ).table
        widths = [0.48, 0.52, 0.48, 0.68, 0.68, 0.68, 0.68, 0.48, 0.54, 0.54]
        for column, width in enumerate(widths):
            table.columns[column].width = Inches(width)
        for column, value in enumerate(headers):
            table.cell(0, column).text = value
        for row, current in enumerate(currents, 1):
            point = points[f"{current:+.1f}"]
            readout = point["readout"]
            values = [
                f"{current:+.1f}",
                f"{point['qspec_gain']:.3g}",
                f"{point['qspec_length_us']:.1f}",
                f"{point['qfreq_seed_MHz']:.2f}",
                f"{point['peak_metrics']['frequency_MHz']:.2f}",
                f"{point['res_base_MHz']:.2f}",
                f"{readout['frequency_MHz']:.2f}",
                f"{readout['length_us']:.1f}",
                f"{readout['gain']:.4f}",
                f"{readout['offset_mhz']:+.2f}",
            ]
            for column, value in enumerate(values):
                table.cell(row, column).text = value
        for row_index, table_row in enumerate(table.rows):
            for cell in table_row.cells:
                cell.text_frame.margin_left = 0
                cell.text_frame.margin_right = 0
                for paragraph in cell.text_frame.paragraphs:
                    paragraph.font.size = Pt(5.7)
                    paragraph.font.bold = row_index == 0
                    paragraph.alignment = PP_ALIGN.CENTER
        minimum_snr = min(
            points[f"{current:+.1f}"]["peak_metrics"]["positive_snr"]
            for current in currents
        )
        first_point = points[f"{currents[0]:+.1f}"]
        actual_start = first_point["actual_frequency_start_MHz"]
        actual_stop = first_point["actual_frequency_stop_MHz"]
        actual_step = first_point["actual_frequency_median_step_MHz"]
        note = slide.shapes.add_textbox(
            Inches(7.48), Inches(5.76), Inches(5.55), Inches(1.18)
        )
        note.text_frame.word_wrap = True
        note.text_frame.text = (
            f"One uninterrupted sweep/current. Actual common hardware axis: "
            f"{actual_start:.3f}-{actual_stop:.3f} MHz at {actual_step:.6f} MHz "
            f"median spacing (requested {map_data['frequency_start_MHz']:.0f}-"
            f"{map_data['frequency_stop_MHz']:.0f} MHz at "
            f"{map_data['frequency_step_MHz']:.3f} MHz); "
            f"{map_data['steps']:,} points, {map_data['reps']:,} reps/point, "
            f"and one {map_data['ssf_shots_per_state']:,}-shot/state built-in "
            f"SSF calibration for the complete row. Minimum local positive-peak "
            f"SNR = {minimum_snr:.1f}. The same qspec gain/length and complete "
            "frequency grid used in the local panels are used here. Color is "
            "calibrated Delta Pe after subtracting only a slow interpolated "
            "20-MHz block-median baseline; there is no row amplitude or peak "
            "normalization. Ordinary bins show their five-sample-smoothed mean. "
            "The +/-1 MHz measured-line bin keeps its local maximum and is repeated "
            "over three display pixels so a sub-MHz line survives slide rendering. "
            "A fixed gamma=0.55 color transform increases low-level contrast. "
            "Canonical HDF5 samples are unchanged; all frequencies are MHz."
        )
        note.text_frame.paragraphs[0].font.size = Pt(7.2)

        add_full_width_plot_slide(
            prs,
            f"{title_label}: wide qspec signal-contrast map",
            magnitude_path,
            "Same single HDF5 sweep, complete actual frequency grid, qspec settings, "
            "currents, and shifted resonator readout as the calibrated-population "
            "map. Color is the absolute SSF-axis displacement from an interpolated "
            "20-MHz block-median baseline after a fixed eleven-sample mean, in raw "
            "I/Q units; there is no row amplitude or peak normalization. The +/-1 "
            "MHz measured-line bin keeps its local maximum and is repeated over "
            "three display pixels; gamma=0.55 only changes color contrast. Raw HDF5 "
            "samples are unchanged.",
            raster_title_stem=f"{device_key}_signal",
        )
        add_full_width_plot_slide(
            prs,
            f"{title_label}: wide qspec peak at every current",
            local_path,
            "These are samples from the same uninterrupted full-range acquisitions, "
            "shown within +/-3 MHz of the validated frequency. Light blue is saved "
            "Pe, dark blue is a fixed five-sample mean, black dashed is the validated "
            "seed used only for SSF preparation, and red is the measured positive "
            "maximum. Each panel has labeled physical Pe limits so weak and low-SSF "
            "cases remain visible; values are not normalized or rescaled.",
            raster_title_stem=f"{device_key}_local",
        )


def add_single_sweep_rabi_slides(prs):
    """Append fixed-sigma current-by-gain Rabi maps and settings."""
    if not SINGLE_RABI_MANIFEST.exists():
        return
    manifest = json.loads(SINGLE_RABI_MANIFEST.read_text(encoding="utf-8"))
    currents = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]
    labels = {
        "q4": "Yoko 3 / physical Q4 / M5",
        "q6": "Yoko 4 / physical Q6 / M6",
    }
    for device_key, label in labels.items():
        title_label = label.replace("physical ", "")
        heatmap_path = (
            SINGLE_RABI_2D
            / f"{device_key}_fixed_sigma_rabi_population_2d.png"
        )
        curves_path = (
            SINGLE_RABI_2D
            / f"{device_key}_fixed_sigma_rabi_population_curves.png"
        )
        map_data = (
            manifest.get("devices", {}).get(device_key, {}).get("map", {})
        )
        points = map_data.get("points", {})
        if (not heatmap_path.exists() or not curves_path.exists()
                or any(
                    points.get(f"{current:+.1f}", {}).get("status")
                    != "complete" for current in currents
                )):
            continue

        slide = prs.slides.add_slide(prs.slide_layouts[6])
        add_raster_title(
            slide,
            f"{title_label}: fixed-sigma Rabi population map",
            f"{device_key}_fixed_rabi",
            output_folder=SINGLE_RABI_2D,
        )
        add_contained_picture(slide, heatmap_path, 0.10, 0.70, 7.20, 6.24)
        heading = slide.shapes.add_textbox(
            Inches(7.48), Inches(0.72), Inches(5.60), Inches(0.30)
        )
        heading.text_frame.text = "Qspec frequency and shifted readout used"
        heading.text_frame.paragraphs[0].font.size = Pt(12)
        heading.text_frame.paragraphs[0].font.bold = True
        headers = [
            "I mA", "qfreq", "fres", "RO", "r us", "rG", "off", "pi",
        ]
        table = slide.shapes.add_table(
            10, len(headers), Inches(7.42), Inches(1.05),
            Inches(5.80), Inches(4.58),
        ).table
        widths = [0.55, 0.75, 0.75, 0.75, 0.62, 0.70, 0.70, 0.70]
        for column, width in enumerate(widths):
            table.columns[column].width = Inches(width)
        for column, value in enumerate(headers):
            table.cell(0, column).text = value
        for row, current in enumerate(currents, 1):
            point = points[f"{current:+.1f}"]
            readout = point["readout"]
            values = [
                f"{current:+.1f}",
                f"{point['qfreq_MHz']:.2f}",
                f"{point['res_base_MHz']:.2f}",
                f"{readout['frequency_MHz']:.2f}",
                f"{readout['length_us']:.1f}",
                f"{readout['gain']:.4f}",
                f"{readout['offset_mhz']:+.2f}",
                f"{point['fitted_pi_amp']:.3f}",
            ]
            for column, value in enumerate(values):
                table.cell(row, column).text = value
        for row_index, table_row in enumerate(table.rows):
            for cell in table_row.cells:
                cell.text_frame.margin_left = 0
                cell.text_frame.margin_right = 0
                for paragraph in cell.text_frame.paragraphs:
                    paragraph.font.size = Pt(6.3)
                    paragraph.font.bold = row_index == 0
                    paragraph.alignment = PP_ALIGN.CENTER

        first_point = points[f"{currents[0]:+.1f}"]
        minimum_span = min(
            points[f"{current:+.1f}"]["population_metrics"][
                "population_span_5_95"
            ] for current in currents
        )
        note = slide.shapes.add_textbox(
            Inches(7.48), Inches(5.76), Inches(5.55), Inches(1.18)
        )
        note.text_frame.word_wrap = True
        note.text_frame.text = (
            f"Fixed sigma={map_data['fixed_sigma_us']:.3f} us for every current "
            f"on this device. Requested gain {map_data['gain_start']:.3f}-"
            f"{map_data['gain_stop']:.3f} with {map_data['gain_steps']} points; "
            f"actual common axis {first_point['actual_gain_start']:.6f}-"
            f"{first_point['actual_gain_stop']:.6f}, median step "
            f"{first_point['actual_gain_median_step']:.8f}. "
            f"{map_data['reps']} reps x {map_data['rounds']} round; "
            f"Rabi relax {map_data['rabi_relax_delay_us']:.0f} us. Each current "
            f"uses {map_data['ssf_shots_per_state']} SSF shots/state with "
            f"{map_data['ssf_relax_delay_us']:.0f} us relax. qfreq is the "
            "observed continuous-wide-qspec population peak; RO is the exact "
            "physical tone used for that qspec row. Color is signed physical "
            f"Pe (g=0/e=1), fixed -0.30 to 1.50 scale, no row normalization; "
            f"minimum 5-95% population span={minimum_span:.3f}. Attenuation dB: "
            "res DAC [10,15], qubit DAC [5,4], ADC 17."
        )
        note.text_frame.paragraphs[0].font.size = Pt(7.1)

        add_full_width_plot_slide(
            prs,
            f"{title_label}: fixed-sigma Rabi curves at every current",
            curves_path,
            f"The same canonical HDF5 rows as the heatmap, shown directly on "
            f"one physical population scale. Sigma is fixed at "
            f"{map_data['fixed_sigma_us']:.3f} us; every row has "
            f"{map_data['gain_steps']} gain points, {map_data['reps']} reps x "
            f"{map_data['rounds']} round, and its own saved "
            f"{map_data['ssf_shots_per_state']}-shot/state SSF calibration. "
            "Light blue is saved Pe, dark blue is a fixed three-sample mean, "
            "red is the saved fit, and the dashed black line is its first pi "
            "extremum. No row is offset, divided, normalized, or rescaled.",
            raster_title_stem=f"{device_key}_fixed_rabi_curves",
            raster_title_folder=SINGLE_RABI_2D,
        )


def add_ramsey_corrected_rabi_slides(prs):
    """Append Ramsey correction evidence and corrected-frequency Rabi maps."""
    if not RAMSEY_RABI_MANIFEST.exists():
        return
    manifest = json.loads(RAMSEY_RABI_MANIFEST.read_text(encoding="utf-8"))
    currents = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]
    labels = {
        "q4": "Yoko 3 / physical Q4 / M5",
        "q6": "Yoko 4 / physical Q6 / M6",
    }
    for device_key, label in labels.items():
        title_label = label.replace("physical ", "")
        map_data = (
            manifest.get("devices", {}).get(device_key, {}).get("map", {})
        )
        points = map_data.get("points", {})
        ramsey_path = (
            RAMSEY_RABI_2D
            / f"{device_key}_ramsey_frequency_correction_curves.png"
        )
        heatmap_path = (
            RAMSEY_RABI_2D
            / f"{device_key}_ramsey_corrected_rabi_population_2d.png"
        )
        curves_path = (
            RAMSEY_RABI_2D
            / f"{device_key}_ramsey_corrected_rabi_population_curves.png"
        )
        if (not ramsey_path.exists() or not heatmap_path.exists()
                or not curves_path.exists()
                or any(
                    points.get(f"{current:+.1f}", {}).get("status")
                    != "complete" for current in currents
                )):
            continue

        ramsey_settings = map_data["ramsey"]
        averaging_groups = {}
        for current in currents:
            ramsey = points[f"{current:+.1f}"]["ramsey"]
            statistics = (int(ramsey["reps"]), int(ramsey["rounds"]))
            averaging_groups.setdefault(statistics, []).append(current)
        averaging_text = format_current_averaging(
            averaging_groups, len(currents)
        )
        maximum_stderr = max(
            points[f"{current:+.1f}"]["ramsey"]["fit_metrics"][
                "independent_fit"
            ]["frequency_stderr_MHz"] for current in currents
        )
        minimum_r2 = min(
            points[f"{current:+.1f}"]["ramsey"]["fit_metrics"][
                "saved_fit_r_squared"
            ] for current in currents
        )
        add_full_width_plot_slide(
            prs,
            f"{title_label}: Ramsey qubit-frequency correction",
            ramsey_path,
            "Canonical round-robin T2RMeasurement.adjust_qspec traces. "
            f"Established sweep retained: {ramsey_settings['start_us']:.1f}-"
            f"{ramsey_settings['stop_us']:.1f} us, "
            f"{ramsey_settings['steps']} points, "
            f"{ramsey_settings['detuning_MHz']:.1f} MHz intentional detuning, "
            f"{ramsey_settings['relax_delay_us']:.0f} us relaxation; averaging "
            f"raised to {ramsey_settings['reps']} reps x "
            f"{ramsey_settings['rounds']} rounds. Correction is seed - fitted "
            f"fringe + detuning. Minimum saved-fit R2={minimum_r2:.3f}; maximum "
            f"independent frequency standard error={maximum_stderr:.4f} MHz. "
            "Each row uses the prior measured pi amplitude with the sigma at "
            "which that pi was calibrated, plus the unchanged shifted readout.",
            raster_title_stem=f"{device_key}_ramsey_correction",
            raster_title_folder=RAMSEY_RABI_2D,
        )

        slide = prs.slides.add_slide(prs.slide_layouts[6])
        add_raster_title(
            slide,
            f"{title_label}: Ramsey-corrected 60 ns Rabi population map",
            f"{device_key}_ramsey_corrected_rabi",
            output_folder=RAMSEY_RABI_2D,
        )
        add_contained_picture(slide, heatmap_path, 0.10, 0.70, 7.18, 6.24)
        heading = slide.shapes.add_textbox(
            Inches(7.40), Inches(0.72), Inches(5.75), Inches(0.30)
        )
        heading.text_frame.text = "Frequencies and shifted readout used"
        heading.text_frame.paragraphs[0].font.size = Pt(12)
        heading.text_frame.paragraphs[0].font.bold = True
        headers = [
            "I", "seed", "f01", "df", "fres", "RO", "r us", "rG", "off",
        ]
        table = slide.shapes.add_table(
            10, len(headers), Inches(7.36), Inches(1.05),
            Inches(5.88), Inches(4.58),
        ).table
        widths = [0.48, 0.69, 0.69, 0.53, 0.66, 0.66, 0.55, 0.60, 0.55]
        for column, width in enumerate(widths):
            table.columns[column].width = Inches(width)
        for column, value in enumerate(headers):
            table.cell(0, column).text = value
        for row, current in enumerate(currents, 1):
            point = points[f"{current:+.1f}"]
            ramsey = point["ramsey"]
            rabi = point["rabi"]
            readout = rabi["readout"]
            values = [
                f"{current:+.1f}",
                f"{ramsey['seed_qfreq_MHz']:.2f}",
                f"{rabi['qfreq_MHz']:.3f}",
                f"{ramsey['frequency_correction_MHz']:+.3f}",
                f"{rabi['res_base_MHz']:.2f}",
                f"{readout['frequency_MHz']:.2f}",
                f"{readout['length_us']:.1f}",
                f"{readout['gain']:.4f}",
                f"{readout['offset_mhz']:+.2f}",
            ]
            for column, value in enumerate(values):
                table.cell(row, column).text = value
        for row_index, table_row in enumerate(table.rows):
            for cell in table_row.cells:
                cell.text_frame.margin_left = 0
                cell.text_frame.margin_right = 0
                for paragraph in cell.text_frame.paragraphs:
                    paragraph.font.size = Pt(5.8)
                    paragraph.font.bold = row_index == 0
                    paragraph.alignment = PP_ALIGN.CENTER

        rabi_settings = map_data["rabi"]
        first_point = points[f"{currents[0]:+.1f}"]["rabi"]
        minimum_span = min(
            points[f"{current:+.1f}"]["rabi"]["population_metrics"][
                "population_span_5_95"
            ] for current in currents
        )
        note = slide.shapes.add_textbox(
            Inches(7.43), Inches(5.73), Inches(5.66), Inches(1.22)
        )
        note.text_frame.word_wrap = True
        note.text_frame.text = (
            f"Fixed sigma={rabi_settings['fixed_sigma_us']:.3f} us for both "
            f"Yokogawas. Gain {rabi_settings['gain_start']:.3f}-"
            f"{rabi_settings['gain_stop']:.3f}, "
            f"{rabi_settings['gain_steps']} requested points; actual common "
            f"axis {first_point['actual_gain_start']:.6f}-"
            f"{first_point['actual_gain_stop']:.6f}, step "
            f"{first_point['actual_gain_median_step']:.8f}. "
            f"{rabi_settings['reps']} reps x {rabi_settings['rounds']} rounds, "
            f"{rabi_settings['relax_delay_us']:.0f} us relax; "
            f"{rabi_settings['ssf_shots_per_state']} SSF shots/state, "
            f"{rabi_settings['ssf_relax_delay_us']:.0f} us relax. Color is "
            "saved signed physical Pe, fixed -0.30 to 1.50, with no row "
            f"normalization; minimum 5-95% span={minimum_span:.3f}. "
            "f01 is the Ramsey-corrected drive; fres is resonator base and RO "
            "is the exact unchanged physical readout tone. Frequencies MHz."
        )
        note.text_frame.paragraphs[0].font.size = Pt(6.7)

        add_full_width_plot_slide(
            prs,
            f"{title_label}: Ramsey-corrected 60 ns Rabi curves",
            curves_path,
            "The same canonical HDF5 rows as the heatmap, ordered by current "
            "and shown on one physical population scale. Light blue is saved "
            "Pe, dark blue is a fixed three-sample display mean, red is the "
            "saved fit when available, and the dashed black line is the fitted "
            "first pi extremum. Sigma is 0.060 us for every row; no curve is "
            "offset, divided, normalized, or rescaled.",
            raster_title_stem=f"{device_key}_ramsey_corrected_rabi_curves",
            raster_title_folder=RAMSEY_RABI_2D,
        )


def add_sigma_coherence_slides(prs):
    """Append fresh Ramsey, tuned-sigma, and coherence-map evidence."""
    if not SIGMA_COHERENCE_MANIFEST.exists():
        return
    manifest = json.loads(
        SIGMA_COHERENCE_MANIFEST.read_text(encoding="utf-8")
    )
    currents = [-10.0, -7.5, -5.0, -2.5, 0.0, 2.5, 5.0, 7.5, 10.0]
    labels = {
        "q4": "Yoko 3 / physical Q4 / M5",
        "q6": "Yoko 4 / physical Q6 / M6",
    }
    coherence_titles = {
        "t1": "T1",
        "t2r": "T2R Ramsey",
        "t2e": "T2E spin echo",
    }
    for device_key, label in labels.items():
        title_label = label.replace("physical ", "")
        map_data = manifest.get("devices", {}).get(device_key, {}).get(
            "map", {}
        )
        points = map_data.get("points", {})
        required_plots = [
            SIGMA_COHERENCE_2D / f"{device_key}_fresh_ramsey_corrections.png",
            SIGMA_COHERENCE_2D / f"{device_key}_sigma_vs_current.png",
            SIGMA_COHERENCE_2D / f"{device_key}_selected_sigma_rabi_curves.png",
        ]
        for experiment in coherence_titles:
            required_plots.extend([
                SIGMA_COHERENCE_2D
                / f"{device_key}_{experiment}_population_2d.png",
                SIGMA_COHERENCE_2D
                / f"{device_key}_{experiment}_population_curves.png",
            ])
        if (
            any(not path.exists() for path in required_plots)
            or any(
                points.get(f"{current:+.1f}", {}).get("status")
                != "complete" for current in currents
            )
        ):
            continue

        ramsey_settings = map_data["ramsey"]
        maximum_stderr = max(
            points[f"{current:+.1f}"]["ramsey"]["fit_metrics"][
                "independent_fit"
            ]["frequency_stderr_MHz"] for current in currents
        )
        minimum_r2 = min(
            points[f"{current:+.1f}"]["ramsey"]["fit_metrics"][
                "saved_fit_r_squared"
            ] for current in currents
        )
        ramsey_averaging_groups = {}
        for current in currents:
            ramsey_result = points[f"{current:+.1f}"]["ramsey"]
            statistics = (
                int(ramsey_result["reps"]),
                int(ramsey_result["rounds"]),
            )
            ramsey_averaging_groups.setdefault(statistics, []).append(current)
        ramsey_averaging_text = format_current_averaging(
            ramsey_averaging_groups, len(currents)
        )
        add_full_width_plot_slide(
            prs,
            f"{title_label}: fresh Ramsey frequency corrections",
            required_plots[0],
            "Fresh canonical T2RMeasurement.adjust_qspec pass seeded by the "
            "previous validated corrected frequency and its matching 60 ns "
            f"Rabi pi pulse. {ramsey_settings['start_us']:.1f}-"
            f"{ramsey_settings['stop_us']:.1f} us, "
            f"{ramsey_settings['steps']} points, "
            f"{ramsey_settings['detuning_MHz']:.1f} MHz intentional detuning, "
            f"{ramsey_settings['relax_delay_us']:.0f} us relaxation. "
            f"Averaging by current: {ramsey_averaging_text}. "
            f"Minimum saved-fit R2={minimum_r2:.3f}; maximum independent "
            f"frequency standard error={maximum_stderr:.4f} MHz. Correction "
            "is seed - fitted fringe + intentional detuning.",
            raster_title_stem=f"{device_key}_fresh_ramsey_2",
            raster_title_folder=SIGMA_COHERENCE_2D,
        )

        slide = prs.slides.add_slide(prs.slide_layouts[6])
        add_raster_title(
            slide,
            f"{title_label}: sigma tuned for pi gain near 0.65",
            f"{device_key}_sigma_current",
            output_folder=SIGMA_COHERENCE_2D,
        )
        add_contained_picture(
            slide, required_plots[1], 0.08, 0.69, 6.55, 6.22
        )
        heading = slide.shapes.add_textbox(
            Inches(6.68), Inches(0.72), Inches(6.50), Inches(0.30)
        )
        heading.text_frame.text = "Final operating point at every current"
        heading.text_frame.paragraphs[0].font.size = Pt(11.5)
        heading.text_frame.paragraphs[0].font.bold = True
        headers = [
            "I", "seed", "f01", "df", "fres", "RO", "sigma", "pi",
        ]
        table = slide.shapes.add_table(
            10, len(headers), Inches(6.62), Inches(1.05),
            Inches(6.64), Inches(4.55),
        ).table
        widths = [0.54, 0.82, 0.82, 0.62, 0.76, 0.76, 0.72, 0.60]
        for column, width in enumerate(widths):
            table.columns[column].width = Inches(width)
        for column, value in enumerate(headers):
            table.cell(0, column).text = value
        for row, current in enumerate(currents, 1):
            point = points[f"{current:+.1f}"]
            ramsey = point["ramsey"]
            selected = point["sigma_calibration"]["selected"]
            readout = point["coherence"]["readout"]
            values = [
                f"{current:+.1f}",
                f"{ramsey['seed_qfreq_MHz']:.2f}",
                f"{ramsey['corrected_qfreq_MHz']:.3f}",
                f"{ramsey['frequency_correction_MHz']:+.3f}",
                f"{point['coherence']['res_base_MHz']:.2f}",
                f"{readout['frequency_MHz']:.2f}",
                f"{selected['sigma_us'] * 1000:.1f} ns",
                f"{selected['pi_gain']:.4f}",
            ]
            for column, value in enumerate(values):
                table.cell(row, column).text = value
        for row_index, table_row in enumerate(table.rows):
            for cell in table_row.cells:
                cell.text_frame.margin_left = 0
                cell.text_frame.margin_right = 0
                for paragraph in cell.text_frame.paragraphs:
                    paragraph.font.size = Pt(5.9)
                    paragraph.font.bold = row_index == 0
                    paragraph.alignment = PP_ALIGN.CENTER
        tuning = map_data["sigma_tuning"]
        note = slide.shapes.add_textbox(
            Inches(6.72), Inches(5.70), Inches(6.38), Inches(1.24)
        )
        note.text_frame.word_wrap = True
        note.text_frame.text = (
            f"Target pi gain={tuning['target_pi_gain']:.3f}; accepted "
            f"{tuning['accepted_pi_gain'][0]:.3f}-"
            f"{tuning['accepted_pi_gain'][1]:.3f}. Each attempt: gain "
            f"{tuning['gain_start']:.3f}-{tuning['gain_stop']:.3f}, "
            f"{tuning['gain_steps']} points, {tuning['reps']} reps x "
            f"{tuning['rounds']} rounds, {tuning['relax_delay_us']:.0f} us "
            f"relax, {tuning['ssf_shots_per_state']} SSF shots/state. "
            "Sigma is current-indexed and does not replace the zero-current "
            "six-row config. Frequencies are MHz; sigma is ns. RO uses each "
            "unchanged shifted readout tone; readout length/gain/offset remain "
            "the values documented on slides 44 and 47. Attenuation dB: res "
            "DAC [10,15], qubit DAC [5,4], ADC 17."
        )
        note.text_frame.paragraphs[0].font.size = Pt(6.7)

        add_full_width_plot_slide(
            prs,
            f"{title_label}: selected-sigma Rabi curves",
            required_plots[2],
            "Final accepted canonical Rabi row at each current. Color is the "
            "saved signed SSF affine population (ground=0, excited=1); red is "
            "the saved fit and black marks its fitted first pi extremum. All "
            "pi gains are inside 0.625-0.675. No trace is offset, divided, "
            "normalized, or rescaled.",
            raster_title_stem=f"{device_key}_selected_sigma_rabi",
            raster_title_folder=SIGMA_COHERENCE_2D,
        )

        plot_index = 3
        for experiment, experiment_title in coherence_titles.items():
            heatmap_path = required_plots[plot_index]
            curves_path = required_plots[plot_index + 1]
            plot_index += 2
            coherence = map_data["coherence"]
            settings = coherence["experiments"][experiment]
            fitted = [
                points[f"{current:+.1f}"]["coherence"]["experiments"][
                    experiment
                ]["fit_value_us"] for current in currents
            ]
            averaging_groups = {}
            for current in currents:
                experiment_result = points[f"{current:+.1f}"][
                    "coherence"
                ]["experiments"][experiment]
                statistics = (
                    int(experiment_result["reps"]),
                    int(experiment_result["rounds"]),
                )
                averaging_groups.setdefault(statistics, []).append(current)
            averaging_text = format_current_averaging(
                averaging_groups, len(currents)
            )
            retry_notes = []
            for current in currents:
                choice = points[f"{current:+.1f}"]["coherence"].get(
                    "selection_choices", {}
                ).get(experiment)
                if choice:
                    outcome = (
                        "high-stat selected"
                        if choice["selected"]
                        == "high-statistics reacquisition"
                        else "prior retained"
                    )
                    retry_notes.append(
                        f"{current:+.1f} mA 2000x4: R2 "
                        f"{choice['prior_r_squared']:.3f}->"
                        f"{choice['candidate_r_squared']:.3f}, {outcome}"
                    )
            retry_text = (
                " Retry comparison: " + "; ".join(retry_notes) + "."
                if retry_notes else ""
            )
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            add_raster_title(
                slide,
                f"{title_label}: {experiment_title} versus current",
                f"{device_key}_{experiment}_coherence",
                output_folder=SIGMA_COHERENCE_2D,
            )
            add_contained_picture(
                slide, heatmap_path, 0.05, 0.72, 6.60, 6.16
            )
            add_contained_picture(
                slide, curves_path, 6.68, 0.72, 6.60, 6.16
            )
            detail = (
                f"Actual delay {settings['start_us']:.2f}-"
                f"{settings['stop_us']:.0f} us, {settings['steps']} points; "
            )
            if "detuning_MHz" in settings:
                detail += f"{settings['detuning_MHz']:.1f} MHz detuning; "
            footer = slide.shapes.add_textbox(
                Inches(0.22), Inches(6.93), Inches(12.88), Inches(0.43)
            )
            footer.text_frame.word_wrap = True
            footer.text_frame.text = (
                detail
                + f"Averaging by current: {averaging_text}; "
                f"{coherence['relax_delay_us']:.0f} us relaxation, "
                f"{coherence['ssf_shots_per_state_per_experiment']} SSF "
                "shots/state per curve. Heatmap axes: current x, actual delay "
                "in us y, saved Pe color. Each current uses its native QICK "
                "delay centers with explicit midpoint edges; there is no "
                "delay-axis interpolation. Fixed "
                "-0.30 to 1.50 color scale and three-point display mean; no "
                "row normalization. Fitted values "
                "from -10 to +10 mA: "
                + ", ".join(f"{value:.3f}" for value in fitted)
                + " us."
                + retry_text
            )
            footer.text_frame.paragraphs[0].font.size = Pt(6.2)


def add_frequency_axis_summary_slides(prs):
    """Append per-row sigma/coherence plots with current and f01 x axes."""
    if not FREQUENCY_SUMMARY_PROVENANCE.exists():
        return
    provenance = json.loads(
        FREQUENCY_SUMMARY_PROVENANCE.read_text(encoding="utf-8")
    )
    slide_pairs = (
        ("sigma", "t1", "selected sigma and T1"),
        ("t2r", "t2e", "T2R and T2E"),
    )
    for device_key in ("q4", "q6"):
        device = provenance.get("devices", {}).get(device_key, {})
        plots = device.get("plots", {})
        rows = device.get("rows", [])
        required_metrics = ("sigma", "t1", "t2r", "t2e")
        required_paths = {
            metric: Path(plots.get(metric, ""))
            for metric in required_metrics
        }
        if (
            len(rows) != 9
            or any(not str(path) or not path.is_file()
                   for path in required_paths.values())
        ):
            continue
        title_label = device["label"].replace("physical ", "")
        for left_metric, right_metric, pair_title in slide_pairs:
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            add_raster_title(
                slide,
                f"{title_label}: {pair_title} with corrected f01 lookup",
                f"{device_key}_{left_metric}_{right_metric}_frequency_axes",
                output_folder=SIGMA_COHERENCE_2D,
            )
            add_contained_picture(
                slide, required_paths[left_metric], 0.18, 0.70, 6.40, 6.22
            )
            add_contained_picture(
                slide, required_paths[right_metric], 6.75, 0.70, 6.40, 6.22
            )
            footer = slide.shapes.add_textbox(
                Inches(0.30), Inches(6.98), Inches(12.73), Inches(0.38)
            )
            footer.text_frame.word_wrap = True
            footer.text_frame.text = (
                "Bottom x axis is Yokogawa current. Top labels are the exact "
                "Ramsey-corrected f01 values from the same manifest row; "
                "they are a per-bias lookup, not a linear frequency transform. "
                "Every plotted metric is paired only within its current row."
            )
            footer.text_frame.paragraphs[0].font.size = Pt(7.2)


def add_frequency_only_summary_slides(prs):
    """Append sigma/coherence plots whose sole x axis is corrected f01."""
    if not FREQUENCY_ONLY_SUMMARY_PROVENANCE.exists():
        return
    provenance = json.loads(
        FREQUENCY_ONLY_SUMMARY_PROVENANCE.read_text(encoding="utf-8")
    )
    slide_pairs = (
        ("sigma", "t1", "selected sigma and T1"),
        ("t2r", "t2e", "T2R and T2E"),
    )
    for device_key in ("q4", "q6"):
        device = provenance.get("devices", {}).get(device_key, {})
        plots = device.get("plots", {})
        rows = device.get("frequency_sorted_rows", [])
        required_metrics = ("sigma", "t1", "t2r", "t2e")
        required_paths = {
            metric: Path(plots.get(metric, ""))
            for metric in required_metrics
        }
        frequencies = [
            row.get("corrected_qfreq_MHz") for row in rows
        ]
        if (
            len(rows) != 9
            or any(not str(path) or not path.is_file()
                   for path in required_paths.values())
            or any(left >= right for left, right in zip(
                frequencies, frequencies[1:]
            ))
        ):
            continue
        title_label = device["label"].replace("physical ", "")
        for left_metric, right_metric, pair_title in slide_pairs:
            slide = prs.slides.add_slide(prs.slide_layouts[6])
            add_raster_title(
                slide,
                f"{title_label}: {pair_title} versus corrected f01",
                f"{device_key}_{left_metric}_{right_metric}_frequency_only",
                output_folder=SIGMA_COHERENCE_2D,
            )
            add_contained_picture(
                slide, required_paths[left_metric], 0.18, 0.70, 6.40, 6.22
            )
            add_contained_picture(
                slide, required_paths[right_metric], 6.75, 0.70, 6.40, 6.22
            )
            footer = slide.shapes.add_textbox(
                Inches(0.30), Inches(6.98), Inches(12.73), Inches(0.38)
            )
            footer.text_frame.word_wrap = True
            footer.text_frame.text = (
                "The sole x axis is the actual Ramsey-corrected f01 in MHz. "
                "Points are ordered from lowest to highest frequency; each "
                "metric remains paired with f01 from the same source row. "
                "Current is not used as an axis or as a secondary label."
            )
            footer.text_frame.paragraphs[0].font.size = Pt(7.2)


def main():
    if "--append-frequency-only-summary" in sys.argv[1:]:
        if not OUT.exists():
            raise FileNotFoundError(OUT)
        if not FREQUENCY_ONLY_SUMMARY_PROVENANCE.exists():
            raise FileNotFoundError(FREQUENCY_ONLY_SUMMARY_PROVENANCE)
        if not PRE_FREQUENCY_ONLY_SUMMARY_DECK.exists():
            current = Presentation(str(OUT))
            if len(current.slides) != 64:
                raise RuntimeError(
                    f"Expected the 64-slide dual-axis deck, found "
                    f"{len(current.slides)} slides"
                )
            shutil.copy2(OUT, PRE_FREQUENCY_ONLY_SUMMARY_DECK)
        prs = Presentation(str(PRE_FREQUENCY_ONLY_SUMMARY_DECK))
        baseline_count = len(prs.slides)
        add_frequency_only_summary_slides(prs)
        if len(prs.slides) != baseline_count + 4:
            raise RuntimeError(
                "Frequency-only summary append did not produce four slides"
            )
        prs.save(str(OUT))
        print(OUT)
        return
    if "--append-frequency-summary" in sys.argv[1:]:
        if not OUT.exists():
            raise FileNotFoundError(OUT)
        if not FREQUENCY_SUMMARY_PROVENANCE.exists():
            raise FileNotFoundError(FREQUENCY_SUMMARY_PROVENANCE)
        if not PRE_FREQUENCY_SUMMARY_DECK.exists():
            current = Presentation(str(OUT))
            if len(current.slides) != 60:
                raise RuntimeError(
                    f"Expected the 60-slide coherence deck, found "
                    f"{len(current.slides)} slides"
                )
            shutil.copy2(OUT, PRE_FREQUENCY_SUMMARY_DECK)
        prs = Presentation(str(PRE_FREQUENCY_SUMMARY_DECK))
        baseline_count = len(prs.slides)
        add_frequency_axis_summary_slides(prs)
        if len(prs.slides) != baseline_count + 4:
            raise RuntimeError(
                "Frequency-summary append did not produce four slides"
            )
        prs.save(str(OUT))
        print(OUT)
        return
    if "--append-sigma-coherence" in sys.argv[1:]:
        if not OUT.exists():
            raise FileNotFoundError(OUT)
        if not PRE_SIGMA_COHERENCE_DECK.exists():
            shutil.copy2(OUT, PRE_SIGMA_COHERENCE_DECK)
        prs = Presentation(str(PRE_SIGMA_COHERENCE_DECK))
        add_sigma_coherence_slides(prs)
        prs.save(str(OUT))
        print(OUT)
        return
    if "--append-ramsey-rabi" in sys.argv[1:]:
        if not OUT.exists():
            raise FileNotFoundError(OUT)
        if not PRE_RAMSEY_RABI_DECK.exists():
            shutil.copy2(OUT, PRE_RAMSEY_RABI_DECK)
        prs = Presentation(str(PRE_RAMSEY_RABI_DECK))
        add_ramsey_corrected_rabi_slides(prs)
        prs.save(str(OUT))
        print(OUT)
        return
    if "--replace-rabi-map-slides" in sys.argv[1:]:
        if not OUT.exists():
            raise FileNotFoundError(OUT)
        if not PRE_RABI_DECK.exists():
            shutil.copy2(OUT, PRE_RABI_DECK)
        prs = Presentation(str(PRE_RABI_DECK))
        add_single_sweep_rabi_slides(prs)
        prs.save(str(OUT))
        print(OUT)
        return
    if "--replace-qspec-map-slides" in sys.argv[1:]:
        if not PRE_WIDE_DECK.exists():
            raise FileNotFoundError(PRE_WIDE_DECK)
        if OUT.exists() and not SEGMENTED_DECK_BACKUP.exists():
            shutil.copy2(OUT, SEGMENTED_DECK_BACKUP)
        prs = Presentation(str(PRE_WIDE_DECK))
        add_single_sweep_qspec_slides(prs)
        add_single_sweep_rabi_slides(prs)
        prs.save(str(OUT))
        print(OUT)
        return
    if "--append-wide-only" in sys.argv[1:]:
        prs = Presentation(str(OUT))
        add_wide_lower_qspec_slides(prs)
        prs.save(str(OUT))
        print(OUT)
        return
    update_original = "--update-original" in sys.argv[1:]
    prs = Presentation(str(ORIGINAL)) if update_original else Presentation()
    if not update_original:
        prs.slide_width = Inches(13.333)
        prs.slide_height = Inches(7.5)
    add_summary(prs)

    res = [newest(FULL / "res_spec_ge_plots", f"*Q_{q}_*res_spec.png") for q in ROWS]
    add_grid_slide(prs, "Resonator spectroscopy — five validated modes", res,
                   "Readout gain [0.48,0.56,0.41,0.33,0.60]; length [10,5,12,5,3] us; offset [-0.52,-0.40,-0.08,+0.44,-0.04] MHz")

    qspec = [newest(FULL / "qubit_spec_ge_plots", f"*Q_{q}_*q{q}.png") for q in [1, 2, 3]]
    qspec += [newest(QR / "qubit_spec_ge_plots", "*Q_5_*q5.png"), newest(Q6SPEC / "qubit_spec_ge_plots", "*Q_6_*q6.png")]
    add_grid_slide(prs, "Qubit spectroscopy — calibrated population", qspec,
                   "Qspec gain [0.001,0.007,0.002,0.004,0.006]; 20 us pulse; 301 points over ±0.8 MHz; 5000 averages")

    rabi = [newest(FULL / "power_rabi_ge_plots", f"*Q_{q}_*q{q}.png") for q in [1, 2, 3]]
    rabi += [newest(QR / "power_rabi_ge_plots", "*Q_5_*q5.png"), newest(QR / "power_rabi_ge_plots", "*Q_6_*q6.png")]
    add_grid_slide(prs, "Amplitude Rabi — calibrated population", rabi,
                   "sigma [0.050,0.101,0.124,0.055,0.062] us; pi gain [0.679,0.649,0.659,0.649,0.649]")

    ssf = [newest(FULL / "ss_repeat_meas_ge" / f"Q{q}", "*.png") for q in ROWS]
    add_grid_slide(prs, "Single-shot fidelity", ssf,
                   "Final RR fidelities: [58.2,34.3,37.2,17.4,21.5]%; readout settings shown on resonator slide")

    t1 = [newest(FULL / "T1_ge", f"*Q_{q}_*q{q}.png") for q in [1, 2, 3]] + [newest(COH / "T1_ge", f"*Q_{q}_*q{q}.png") for q in [5, 6]]
    add_grid_slide(prs, "T1", t1, "Analog I/Q plots from round robin; M5/M6 use 800 averages and tightened windows")
    t2r = [newest(FULL / "Ramsey_ge_1", f"*Q_{q}_*q{q}.png") for q in [1, 2, 3]] + [newest(COH / "Ramsey_ge_1", f"*Q_{q}_*q{q}.png") for q in [5, 6]]
    add_grid_slide(prs, "T2 Ramsey", t2r, "0.5 MHz detuning; ≥10 samples/fringe; M5/M6 use 1000 averages")
    t2e = [newest(FULL / "SpinEcho_ge", f"*Q_{q}_*q{q}.png") for q in [1, 2, 3]] + [newest(COH / "SpinEcho_ge", f"*Q_{q}_*q{q}.png") for q in [5, 6]]
    add_grid_slide(prs, "T2 echo", t2e, "0.5 MHz phase cycling; ≥10 samples/fringe; M5/M6 use 1000 averages")

    add_grid_slide(prs, "Final readout-length optimization", [LENOPT / f"readout_length_Q{q}.png" for q in ROWS],
                   "Chosen shortest plateau lengths: Q1 10 us, Q2 5 us, Q3 12 us, Q4/M5 5 us, Q6 3 us")
    add_grid_slide(prs, "Final readout gain × frequency optimization", [GFOPT / f"gain_frequency_Q{q}.png" for q in ROWS],
                   "Chosen gain/offset: Q1 .48/-.52, Q2 .56/-.40, Q3 .41/-.08, Q4/M5 .33/+.44, Q6 .60/-.04")

    add_manifest_flux_tables(prs)
    # Static checkpoint slides contain superseded intermediate values. The
    # clean deck uses only manifest-backed final values and canonical HDF5 data.
    add_final_q4_minus_ten(prs)
    add_flux_2d_maps(prs)
    add_single_sweep_qspec_slides(prs)
    add_single_sweep_rabi_slides(prs)
    add_ramsey_corrected_rabi_slides(prs)
    add_sigma_coherence_slides(prs)
    add_frequency_axis_summary_slides(prs)
    add_frequency_only_summary_slides(prs)

    output_path = ORIGINAL if update_original else OUT
    try:
        prs.save(str(output_path))
        saved_path = output_path
    except PermissionError:
        saved_path = output_path.with_name(f"{output_path.stem}_updated{output_path.suffix}")
        prs.save(str(saved_path))
    print(saved_path)


if __name__ == "__main__":
    main()
