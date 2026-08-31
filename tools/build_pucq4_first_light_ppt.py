"""Build the reusable PUCQ4 first-light characterization deck.

This is analysis-only: it reads existing PNG artifacts and writes one PPTX.
"""

from pathlib import Path
import sys

from PIL import Image
from pptx import Presentation
from pptx.enum.text import PP_ALIGN
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

ROWS = [1, 2, 3, 5, 6]
SUMMARY = [
    ("Q1 / M1", "R1", 8919.830, 5517.694, 31.45, 16.53, 37.13),
    ("Q2 / M2", "R2", 8951.650, 5438.714, 8.77, 11.71, 15.88),
    ("Q3 / M3", "R3", 8974.970, 5557.334, 45.79, 42.85, 77.22),
    ("Q4 / M5", "R5", 9014.590, 7084.778, 7.63, 5.28, 11.26),
    ("Q6 / M6", "R6", 9059.310, 6522.540, 3.34, 1.53, 4.19),
]

FLUX_Q4 = [
    ("0.0", 7084.7785, 0.1, 0.05996, 0.63871, None, None, None, 5.96, 4.33, 11.35, "complete"),
    ("-2.5", 6945.5141, 0.1, 0.05996, 0.63871, 8.0, 0.57, 0.552, 11.22, 3.48, 9.75, "complete"),
    ("-5.0", 6725.3975, 0.2, 0.08530, 0.64869, 7.0, 0.89, 0.572, 15.53, 2.40, 7.74, "complete"),
    ("-7.5", 6423.2222, 0.1, 0.05911, 0.64869, 7.0, 0.9267, 0.666, 19.09, 1.86, 6.01, "complete"),
    ("+2.5", 7143.8491, 0.1, 0.05264, 0.65867, 5.0, 0.95, 0.6296, None, None, None, "complete"),
    ("+5.0", 7121.7063, 0.1, 0.07321, 0.66865, None, None, None, None, None, None, "complete"),
    ("+7.5", 7022.3193, 0.1, 0.05445, 0.65867, 6.0, 0.9033, 0.6913, None, None, None, "complete"),
    ("+10.0", 6841.5394, 0.1, 0.05911, 0.62873, None, None, None, None, None, None, "complete"),
]


def newest(folder, pattern):
    paths = list(folder.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No {pattern} under {folder}")
    return max(paths, key=lambda p: p.stat().st_mtime)


def add_title(slide, title, subtitle=None):
    box = slide.shapes.add_textbox(Inches(0.3), Inches(0.1), Inches(12.7), Inches(0.55))
    p = box.text_frame.paragraphs[0]
    p.text = title
    p.font.size = Pt(25)
    p.font.bold = True
    if subtitle:
        sub = slide.shapes.add_textbox(Inches(0.35), Inches(7.12), Inches(12.5), Inches(0.25))
        p = sub.text_frame.paragraphs[0]
        p.text = subtitle
        p.font.size = Pt(8)


def add_grid_slide(prs, title, paths, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, title, notes)
    positions = [(0.25, 0.72), (4.55, 0.72), (8.85, 0.72), (2.4, 3.88), (6.7, 3.88)]
    for path, (x, y) in zip(paths, positions):
        add_contained_picture(slide, path, x, y, 4.15, 3.0)
    return slide


def add_three_plot_slide(prs, title, paths, captions, notes):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, title, notes)
    for path, caption, x in zip(paths, captions, (0.25, 4.55, 8.85)):
        add_contained_picture(slide, path, x, 1.0, 4.15, 4.9)
        box = slide.shapes.add_textbox(Inches(x), Inches(6.0), Inches(4.15), Inches(0.45))
        p = box.text_frame.paragraphs[0]
        p.text = caption
        p.font.size = Pt(12)
        p.alignment = PP_ALIGN.CENTER
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
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "PUCQ4 first light at Yoko current = 0 A")
    rows, cols = len(SUMMARY) + 1, 7
    table = slide.shapes.add_table(rows, cols, Inches(0.35), Inches(1.1), Inches(12.6), Inches(3.7)).table
    headers = ["Physical / row", "Resonator", "fres (MHz)", "f01 (MHz)", "T1 (us)", "T2R (us)", "T2E (us)"]
    for c, text in enumerate(headers):
        table.cell(0, c).text = text
    for r, values in enumerate(SUMMARY, 1):
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


def add_flux_checkpoint_table(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_title(slide, "Yoko 3 / physical Q4 (M5) flux-sweep checkpoint")
    headers = ["I (mA)", "f01 (MHz)", "FWHM", "sigma (us)", "pi gain", "RO len/gain", "SSF", "T1", "T2R", "T2E", "status"]
    table = slide.shapes.add_table(len(FLUX_Q4) + 1, len(headers), Inches(0.18), Inches(1.0), Inches(12.95), Inches(3.5)).table
    for col, value in enumerate(headers):
        table.cell(0, col).text = value
    for row, values in enumerate(FLUX_Q4, 1):
        current, fq, width, sigma, pi_amp, ro_len, ro_gain, ssf, t1, t2r, t2e, status = values
        rendered = [
            current, f"{fq:.4f}", f"{width:.1f} MHz", f"{sigma:.5f}",
            "—" if pi_amp is None else f"{pi_amp:.5f}",
            "—" if ro_len is None else f"{ro_len:.0f} us / {ro_gain:.2f}",
            "—" if ssf is None else f"{ssf:.3f}",
            "—" if t1 is None else f"{t1:.2f}", "—" if t2r is None else f"{t2r:.2f}",
            "—" if t2e is None else f"{t2e:.2f}", status,
        ]
        for col, value in enumerate(rendered):
            table.cell(row, col).text = value
    for row in table.rows:
        for cell in row.cells:
            for paragraph in cell.text_frame.paragraphs:
                paragraph.font.size = Pt(9)
                paragraph.alignment = PP_ALIGN.CENTER
    note = slide.shapes.add_textbox(Inches(0.4), Inches(4.8), Inches(12.5), Inches(1.6))
    note.text_frame.text = (
        "The −7.5 mA blocker was resolved by calibrating Rabi from averaged raw I/Q before SSF optimization. "
        "The −10 mA Q4 trace failed visual/fit-quality acceptance and remains diagnostic-only. "
        "Q4 is complete at every other deck current; Q6 is complete at 0 and +2.5 mA, while its other current points remain unresolved or unmeasured. "
        "Both sources were returned to 0 A with outputs off."
    )
    note.text_frame.paragraphs[0].font.size = Pt(17)


def add_flux_checkpoint_plots(prs):
    currents = [("+00.0", "0 mA"), ("-02.5", "−2.5 mA"), ("-05.0", "−5 mA")]
    calibration_runs = [
        "flux_yoko3_q4_+00.0mA_calibration",
        "flux_yoko3_q4_-02.5mA_post_readout_calibration",
        "flux_yoko3_q4_-05.0mA_post_readout_calibration",
    ]
    qspec = [newest_flux_artifact(run, "qubit_spec_ge_plots", "*q5.png") for run in calibration_runs]
    add_three_plot_slide(prs, "Q4/M5 qubit spectroscopy versus Yoko 3 current", qspec,
                         [label for _, label in currents],
                         "Accepted centers: 7084.7785, 6945.5141, 6725.3975 MHz; FWHM: 0.1, 0.1, 0.2 MHz")

    rabi_runs = ["flux_yoko3_q4_+00.0mA_rabi_retry1", "flux_yoko3_q4_-02.5mA_post_readout_calibration", "flux_yoko3_q4_-05.0mA_rabi_retry2"]
    rabi = [newest_flux_artifact(run, "power_rabi_ge_plots", "*q5.png") for run in rabi_runs]
    add_three_plot_slide(prs, "Q4/M5 amplitude Rabi versus Yoko 3 current", rabi,
                         [label for _, label in currents],
                         "sigma: 0.05996, 0.05996, 0.08530 us; pi gain: 0.63871, 0.63871, 0.64869")

    for folder, title, values in [
        ("T1_ge", "Q4/M5 T1 versus Yoko 3 current", "5.96, 11.22, 15.53 us"),
        ("Ramsey_ge_1", "Q4/M5 T2 Ramsey versus Yoko 3 current", "4.33, 3.48, 2.40 us"),
        ("SpinEcho_ge", "Q4/M5 T2 echo versus Yoko 3 current", "11.35, 9.75, 7.74 us"),
    ]:
        paths = [newest_flux_artifact(f"flux_yoko3_q4_{tag}mA_coherence", folder, "*q5.png") for tag, _ in currents]
        add_three_plot_slide(prs, title, paths, [label for _, label in currents], f"Accepted fitted values: {values}")

    length_paths = [
        newest_flux_artifact("flux_yoko3_q4_-02.5mA_readout_length2", "", "readout_length_Q5.png"),
        newest_flux_artifact("flux_yoko3_q4_-05.0mA_readout_length5", "", "readout_length_Q5.png"),
    ]
    add_three_plot_slide(prs, "Q4/M5 final current-specific readout-length optimization", length_paths,
                         ["−2.5 mA: 8 us / gain 0.57 / +0.08 MHz", "−5 mA: 7 us / gain 0.89 / +0.20 MHz"],
                         "Shortest plateau lengths shown; measured SSF 0.552 and 0.572. Third panel intentionally blank.")

    gain_paths = [
        newest_flux_artifact("flux_yoko3_q4_-02.5mA_readout_gf2", "", "gain_frequency_Q5.png"),
        newest_flux_artifact("flux_yoko3_q4_-05.0mA_readout_gf5", "", "gain_frequency_Q5.png"),
    ]
    add_three_plot_slide(prs, "Q4/M5 final gain × readout-frequency optimization", gain_paths,
                         ["−2.5 mA: gain 0.57 / +0.08 MHz", "−5 mA: gain 0.89 / +0.20 MHz"],
                         "Final completed optimization iterations only. Third panel intentionally blank.")

    resolved_paths = [
        newest_flux_artifact("flux_yoko3_q4_-07.5mA_post_readout_calibration", "qubit_spec_ge_plots", "*q5.png"),
        newest_flux_artifact("flux_yoko3_q4_-07.5mA_post_readout_calibration", "power_rabi_ge_plots", "*q5.png"),
        newest_flux_artifact("flux_yoko3_q4_-07.5mA_coherence", "T1_ge", "*q5.png"),
        newest_flux_artifact("flux_yoko3_q4_-07.5mA_coherence", "Ramsey_ge_1", "*q5.png"),
        newest_flux_artifact("flux_yoko3_q4_-07.5mA_coherence", "SpinEcho_ge", "*q5.png"),
    ]
    add_grid_slide(prs, "Resolved −7.5 mA Q4/M5 characterization", resolved_paths,
                   "f01 6423.2222 MHz; sigma 0.05911 us; pi 0.64869; SSF 0.666; T1 19.09 us; T2R 1.86 us; T2E 6.01 us")


def add_new_flux_results(prs):
    q4_runs = [
        "flux_yoko3_q4_+02.5mA_post_readout_calibration",
        "flux_yoko3_q4_+05.0mA_post_readout_calibration",
        "flux_yoko3_q4_+07.5mA_post_readout_calibration",
        "flux_yoko3_q4_+10.0mA_post_readout_calibration",
    ]
    q4_paths = [newest_flux_artifact(run, "qubit_spec_ge_plots", "*q5.png") for run in q4_runs]
    add_grid_slide(prs, "New accepted Q4/M5 positive-current qspec", q4_paths,
                   "+2.5, +5, +7.5, +10 mA; accepted f01 7143.8491, 7121.7063, 7022.3193, 6841.5394 MHz")

    q6_run = "flux_yoko4_q6_+02.5mA_post_readout_calibration"
    q6_coh = "flux_yoko4_q6_+02.5mA_coherence"
    q6_paths = [
        newest_flux_artifact(q6_run, "qubit_spec_ge_plots", "*q6.png"),
        newest_flux_artifact(q6_run, "power_rabi_ge_plots", "*q6.png"),
        newest_flux_artifact(q6_coh, "T1_ge", "*q6.png"),
        newest_flux_artifact(q6_coh, "Ramsey_ge_1", "*q6.png"),
        newest_flux_artifact(q6_coh, "SpinEcho_ge", "*q6.png"),
    ]
    add_grid_slide(prs, "Accepted Q6/M6 at +2.5 mA", q6_paths,
                   "f01 6842.1315 MHz; sigma 0.05800 us; pi 0.60877; readout 12 us / 0.7967 / -0.52 MHz; SSF 0.5753")

    rejected_q6_paths = [
        newest_flux_artifact("flux_yoko4_q6_-10.0mA_wide_locator_diagnostic", "qubit_spec_ge_plots", "*q6.png"),
        newest_flux_artifact("flux_yoko4_q6_-10.0mA_raw_iq_rabi", "power_rabi_ge_plots", "*q6.png"),
        newest_flux_artifact("flux_yoko4_q6_-07.5mA_raw_iq_rabi", "power_rabi_ge_plots", "*q6.png"),
        newest_flux_artifact("flux_yoko4_q6_+10.0mA_raw_iq_rabi", "power_rabi_ge_plots", "*q6.png"),
    ]
    add_grid_slide(
        prs,
        "Q6/M6 rejected-current evidence",
        rejected_q6_paths,
        "-10 mA: raw-I/Q sigma iteration reached 0.0986291 us / pi 0.688610, but final qspec/Rabi revalidation failed; "
        "the pulse seed is stored separately as revalidation_required. -7.5 and +10 mA remain rejected diagnostic branches.",
    )


def add_flux_2d_maps(prs):
    slide_specs = [
        ("qspec", "Yokogawa current × qubit-frequency maps",
         "X = bias current, Y = actual swept qubit frequency, color = measured signal magnitude. Blank areas were not acquired."),
        ("res_spec", "Yokogawa current × resonator-frequency maps",
         "X = bias current, Y = actual swept resonator frequency, color = measured signal magnitude. No interpolation across currents."),
        ("found_frequencies_vs_current", "Found spectroscopy centers versus bias current",
         "Separate accepted fitted qubit-transition and resonator-center curves; failed points are omitted rather than inferred."),
        ("qspec_fwhm_vs_current", "Qspec FWHM versus bias current",
         "Accepted qspec linewidths parsed from the final calibration logs; shading marks the 0.1-0.2 MHz target."),
        ("rabi", "Yokogawa current × Rabi-gain maps",
         "Measured I/Q projected onto the dominant response axis. Blank columns have no numeric acquisition."),
        ("sigma_pi", "Current-indexed sigma and pi-amplitude calibration",
         "Required pi window 0.6-0.7. Q6 -10 mA diamond: sigma 0.0986291 us / raw-IQ pi 0.688610; revalidation required."),
        ("t1", "Yokogawa current × T1-delay maps",
         "Numeric round-robin delay traces only; missing coherence rows remain blank."),
        ("t2r", "Yokogawa current × T2 Ramsey-delay maps",
         "Numeric round-robin delay traces only; configured detuning and sampling are retained in each HDF5 provenance record."),
        ("t2e", "Yokogawa current × T2 echo-delay maps",
         "Numeric round-robin delay traces only; missing/rejected rows are not synthesized."),
    ]
    for stem, title, note in slide_specs:
        suffix = "_2d.png" if stem in {"qspec", "res_spec", "rabi", "sigma_pi", "t1", "t2r", "t2e"} else ".png"
        paths = [FLUX_2D / f"q4_{stem}{suffix}", FLUX_2D / f"q6_{stem}{suffix}"]
        add_grid_slide(prs, title, paths, note)


def main():
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

    add_flux_checkpoint_table(prs)
    add_flux_checkpoint_plots(prs)
    add_new_flux_results(prs)
    add_flux_2d_maps(prs)

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
