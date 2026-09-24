"""One reusable PowerPoint for the new RR reproduction campaign (offline)."""
import argparse
import datetime
import json
from pathlib import Path
import sys

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pptx import Presentation
from pptx.util import Inches

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools import build_pucq4_chip_summary as B
from tools.audit_pucq4_round_robin import ROOT, ARTIFACTS, KINDS
from tools.analyze_qspec_iq_series import decode_numeric

OUTPUT = ROOT / "PUCQ4_round_robin_validation_20260908.pptx"
TITLES = dict(res_ge="Resonator spectroscopy", qspec_ge="Qubit spectroscopy",
              rabi_ge="Amplitude Rabi", ss_ge="Single-shot readout",
              t1_ge="T1", t2_ge="T2R / Ramsey", t2e_ge="T2E / echo")
GRID = [-10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10]


def load(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def fmt(value, places=3):
    return "pending" if value is None else ("%.*f" % (places, value))


def status_text(result):
    if not result:
        return "pending"
    return {"validated_change": "repeat checked", "validated_tracking": "tracking verified"}.get(result["status"], result["status"])


def experiments(result):
    return result.get("experiments", {}) if result else {}


def metric(result, kind, key, places=3):
    return fmt(experiments(result).get(kind, {}).get(key), places)


def config(result, kind, key, places=3):
    return fmt(experiments(result).get(kind, {}).get("config", {}).get(key), places)


def measurement_passes(result, kind):
    """Keep unresolved measurements and their dependent calibrations out of summaries."""
    if not result:
        return False
    prefixes = dict(res_ge=("res_ge:", "res spec:"),
                    qspec_ge=("qspec_ge:", "qspec:"),
                    rabi_ge=("rabi_ge:", "Rabi:"), ss_ge=("ss_ge:", "SSF:"))
    dependencies = [kind]
    if kind == "qspec_ge":
        dependencies += ["res_ge"]
    elif kind == "rabi_ge":
        dependencies += ["res_ge", "qspec_ge"]
    elif kind == "ss_ge":
        dependencies += ["res_ge", "qspec_ge", "rabi_ge"]
    elif kind in ("t1_ge", "t2_ge", "t2e_ge"):
        dependencies += ["res_ge", "qspec_ge", "rabi_ge", "ss_ge"]
    blocked = tuple(p for name in dependencies for p in prefixes.get(name, (name + ":",)))
    return kind in experiments(result) and not any(issue.startswith(blocked) for issue in result["issues"])


def checked_metric(result, kind, key, places=3):
    """Show passed measurements from a complete chain without endorsing bad fits."""
    if not result:
        return "pending"
    if not measurement_passes(result, kind):
        return "review"
    return metric(result, kind, key, places)


def checked_config(result, kind, key, places=3):
    if not result:
        return "pending"
    return config(result, kind, key, places) if measurement_passes(result, kind) else "review"


def table_slide(prs, title, headers, rows, sources, footer="", font=11):
    slide = B.slide_base(prs, title, footer, sources)
    height = 5.95 if len(rows) >= 5 else .62 * (len(rows) + 1)
    B.table(slide, headers, rows, y=.92, height=height, font_size=font)
    return slide


def selected_results(manifest):
    zero, bias = {}, {4: {}, 5: {}}
    # Review attempts remain visible and explicitly labelled until resolved.
    for item in manifest["attempts"]:
        result = load(item["audit"])
        if not result.get("complete_chain"):
            continue
        q, current = result["qubit_index"], result["current_mA"]
        if result["yoko3_current_mA"] == result["yoko4_current_mA"] == 0:
            zero[q] = result
        elif q in (4, 5) and current and (result["yoko4_current_mA"] if q == 4 else result["yoko3_current_mA"]) == 0:
            bias[q][current] = result
    for q, path in manifest["zero"].items():
        zero[int(q)] = load(path)
    for q, mapping in manifest["bias"].items():
        bias[int(q)].update({float(i): load(p) for i, p in mapping.items()})
    for q in (4, 5):
        if q in zero:
            bias[q][0.0] = zero[q]
    return zero, bias


def closeout_slide(prs, manifest):
    """Show measured coverage separately from accepted calibration rows."""
    closeout = manifest.get("closeout")
    if not closeout:
        return
    accepted_states = {"pass", "validated_change", "validated_tracking"}

    def accepted(path):
        result = load(path)
        return (result.get("complete_chain", False) and not result["issues"]
                and result["status"] in accepted_states)

    zero_count = sum(accepted(path) for path in manifest["zero"].values())
    rows = [["Zero current: M1--M6", "%d / 6" % zero_count,
             "Complete chains accepted in this campaign"]]
    for q, label in ((4, "Q4 / M5 / Yoko 3"), (5, "Q6 / M6 / Yoko 4")):
        mapping = manifest["bias"][str(q)]
        passed = {float(current) for current, path in mapping.items()
                  if float(current) != 0 and accepted(path)}
        remaining = ["%+.1f" % current for current in GRID
                     if current and current not in passed]
        rows.append([label + ": nonzero currents", "%d / 8" % len(passed),
                     "Unresolved/unmeasured (mA): " + ", ".join(remaining)
                     if remaining else "All documented nonzero currents accepted"])
    for q, y3, y4 in ((0, 10, 0), (1, 10, 0), (2, 0, -10), (3, 0, -10)):
        matches = [item for item in manifest.get("controls", [])
                   if item["qubit_index"] == q and item["yoko3_current_mA"] == y3
                   and item["yoko4_current_mA"] == y4]
        status = "accepted" if any(accepted(item["audit"]) for item in matches) else (
            "review" if matches else "unmeasured")
        rows.append(["M%d endpoint control" % (q + 1), status,
                     "Yoko 3 %+.1f mA / Yoko 4 %+.1f mA" % (y3, y4)])
    final = manifest.get("final_default_validation")
    if final:
        detail = ("%d / 6 chains pass all gates; %s" % (
            len(final["accepted_rows"]), Path(final["run"]).name)
            if final.get("audits") else "Acquisition/audit pending; normal defaults, no diagnostic overrides")
        rows.append(["Final all-six default pass", final["status"],
                     detail])
    else:
        rows.append(["Final all-six default pass", "pending", "No final default pass recorded"])
    table_slide(prs, "Validation coverage and closeout",
                ["Required evidence", "Accepted / status", "Coverage detail"], rows,
                manifest, closeout["summary"], 11)


def final_default_slide(prs, manifest, records):
    """Keep every final-run outcome visible even when an older row is selected."""
    if not records:
        return
    rows = []
    for q in range(6):
        r = records.get(q)
        if not r:
            rows.append(["M%d / %d" % (q + 1, q), "missing", "No final audit recorded"])
            continue
        rows.append(["M%d / %d" % (q + 1, q), status_text(r),
                     "; ".join(r["issues"]) if r["issues"] else "All raw, calibration, fit and comparison gates pass"])
    final = manifest["final_default_validation"]
    table_slide(prs, "Final all-six default pass | every outcome",
                ["Row / index", "Audit", "Final-run findings"], rows, final,
                "One zero-current RR pass, no diagnostic overrides: %s. Earlier accepted rows do not hide final failures." %
                Path(final["run"]).name, 10)


def zero_slides(prs, zero, manifest):
    accepted = {int(q): load(path) for q, path in manifest["zero"].items()}
    rows = []
    for q in range(6):
        r = accepted.get(q, zero.get(q))
        rows.append(["M%d / %d" % (q + 1, q), {4: "Q4 / Y3", 5: "Q6 / Y4"}.get(q, "unassigned"),
                     checked_metric(r, "res_ge", "found_frequency_MHz"), checked_metric(r, "qspec_ge", "center_MHz", 6),
                     checked_metric(r, "t1_ge", "tau_us"), checked_metric(r, "t2_ge", "tau_us"),
                     checked_metric(r, "t2e_ge", "tau_us"), status_text(r)])
    table_slide(prs, "PUCQ4 | verified measurements at zero Yoko current",
                ["Row / index", "Physical ID", "Res (MHz)", "Qubit (MHz)", "T1 (us)", "T2R (us)", "T2E (us)", "Audit"],
                rows, zero, "M1--M6 = displayed Q1--Q6; Python indices0--5. Physical Q4 is M5. Unresolved checks withhold affected values; review traces follow.")
    closeout_slide(prs, manifest)
    final_records = {int(q): load(path) for q, path in
                     manifest.get("final_default_validation", {}).get("audits", {}).items()}
    final_default_slide(prs, manifest, final_records)
    evidence_zero = final_records or zero
    evidence_title = "Final default zero current" if final_records else "Zero current"
    slide = B.slide_base(prs, "Run the documented configuration", "Task state: " + manifest["task_status"], manifest)
    B.textbox(slide, "In round_robin_benchmark.py, set:\n\nYOKO3_CURRENT_MA = 0.0\nYOKO4_CURRENT_MA = 0.0\nQs_to_look_at = [0, 1, 2, 3, 4, 5]\n\nThen run the script once.", .6, 1.05, 7.5, 3.2, 22)
    B.textbox(slide, "Bias choices: -10, -7.5, -5, -2.5, 0, 2.5, 5, 7.5, 10 mA.\n"
              "set_yoko_bias() ramps/verifies at 0.5 mA/s; Yoko 3 +/-10 mA waits 600 s.\n"
              "The script selects pulse/readout values in system_config.py and\n"
              "acquisition profiles in expt_config.py; both sources return to zero/off.\n"
              "Readout retains its calibrated frequency; offset is recorded from the new dip.\n"
              "Current sweeps use one nonzero Yoko at a time; joint biases are unverified.\n"
              "Normal flags: res spec, qspec, Rabi, SSF, T1, T2R, T2E.\n"
              "One pass; calibrated population plots; original experiment classes/QICK programs.", .6, 4.4, 12.0, 2.4, 16)
    comparison = []
    for q, r in sorted(evidence_zero.items()):
        values = ["M%d" % (q + 1)]
        for kind in ("t1_ge", "t2_ge", "t2e_ge"):
            old = r["reference"][kind]
            new = r["experiments"][kind]
            values.extend(["%.2f -> %.2f +/- %.2f" % (old["tau_us"], new["tau_us"], new["error_us"]),
                           "%+.1f%%" % r["comparisons"][kind]["change_percent"]])
        values.append(status_text(r))
        comparison.append(values)
    if comparison:
        table_slide(prs, evidence_title + " | comparison with the previous PowerPoint",
                    ["Row", "T1: prior -> new (us)", "Change", "T2R: prior -> new (us)", "Change",
                     "T2E: prior -> new (us)", "Change", "Audit"], comparison, evidence_zero,
                    "Review threshold: larger of 30% of prior tau or 3 combined fit standard errors. Physical changes remain visible.", 10)
    settings_slides(prs, evidence_title, [("M%d" % (q + 1), r) for q, r in sorted(evidence_zero.items())])
    for kind in KINDS:
        available = [(q, r) for q, r in sorted(evidence_zero.items()) if "plot" in r["experiments"].get(kind, {})]
        per_slide = 1 if kind == "ss_ge" else 2
        for start in range(0, len(available), per_slide):
            subset = available[start:start + per_slide]
            slide = B.slide_base(prs, evidence_title + " | " + TITLES[kind],
                                 "Original RR figures. Units/configurations and canonical HDF5 are in slide notes.",
                                 {q: r for q, r in subset})
            for col, (q, r) in enumerate(subset):
                e = r["experiments"][kind]
                width = 12.7 if per_slide == 1 else 6.1
                x = .3 + 6.4 * col
                B.textbox(slide, "M%d / index %d | %s | %s" % (q + 1, q, Path(r["run"]).name, status_text(r)), x, .88, width, .4, 12, True)
                B.picture(slide, e["plot"], x, 1.4, width, 4.95,
                          crop=B.RESONATOR_CROP if kind == "res_ge" else None)
                c = e["config"]
                annotation = "RO %.4f MHz | gain %.4g | %.3g us | sigma %.6g us" % (
                    c.get("res_freq_ge", e.get("found_frequency_MHz", 0)), c["res_gain_ge"], c["res_length"], c["sigma"])
                if kind == "res_ge":
                    annotation = "Sweep %.3f--%.3f MHz | gain %.4g | %.3g us" % (
                        e["actual_axis"]["start"], e["actual_axis"]["stop"], c["res_gain_ge"], c["res_length"])
                B.textbox(slide, annotation, x, 6.48, width, .36, 10)


def settings_slides(prs, title, labelled_results):
    if not labelled_results:
        return
    rows, readout = [], []
    for label, r in labelled_results:
        e = r["experiments"]
        if r["status"] == "validated_tracking":
            label += " (tracking)"
        elif r["status"] not in ("pass", "validated_change"):
            label += " (review)"
        rows.append([label, checked_config(r, "rabi_ge", "qubit_freq_ge", 6), config(r, "qspec_ge", "qubit_gain_ge", 6),
                     config(r, "qspec_ge", "qubit_length_ge", 1), checked_metric(r, "qspec_ge", "fwhm_MHz", 4),
                     checked_metric(r, "qspec_ge", "fwhm_error_MHz", 4), checked_config(r, "rabi_ge", "sigma", 7),
                     checked_metric(r, "rabi_ge", "measured_pi_gain", 5)])
        readout.append([label, metric(r, "res_ge", "found_frequency_MHz", 4), config(r, "rabi_ge", "res_freq_ge", 4),
                        config(r, "rabi_ge", "res_freq_offset_ge", 3), config(r, "rabi_ge", "res_gain_ge", 4),
                        config(r, "rabi_ge", "res_length", 2), checked_metric(r, "ss_ge", "contrast", 4),
                        e["ss_ge"]["shots_per_state"]])
    table_slide(prs, title + " | qubit pulse and linewidth settings",
                ["Row/current", "Pulse freq (MHz)", "Qspec gain", "Qspec (us)", "FWHM (MHz)", "+/- (MHz)", "Sigma (us)", "Pi gain"],
                rows, dict(labelled_results), "Pulse frequency is the value actually used for Rabi/coherence. FWHM belongs to this current/gain trace. Gaussian duration = 4 sigma.", 10)
    table_slide(prs, title + " | readout settings", ["Row/current", "Res dip (MHz)", "Readout (MHz)", "Offset (MHz)", "Gain", "Length (us)", "SSF contrast", "Shots/state"],
                readout, dict(labelled_results), "Readout = measured dip + offset. Nominal attenuation arguments: res DAC10/15 dB; qubit DAC5/4 dB; ADC17 dB. Existing RF path retained.", 11)
    for kinds, label in ((KINDS[:3], "spectroscopy / Rabi"), (KINDS[4:], "coherence")):
        rows = []
        for identity, r in labelled_results:
            for kind in kinds:
                e, c = r["experiments"][kind], r["experiments"][kind]["config"]
                axis = e["actual_axis"]
                short_name = dict(res_ge="Res spec", qspec_ge="Qspec", rabi_ge="Rabi",
                                  t1_ge="T1", t2_ge="T2R", t2e_ge="T2E")[kind]
                rows.append([identity, short_name, "%.5f\n%.5f" % (axis["start"], axis["stop"]),
                             axis["points"], "%.0f x %.0f" % (c["reps"], c["rounds"]), fmt(c["relax_delay"], 0),
                             e.get("state_calibration", {}).get("shots_per_state", "--"),
                             e.get("state_calibration", {}).get("relax_delay_us", "--"),
                             fmt(c.get("ramsey_freq"), 6) if kind in KINDS[5:] else "--"])
        for start in range(0, len(rows), 15):
            table_slide(prs, title + " | acquisition settings: " + label,
                        ["Row/current", "Experiment", "Axis start / stop", "Points", "Reps x rounds", "Relax (us)", "Cal shots/state", "Cal relax (us)", "Phase (MHz)"],
                        rows[start:start + 15], dict(labelled_results),
                        "Axis units: MHz for spectra, normalized gain for Rabi, us for coherence. Calibration relaxation and full configs are saved in HDF5.", 9)
    fit_rows, population_starts = [], []
    for identity, r in labelled_results:
        entry = r["experiments"]["t2_ge"]
        setting = entry.get("fit_initialization")
        if setting:
            fit_rows.append([identity, fmt(entry["config"]["ramsey_freq"], 9),
                             fmt(setting["frequency_guess_MHz"], 6),
                             fmt(setting["fitted_frequency_MHz"], 6),
                             fmt(setting["unseeded_tau_us"], 3),
                             checked_metric(r, "t2_ge", "tau_us"),
                             metric(r, "t2_ge", "error_us")])
            starts = setting.get("starting_values", {"f": setting["frequency_guess_MHz"]})
            if "amp" in starts:
                population_starts.append([identity, fmt(starts["phase"], 3),
                                          fmt(starts["amp"], 3), fmt(starts["initial_offset"], 3),
                                          fmt(starts["final_offset"], 3)])
    if fit_rows:
        table_slide(prs, title + " | Ramsey fit configuration",
                    ["Row/current", "Phase (MHz)", "Fit start (MHz)", "Free fit freq (MHz)",
                     "Automatic T2R (us)", "Configured T2R (us)", "+/- (us)"],
                    fit_rows, dict(labelled_results),
                    "The existing fitter receives recorded starting values; all parameters remain free. Its original curve is preserved in HDF5 as Fit Unseeded. Class and model unchanged.", 11)
    if population_starts:
        table_slide(prs, title + " | Ramsey population fit starting values",
                    ["Row/current", "Phase start (rad)", "Amplitude start", "Initial offset start", "Final offset start"],
                    population_starts, dict(labelled_results),
                    "Amplitude and offsets use calibrated population units. Starting values initialize free parameters; no T2 starting guess or constraint is supplied.", 12)


def spectroscopy_target_slides(prs, zero, bias):
    """Report the requested tuning targets separately from reproduction audits."""
    points = [("M%d / zero" % (q + 1), r) for q, r in sorted(zero.items())]
    points += [("M%d / %+.1f mA" % (q + 1, current), r)
               for q, results in bias.items() for current, r in sorted(results.items()) if current]
    rows, records = [], []
    for label, result in points:
        e = result["experiments"]["qspec_ge"]
        width, peak = e["fwhm_MHz"], e["peak_population"]
        width_in_target = .1 <= width <= .2
        peak_in_target = .3 <= peak <= .5
        findings = []
        if not measurement_passes(result, "qspec_ge"):
            findings.append("spectrum under review")
        if not width_in_target:
            findings.append("width below target" if width < .1 else "width above target")
        if not peak_in_target:
            findings.append("peak below target" if peak < .3 else "peak above target")
        rows.append([label, "%.4f +/- %.4f" % (width, e["fwhm_error_MHz"]),
                     "%.3f" % peak, "%.6g" % e["config"]["qubit_gain_ge"],
                     "; ".join(findings) or "both targets met"])
        records.append(dict(label=label, run=result["run"], qspec_h5=e["h5"], fwhm_MHz=width,
                            peak_population=peak, width_in_target=width_in_target,
                            peak_in_target=peak_in_target, findings=findings))
    source = dict(width_target_MHz=[.1, .2], peak_population_target=[.3, .5], records=records,
                  interpretation="Strict nominal target comparison, separate from measurement quality/reproduction gates. "
                  "Outside-target values remain explicit; accepted reproduction does not imply all tuning targets were met.")
    (ARTIFACTS / "spectroscopy_target_review.json").write_text(json.dumps(source, indent=2))
    for start in range(0, len(rows), 12):
        table_slide(prs, "Spectroscopy tuning targets | remaining deviations",
                    ["Operating point", "FWHM +/- (MHz)", "Fitted peak pop.", "Qspec gain", "Target assessment"],
                    rows[start:start + 12], source,
                    "Targets: FWHM 0.1--0.2 MHz; calibrated peak population 0.3--0.5. Reproduction acceptance does not imply these tuning targets were met.", 11)


def raw(result, kind):
    e = result["experiments"][kind]
    with h5py.File(e["h5"], "r") as handle:
        g = handle["Q%d" % (result["qubit_index"] + 1)]
        if kind == "res_ge":
            return decode_numeric(g["freq_pts"]) + decode_numeric(g["freq_center"])[0], decode_numeric(g["Amps"])
        name = "Frequencies" if kind == "qspec_ge" else "Gains" if kind == "rabi_ge" else "Delay Times"
        return decode_numeric(g[name]), decode_numeric(g["Population"])


def current_map(q, rows, kind):
    current = [i for i in GRID if measurement_passes(rows.get(i), kind)]
    arrays = [raw(rows[i], kind) for i in current]
    if kind == "qspec_ge":
        arrays = [(x - rows[i]["experiments"][kind]["center_MHz"], y) for i, (x, y) in zip(current, arrays)]
    grid = np.linspace(min(x[0] for x, y in arrays), max(x[-1] for x, y in arrays), 601)
    z = np.full((len(GRID), len(grid)), np.nan)
    for i, (x, y) in zip(current, arrays):
        z[GRID.index(i)] = np.interp(grid, x, y, left=np.nan, right=np.nan)
    fig, axes = plt.subplots(1, 2 if kind == "qspec_ge" else 1, figsize=(12.5, 5.6), squeeze=False)
    ax = axes[0, 0]
    display = ax.imshow(z, aspect="auto", origin="lower", interpolation="none", cmap="viridis",
                        extent=(grid[0], grid[-1], -.5, len(GRID) - .5))
    ax.set_yticks(range(len(GRID)))
    ax.set_yticklabels(["%+.1f" % i for i in GRID])
    ax.set_ylabel("Current (mA)")
    ax.set_xlabel("Frequency - fitted f_ge (MHz)" if kind == "qspec_ge" else
                  "Frequency (MHz)" if kind == "res_ge" else "DAC gain" if kind == "rabi_ge" else "Delay (us)")
    ax.set_title(TITLES[kind] + " | new round-robin measurements")
    fig.colorbar(display, ax=ax, label="Raw readout amplitude" if kind == "res_ge" else "Calibrated population")
    if kind == "qspec_ge":
        ax2 = axes[0, 1]
        ax2.plot(GRID, [rows[i]["experiments"][kind]["center_MHz"] if i in current else np.nan for i in GRID], "o-")
        ax2.set(xlabel="Current (mA)", ylabel="Qubit frequency (MHz)", title="Measured centers of local scan windows")
        ax2.grid(alpha=.2)
    fig.tight_layout()
    path = ARTIFACTS / ("M%d_%s_map.png" % (q + 1, kind))
    fig.savefig(path, dpi=190)
    plt.close(fig)
    return path


def bias_slides(prs, bias):
    for q, results in bias.items():
        if not results:
            continue
        label = "Yoko %d / physical Q%d / M%d" % (3 if q == 4 else 4, 4 if q == 4 else 6, q + 1)
        rows = []
        for current in GRID:
            r = results.get(current)
            row = ["%+.1f" % current]
            for kind in KINDS[4:]:
                value = checked_metric(r, kind, "tau_us")
                row.append(value + " +/- " + checked_metric(r, kind, "error_us") if measurement_passes(r, kind) else value)
                row.append("%+.1f%%" % r["comparisons"][kind]["change_percent"] if measurement_passes(r, kind) else value)
            row.append(status_text(r))
            rows.append(row)
        table_slide(prs, label + " | coherence comparison", ["mA", "T1 (us)", "Change", "T2R (us)", "Change", "T2E (us)", "Change", "Audit"],
                    rows, results, "Other Yoko = 0 mA. Errors are fit standard errors. Each row links to its exact RR pass in slide notes.", 10)
        settings_slides(prs, label, [("%+.1f mA" % i, r) for i, r in sorted(results.items())])
        for kind in ("res_ge", "qspec_ge", "rabi_ge", "t1_ge", "t2_ge", "t2e_ge"):
            path = current_map(q, results, kind)
            B.plot_slide(prs, label + " | " + TITLES[kind] + " map", [path],
                         "Validated RR samples; interpolation only inside each acquired window. Blank rows are pending or under review. "
                         + ("Local qspec windows are aligned to fitted frequency to show linewidth." if kind == "qspec_ge" else ""), results)
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
        current = GRID
        accepted = {i for i in current if measurement_passes(results.get(i), "rabi_ge")}
        axes[0].plot(current, [results[i]["experiments"]["rabi_ge"]["config"]["sigma"] if i in accepted else np.nan for i in current], "o-")
        axes[0].set(xlabel="Current (mA)", ylabel="Gaussian sigma (us)")
        axes[1].plot(current, [results[i]["experiments"]["rabi_ge"]["measured_pi_gain"] if i in accepted else np.nan for i in current], "o-")
        axes[1].axhspan(.6, .7, alpha=.15, color="green")
        axes[1].set(xlabel="Current (mA)", ylabel="Measured pi gain")
        for ax in axes: ax.grid(alpha=.2)
        fig.tight_layout()
        path = ARTIFACTS / ("M%d_sigma.png" % (q + 1))
        fig.savefig(path, dpi=190)
        plt.close(fig)
        B.plot_slide(prs, label + " | Gaussian pulse calibration", [path], "Green band = accepted pi gain 0.6--0.7. Pulse length = 4 sigma.", results)


def followup_slides(prs, manifest):
    zero, bias = selected_results(manifest)
    reviewed = [r for r in list(zero.values()) +
                [r for rows in bias.values() for current, r in rows.items() if current] +
                [load(item["audit"]) for item in manifest.get("controls", [])]
                if r.get("review_decision") or r.get("review_decisions")]
    changes = []
    for r in reviewed:
        for decision in r.get("review_decisions") or [r["review_decision"]]:
            kind = decision["kind"]
            if kind == "qspec_ge":
                prior = r["reference"]["qspec"]["center_MHz"]
                measured = r["experiments"][kind]["center_MHz"]
                repeat = decision["confirmation_center_MHz"]
                unit, places = "MHz", 6
            elif kind == "ss_ge":
                prior = r["reference"]["ssf_contrast"]
                measured = r["experiments"][kind]["contrast"]
                repeat = decision["confirmation_contrast"]
                unit, places = "fraction", 4
            else:
                prior = r["reference"][kind]["tau_us"]
                measured = r["experiments"][kind]["tau_us"]
                repeat = decision["confirmation_tau_us"]
                unit, places = "us", 3
            identity = "M%d / %+.1f mA" % (r["qubit_index"] + 1, r["current_mA"])
            if r["qubit_index"] not in (4, 5) and (r["yoko3_current_mA"] or r["yoko4_current_mA"]):
                identity = "M%d / Y3 %+.1f / Y4 %+.1f mA" % (
                    r["qubit_index"] + 1, r["yoko3_current_mA"], r["yoko4_current_mA"])
            changes.append([identity,
                            TITLES[kind], fmt(prior, places), fmt(measured, places),
                            fmt(repeat, places), unit,
                            "repeat within old band" if decision.get("type") ==
                            "consistent repeat; reference band overlap" else
                            "frequency varies" if decision.get("type") ==
                            "variable frequency; tracking validated" else "change confirmed"])
    for start in range(0, len(changes), 12):
        table_slide(prs, "Independent checks of differences from previous measurements",
                    ["Operating point", "Measurement", "Previous", "Full RR", "Independent repeat", "Unit", "Decision"],
                    changes[start:start + 12], reviewed,
                    "Original comparison flags and confirmation sources remain in the reviewed audits. Fit/raw/calibration gates were retained; no physical cause is assigned.", 11)
    for r in reviewed:
        for decision in r.get("review_decisions") or [r["review_decision"]]:
            if decision.get("type") != "variable frequency; tracking validated":
                continue
            low, high = decision["observed_center_range_MHz"]
            slide = B.slide_base(prs, "M%d | spectroscopy tracks a varying frequency" % (r["qubit_index"] + 1),
                                 "All original comparisons and raw traces are retained. No physical cause is assigned.", decision)
            B.textbox(slide, "At Yoko 3 +10 mA, four resolved spectra span %.6f--%.6f MHz.\n\n"
                      "The two scans with matching 600 s settling differ by %.3f MHz. This exceeds the strict fixed-shift repeat tolerance of %.3f MHz.\n\n"
                      "Two full round-robin sequences reproduce Rabi, readout and all three coherence measurements while using each newly measured qubit frequency.\n\n"
                      "Keep qspec enabled in every normal pass. The table uses the complete settled run; the observed frequency range is separate from its fit uncertainty." %
                      (low, high, decision["settled_repeat_separation_MHz"], decision["static_repeat_limit_MHz"]),
                      .65, 1.2, 12.0, 5.4, 22)
        if r["qubit_index"] in (4, 5) or not (r["yoko3_current_mA"] or r["yoko4_current_mA"]):
            continue
        for decision in r.get("review_decisions") or [r["review_decision"]]:
            kind = decision["kind"]
            repeat = load(decision["confirmation_audit"])
            B.plot_slide(prs, "Control M%d | independent %s repeat" % (r["qubit_index"] + 1, TITLES[kind]),
                         [r["experiments"][kind]["plot"], repeat["experiments"][kind]["plot"]],
                         "Y3 %+.1f / Y4 %+.1f mA. Left: full RR; right: exact-setting independent repeat. %s" % (
                             r["yoko3_current_mA"], r["yoko4_current_mA"], decision["explanation"]), decision)
    for item in manifest.get("diagnostic_figures", []):
        B.plot_slide(prs, item["title"], item.get("plots", [item["plot"]]), item["caption"], item)
    for index, source in enumerate(manifest.get("readout_grids", [])):
        grid = load(source)
        points = grid.get("points", [])
        if not points:
            continue
        keys = ("res_freq_ge", "res_gain_ge", "res_length")
        labels = dict(res_freq_ge="Readout frequency (MHz)", res_gain_ge="Readout gain", res_length="Readout length (us)")
        varied = [k for k in keys if len({p["config"][k] for p in points}) > 1]
        fig, ax = plt.subplots(figsize=(11.5, 5.5))
        if len(varied) > 1:
            artist = ax.scatter([p["config"][varied[0]] for p in points],
                                [p["config"][varied[1]] for p in points],
                                c=[p["contrast"] for p in points], s=150, marker="s", cmap="viridis")
            ax.set(xlabel=labels[varied[0]], ylabel=labels[varied[1]])
            ax.ticklabel_format(axis="both", style="plain", useOffset=False)
            fig.colorbar(artist, ax=ax, label="Measured SSF contrast")
        else:
            key = varied[0] if varied else keys[0]
            ordered = sorted(points, key=lambda p: p["config"][key])
            coordinates = [p["config"][key] for p in ordered]
            style = "o" if len(set(coordinates)) < len(coordinates) else "o-"
            ax.plot(coordinates, [p["contrast"] for p in ordered], style)
            ax.set(xlabel=labels[key], ylabel="Measured SSF contrast")
            ax.ticklabel_format(axis="x", style="plain", useOffset=False)
            ax.grid(alpha=.2)
        first = points[0]["settings"]
        ax.set_title("M%d | Yoko 3 %s mA, Yoko 4 %s mA | readout scan" % (
            int(first["PUCQ4_QS"]) + 1,
            first.get("PUCQ4_YOKO3_CURRENT_MA", "0"),
            first.get("PUCQ4_YOKO4_CURRENT_MA", "0")))
        fig.tight_layout()
        target = ARTIFACTS / ("readout_followup_%02d.png" % index)
        fig.savefig(target, dpi=190)
        plt.close(fig)
        fixed = ["%s = %s" % (labels[k], fmt(points[0]["config"][k], 4))
                 for k in keys if k not in varied]
        caption = ("Fixed readout: %s. Each point: %s shots/state, %s us relaxation. "
                   "Actual SS-only RR data; complete-sequence confirmation is reported separately." % (
                       "; ".join(fixed), first.get("PUCQ4_SSF_SHOTS", "recorded"),
                       first.get("PUCQ4_SSF_RELAX_DELAY_US", "recorded")))
        B.plot_slide(prs, "Readout follow-up | scan %d" % (index + 1), [target],
                     caption, grid)
    if manifest.get("controls"):
        rows = []
        records = []
        for control in manifest["controls"]:
            r = load(control["audit"])
            records.append(r)
            rows.append(["M%d" % (r["qubit_index"] + 1), "%+.1f / %+.1f" % (r["yoko3_current_mA"], r["yoko4_current_mA"]),
                         checked_metric(r, "qspec_ge", "center_MHz", 5), checked_metric(r, "qspec_ge", "fwhm_MHz", 4),
                         checked_metric(r, "rabi_ge", "measured_pi_gain", 4), checked_metric(r, "t1_ge", "tau_us"),
                         checked_metric(r, "t2_ge", "tau_us"), checked_metric(r, "t2e_ge", "tau_us"), status_text(r)])
        for start in range(0, len(rows), 12):
            table_slide(prs, "Control rows at nonzero Yoko biases",
                        ["Row", "Y3/Y4 (mA)", "Qubit (MHz)", "FWHM (MHz)", "Pi gain", "T1 (us)", "T2R (us)", "T2E (us)", "Audit"],
                        rows[start:start + 12], records, "Normal RR settings at the selected currents. Failed checks withhold affected values; findings and raw traces remain in the source audit.", 10)
        for r in records:
            for kind in KINDS:
                e = r["experiments"].get(kind, {})
                if not measurement_passes(r, kind) and e.get("plot"):
                    B.plot_slide(prs, "Control M%d | Y3 %+.1f / Y4 %+.1f mA | %s" % (
                        r["qubit_index"] + 1, r["yoko3_current_mA"], r["yoko4_current_mA"], TITLES[kind]),
                        [e["plot"]], "Original RR trace retained for review. Findings: " + "; ".join(r["issues"]), r)
    changes = manifest.get("configuration_changes", [])
    for start in range(0, len(changes), 14):
        table_slide(prs, "Persistent configuration changes",
                    ["Row/current", "Setting", "Old", "New", "Unit", "Source"],
                    [[c["row"], c["setting"], c["old"], c["new"], c["unit"], c["file"]] for c in changes[start:start + 14]],
                    changes, "Underlying experiment classes and QICK programs are unchanged from the task-start snapshot.", 10)


def build():
    manifest = load(ARTIFACTS / "validation_manifest.json")
    zero, bias = selected_results(manifest)
    prs = Presentation()
    prs.slide_width, prs.slide_height = Inches(13.333), Inches(7.5)
    zero_slides(prs, zero, manifest)
    spectroscopy_target_slides(prs, zero, bias)
    bias_slides(prs, bias)
    followup_slides(prs, manifest)
    issues = [("M%d / %+.1f mA" % (r["qubit_index"] + 1, r["current_mA"]), issue)
              for r in list(zero.values()) + [r for rows in bias.values() for i, r in rows.items() if i]
              for issue in r["issues"]]
    for control in manifest.get("controls", []):
        r = load(control["audit"])
        issues.extend(("M%d / Y3 %+.1f, Y4 %+.1f mA" %
                       (r["qubit_index"] + 1, r["yoko3_current_mA"], r["yoko4_current_mA"]), issue)
                      for issue in r["issues"])
    for path in manifest.get("final_default_validation", {}).get("audits", {}).values():
        r = load(path)
        issues.extend(("Final default M%d / zero current" % (r["qubit_index"] + 1), issue)
                      for issue in r["issues"])
    if issues:
        for start in range(0, len(issues), 12):
            table_slide(prs, "Results requiring review", ["Operating point", "Finding"], issues[start:start + 12], manifest,
                        "A completed acquisition does not by itself prove agreement. These results remain visibly flagged.", 12)
    staging = ARTIFACTS / "report_staging.pptx"
    prs.save(str(staging))
    check = Presentation(str(staging))
    assert len(check.slides) == len(prs.slides)
    staging.replace(OUTPUT)
    (ARTIFACTS / "report_build.json").write_text(json.dumps(dict(
        updated=datetime.datetime.now().isoformat(), output=str(OUTPUT), slides=len(prs.slides),
        zero_rows=len(zero), bias_rows={q: len(v) for q, v in bias.items()}, issues=issues), indent=2))
    print(json.dumps(dict(output=str(OUTPUT), slides=len(prs.slides), issues=len(issues))))


if __name__ == "__main__":
    build()
