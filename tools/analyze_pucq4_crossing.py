"""Offline complex-IQ line extraction, bias maps and conditional coupled-mode fit.

All reported data come from canonical paired drive-on/off HDF5 records. A
two-mode fit is attempted only with two resolved peaks at several bias points.
Physical Q5 identity is kept separate from an observed resonator-row coupling.
"""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks, peak_widths, savgol_filter


def close_pair(f, iq, noise, seeds):
    """Fit two detected nearby peaks together, retaining a one-line null model."""
    df = float(np.median(np.diff(f)))
    centers = np.sort(np.asarray(seeds, dtype=float))
    half = max(.6, 15 * df)
    use = (f >= centers[0] - half) & (f <= centers[1] + half)
    ff, yy = f[use], iq[use] / noise
    if len(ff) < 20:
        return None
    xx = (ff - np.mean(ff)) / np.ptp(ff)
    def solve(theta, return_coefficient=False):
        profiles = [1 / (1 + (2 * (ff - c) / w) ** 2)
                    for c, w in np.asarray(theta).reshape(-1, 2)]
        design = np.column_stack(profiles + [np.ones(len(ff)), xx])
        coefficient = np.linalg.lstsq(design, yy, rcond=None)[0]
        if return_coefficient:
            return coefficient
        return (design @ coefficient - yy).ravel()
    middle = float(np.mean(centers))
    bounds = ([ff[2], df * 1.01, middle, df * 1.01],
              [middle, half * 1.5, ff[-3], half * 1.5])
    fits = [least_squares(solve, [centers[0], width, centers[1], width], bounds=bounds)
            for width in (max(3.1 * df, .08), .2, .4)]
    fit = min(fits, key=lambda item: np.sum(item.fun ** 2))
    single = [least_squares(solve, [center, width],
              bounds=([ff[2], df * 1.01], [ff[-3], np.ptp(ff)]))
              for center in (centers[0], middle, centers[1]) for width in (.15, .5)]
    chi = float(np.sum(fit.fun ** 2))
    null_chi = min(float(np.sum(item.fun ** 2)) for item in single)
    delta_bic = null_chi - chi - 4 * np.log(fit.fun.size)
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * max(1, chi / (fit.fun.size - 12))
    errors = np.sqrt(np.diag(covariance))
    separation = float(fit.x[2] - fit.x[0])
    separation_error = float(np.sqrt(max(0, covariance[0, 0] + covariance[2, 2] - 2 * covariance[0, 2])))
    coefficients = solve(fit.x, True)
    rms = float(np.sqrt(chi / fit.fun.size))
    widths = fit.x[[1, 3]]
    # Resolving two maxima and beating a single-line fit is required. Merely
    # decomposing a broad peak into two fitted components is not a splitting.
    accepted = bool(fit.success and not np.any(fit.active_mask)
        and np.linalg.matrix_rank(fit.jac) == 4 and delta_bic > 10 and rms < 2
        and separation > max(widths) and separation > 3 * separation_error)
    entries = []
    for index in range(2):
        pos = index * 2
        snr = float(np.linalg.norm(coefficients[index]))
        width = float(fit.x[pos + 1])
        accepted = bool(accepted and snr >= 5 and width >= 3 * df
            and width < half * 1.3 and errors[pos] < width / 3
            and errors[pos + 1] < width / 2)
        entries.append(dict(center_MHz=float(fit.x[pos]), center_error_MHz=float(errors[pos]),
            FWHM_MHz=width, FWHM_error_MHz=float(errors[pos + 1]), snr=snr,
            residual_rms_noise_units=rms, samples_per_FWHM=width / df,
            fit_window_MHz=[float(ff[0]), float(ff[-1])],
            fit_method="joint two-line complex-IQ", delta_BIC_over_single=delta_bic,
            joint_center_covariance_MHz2=covariance[np.ix_([0, 2], [0, 2])].tolist()))
    if not accepted:
        return None
    for entry in entries:
        entry["accepted"] = True
    return entries


def read_record(record):
    arrays = {}
    for kind, path in record["traces"].items():
        with h5py.File(path, "r") as handle:
            group = handle["Q%d" % record["row"]]
            arrays[kind] = dict(f=np.asarray(group["Frequencies"][0], dtype=float),
                iq=np.column_stack((group["I"][0], group["Q"][0])),
                config=json.loads(group["Exp Config"][0]),
                acquired_unix=float(group["Dates"][0]),
                raw_shape=list(group["Raw IQ"].shape))
    np.testing.assert_allclose(arrays["drive_on"]["f"], arrays["drive_off"]["f"], atol=1e-8)
    f = arrays["drive_on"]["f"]
    if len(f) != record["requested"]["steps"] or not np.all(np.isfinite(f)) or np.any(np.diff(f) <= 0):
        raise ValueError("Unexpected acquired frequency grid")
    for kind, data in arrays.items():
        if data["iq"].shape != (len(f), 2) or not np.all(np.isfinite(data["iq"])):
            raise ValueError("Invalid saved IQ array")
        if data["config"]["acquisition_kind"] != kind:
            raise ValueError("Reference label mismatch")
        if abs(data["config"]["actual_yoko4_current_mA"] - record["current_mA"]) > 1e-4:
            raise ValueError("Actual bias mismatch")
    recorded_current_mA(arrays)
    return f, arrays["drive_on"]["iq"] - arrays["drive_off"]["iq"], arrays


def recorded_current_mA(arrays):
    """Use the recorded source setting, retaining requests separately downstream."""
    on = float(arrays["drive_on"]["config"]["actual_yoko4_current_mA"])
    off = float(arrays["drive_off"]["config"]["actual_yoko4_current_mA"])
    if not np.isfinite(on) or not np.isfinite(off) or abs(on - off) > 1e-9:
        raise ValueError("Drive-on/off source settings differ or are nonfinite")
    return on


def narrow_line_detail(f, iq, point, output, number):
    peaks = [p for p in point["metrics"]["candidates"] if p["accepted"]]
    limited = False
    if not peaks:
        peaks = [p for p in point["metrics"]["candidates"] if p["snr"] >= 5
                 and p["samples_per_FWHM"] < 3 and p.get("fit_success")
                 and p.get("fit_jacobian_rank") == 2
                 and p["residual_rms_noise_units"] < 2
                 and p["center_error_MHz"] < p["FWHM_MHz"] / 3
                 and p["FWHM_error_MHz"] < p["FWHM_MHz"] / 2
                 and not any(p.get("active_parameter_bounds", []))]
        limited = bool(peaks)
    if len(peaks) != 1 or np.ptp(f) / peaks[0]["FWHM_MHz"] <= 100:
        return
    peak = peaks[0]
    center, width = peak["center_MHz"], peak["FWHM_MHz"]
    lo, hi = peak["fit_window_MHz"]
    use = (f >= lo) & (f <= hi)
    def design(axis):
        return np.column_stack((1 / (1 + (2 * (axis - center) / width) ** 2),
                                np.ones(len(axis)), (axis - center) / (hi - lo)))
    coefficient = np.linalg.lstsq(design(f[use]), iq[use], rcond=None)[0]
    half = max(4 * width, 15 * point["metrics"]["frequency_step_MHz"])
    shown = np.abs(f - center) <= half
    dense = np.linspace(f[shown][0], f[shown][-1], 401)
    model = design(dense) @ coefficient
    fig, axes = plt.subplots(2, 1, figsize=(9, 6), sharex=True)
    for index, axis in enumerate(axes):
        axis.plot(f[shown], iq[shown, index], "o", ms=4, label="Measured samples")
        axis.plot(dense, model[:, index], "--" if limited else "-", color="black",
                  label="Unaccepted candidate fit" if limited else "Complex Lorentzian fit")
        axis.set_ylabel("On - off %s (ADC units)" % ("I" if index == 0 else "Q"))
    axes[0].legend()
    axes[0].set_title("M%d %s at %.6f mA\n%.2f kHz bins | %.2f samples/FWHM" % (
        point["row"], "SAMPLING-LIMITED CANDIDATE" if limited else "detail", point["current_mA"],
        1000 * point["metrics"]["frequency_step_MHz"], peak["samples_per_FWHM"]))
    axes[1].set_xlabel("Spectroscopy frequency (MHz)")
    axes[1].ticklabel_format(axis="x", style="plain", useOffset=False)
    fig.tight_layout()
    path = output / ("M%d_trace%03d_detail.png" % (point["row"], number))
    fig.savefig(path, dpi=180); plt.close(fig)
    point["detail_plot"] = str(path)
    point["detail_status"] = "sampling limited candidate" if limited else "accepted line"


def line_candidates(f, iq):
    """Shared Lorentzian centers/widths fit to signed I and Q, never |IQ|."""
    df = float(np.median(np.diff(f)))
    x = (f - np.mean(f)) / np.ptp(f)
    baseline = np.column_stack((np.ones(len(f)), x))
    residual = iq - baseline @ np.linalg.lstsq(baseline, iq, rcond=None)[0]
    noise = np.median(np.abs(np.diff(residual, axis=0) - np.median(np.diff(residual, axis=0), axis=0)), axis=0) / (0.67448975 * np.sqrt(2))
    noise = np.maximum(noise, 1e-9)
    whitened = residual / noise
    _, _, axes = np.linalg.svd(whitened, full_matrices=False)
    projected = whitened @ axes[0]
    window = max(5, int(round(0.10 / df)) | 1)
    window = min(window, 21, len(f) // 2 * 2 - 1)
    smooth = savgol_filter(projected, window, 2)
    # A sparse line need not dominate the covariance of an entire wide scan:
    # global PCA can choose nearly perpendicular noise and hide a real line.
    # Detect with both signed components; still fit the original signed IQ.
    score = np.linalg.norm(savgol_filter(whitened, window, 2, axis=0), axis=1)
    index, props = find_peaks(score, height=3.5, prominence=3.0,
                             distance=max(3, int(round(.08 / df))))
    candidates = []
    for peak in index[np.argsort(props["prominences"])[::-1]][:8]:
        detected_width = float(peak_widths(score, [peak], rel_height=.5)[0][0]) * df
        half = min(max(.6, 15 * df, 2.5 * detected_width), np.ptp(f) / 3)
        use = np.abs(f - f[peak]) <= half
        ff, yy = f[use], iq[use]
        if len(ff) < 12 or peak < 4 or peak > len(f) - 5:
            continue
        center0 = f[peak]
        # Variable projection solves the independent complex amplitudes and
        # linear backgrounds exactly for each common center/linewidth pair.
        xx = (ff - center0) / half
        def profile(center, width):
            return 1 / (1 + (2 * (ff - center) / width) ** 2)
        def solve(theta, return_fit=False):
            design = np.column_stack((profile(*theta), np.ones(len(ff)), xx))
            coefficient = np.linalg.lstsq(design, yy / noise, rcond=None)[0]
            fitted = design @ coefficient
            if return_fit:
                return fitted * noise, coefficient
            return (fitted - yy / noise).ravel()
        best = None
        for width0 in (.12, .3, .6):
            fit = least_squares(solve, (center0, min(width0, half)),
                   bounds=([ff[2], max(df * 1.01, .005)], [ff[-3], half * 1.5]))
            if best is None or np.sum(fit.fun ** 2) < np.sum(best.fun ** 2):
                best = fit
        fitted, coefficient = solve(best.x, True)
        dof = len(best.fun) - 8
        covariance = np.linalg.pinv(best.jac.T @ best.jac) * np.sum(best.fun ** 2) / dof
        error = np.sqrt(np.diag(covariance))
        width = best.x[1]
        snr = float(np.linalg.norm(coefficient[0]))
        rms = float(np.sqrt(np.mean(best.fun ** 2)))
        rank = int(np.linalg.matrix_rank(best.jac))
        accepted = bool(best.success and rank == 2 and not np.any(best.active_mask)
                        and snr >= 5 and width >= 3 * df and width < half * 1.3
                        and error[0] < width / 3 and error[1] < width / 2 and rms < 2)
        entry = dict(center_MHz=float(best.x[0]), center_error_MHz=float(error[0]),
            FWHM_MHz=float(width), FWHM_error_MHz=float(error[1]), snr=snr,
            residual_rms_noise_units=rms, samples_per_FWHM=float(width / df),
            accepted=accepted, fit_window_MHz=[float(ff[0]), float(ff[-1])],
            fit_success=bool(best.success), fit_jacobian_rank=rank,
            active_parameter_bounds=best.active_mask.tolist())
        if not any(abs(entry["center_MHz"] - old["center_MHz"]) < min(width, old["FWHM_MHz"]) / 2 for old in candidates):
            candidates.append(entry)
    # The independent local windows overlap for close branches. Fit that pair
    # jointly so each line cannot bias the other's center or linewidth.
    ordered = np.sort(index[np.argsort(props["prominences"])[::-1]][:8])
    for left, right in zip(ordered[:-1], ordered[1:]):
        if f[right] - f[left] >= 1.2 or left < 4 or right > len(f) - 5:
            continue
        pair = close_pair(f, iq, noise, [f[left], f[right]])
        if pair is not None:
            lo, hi = pair[0]["fit_window_MHz"]
            candidates = [old for old in candidates if not lo <= old["center_MHz"] <= hi]
            candidates.extend(pair)
    return dict(candidates=sorted(candidates, key=lambda item: item["center_MHz"]),
                noise_IQ=noise.tolist(), frequency_step_MHz=df,
                projected_snr_max=float(np.max(np.abs(smooth))),
                vector_snr_max=float(np.max(score))), projected


def mode_frequencies(parameters, current, varying_fixed=False):
    t0, slope, fixed, coupling = parameters[:4]
    tunable = t0 + slope * current
    fixed = fixed + (parameters[4] * current if varying_fixed else 0)
    middle = (tunable + fixed) / 2
    shift = np.sqrt(((tunable - fixed) / 2) ** 2 + coupling ** 2)
    return np.column_stack((middle - shift, middle + shift))


def segmented_pair_plot(pair, points, output, label):
    """Show actual signed IQ in separate measured windows without joining gaps."""
    originals = [p for p in points if p["traces"] in pair["traces"]]
    originals.sort(key=measured_window)
    fig, axes = plt.subplots(2, len(originals), figsize=(12, 6), squeeze=False)
    for column, point in enumerate(originals):
        manifest = json.loads(Path(point["source_manifest"]).read_text(encoding="utf-8"))
        f, iq, _ = read_record(manifest["records"][point["source_record_index"]])
        candidates = [c for c in point["metrics"]["candidates"] if c["accepted"]]
        for component in range(2):
            axis = axes[component, column]
            axis.plot(f, iq[:, component], ".-", ms=3, lw=.7)
            for peak in candidates:
                axis.axvline(peak["center_MHz"], color="black", ls="--", lw=.8)
            axis.set_ylabel("On - off %s (ADC units)" % ("I" if component == 0 else "Q"))
            axis.ticklabel_format(axis="x", useOffset=False, style="plain")
            axis.locator_params(axis="x", nbins=4)
        axes[1, column].set_xlabel("Measured drive frequency (MHz)")
        axes[0, column].set_title("%d averages/state | %d frequency samples\n" %
            (point["requested"]["reps"], len(f)) + "\n".join(
            "%.6f +/- %.6f MHz\nFWHM %.4f MHz | SNR %.2f" %
            (c["center_MHz"], c["center_error_MHz"], c["FWHM_MHz"], c["snr"])
            for c in candidates), fontsize=11)
    fig.suptitle("%s | M%d | %.6f mA source setting | gain %g\n"
                 "Separate sequential windows; intervening gap unmeasured; conditional fit errors" %
                 (label, pair["row"], pair["current_mA"], pair["gain"]), fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, .89))
    path = output / ("%s_M%d_%+.6f_g%g_pair.png" %
                     (label, pair["row"], pair["current_mA"], pair["gain"]))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return str(path)


def measured_window(point):
    """Prefer the saved QICK axis; legacy/synthetic points use requested bounds."""
    return tuple(point.get("actual_frequency_window_MHz", (
        point["requested"]["start_MHz"], point["requested"]["stop_MHz"])))


def merge_paired_windows(visits):
    """Combine explicitly tagged disjoint windows on the available readouts.

    This is analysis-only: original per-window points and raw paths remain
    unchanged. Untagged repeats, missing windows, mixed visits/settings/sources,
    contaminated references, and overlapping windows cannot form a pair.
    """
    flat = [p for group in visits for p in group]
    tags = {p["requested"].get("paired_windows_id") for p in flat}
    if len(tags) != 1 or not next(iter(tags), None):
        return None, "Window group needs one explicit visit identifier"
    sources = {p.get("source_manifest") for p in flat}
    settings = {(p["current_mA"], p["gain"], p["requested"]["length_us"],
                 p["requested"].get("relax_us", 100)) for p in flat}
    if len(sources) != 1 or None in sources or len(settings) != 1:
        return None, "Window group mixes source manifests or drive settings"
    indices = [p.get("source_record_index") for p in flat]
    if any(i is not None for i in indices):
        if (any(i is None for i in indices) or len(set(indices)) != len(indices)
                or max(indices) - min(indices) + 1 != len(indices)):
            return None, "Window group is not one contiguous acquisition visit"
    if any(p.get("control_review_required") for p in flat):
        return None, "Window group reference needs review"
    coverage, ordered = [], []
    for group in visits:
        group = sorted(group, key=measured_window)
        windows = [measured_window(p) for p in group]
        if (len(windows) < 2 or any(lo >= hi for lo, hi in windows)
                or any(a[1] >= b[0] for a, b in zip(windows, windows[1:]))):
            return None, "Missing, repeated or overlapping frequency window"
        for point, (lo, hi) in zip(group, windows):
            if any(c["accepted"] and not lo < c["center_MHz"] < hi
                   for c in point["metrics"]["candidates"]):
                return None, "Accepted line outside its measured window"
        coverage.append(windows)
        ordered.append(group)
    if any(windows != coverage[0] for windows in coverage[1:]):
        return None, "Readouts do not have identical measured windows"
    records = []
    for group, windows in zip(ordered, coverage):
        merged = dict(group[0])
        merged["metrics"] = dict(candidates=[dict(c, readout_row=p["row"],
                acquired_unix=p["acquired_unix"], measured_window_MHz=list(window))
                for p, window in zip(group, windows) for c in p["metrics"]["candidates"]])
        merged["frequency_windows_MHz"] = windows
        merged["coverage_note"] = "Matched disjoint measured windows; intervening gaps are unmeasured"
        merged["traces"] = [p["traces"] for p in group]
        merged["window_requests"] = [dict(p["requested"],
            requested_current_mA=p.get("requested_current_mA", p["current_mA"]))
            for p in group]
        merged["window_source_record_indices"] = [p.get("source_record_index") for p in group]
        merged["window_acquisition_times"] = [p["acquired_unix"] for p in group]
        merged["acquired_unix"] = max(merged["window_acquisition_times"])
        records.append(merged)
    return records, None


def coalesce_tagged_windows(points):
    """Form within-readout spectra for an explicitly tagged held-bias visit.

    All available readouts must have complete matching coverage and one
    contiguous source visit. A single readout can independently resolve both
    lines; cross-readout pairing remains a separate review.
    """
    collected, review = [], []
    for current in sorted({p["current_mA"] for p in points}):
        local = [p for p in points if p["current_mA"] == current]
        if not any(p["requested"].get("paired_windows_id") for p in local):
            collected.extend(local)
            continue
        visits = [[p for p in local if p["row"] == row]
                  for row in sorted({p["row"] for p in local})]
        records, reason = merge_paired_windows(visits)
        check = dict(current_mA=current, reason=reason)
        review.append(check)
        if reason is None:
            for record in records:
                record["readout_rows"] = [record["row"]]
                record["selection"] = "Both measured windows in M%d" % record["row"]
                peaks = sorted([c for c in record["metrics"]["candidates"] if c["accepted"]],
                               key=lambda c: c["center_MHz"])
                if len(peaks) == 2:
                    separation = peaks[1]["center_MHz"] - peaks[0]["center_MHz"]
                    error = np.hypot(*(c["center_error_MHz"] for c in peaks))
                    record["window_pair_review_required"] = bool(
                        separation <= max(c["FWHM_MHz"] for c in peaks) or separation <= 3 * error)
            if len(records) > 1:
                rows = {record["row"] for record in records}
                if len(rows) != 2 or 6 not in rows:
                    check["cross_readout_reason"] = "Unsupported readout combination for window-pair corroboration"
                    corroborated = False
                else:
                    fixed_row = next(row for row in rows if row != 6)
                    paired, cross_review = pair_readouts(local, fixed_row)
                    check["cross_readout_review"] = cross_review
                    corroborated = bool(paired)
                if not corroborated:
                    for record in records:
                        record["window_pair_review_required"] = True
            collected.extend(records)
            check["frequency_windows_MHz"] = records[0]["frequency_windows_MHz"]
    return collected, review


def map_windows_do_not_overlap(points):
    """Permit tagged disjoint segments, but never paint repeated visits together."""
    for current in {p["current_mA"] for p in points}:
        group = [p for p in points if p["current_mA"] == current]
        if len(group) == 1:
            continue
        tags = {p["requested"].get("paired_windows_id") for p in group}
        windows = sorted(measured_window(p) for p in group)
        if (len(tags) != 1 or not next(iter(tags), None)
                or any(lo >= hi for lo, hi in windows)
                or any(a[1] >= b[0] for a, b in zip(windows, windows[1:]))):
            return False
    return True


def pair_readouts(points, fixed_row):
    """Use resolved frequencies from sequential readouts at the same drive settings.

    Prefer a measured doublet in one readout. Otherwise two distinct, resolved
    single lines can supply the pair. Never average coincident lines into two
    modes, merge repeated visits, or mix spectroscopy powers/pulse lengths.
    """
    paired, review = [], []
    for current in sorted(set(p["current_mA"] for p in points)):
        visits = [[p for p in points if p["current_mA"] == current and p["row"] == row]
                  for row in (6, fixed_row)]
        check = dict(current_mA=current)
        review.append(check)
        if any(p["requested"].get("paired_windows_id") for group in visits for p in group):
            records, reason = merge_paired_windows(visits)
            if reason:
                check["reason"] = reason
                continue
            check["common_frequency_windows_MHz"] = records[0]["frequency_windows_MHz"]
            check["coverage_note"] = records[0]["coverage_note"]
        else:
            if any(len(v) != 1 for v in visits):
                check["reason"] = "Missing or repeated readout visit"
                continue
            records = [v[0] for v in visits]
        settings = [(p["gain"], p["requested"]["length_us"], p["requested"].get("relax_us", 100))
                    for p in records]
        if settings[0] != settings[1] or any(p.get("control_review_required") for p in records):
            check["reason"] = "Different drive settings or reference needs review"
            continue
        peaks = [[c for c in p["metrics"]["candidates"] if c["accepted"]] for p in records]
        doublets = [(index, group) for index, group in enumerate(peaks) if len(group) == 2]
        if doublets:
            if len(doublets) == 2:
                ordered = [sorted(group, key=lambda c: c["center_MHz"]) for _, group in doublets]
                if any(abs(a["center_MHz"] - b["center_MHz"]) > 3 * np.hypot(a["center_error_MHz"], b["center_error_MHz"])
                       for a, b in zip(*ordered)):
                    check["reason"] = "Doublet centers disagree between readouts; check temporal or line-shape effects"
                    continue
            index, selected = max(doublets, key=lambda item: min(c["snr"] for c in item[1]))
            selection = "Both lines resolved in M%d" % records[index]["row"]
            other = peaks[1 - index]
            if any(not any(abs(a["center_MHz"] - b["center_MHz"]) <= max(a["FWHM_MHz"], b["FWHM_MHz"],
                           3 * np.hypot(a["center_error_MHz"], b["center_error_MHz"])) for b in selected) for a in other):
                check["reason"] = "Additional unmatched line in the other readout"
                continue
        elif all(len(group) == 1 for group in peaks):
            selected = [group[0] for group in peaks]
            selection = "One resolved line from each sequential readout"
        else:
            check["reason"] = "No unambiguous pair of accepted lines"
            continue
        selected = sorted(selected, key=lambda c: c["center_MHz"])
        separation = selected[1]["center_MHz"] - selected[0]["center_MHz"]
        uncertainty = np.hypot(*(c["center_error_MHz"] for c in selected))
        coverage = [p.get("frequency_windows_MHz", [measured_window(p)]) for p in records]
        covered = all(any(lo < c["center_MHz"] < hi for lo, hi in windows)
                      for windows in coverage for c in selected)
        if (separation <= max(c["FWHM_MHz"] for c in selected) or separation <= 3 * uncertainty
                or not covered):
            check["reason"] = "Lines overlap within linewidth/error or lack common frequency coverage"
            continue
        times = [p["acquired_unix"] for p in records]
        check.update(selection=selection, separation_MHz=separation,
                     readout_completion_separation_s=abs(times[1] - times[0]))
        paired.append(dict(current_mA=current, metrics=dict(candidates=selected),
            readout_rows=[6, fixed_row], gain=settings[0][0], selection=selection,
            readout_traces=[p["traces"] for p in records], acquired_unix=times,
            frequency_coverage_MHz=coverage,
            coverage_note=check.get("coverage_note", "Single continuous window per readout"),
            window_acquisition_times=[p.get("window_acquisition_times", [p["acquired_unix"]])
                                      for p in records]))
    return paired, review


def bias_reversal(points):
    """A-B-A center contrast, with a time-interpolated baseline from the A visits."""
    if (len(points) != 3 or points[0]["current_mA"] != points[2]["current_mA"]
            or points[0]["current_mA"] == points[1]["current_mA"]):
        return None
    selected = [[c for c in p["metrics"]["candidates"] if c["accepted"]] for p in points]
    if any(len(peaks) != 1 for peaks in selected) or any(p.get("control_review_required") for p in points):
        return dict(status="line_review_required", reason="A-B-A needs one accepted, reference-clean line per visit")
    centers = np.array([p[0]["center_MHz"] for p in selected])
    errors = np.array([p[0]["center_error_MHz"] for p in selected])
    times = np.array([p["acquired_unix"] for p in points])
    if np.any(np.diff(times) <= 0):
        raise ValueError("A-B-A records must retain chronological order")
    weight = float((times[1] - times[0]) / (times[2] - times[0]))
    baseline = (1 - weight) * centers[0] + weight * centers[2]
    contrast = float(centers[1] - baseline)
    uncertainty = float(np.sqrt(errors[1] ** 2 + ((1 - weight) * errors[0]) ** 2 + (weight * errors[2]) ** 2))
    repeat_error = float(np.hypot(errors[0], errors[2]))
    stable = abs(centers[2] - centers[0]) <= max(.05, 3 * repeat_error)
    threshold = max(.05, 3 * uncertainty)
    status = ("temporal_instability" if not stable else
              "bias_response_supported" if abs(contrast) > threshold else "bias_response_not_resolved")
    return dict(status=status, centers_MHz=centers.tolist(), center_errors_MHz=errors.tolist(),
        A_bias_mA=points[0]["current_mA"], B_bias_mA=points[1]["current_mA"],
        B_minus_interpolated_A_MHz=contrast, contrast_standard_error_MHz=uncertainty,
        contrast_threshold_MHz=threshold, A_repeat_difference_MHz=float(centers[2] - centers[0]),
        A_repeat_standard_error_MHz=repeat_error, interpolated_A_MHz=float(baseline),
        interpolation_weight=weight,
        interpretation="A center-stability control, not a splitting or coupling measurement. Linear drift interpolation does not remove abrupt frequency changes.")


def coupled_fit(points):
    pairs = [point for point in points if not point.get("control_review_required")
             and not point.get("window_pair_review_required")
             and len([c for c in point["metrics"]["candidates"] if c["accepted"]]) == 2]
    if len(pairs) < 7 or len(set(p["current_mA"] for p in pairs)) < 7:
        reason = ("No sampled bias has two accepted lines that pass the reference check" if not pairs
                  else "Only %d distinct biases have two accepted lines; seven are required for this fit" % len(set(p["current_mA"] for p in pairs)))
        return dict(status="unresolved", reason=reason)
    pairs.sort(key=lambda p: p["current_mA"])
    current = np.array([p["current_mA"] for p in pairs])
    center_current = np.mean(current)
    x = current - center_current
    peaks = [sorted([c for c in p["metrics"]["candidates"] if c["accepted"]],
                    key=lambda c: c["center_MHz"]) for p in pairs]
    freq = np.array([[c["center_MHz"] for c in pp] for pp in peaks])
    errors = np.array([[c["center_error_MHz"] for c in pp] for pp in peaks])
    errors = np.maximum(errors, 1e-4)
    whiteners = []
    for pair, error in zip(peaks, errors):
        matrix = np.diag(error ** 2)
        if (pair[0].get("fit_method") == "joint two-line complex-IQ"
                and pair[0].get("joint_center_covariance_MHz2") == pair[1].get("joint_center_covariance_MHz2")):
            matrix = np.asarray(pair[0]["joint_center_covariance_MHz2"])
            values, axes = np.linalg.eigh(matrix)
            matrix = (axes * np.maximum(values, 1e-8)) @ axes.T
        whiteners.append(np.linalg.inv(np.linalg.cholesky(matrix)))
    whiteners = np.asarray(whiteners)
    def residual(parameters, varying_fixed=False):
        delta = mode_frequencies(parameters, x, varying_fixed) - freq
        return np.einsum("nij,nj->ni", whiteners, delta).ravel()
    separation = freq[:, 1] - freq[:, 0]
    closest = int(np.argmin(separation))
    if closest < 2 or closest > len(pairs) - 3:
        return dict(status="unresolved", reason="Minimum observed separation is not bracketed by enough bias points")
    mid = float(np.mean(freq[closest]))
    starting = np.array([mid, 250.0, mid, float(separation[closest] / 2)])
    fit = least_squares(residual, starting,
            bounds=([freq.min() - 100, 0, freq.min() - 100, 0], [freq.max() + 100, 1500, freq.max() + 100, 100]))
    chi = float(np.sum(fit.fun ** 2))
    dof = freq.size - 4
    covariance = np.linalg.pinv(fit.jac.T @ fit.jac) * max(1.0, chi / dof)
    parameter_errors = np.sqrt(np.diag(covariance))
    resonance = center_current + (fit.x[2] - fit.x[0]) / fit.x[1]
    gradient = np.array([-1 / fit.x[1], -(fit.x[2] - fit.x[0]) / fit.x[1] ** 2, 1 / fit.x[1], 0])
    resonance_error = float(np.sqrt(gradient @ covariance @ gradient))
    varying = least_squares(lambda p: residual(p, True), np.r_[fit.x, 0.0],
            bounds=([freq.min() - 100, 0, freq.min() - 100, 0, -500], [freq.max() + 100, 1500, freq.max() + 100, 100, 500]))
    delta_aic = chi + 8 - (float(np.sum(varying.fun ** 2)) + 10)
    resolved = bool(fit.success and not np.any(fit.active_mask)
                    and np.linalg.matrix_rank(fit.jac) == 4
                    and current[1] < resonance < current[-2]
                    and fit.x[3] > 3 * parameter_errors[3]
                    and chi / dof < 10 and delta_aic < 6)
    return dict(status="resolved" if resolved else "model_review_required",
        resonance_bias_mA=float(resonance), resonance_bias_standard_error_mA=resonance_error,
        J_Hz=float(fit.x[3] * 1e6), J_standard_error_Hz=float(parameter_errors[3] * 1e6),
        minimum_splitting_Hz=float(2 * fit.x[3] * 1e6),
        minimum_splitting_standard_error_Hz=float(2 * parameter_errors[3] * 1e6),
        fitted_parameters_MHz=fit.x.tolist(), parameter_covariance=covariance.tolist(),
        current_origin_mA=float(center_current), reduced_chi_squared=chi / dof,
        fit_jacobian_rank=int(np.linalg.matrix_rank(fit.jac)),
        active_parameter_bounds=fit.active_mask.tolist(),
        fixed_frequency_test=dict(delta_AIC_in_favor_of_linear_fixed=delta_aic,
                                 fitted_fixed_slope_MHz_per_mA=float(varying.x[4])),
        uncertainty_note="Conditional least-squares standard errors; temporal/current drift and line-shape systematics are not included.",
        points=pairs)


def coupled_plot(key, fit, output):
    if fit["status"] != "resolved":
        return
    pairs = fit["points"]
    current = np.array([p["current_mA"] for p in pairs])
    peaks = [[c for c in p["metrics"]["candidates"] if c["accepted"]] for p in pairs]
    dense = np.linspace(current.min(), current.max(), 401)
    relative = dense - fit["current_origin_mA"]
    parameters = np.asarray(fit["fitted_parameters_MHz"])
    modes = mode_frequencies(parameters, relative)
    fig, axis = plt.subplots(figsize=(10, 6))
    for branch in range(2):
        axis.errorbar(current, [p[branch]["center_MHz"] for p in peaks],
                      yerr=[p[branch]["center_error_MHz"] for p in peaks],
                      fmt="o", color="C%d" % branch, capsize=3)
        axis.plot(dense, modes[:, branch], color="C%d" % branch,
                  label="Fitted coupled branches" if branch == 0 else None)
    axis.plot(dense, parameters[0] + parameters[1] * relative, "--", color="0.5", label="Fitted bare modes")
    axis.axhline(parameters[2], linestyle="--", color="0.5")
    axis.axvline(fit["resonance_bias_mA"], linestyle=":", color="black", label="Fitted minimum separation")
    axis.set(xlabel="Yoko4 source setting (mA)", ylabel="Spectroscopy frequency (MHz)",
             title="%s\n2|J| = %.4f +/- %.4f MHz (conditional SE)" %
             (key, fit["minimum_splitting_Hz"] / 1e6, fit["minimum_splitting_standard_error_Hz"] / 1e6))
    axis.ticklabel_format(axis="both", style="plain", useOffset=False)
    axis.legend()
    fig.tight_layout()
    path = output / (key + "_coupled_fit.png")
    fig.savefig(path, dpi=180); plt.close(fig)
    fit["plot"] = str(path)


def analyze(manifest_paths, output):
    output.mkdir(parents=True, exist_ok=True)
    points = []
    manifests = []
    for path in manifest_paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        manifests.append(manifest)
        for record_index, record in enumerate(manifest["records"]):
            if record["status"] != "complete":
                continue
            f, iq, raw = read_record(record)
            metrics, projected = line_candidates(f, iq)
            reference, _ = line_candidates(f, raw["drive_off"]["iq"])
            overlaps = [off for off in reference["candidates"] if off["accepted"]
                and any(on["accepted"] and abs(off["center_MHz"] - on["center_MHz"])
                        < max(off["FWHM_MHz"], on["FWHM_MHz"]) for on in metrics["candidates"])]
            point = dict(row=record["row"], current_mA=recorded_current_mA(raw),
                requested_current_mA=record["current_mA"], gain=record["requested"]["gain"],
                requested=record["requested"], readout_MHz=record["readout_MHz"],
                actual_frequency_window_MHz=[float(f[0]), float(f[-1])],
                source_record_index=record_index,
                source_level_readback_mA=raw["drive_on"]["config"]["actual_yoko4_current_mA"],
                acquired_unix=raw["drive_on"]["acquired_unix"],
                drive_off_control=reference, control_review_required=bool(overlaps),
                metrics=metrics, source_manifest=str(path), traces=record["traces"], plot=record["plot"])
            points.append(point)
            narrow_line_detail(f, iq, point, output, len(points))
    source = Path(__file__).read_bytes()
    (output / "analyze_pucq4_crossing.py").write_bytes(source)
    result = dict(source_manifests=[str(p) for p in manifest_paths], points=points, maps=[],
        coupled_fits={}, power_comparisons=[], bias_reversal_checks=[], map_omissions=[],
        paired_readout_reviews=[], segmented_readout_reviews=[],
        created_UTC=datetime.now(timezone.utc).isoformat(),
        analysis_code_sha256=hashlib.sha256(source).hexdigest(),
        current_axis="Recorded :SOURCE:LEVEL? setting in mA; not an independent ammeter. Requested current is retained separately; original acquisition plot titles may show the request.")
    window_cache = {}
    for row, gain, source in sorted(set((p["row"], p["gain"], p["source_manifest"]) for p in points)):
        subset = [p for p in points if p["row"] == row and p["gain"] == gain and p["source_manifest"] == source]
        series = Path(source).parents[2].name
        fit_subset = subset
        if any(p["requested"].get("paired_windows_id") for p in subset):
            cache_key = (source, gain)
            if cache_key not in window_cache:
                window_cache[cache_key] = coalesce_tagged_windows([
                    p for p in points if p["source_manifest"] == source and p["gain"] == gain])
            grouped, window_review = window_cache[cache_key]
            fit_subset = [p for p in grouped if p["row"] == row]
            accepted_pairs = [p for p in fit_subset if p.get("frequency_windows_MHz")
                and not p.get("window_pair_review_required")
                and len([c for c in p["metrics"]["candidates"] if c["accepted"]]) == 2]
            for pair in accepted_pairs:
                pair["measured_windows_plot"] = segmented_pair_plot(pair, subset, output, series)
            result["segmented_readout_reviews"].append(dict(
                series="%s_M%d_windows_gain%g" % (series, row, gain), source_label=series,
                accepted_pairs=accepted_pairs, review=window_review))
        row_fit = coupled_fit(fit_subset)
        result["coupled_fits"]["%s_M%d_gain%g" % (series, row, gain)] = row_fit
        reversal = bias_reversal(subset)
        if reversal is not None:
            reversal.update(series=series, row=row, gain=gain)
            if "centers_MHz" in reversal:
                fig, axis = plt.subplots(figsize=(10, 6))
                locations = [0, reversal["interpolation_weight"], 1]
                axis.errorbar(locations, reversal["centers_MHz"],
                              yerr=reversal["center_errors_MHz"], fmt="o", capsize=4)
                axis.plot([0, 1], np.array(reversal["centers_MHz"])[[0, 2]], "--", color="gray", label="A-visit linear drift baseline")
                axis.plot(locations[1], reversal["interpolated_A_MHz"], "x", color="gray")
                axis.set_xticks(locations)
                axis.set_xticklabels(["A1\n%.6f mA" % reversal["A_bias_mA"], "B\n%.6f mA" % reversal["B_bias_mA"], "A2\n%.6f mA" % reversal["A_bias_mA"]])
                axis.ticklabel_format(axis="y", style="plain", useOffset=False)
                axis.set(ylabel="Fitted line center (MHz)", xlabel="A-B-A visits (spacing follows elapsed time)",
                    title="M%d bias reversal: B - interpolated A = %.4f +/- %.4f MHz" % (row, reversal["B_minus_interpolated_A_MHz"], reversal["contrast_standard_error_MHz"]))
                axis.legend(); fig.tight_layout()
                path = output / ("%s_M%d_bias_reversal.png" % (series, row))
                fig.savefig(path, dpi=180); plt.close(fig)
                reversal["plot"] = str(path)
            result["bias_reversal_checks"].append(reversal)
        if len(set(p["current_mA"] for p in subset)) < 2:
            continue
        if not map_windows_do_not_overlap(subset):
            result["map_omissions"].append(dict(series=series, row=row, gain=gain,
                reason="Repeated visits are retained as individual traces/control plots; a map would overlap them"))
            continue
        # Plot each saved sample with its real bias/frequency coordinate; this
        # also handles deliberately shifted locator windows without stitching.
        fig, axis = plt.subplots(figsize=(10, 6))
        frequencies, biases, responses = [], [], []
        for point in subset:
            original = next(r for m in manifests for r in m["records"] if r.get("traces") == point["traces"])
            ff, zz, _ = read_record(original)
            frequencies.append(ff); biases.append(point["current_mA"])
            responses.append(np.linalg.norm(zz, axis=1))
        limit = float(np.percentile(np.concatenate(responses), 99.5))
        unique_bias = np.unique(biases)
        edges = np.r_[unique_bias[0] - (unique_bias[1] - unique_bias[0]) / 2,
                      (unique_bias[1:] + unique_bias[:-1]) / 2,
                      unique_bias[-1] + (unique_bias[-1] - unique_bias[-2]) / 2]
        for bias, ff, response in zip(biases, frequencies, responses):
            position = int(np.searchsorted(unique_bias, bias))
            f_edges = np.r_[ff[0] - (ff[1] - ff[0]) / 2,
                            (ff[1:] + ff[:-1]) / 2,
                            ff[-1] + (ff[-1] - ff[-2]) / 2]
            dots = axis.pcolormesh(edges[position:position + 2], f_edges,
                    response[:, None], cmap="viridis", vmin=0, vmax=limit,
                    shading="flat", rasterized=True)
        axis.set(xlabel="Yoko4 source setting (mA)", ylabel="Spectroscopy frequency (MHz)",
                 title="M%d readout; shared drive gain %g" % (row, gain))
        if len(unique_bias) <= 9:
            axis.set_xticks(unique_bias)
            axis.set_xticklabels(["%.6f" % bias for bias in unique_bias],
                                 rotation=30 if len(unique_bias) > 4 else 0)
        fig.colorbar(dots, ax=axis, label="Magnitude of drive-on minus drive-off IQ (ADC units)")
        fig.tight_layout()
        path = output / ("%s_M%d_g%g_map.png" % (series, row, gain))
        fig.savefig(path, dpi=180); plt.close(fig)
        result["maps"].append(str(path))
    for source in sorted(set(p["source_manifest"] for p in points)):
        series = Path(source).parents[2].name
        for fixed_row in sorted(set(p["row"] for p in points if p["source_manifest"] == source and p["row"] != 6)):
            for gain in sorted(set(p["gain"] for p in points if p["source_manifest"] == source and p["row"] == fixed_row)):
                subset = [p for p in points if p["source_manifest"] == source
                          and p["row"] in (6, fixed_row) and p["gain"] == gain]
                if set(p["row"] for p in subset) != {6, fixed_row}:
                    continue
                paired, review = pair_readouts(subset, fixed_row)
                key = "%s_M6_M%d_gain%g" % (series, fixed_row, gain)
                fit = coupled_fit(paired)
                fit["readout_note"] = "Sequential readouts at a common shared-drive gain/pulse. Missing/overlapping lines are not a doublet. Fit errors exclude temporal shifts between readouts."
                result["coupled_fits"][key] = fit
                result["paired_readout_reviews"].append(dict(series=key, accepted_pairs=paired, review=review))
    for row, bias in sorted(set((p["row"], p["current_mA"]) for p in points)):
        subset = [p for p in points if p["row"] == row and p["current_mA"] == bias]
        if len(set(p["gain"] for p in subset)) < 2:
            continue
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
        for point in sorted(subset, key=lambda p: p["gain"]):
            original = next(r for m in manifests for r in m["records"] if r.get("traces") == point["traces"])
            ff, zz, _ = read_record(original)
            label = "gain %g; %d averages/state" % (point["gain"], point["requested"]["reps"])
            for index, axis in enumerate(axes):
                axis.plot(ff, zz[:, index], label=label, linewidth=1)
        axes[0].set(title="M%d readout at %.6f mA: drive-power comparison" % (row, bias),
                    ylabel="Drive on - off I (ADC units)")
        axes[1].set(ylabel="Drive on - off Q (ADC units)", xlabel="Spectroscopy frequency (MHz)")
        axes[0].legend()
        fig.tight_layout()
        path = output / ("M%d_I%.6f_power.png" % (row, bias))
        fig.savefig(path, dpi=180); plt.close(fig)
        result["power_comparisons"].append(str(path))
    if any(c["accepted"] for p in points for c in p["metrics"]["candidates"]):
        fig, axis = plt.subplots(figsize=(10, 6))
        for row, gain in sorted(set((p["row"], p["gain"]) for p in points)):
            subset = [p for p in points if p["row"] == row and p["gain"] == gain]
            current, centers, errors = [], [], []
            for point in subset:
                for peak in point["metrics"]["candidates"]:
                    if peak["accepted"]:
                        current.append(point["current_mA"])
                        centers.append(peak["center_MHz"])
                        errors.append(peak["center_error_MHz"])
            if centers:
                axis.errorbar(current, centers, yerr=errors, fmt="o", capsize=3,
                              label="M%d; gain %g" % (row, gain))
        axis.set(xlabel="Yoko4 source setting (mA)", ylabel="Fitted line center (MHz)",
                 title="Accepted individual line centers (conditional standard errors)")
        axis.legend()
        fig.tight_layout()
        path = output / "line_centers.png"
        fig.savefig(path, dpi=180); plt.close(fig)
        result["line_centers_plot"] = str(path)
    for key, fit in result["coupled_fits"].items():
        coupled_plot(key, fit, output)
    destination = output / "analysis.json"
    destination.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(destination), points=len(points), fits={k:v["status"] for k,v in result["coupled_fits"].items()})))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    analyze(args.manifest, args.output)


if __name__ == "__main__":
    main()
