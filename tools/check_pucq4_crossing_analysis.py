"""Synthetic scientific checks for complex line extraction and coupling recovery."""

import copy
import numpy as np
from analyze_pucq4_crossing import (line_candidates, mode_frequencies, coupled_fit,
    bias_reversal, pair_readouts, map_windows_do_not_overlap, coalesce_tagged_windows,
    recorded_current_mA)


def main():
    raw = {kind: dict(config=dict(actual_yoko4_current_mA=-5.6566))
           for kind in ("drive_on", "drive_off")}
    assert recorded_current_mA(raw) == -5.6566
    assert recorded_current_mA(raw) != -5.656667
    for bad in (-5.6565, float("nan")):
        raw["drive_off"]["config"]["actual_yoko4_current_mA"] = bad
        try:
            recorded_current_mA(raw)
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched or nonfinite source settings accepted")
    rng = np.random.default_rng(876)
    wide = np.linspace(5380, 5500, 2401)
    sparse = rng.normal(0, 1, (len(wide), 2))
    sparse[::37, 1] += 5 * np.where(np.arange(len(sparse[::37])) % 2, 1, -1)
    sparse[:, 0] += 12 / (1 + (2 * (wide - 5434.64) / .18) ** 2)
    metrics, _ = line_candidates(wide, sparse)
    assert any(c["accepted"] and abs(c["center_MHz"] - 5434.64) < .03
               for c in metrics["candidates"]), metrics
    rng = np.random.default_rng(876)
    frequencies = np.linspace(5500, 5508, 1601)
    iq = rng.normal(0, 1, (len(frequencies), 2))
    for center, amplitude in ((5503.0, [80, -40]), (5505.0, [-30, 100])):
        iq += np.outer(1 / (1 + (2 * (frequencies - center) / .15) ** 2), amplitude)
    metrics, _ = line_candidates(frequencies, iq)
    peaks = [p for p in metrics["candidates"] if p["accepted"]]
    assert len(peaks) == 2, metrics
    np.testing.assert_allclose([p["center_MHz"] for p in peaks], [5503, 5505], atol=.005)
    # Realistic near-minimum separation makes independent fit windows overlap.
    frequencies = np.linspace(5593, 5596, 151)
    iq = rng.normal(0, 1, (len(frequencies), 2))
    for center, amplitude in ((5594.30, [18, 12]), (5594.66, [15, 8])):
        iq += np.outer(1 / (1 + (2 * (frequencies - center) / .12) ** 2), amplitude)
    metrics, _ = line_candidates(frequencies, iq)
    peaks = [p for p in metrics["candidates"] if p["accepted"]]
    assert len(peaks) == 2, metrics
    np.testing.assert_allclose([p["center_MHz"] for p in peaks], [5594.30, 5594.66], atol=.02)
    assert all(p["fit_method"] == "joint two-line complex-IQ" for p in peaks), peaks
    iq = rng.normal(0, 1, (len(frequencies), 2))
    iq += np.outer(1 / (1 + (2 * (frequencies - 5594.5) / .35) ** 2), [25, 15])
    metrics, _ = line_candidates(frequencies, iq)
    assert len([p for p in metrics["candidates"] if p["accepted"]]) == 1, metrics
    iq = rng.normal(0, 1, (len(frequencies), 2))
    iq += np.outer(1 / (1 + (2 * (frequencies - 5594.5) / .83) ** 2), [20, 15])
    metrics, _ = line_candidates(frequencies, iq)
    peaks = [p for p in metrics["candidates"] if p["accepted"]]
    assert len(peaks) == 1 and abs(peaks[0]["FWHM_MHz"] - .83) < .15, metrics
    assert coupled_fit([])["status"] == "unresolved"
    truth = [5550, 250, 5550, .8]
    points = []
    for bias in np.linspace(-.03, .03, 19):
        pair = mode_frequencies(truth, np.array([bias]))[0] + rng.normal(0, .002, 2)
        points.append(dict(current_mA=float(-5.1 + bias), metrics=dict(candidates=[
            dict(center_MHz=float(f), center_error_MHz=.003, FWHM_MHz=.15, accepted=True) for f in pair])))
    result = coupled_fit(points)
    assert result["status"] == "resolved", result
    assert abs(result["J_Hz"] / 1e6 - .8) < .02
    assert abs(result["resonance_bias_mA"] + 5.1) < .0001
    assert result["minimum_splitting_standard_error_Hz"] > 0
    # The same known branches may be visible one per readout. Retain their
    # distinct frequencies at common power, but never count a coincident line
    # twice or mix repeated visits / different drive settings.
    readouts = []
    for index, point in enumerate(points):
        for branch, row in enumerate((6, 3)):
            readouts.append(dict(row=row, gain=.006, current_mA=point["current_mA"],
                acquired_unix=1000 + index * 120 + branch * 30, traces={},
                requested=dict(length_us=20, start_MHz=5540, stop_MHz=5560),
                metrics=dict(candidates=[dict(point["metrics"]["candidates"][branch], snr=12)])))
    paired, review = pair_readouts(readouts, 3)
    assert len(paired) == 19 and coupled_fit(paired)["status"] == "resolved", review
    # Explicit disjoint windows retain identical coverage on both readouts.
    # Known branches remain recoverable without inventing samples in the gap.
    segmented = []
    for offset in range(0, len(readouts), 2):
        records = readouts[offset:offset + 2]
        centers = [r["metrics"]["candidates"][0]["center_MHz"] for r in records]
        for row_index, record in enumerate(records):
            for window_index, center in enumerate(centers):
                part = copy.deepcopy(record)
                part["source_manifest"] = "synthetic/common/manifest.json"
                part["source_record_index"] = len(segmented)
                part["requested"].update(start_MHz=center - .4, stop_MHz=center + .4,
                                          paired_windows_id="visit_%d" % offset)
                if row_index != window_index:
                    part["metrics"]["candidates"] = []
                part["acquired_unix"] += window_index * 5
                segmented.append(part)
    snapshot = copy.deepcopy(segmented)
    paired_windows, review = pair_readouts(segmented, 3)
    assert len(paired_windows) == 19, review
    assert coupled_fit(paired_windows)["status"] == "resolved"
    assert abs(coupled_fit(paired_windows)["J_Hz"] - result["J_Hz"]) < 1e-5
    assert segmented == snapshot, "Window grouping must not mutate original evidence"
    assert all(len(p["frequency_coverage_MHz"][0]) == 2 for p in paired_windows)
    only_fixed = copy.deepcopy([p for p in segmented if p["row"] == 3])
    for index in range(0, len(only_fixed), 2):
        only_fixed[index]["metrics"] = copy.deepcopy(segmented[2 * index]["metrics"])
    only_fixed_before = copy.deepcopy(only_fixed)
    self_spectra, self_review = coalesce_tagged_windows(only_fixed)
    assert len(self_spectra) == 19 and not any(r["reason"] for r in self_review)
    assert coupled_fit(self_spectra)["status"] == "resolved"
    assert abs(coupled_fit(self_spectra)["J_Hz"] - result["J_Hz"]) < 1e-5
    assert only_fixed == only_fixed_before
    # A strong control and weak branch can use different averaging while
    # retaining matched power/timing and each window's exact provenance.
    mixed_averaging = copy.deepcopy(only_fixed[:2])
    for point, repetitions in zip(mixed_averaging, (16000, 32000)):
        point["row"] = 6
        point["requested"]["reps"] = repetitions
        point["requested_current_mA"] = point["current_mA"] + .000067
    mixed_pairs, mixed_review = coalesce_tagged_windows(mixed_averaging)
    assert len(mixed_pairs) == 1 and not mixed_review[0]["reason"]
    assert not mixed_pairs[0]["window_pair_review_required"]
    assert [r["reps"] for r in mixed_pairs[0]["window_requests"]] == [16000, 32000]
    assert all(r["requested_current_mA"] == mixed_averaging[0]["requested_current_mA"]
               for r in mixed_pairs[0]["window_requests"])
    mixed_pairs[0]["window_requests"][0]["reps"] = 1
    assert mixed_averaging[0]["requested"]["reps"] == 16000
    assert len(coalesce_tagged_windows(only_fixed[1:])[0]) == 18
    assert len(coalesce_tagged_windows(only_fixed + [copy.deepcopy(only_fixed[0])])[0]) == 18
    invalid_self = copy.deepcopy(only_fixed)
    invalid_self[0]["metrics"]["candidates"] = [dict(
        invalid_self[1]["metrics"]["candidates"][0],
        center_MHz=(invalid_self[0]["requested"]["start_MHz"] + invalid_self[0]["requested"]["stop_MHz"]) / 2,
        FWHM_MHz=100)]
    reviewed, _ = coalesce_tagged_windows(invalid_self)
    assert reviewed[0]["window_pair_review_required"], "Wide overlapping lines are not a resolved pair"
    conflicting_self = copy.deepcopy(segmented)
    conflicting_self[1]["metrics"] = copy.deepcopy(conflicting_self[3]["metrics"])
    conflicting_self[1]["metrics"]["candidates"][0]["center_MHz"] += .2
    conflicting_self[2]["metrics"] = copy.deepcopy(conflicting_self[0]["metrics"])
    reviewed, _ = coalesce_tagged_windows(conflicting_self)
    assert all(p["window_pair_review_required"] for p in reviewed[:2]), "Retain cross-readout conflict"
    row_windows = [p for p in segmented if p["row"] == 6]
    assert map_windows_do_not_overlap(row_windows)
    assert not map_windows_do_not_overlap(row_windows + [copy.deepcopy(row_windows[0])])
    untagged = copy.deepcopy(row_windows)
    del untagged[0]["requested"]["paired_windows_id"]
    assert not map_windows_do_not_overlap(untagged)
    assert map_windows_do_not_overlap([p for p in readouts if p["row"] == 6])
    for field, value in (("gain", .003), ("source_manifest", "different-run"),
                         ("control_review_required", True)):
        invalid = copy.deepcopy(segmented)
        invalid[0][field] = value
        assert len(pair_readouts(invalid, 3)[0]) == 18, field
    for field, value in (("paired_windows_id", "other-visit"),
                         ("stop_MHz", segmented[0]["requested"]["stop_MHz"] + .01),
                         ("length_us", 30)):
        invalid = copy.deepcopy(segmented)
        invalid[0]["requested"][field] = value
        assert len(pair_readouts(invalid, 3)[0]) == 18, field
    assert len(pair_readouts(segmented[1:], 3)[0]) == 18
    assert len(pair_readouts(segmented + [copy.deepcopy(segmented[0])], 3)[0]) == 18
    invalid = copy.deepcopy(segmented)
    del invalid[0]["requested"]["paired_windows_id"]
    assert len(pair_readouts(invalid, 3)[0]) == 18
    invalid = copy.deepcopy(segmented)
    invalid[0]["metrics"]["candidates"][0]["center_MHz"] = (
        segmented[0]["requested"]["stop_MHz"] + segmented[1]["requested"]["start_MHz"]) / 2
    assert len(pair_readouts(invalid, 3)[0]) == 18
    invalid = copy.deepcopy(segmented)
    invalid[1]["requested"]["start_MHz"] = invalid[0]["requested"]["stop_MHz"] - .1
    assert len(pair_readouts(invalid, 3)[0]) == 18
    invalid = copy.deepcopy(segmented)
    # A second readout must not hide an additional, unmatched accepted line.
    invalid[1]["metrics"]["candidates"] = copy.deepcopy(invalid[3]["metrics"]["candidates"])
    invalid[2]["metrics"]["candidates"] = copy.deepcopy(invalid[0]["metrics"]["candidates"])
    extra = copy.deepcopy(invalid[2]["metrics"]["candidates"][0])
    extra["center_MHz"] += .3
    invalid[2]["metrics"]["candidates"].append(extra)
    assert len(pair_readouts(invalid, 3)[0]) == 18
    invalid = copy.deepcopy(segmented)
    invalid[0]["actual_frequency_window_MHz"] = [
        invalid[0]["requested"]["start_MHz"] + .01, invalid[0]["requested"]["stop_MHz"]]
    assert len(pair_readouts(invalid, 3)[0]) == 18, "Actual coverage overrides requested bounds"
    invalid = copy.deepcopy(segmented)
    invalid[0]["source_record_index"] = 10000
    assert len(pair_readouts(invalid, 3)[0]) == 18, "Do not merge a return visit"
    mixed = copy.deepcopy(readouts)
    mixed[1]["gain"] = .003
    assert len(pair_readouts(mixed, 3)[0]) == 18
    assert len(pair_readouts(readouts + [copy.deepcopy(readouts[0])], 3)[0]) == 18
    coincident = copy.deepcopy(readouts)
    for index in range(0, len(coincident), 2):
        coincident[index + 1]["metrics"] = copy.deepcopy(coincident[index]["metrics"])
    assert not pair_readouts(coincident, 3)[0]
    doublets = copy.deepcopy(readouts[:2])
    two_lines = [copy.deepcopy(r["metrics"]["candidates"][0]) for r in doublets]
    for record in doublets:
        record["metrics"]["candidates"] = copy.deepcopy(two_lines)
    assert len(pair_readouts(doublets, 3)[0]) == 1
    for candidate in doublets[1]["metrics"]["candidates"]:
        candidate["center_MHz"] += .2
    assert not pair_readouts(doublets, 3)[0]
    for point in points:
        point["control_review_required"] = True
    assert coupled_fit(points)["status"] == "unresolved"
    for point in points:
        del point["control_review_required"]
    # A linear fixed-qubit drift should be flagged rather than silently folded
    # into a constant-fixed-frequency coupling estimate.
    for bias, point in zip(np.linspace(-.03, .03, 19), points):
        pair = mode_frequencies(truth + [20], np.array([bias]), True)[0]
        for candidate, value in zip(point["metrics"]["candidates"], pair):
            candidate["center_MHz"] = float(value)
    assert coupled_fit(points)["status"] == "model_review_required"
    for bias, point in zip(np.linspace(-.03, .03, 19), points):
        pair = mode_frequencies([5550, 250, 5550, 0], np.array([bias]))[0]
        for candidate, value in zip(point["metrics"]["candidates"], pair):
            candidate["center_MHz"] = float(value)
    assert coupled_fit(points)["status"] != "resolved"
    control = [dict(current_mA=bias, acquired_unix=t, metrics=dict(candidates=[dict(
        accepted=True, center_MHz=center, center_error_MHz=.005)]))
        for bias, t, center in ((-5, 0, 5500), (-4.99, 1, 5500.103), (-5, 2, 5500.006))]
    review = bias_reversal(control)
    assert review["status"] == "bias_response_supported", review
    assert abs(review["B_minus_interpolated_A_MHz"] - .1) < 1e-8
    control[1]["metrics"]["candidates"][0]["center_MHz"] = 5500.003
    assert bias_reversal(control)["status"] == "bias_response_not_resolved"
    control[2]["metrics"]["candidates"][0]["center_MHz"] += .2
    assert bias_reversal(control)["status"] == "temporal_instability"
    print("PASS: signed complex peaks, known-J recovery, continuous/disjoint-window pairing, coverage/repeat/source/power/reference/drift rejection and conditional uncertainty")


if __name__ == "__main__":
    main()
