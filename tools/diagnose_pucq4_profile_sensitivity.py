"""Offline, retrospective split-block profile diagnostics; never promotes fits.

The discovery set selects a line profile and signed IQ direction. Validation uses
disjoint ON *and* OFF blocks. Conditional sign tests require symmetric independent
block errors; sequential ON/OFF acquisition can still confound a physical claim.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import t as student_t

from analyze_pucq4_crossing import read_record, line_candidates


def bank(f):
    """Unit-length Lorentzian templates orthogonal to constant/linear baseline."""
    df, span = float(np.median(np.diff(f))), float(np.ptp(f))
    baseline = np.column_stack((np.ones(len(f)), (f - np.mean(f)) / span))
    q, _ = np.linalg.qr(baseline)
    centers = f[5:-5]
    widths = np.geomspace(max(.06, 3 * df), min(1.2, span / 3), 16)
    cc, ww = np.meshgrid(centers, widths)
    shapes = 1 / (1 + (2 * (f[None, :] - cc.ravel()[:, None]) / ww.ravel()[:, None]) ** 2)
    shapes -= (shapes @ q) @ q.T
    shapes /= np.linalg.norm(shapes, axis=1)[:, None]
    return shapes, cc.ravel(), ww.ravel()


def select_profile(f, discovery, templates=None):
    shapes, centers, widths = bank(f) if templates is None else templates
    difference = np.diff(discovery, axis=0)
    noise = np.median(np.abs(difference - np.median(difference, axis=0)), axis=0) / (.67448975 * np.sqrt(2))
    noise = np.maximum(noise, 1e-12)
    coefficient = shapes @ (discovery / noise)
    norms = np.linalg.norm(coefficient, axis=1)
    index = int(np.argmax(norms))
    direction = coefficient[index] / max(norms[index], 1e-12)
    weight = shapes[index, :, None] * (direction / noise)[None, :]
    return weight, dict(center_MHz=float(centers[index]), FWHM_MHz=float(widths[index]),
                        discovery_profile_score=float(norms[index]), noise_IQ=noise.tolist(),
                        signed_whitened_IQ_direction=direction.tolist())


def summary(values, rng, permutations=32767):
    values = np.asarray(values, dtype=float)
    n = len(values)
    mean, sd = float(values.mean()), float(values.std(ddof=1))
    se = max(sd / np.sqrt(n), 1e-15)
    score = mean / se
    # For fixed observations, sign-flip t is monotonic in the signed mean.
    exceed = 0
    for start in range(0, permutations, 2048):
        count = min(2048, permutations - start)
        signs = rng.integers(0, 2, size=(count, n)) * 2 - 1
        exceed += int(np.count_nonzero(signs @ values >= values.sum()))
    lag = float(np.corrcoef(values[:-1], values[1:])[0, 1]) if sd > 0 else 0.0
    left, right = np.array_split(values, 2)
    drift_se = np.sqrt(left.var(ddof=1) / len(left) + right.var(ddof=1) / len(right))
    drift = float((right.mean() - left.mean()) / max(drift_se, 1e-15))
    return dict(mean=mean, standard_error=se, t=score,
                conditional_t_p_one_sided=float(student_t.sf(score, n - 1)),
                conditional_sign_p_one_sided=(exceed + 1) / (permutations + 1),
                lag1=lag, chronological_half_difference_SE=drift,
                dependence_flag=bool(abs(lag) > .35 or abs(drift) > 3),
                first_half_mean=float(left.mean()), second_half_mean=float(right.mean()),
                values=values.tolist())


def evaluate(f, on, off, rng, family=8, permutations=32767, templates=None):
    """Inputs are independent chronological block means, equal ON/OFF counts."""
    if on.shape != off.shape or on.ndim != 3 or on.shape[0] != 64:
        raise ValueError("Expected 64 blocks with shape (64, frequencies, 2)")
    train = (on[::2] - off[::2]).mean(axis=0)
    weight, selection = select_profile(f, train, templates)
    # Off-reference noise is separate in discovery and validation.
    amplitude = np.einsum("bfi,fi->b", on[1::2] - off[1::2], weight)
    reference = np.einsum("bfi,fi->b", off[1::2], weight)
    check = summary(amplitude, rng, permutations)
    off_check = summary(reference, rng, permutations)
    off_p = float(2 * student_t.sf(abs(off_check["t"]), len(reference) - 1))
    adjusted = min(1.0, family * check["conditional_sign_p_one_sided"])
    qualified = bool(adjusted < .01 and off_p > .01
                     and not check["dependence_flag"] and not off_check["dependence_flag"]
                     and check["first_half_mean"] > 0 and check["second_half_mean"] > 0)
    return dict(selection=selection, validation=check, off_reference=off_check,
                off_reference_p_two_sided=off_p, family_adjusted_conditional_p=adjusted,
                diagnostic_controls_pass=qualified, accepted_scientific_line=False,
                status="retrospective diagnostic only; prospective confirmation required"), weight


def load_blocks(record, block_count=64):
    f, delta, saved = read_record(record)
    reps = record["requested"]["reps"]
    if block_count < 4 or block_count % 2 or reps % block_count:
        raise ValueError("Repetition count must divide into an even number of equal blocks")
    blocks = {}
    normalizations = {}
    for kind, path in record["traces"].items():
        with h5py.File(path, "r") as handle:
            raw = handle["Q%d" % record["row"]]["Raw IQ"]
            expected = (1, reps, len(f), 1, 2)
            if raw.shape != expected:
                raise ValueError("Unexpected raw shape: %r != %r" % (raw.shape, expected))
            means = np.asarray([raw[0, i * (reps // block_count):(i + 1) * (reps // block_count), :, 0, :].mean(axis=0)
                                for i in range(block_count)])
        full = means.mean(axis=0)
        iq = saved[kind]["iq"]
        norm = float(np.sum(full * iq) / np.sum(iq ** 2))
        if not np.isfinite(norm) or norm <= 0:
            raise ValueError("Invalid normalization")
        np.testing.assert_allclose(full / norm, iq, rtol=1e-8, atol=1e-9)
        blocks[kind] = means / norm
        normalizations[kind] = norm
    np.testing.assert_allclose(blocks["drive_on"].mean(axis=0) - blocks["drive_off"].mean(axis=0), delta,
                               rtol=1e-8, atol=1e-9)
    np.testing.assert_allclose(normalizations["drive_on"], normalizations["drive_off"], rtol=1e-8)
    return f, delta, blocks, normalizations


def inspect(record, rng):
    f, delta, blocks, normalizations = load_blocks(record)
    result, weight = evaluate(f, blocks["drive_on"], blocks["drive_off"], rng)
    result.update(row=record["row"], current_mA=record["current_mA"], requested=record["requested"],
                  traces=record["traces"], normalization=normalizations, blocks_per_state=64,
                  repetitions_per_block=record["requested"]["reps"] // 64, raw_average_reconstruction="PASS",
                  original_metrics=line_candidates(f, delta)[0])
    return result, f, blocks, weight


def simulation(seed=5192026):
    """Check detection, null behavior, and explicit rejection of severe drift."""
    rng = np.random.default_rng(seed)
    f = np.linspace(0, 2.4, 121)
    templates = bank(f)
    profile = 1 / (1 + (2 * (f - 1.2) / .36) ** 2)
    counts = {}
    for scenario in ("white_null", "correlated_null", "planted_line", "drifting_line"):
        diagnostics = []
        for _ in range(80):
            on, off = rng.normal(size=(2, 64, len(f), 2))
            if scenario == "correlated_null":
                # Frequency-correlated block noise must survive projection.
                on += rng.normal(size=(64, 1, 2)) * profile[None, :, None] * 3
                off += rng.normal(size=(64, 1, 2)) * profile[None, :, None] * 3
            if scenario in ("planted_line", "drifting_line"):
                strength = np.full(64, .5) if scenario == "planted_line" else np.linspace(0, 4, 64)
                on += strength[:, None, None] * profile[None, :, None] * np.array([1, -.7])
            result, _ = evaluate(f, on, off, rng, permutations=4095, templates=templates)
            diagnostics.append(result)
        counts[scenario] = dict(trials=len(diagnostics),
            controls_pass=sum(item["diagnostic_controls_pass"] for item in diagnostics),
            dependence_flags=sum(item["validation"]["dependence_flag"] for item in diagnostics))
    passed = bool(counts["white_null"]["controls_pass"] <= 3
                  and counts["correlated_null"]["controls_pass"] <= 3
                  and counts["planted_line"]["controls_pass"] >= 50
                  and counts["drifting_line"]["controls_pass"] <= 3
                  and counts["drifting_line"]["dependence_flags"] >= 75)
    return dict(seed=seed, scenarios=counts, passed=passed,
                note="Finite implementation checks, not a calibrated tail guarantee for hardware noise")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    code = Path(__file__).read_bytes()
    (args.output / Path(__file__).name).write_bytes(code)
    if args.self_test:
        tests = simulation()
        (args.output / "validation.json").write_text(json.dumps(tests, indent=2), encoding="utf-8")
        print(json.dumps(tests), flush=True)
        if not tests["passed"]:
            raise SystemExit("Synthetic validation failed; do not inspect real data with this version")
    rng = np.random.default_rng(5192026)
    results = []
    for manifest_path in args.manifest:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        for index, record in enumerate(manifest["records"]):
            if record["status"] != "complete":
                raise ValueError("Incomplete source record")
            result, f, blocks, weight = inspect(record, rng)
            result.update(source_manifest=str(manifest_path), source_record_index=index)
            results.append(result)
            fig, axes = plt.subplots(1, 2, figsize=(12, 4))
            for group, label in ((slice(None, None, 2), "Discovery"), (slice(1, None, 2), "Validation")):
                delta = (blocks["drive_on"][group] - blocks["drive_off"][group]).mean(axis=0)
                direction = np.asarray(result["selection"]["signed_whitened_IQ_direction"])
                axes[0].plot(f, (delta / result["selection"]["noise_IQ"]) @ direction, ".-", ms=3, label=label)
            axes[0].set(xlabel="Frequency (MHz)", ylabel="Signed projection (discovery point-noise units)")
            axes[0].ticklabel_format(axis="x", useOffset=False, style="plain")
            axes[0].legend()
            axes[1].plot(np.arange(1, 33), result["validation"]["values"], "o-", label="ON - OFF")
            axes[1].axhline(0, color="gray")
            axes[1].set(xlabel="Chronological validation block", ylabel="Frozen-profile amplitude (arbitrary units)")
            fig.suptitle("M%d | %.6f mA | %s window\nRetrospective diagnostic; original acceptance unchanged" %
                         (record["row"], record["current_mA"], "lower" if index == 0 else "upper"))
            fig.tight_layout()
            plot = args.output / ("profile_%02d.png" % len(results))
            fig.savefig(plot, dpi=160)
            plt.close(fig)
            result["plot"] = str(plot)
            print(json.dumps(dict(source=manifest_path.parts[-4], row=result["row"], window=index,
                center=result["selection"]["center_MHz"], t=result["validation"]["t"],
                p_adjusted=result["family_adjusted_conditional_p"],
                off_p=result["off_reference_p_two_sided"],
                dependence=result["validation"]["dependence_flag"],
                controls_pass=result["diagnostic_controls_pass"])), flush=True)
            (args.output / "diagnostics.json").write_text(json.dumps(dict(records=results,
                code_sha256=hashlib.sha256(code).hexdigest(), accepted_scientific_lines=0,
                complete=False), indent=2), encoding="utf-8")
    if results:
        if len(results) != 8:
            raise ValueError("Predeclared real-data family requires exactly eight windows")
        (args.output / "diagnostics.json").write_text(json.dumps(dict(records=results,
            code_sha256=hashlib.sha256(code).hexdigest(), accepted_scientific_lines=0,
            complete=True, limitation="Previously inspected data; conditional noise assumptions; no prospective confirmation"),
            indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
