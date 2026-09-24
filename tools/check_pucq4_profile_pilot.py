"""Prospective fixed-profile pilot assessment, independent of the old detector.

Read only saved canonical records. Require every planned repeat and control;
never pool failed visits or change a prediction after seeing pilot data.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.stats import t as student_t

from diagnose_pucq4_profile_sensitivity import load_blocks, summary
from analyze_pucq4_crossing import line_candidates


def frozen_weight(f, prediction):
    center, width = prediction["center_MHz"], prediction["FWHM_MHz"]
    shape = 1 / (1 + (2 * (f - center) / width) ** 2)
    baseline = np.column_stack((np.ones(len(f)), (f - np.mean(f)) / np.ptp(f)))
    shape -= baseline @ np.linalg.lstsq(baseline, shape, rcond=None)[0]
    shape /= np.linalg.norm(shape)
    return shape[:, None] * (np.asarray(prediction["signed_whitened_IQ_direction"])
                            / prediction["noise_IQ"])[None, :]


def assess_blocks(f, on, off, prediction, family=4):
    weight = frozen_weight(f, prediction)
    delta = on - off
    values = np.einsum("bfi,fi->b", delta, weight)
    references = np.einsum("bfi,fi->b", off, weight)
    rng = np.random.default_rng(15092026)
    signal = summary(values, rng)
    reference = summary(references, rng)
    p_adjusted = min(1.0, family * signal["conditional_sign_p_one_sided"])
    ref_p = float(2 * student_t.sf(abs(reference["t"]), len(references) - 1))
    passed = bool(signal["t"] >= 5 and p_adjusted < .01 and ref_p > .01
                  and not signal["dependence_flag"] and not reference["dependence_flag"]
                  and signal["first_half_mean"] > 0 and signal["second_half_mean"] > 0)
    return dict(signal=signal, reference=reference, reference_p_two_sided=ref_p,
                family_adjusted_p=p_adjusted, fixed_profile_controls_pass=passed)


def fit_profile(f, iq, prediction):
    """Conditional parameter estimation after the fixed-profile test, not detection."""
    center = prediction["center_MHz"]
    x = (f - center) / np.ptp(f)
    diff = np.diff(iq, axis=0)
    noise = np.maximum(np.median(np.abs(diff - np.median(diff, axis=0)), axis=0)
                       / (.67448975 * np.sqrt(2)), 1e-12)
    def residual(theta):
        shape = 1 / (1 + (2 * (f - theta[0]) / theta[1]) ** 2)
        design = np.column_stack((shape, np.ones(len(f)), x))
        y = iq / noise
        return (design @ np.linalg.lstsq(design, y, rcond=None)[0] - y).ravel()
    fit = least_squares(residual, [center, prediction["FWHM_MHz"]],
                        bounds=([center - .2, .06], [center + .2, .8]))
    rms = float(np.sqrt(np.mean(fit.fun ** 2)))
    cov = np.linalg.pinv(fit.jac.T @ fit.jac) * np.sum(fit.fun ** 2) / (fit.fun.size - 8)
    err = np.sqrt(np.diag(cov))
    passed = bool(fit.success and not np.any(fit.active_mask)
                  and np.linalg.matrix_rank(fit.jac) == 2 and rms < 2
                  and err[0] < fit.x[1] / 3 and err[1] < fit.x[1] / 2
                  and fit.x[1] >= 3 * np.median(np.diff(f)))
    return dict(center_MHz=float(fit.x[0]), FWHM_MHz=float(fit.x[1]),
                center_error_MHz=float(err[0]), FWHM_error_MHz=float(err[1]),
                residual_rms_noise_units=rms, parameter_quality_pass=passed,
                uncertainty="Conditional local-fit errors; no independent drift model")


def self_test():
    rng = np.random.default_rng(9122026)
    f = np.linspace(5430.7, 5432.7, 101)
    prediction = dict(center_MHz=5431.7, FWHM_MHz=.2, noise_IQ=[1, 1],
                      signed_whitened_IQ_direction=[1, 0])
    on, off = rng.normal(size=(2, 64, 101, 2))
    shape = 1 / (1 + (2 * (f - 5431.7) / .2) ** 2)
    null = assess_blocks(f, on, off, prediction)
    signal_on = on + shape[None, :, None] * np.array([1.0, 0.0])
    signal = assess_blocks(f, signal_on, off, prediction)
    wrong = dict(prediction, signed_whitened_IQ_direction=[-1, 0])
    reversed_phase = assess_blocks(f, signal_on, off, wrong)
    contaminated_off = off + shape[None, :, None] * np.array([1.0, 0.0])
    contaminated = assess_blocks(f, signal_on + shape[None, :, None] * np.array([1.0, 0.0]),
                                 contaminated_off, prediction)
    fitted = fit_profile(f, (signal_on - off).mean(axis=0), prediction)
    assert not null["fixed_profile_controls_pass"]
    assert signal["fixed_profile_controls_pass"]
    assert not reversed_phase["fixed_profile_controls_pass"]
    assert not contaminated["fixed_profile_controls_pass"]
    assert fitted["parameter_quality_pass"] and abs(fitted["center_MHz"] - 5431.7) < .03
    print("PASS: frozen positive/null/phase/reference controls and planted center recovery")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--spec", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    spec = json.loads(args.spec.read_text(encoding="utf-8"))
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    records = manifest["records"]
    if manifest["status"] != "complete" or len(records) != 4:
        raise ValueError("Require the complete four-window pilot, never a favorable subset")
    results = []
    for index, record in enumerate(records):
        branch = ("lower", "upper")[index % 2]
        prediction = spec["predictions"][branch]
        if record["status"] != "complete" or record["row"] != 2:
            raise ValueError("Unexpected row or incomplete window")
        requested = record["requested"]
        for key, value in spec["traces"][branch].items():
            if requested.get(key) != value:
                raise ValueError("Pilot differs from frozen trace: " + key)
        if abs(record["current_mA"] - spec["current_mA"]) > 1e-4:
            raise ValueError("Wrong pilot current")
        f, delta, blocks, norm = load_blocks(record)
        result = assess_blocks(f, blocks["drive_on"], blocks["drive_off"], prediction)
        result.update(branch=branch, repeat=index // 2 + 1, traces=record["traces"],
                      normalization=norm, fit=fit_profile(f, delta, prediction),
                      original_metrics=line_candidates(f, delta)[0])
        result["window_pass"] = bool(result["fixed_profile_controls_pass"]
                                      and result["fit"]["parameter_quality_pass"])
        results.append(result)
    repeat_checks = []
    for branch, left, right in (("lower", results[0], results[2]), ("upper", results[1], results[3])):
        difference = abs(left["fit"]["center_MHz"] - right["fit"]["center_MHz"])
        limit = max(.05, 3 * np.hypot(left["fit"]["center_error_MHz"], right["fit"]["center_error_MHz"]))
        repeat_checks.append(dict(branch=branch, difference_MHz=difference, limit_MHz=float(limit),
                                  passed=bool(difference <= limit)))
    passed = bool(all(item["window_pass"] for item in results) and all(item["passed"] for item in repeat_checks))
    output = dict(pilot_pass=passed, records=results, repeat_checks=repeat_checks,
                  spec_sha256=hashlib.sha256(args.spec.read_bytes()).hexdigest(),
                  analyzer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  source_manifest=str(args.manifest),
                  limitation="Fixed-profile prospective test; no physical identity or coupling assignment")
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(output, handle, indent=2)
    print(json.dumps(dict(pilot_pass=passed, output=str(args.output),
                         windows=[dict(branch=r["branch"], repeat=r["repeat"], t=r["signal"]["t"],
                                       passed=r["window_pass"]) for r in results])))


if __name__ == "__main__":
    main()
