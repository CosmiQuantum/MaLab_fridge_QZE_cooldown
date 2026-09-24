"""Offline descriptive signal/noise budget; never accepts or promotes a line."""

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from diagnose_pucq4_profile_sensitivity import load_blocks, select_profile
from check_pucq4_profile_pilot import fit_profile


def profile_coefficients(f, blocks, center, width):
    shape = 1 / (1 + (2 * (f - center) / width) ** 2)
    design = np.column_stack((shape, np.ones(len(f)), (f - f.mean()) / np.ptp(f)))
    weights = np.linalg.pinv(design)[0]
    return np.einsum("f,bfi->bi", weights, blocks), weights


def self_test():
    f = np.linspace(-1, 1, 101)
    shape = 1 / (1 + (2 * f / .2) ** 2)
    blocks = np.tile(shape[None, :, None] * np.array([.03, -.04]), (128, 1, 1))
    blocks += np.array([10., -20.])[None, None, :] + f[None, :, None] * 17
    got, _ = profile_coefficients(f, blocks, 0, .2)
    np.testing.assert_allclose(got, np.tile([.03, -.04], (128, 1)), atol=1e-12)
    null, _ = profile_coefficients(f, blocks - shape[None, :, None] * [.03, -.04], 0, .2)
    np.testing.assert_allclose(null, 0, atol=1e-12)


def inspect(record):
    f, delta, blocks, norms = load_blocks(record, block_count=128)
    discovery = (blocks["drive_on"][::2] - blocks["drive_off"][::2]).mean(axis=0)
    _, prediction = select_profile(f, discovery)
    # Selection and fitted shape are learned ONLY on discovery blocks.
    if not .06 <= prediction["FWHM_MHz"] <= .8:
        return dict(status="unsupported discovery width", prediction=prediction), None
    fitted = fit_profile(f, discovery, prediction)
    center, width = fitted["center_MHz"], fitted["FWHM_MHz"]
    on, weights = profile_coefficients(f, blocks["drive_on"], center, width)
    off, _ = profile_coefficients(f, blocks["drive_off"], center, width)
    direction = (on - off)[::2].mean(axis=0)
    direction /= max(np.linalg.norm(direction), 1e-15)
    on_values, off_values = on @ direction, off @ direction
    values = (on_values - off_values)[1::2]
    se = values.std(ddof=1) / np.sqrt(len(values))
    nblock = record["requested"]["reps"] // 128
    # These separate block acquisitions are not time-paired measurements.
    separate_se = np.sqrt(on_values[1::2].var(ddof=1) + off_values[1::2].var(ddof=1)) / 8
    noise_on = np.std(blocks["drive_on"][1::2], axis=0, ddof=1)
    noise_off = np.std(blocks["drive_off"][1::2], axis=0, ddof=1)
    out = dict(status="descriptive_only", accepted_scientific_line=False,
        row=record["row"], current_mA=record["current_mA"], requested=record["requested"],
        discovery_fit=fitted, amplitude_ADC=float(values.mean()), amplitude_SE_ADC=float(se),
        independent_variance_SE_ADC=float(separate_se), descriptive_t=float(values.mean() / se),
        direction_IQ=direction.tolist(), on_amplitude_ADC=float(on_values[1::2].mean()),
        off_amplitude_ADC=float(off_values[1::2].mean()),
        on_single_shot_profile_SD_ADC=float(on_values[1::2].std(ddof=1) * np.sqrt(nblock)),
        off_single_shot_profile_SD_ADC=float(off_values[1::2].std(ddof=1) * np.sqrt(nblock)),
        median_single_shot_point_SD_ADC=dict(on=(np.median(noise_on, axis=0)*np.sqrt(nblock)).tolist(),
                                           off=(np.median(noise_off, axis=0)*np.sqrt(nblock)).tolist()),
        on_off_baseline_ADC=dict(on=blocks["drive_on"].mean(axis=(0, 1)).tolist(),
                                 off=blocks["drive_off"].mean(axis=(0, 1)).tolist()),
        block_on_ADC=on_values.tolist(), block_off_ADC=off_values.tolist(),
        normalization=norms, traces=record["traces"],
        limitation="Retrospective discovery/validation; sequential ON/OFF cannot distinguish bias from time or prove causation")
    return out, (f, delta @ direction, center, width, on_values, off_values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    self_test()
    args.output.mkdir(parents=True, exist_ok=False)
    results = []
    for path in args.manifest:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for index, record in enumerate(manifest["records"]):
            if record["status"] != "complete":
                results.append(dict(manifest=str(path), index=index, status="incomplete_record_preserved"))
                continue
            out, plotdata = inspect(record)
            out.update(manifest=str(path), manifest_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), index=index)
            if plotdata is not None:
                f, y, center, width, on, off = plotdata
                fig, axes = plt.subplots(1, 2, figsize=(12, 4))
                axes[0].plot(f, y, ".-")
                axes[0].axvline(center, color="gray")
                axes[0].set(xlabel="Frequency (MHz)", ylabel="Signed delta IQ (ADC units)")
                axes[0].ticklabel_format(axis="x", useOffset=False)
                axes[1].plot(on, label="ON (later acquisition)")
                axes[1].plot(off, label="OFF (earlier acquisition)")
                axes[1].set(xlabel="Block within its own acquisition", ylabel="Profile peak amplitude (ADC units)")
                axes[1].legend()
                fig.suptitle("%s record %d: %.6f mA; descriptive only" % (path.parents[2].name, index, record["current_mA"]))
                fig.tight_layout()
                plot = args.output / ("budget_%02d.png" % len(results))
                fig.savefig(plot, dpi=150)
                plt.close(fig)
                out["plot"] = str(plot)
            results.append(out)
            print(json.dumps({k: v for k, v in out.items() if k in (
                "manifest", "index", "current_mA", "amplitude_ADC", "amplitude_SE_ADC",
                "descriptive_t", "discovery_fit", "on_single_shot_profile_SD_ADC",
                "off_single_shot_profile_SD_ADC", "direction_IQ")}), flush=True)
    output = dict(records=results, self_test="PASS", accepted_scientific_lines=0,
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (args.output / "budget.json").write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
