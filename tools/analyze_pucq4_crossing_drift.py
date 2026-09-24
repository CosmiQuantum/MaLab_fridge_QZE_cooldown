"""Inspect chronological repetition blocks in saved crossing IQ, without hardware.

The installed QICK AveragerProgramV2 uses the reps loop outermost and get_raw
returns integer accumulated IQ in loop/time order. Validate this shape and the
normalization against the saved full average before inspecting any blocks.
"""

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from analyze_pucq4_crossing import line_candidates, read_record


def inspect(record, blocks=4):
    frequencies, full_delta, saved = read_record(record)
    arrays = {}
    for kind, path in record["traces"].items():
        with h5py.File(path, "r") as handle:
            arrays[kind] = np.asarray(handle["Q%d" % record["row"]]["Raw IQ"][0])
    expected = (record["requested"]["reps"], len(frequencies), 1, 2)
    if any(array.shape != expected for array in arrays.values()):
        raise ValueError("Raw loop shape differs from the inspected QICK reps/frequency/readout/IQ order")
    off = arrays["drive_off"].mean(axis=0)[:, 0, :]
    raw_delta = arrays["drive_on"].mean(axis=0)[:, 0, :] - off
    # Recover the common integration normalization algebraically and verify
    # every point. No clock rate or replacement readout setting is assumed.
    normalization = float(np.sum(raw_delta * full_delta) / np.sum(full_delta ** 2))
    if normalization <= 0 or not np.isfinite(normalization):
        raise ValueError("Invalid IQ normalization")
    np.testing.assert_allclose(raw_delta / normalization, full_delta, rtol=1e-8, atol=1e-9)
    results = []
    for index, indices in enumerate(np.array_split(np.arange(expected[0]), blocks)):
        block = arrays["drive_on"][indices].mean(axis=0)[:, 0, :]
        metrics, _ = line_candidates(frequencies, (block - off) / normalization)
        results.append(dict(block=index + 1, first_rep=int(indices[0]), last_rep=int(indices[-1]),
                            repetitions=len(indices), metrics=metrics))
    return dict(row=record["row"], current_mA=record["current_mA"], gain=record["requested"]["gain"],
                normalization=normalization, raw_shape=list(expected), blocks=results,
                acquired_unix=saved["drive_on"]["acquired_unix"], traces=record["traces"],
                note="Each on block uses the full saved off reference. Conditional block errors share reference noise; no independent drift significance or coupling is inferred.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path, nargs="+")
    parser.add_argument("--row", type=int, required=True)
    parser.add_argument("--gain", type=float, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    source = Path(__file__).read_bytes()
    (args.output / Path(__file__).name).write_bytes(source)
    fit_source_path = Path(__file__).with_name("analyze_pucq4_crossing.py")
    fit_source = fit_source_path.read_bytes()
    (args.output / fit_source_path.name).write_bytes(fit_source)
    records = []
    for path in args.manifest:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        for record in manifest["records"]:
            if (record["status"] == "complete" and record["row"] == args.row
                    and record["requested"]["gain"] == args.gain):
                result = inspect(record)
                result["source_manifest"] = str(path)
                records.append(result)
    for index, record in enumerate(records):
        fig, axis = plt.subplots(figsize=(9, 3))
        for block in record["blocks"]:
            for peak in block["metrics"]["candidates"]:
                if peak["accepted"]:
                    axis.errorbar(block["block"], peak["center_MHz"], yerr=peak["center_error_MHz"], fmt="o", capsize=4, color="C0")
        axis.set(title="M%d | %.6f mA | trace completed %s local" % (record["row"], record["current_mA"], datetime.fromtimestamp(record["acquired_unix"]).strftime("%H:%M:%S")),
                 ylabel="Line center (MHz)", xlabel="Chronological quarter of drive-on repetitions", xticks=[1, 2, 3, 4])
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
        fig.tight_layout()
        plot = args.output / ("blocks_%02d.png" % (index + 1))
        fig.savefig(plot, dpi=180); plt.close(fig)
        record["plot"] = str(plot)
    fig, axes = plt.subplots(max(1, len(records)), 1, figsize=(9, max(3, len(records) * 2.5)), squeeze=False)
    for axis, record in zip(axes[:, 0], records):
        for block in record["blocks"]:
            peaks = [p for p in block["metrics"]["candidates"] if p["accepted"]]
            for peak in peaks:
                axis.errorbar(block["block"], peak["center_MHz"], yerr=peak["center_error_MHz"], fmt="o", capsize=4, color="C0")
        axis.set(title="%s | M%d | %.6f mA | gain %g" % (Path(record["source_manifest"]).parents[2].name,
                 record["row"], record["current_mA"], record["gain"]), ylabel="Line center (MHz)", xticks=[1, 2, 3, 4])
        axis.ticklabel_format(axis="y", style="plain", useOffset=False)
    axes[-1, 0].set_xlabel("Chronological quarter of drive-on repetitions (accepted fits only)")
    fig.tight_layout()
    plot = args.output / "chronological_blocks.png"
    fig.savefig(plot, dpi=180); plt.close(fig)
    output = args.output / "block_analysis.json"
    output.write_text(json.dumps(dict(records=records, plot=str(plot),
        code_sha256=hashlib.sha256(source).hexdigest(), fit_code_sha256=hashlib.sha256(fit_source).hexdigest()), indent=2), encoding="utf-8")
    print(json.dumps(dict(output=str(output), traces=len(records),
        accepted_blocks=sum(any(p["accepted"] for p in b["metrics"]["candidates"])
                            for r in records for b in r["blocks"]))))


if __name__ == "__main__":
    main()
