"""Audit final PUCQ4 flux-campaign qspec traces from canonical HDF5 files.

This utility is hardware-free.  It follows each manifest point's selected
post-readout qspec log to the corresponding round-robin HDF5 file, then reports
the shared-I/Q fit and absolute SSF-population peak without normalizing rows.
"""

import argparse
import json
from pathlib import Path

import h5py

from analyze_qspec_iq_series import (
    decode_numeric,
    gain_from_h5,
    joint_fit,
    population_metrics,
)


DEFAULT_ROOT = Path(
    r"M:\_Data\20250822 - Olivia\pucq4_run_started_Aug_3\PUCQ4\pucq4_first_light"
)


def selected_qspec_log(point):
    logs = point.get("logs", {})
    return (
        logs.get("post_readout_qspec")
        or logs.get("post_readout_calibration")
        or logs.get("calibration")
    )


def newest_h5(root, log_path):
    run_dir = root / f"flux_{Path(log_path).stem}"
    files = sorted(
        run_dir.glob("*/study_data/Data_h5/qspec_ge/*.h5"),
        key=lambda path: path.stat().st_mtime,
    )
    if not files:
        raise FileNotFoundError(f"No qspec HDF5 under {run_dir}")
    return files[-1]


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", choices=("q4", "q6"), default="q4")
    parser.add_argument("--half-window-mhz", type=float, default=0.8)
    return parser.parse_args()


def main():
    args = parse_args()
    manifest_path = args.root / "yoko_flux_campaign" / "flux_campaign_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    points = manifest["devices"][args.device].get("points", {})
    group_name = "Q5" if args.device == "q4" else "Q6"
    print(
        "current_mA\tstatus\tgain\tseed_MHz\tjoint_center_MHz\tjoint_FWHM_MHz"
        "\tjoint_SNR\tpop_baseline\tpop_peak\tpop_excursion\tpop_SNR\th5"
    )
    for key, point in sorted(points.items(), key=lambda item: float(item[0])):
        status = point.get("status", "missing")
        log_path = selected_qspec_log(point)
        if status != "complete" or not log_path:
            print(f"{key}\t{status}\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA\tNA")
            continue
        path = newest_h5(args.root, log_path)
        with h5py.File(path, "r") as handle:
            group = handle[group_name]
            frequency = decode_numeric(group["Frequencies"])
            i_data = decode_numeric(group["I"])
            q_data = decode_numeric(group["Q"])
            length = min(frequency.size, i_data.size, q_data.size)
            frequency = frequency[:length]
            i_data = i_data[:length]
            q_data = q_data[:length]
            seed = float(point["qfreq_MHz"])
            fit = joint_fit(
                frequency, i_data, q_data, seed, args.half_window_mhz,
            )
            gain = gain_from_h5(group)
            metrics = None
            if "Population" in group:
                population = decode_numeric(group["Population"])
                length = min(length, population.size)
                metrics = population_metrics(
                    frequency[:length], population[:length],
                    seed, args.half_window_mhz,
                )
        if metrics is None:
            population_fields = "NA\tNA\tNA\tNA"
        else:
            population_fields = (
                f"{metrics['baseline']:.6f}\t{metrics['peak']:.6f}\t"
                f"{metrics['excursion']:.6f}\t{metrics['snr']:.3f}"
            )
        print(
            f"{key}\t{status}\t{gain:.9g}\t{seed:.9f}\t"
            f"{fit['center_MHz']:.9f}\t{fit['fwhm_MHz']:.9f}\t"
            f"{fit['vector_snr']:.3f}\t{population_fields}\t{path}"
        )


if __name__ == "__main__":
    main()
