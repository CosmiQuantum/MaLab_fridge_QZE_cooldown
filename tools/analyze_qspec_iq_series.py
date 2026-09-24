"""Joint-I/Q diagnostics for persisted qubit-spectroscopy gain sweeps.

This utility is hardware-free.  It fits one Lorentzian center and width shared
by I and Q, while allowing each quadrature its own amplitude and linear
baseline.  Sharing the line shape prevents a one-bin excursion in one
quadrature from silently replacing a seeded response visible in both channels.
"""

import argparse
import glob
import re
from pathlib import Path

import h5py
import numpy as np
from scipy.optimize import least_squares


def decode_numeric(dataset):
    value = dataset[0] if dataset.shape else dataset[()]
    if isinstance(value, np.ndarray):
        return np.asarray(value, dtype=float).ravel()
    if isinstance(value, (bytes, np.bytes_)):
        text = value.decode("utf-8")
    else:
        text = str(value)
    fields = [field for field in re.split(r"[,\s\[\]]+", text.strip()) if field]
    return np.asarray([float(field) for field in fields], dtype=float)


def decode_text(dataset):
    value = dataset[0] if dataset.shape else dataset[()]
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return str(value)


def robust_noise(values):
    differences = np.diff(np.asarray(values, dtype=float))
    median = np.median(differences)
    mad = np.median(np.abs(differences - median))
    estimate = mad / (0.67448975 * np.sqrt(2.0))
    return max(float(estimate), np.finfo(float).eps)


def population_metrics(frequency, population, seed_mhz, half_window_mhz):
    """Measure a resolved population peak without renormalizing the trace."""
    order = np.argsort(frequency)
    frequency = np.asarray(frequency, dtype=float)[order]
    population = np.asarray(population, dtype=float)[order]
    finite = np.isfinite(frequency) & np.isfinite(population)
    frequency = frequency[finite]
    population = population[finite]
    if frequency.size < 20:
        raise ValueError("fewer than 20 finite population samples")

    frequency_step = float(np.median(np.diff(frequency)))
    smooth_points = max(3, int(round(0.025 / max(frequency_step, 1e-12))))
    if smooth_points % 2 == 0:
        smooth_points += 1
    padding = smooth_points // 2
    smoothed = np.convolve(
        np.pad(population, (padding, padding), mode="edge"),
        np.ones(smooth_points) / smooth_points,
        mode="valid",
    )

    distance = np.abs(frequency - seed_mhz)
    local = distance <= half_window_mhz
    background = (distance >= 0.7 * half_window_mhz) & (distance <= half_window_mhz)
    if local.sum() < 5 or background.sum() < 10:
        raise ValueError("insufficient local/background samples for population metrics")
    baseline = float(np.median(smoothed[background]))
    peak = float(np.max(smoothed[local]))
    mad = float(np.median(np.abs(smoothed[background] - baseline)))
    noise = max(1.4826 * mad, np.finfo(float).eps)
    return {
        "baseline": baseline,
        "peak": peak,
        "excursion": peak - baseline,
        "snr": (peak - baseline) / noise,
        "smooth_points": smooth_points,
    }


def joint_fit(frequency, i_data, q_data, seed_mhz, half_window_mhz):
    keep = np.abs(frequency - seed_mhz) <= half_window_mhz
    x = np.asarray(frequency[keep], dtype=float)
    i_values = np.asarray(i_data[keep], dtype=float)
    q_values = np.asarray(q_data[keep], dtype=float)
    if x.size < 20:
        raise ValueError("fewer than 20 samples in the requested fit window")

    x_rel = x - seed_mhz
    i_noise = robust_noise(i_values)
    q_noise = robust_noise(q_values)
    edge_count = max(5, min(30, x.size // 8))
    edge = np.r_[0:edge_count, x.size - edge_count:x.size]
    i_slope, i_base = np.polyfit(x_rel[edge], i_values[edge], 1)
    q_slope, q_base = np.polyfit(x_rel[edge], q_values[edge], 1)

    def model(parameters):
        center, half_width, i_amp, i_offset, i_gradient, q_amp, q_offset, q_gradient = parameters
        line = 1.0 / (1.0 + ((x - center) / half_width) ** 2)
        return (
            i_offset + i_gradient * x_rel + i_amp * line,
            q_offset + q_gradient * x_rel + q_amp * line,
        )

    def residual(parameters):
        i_fit, q_fit = model(parameters)
        return np.r_[(i_fit - i_values) / i_noise, (q_fit - q_values) / q_noise]

    center_bound = min(half_window_mhz, max(0.20, half_window_mhz * 0.75))
    frequency_step = float(np.median(np.diff(x)))
    lower = [
        seed_mhz - center_bound, frequency_step / 2.0,
        -np.inf, -np.inf, -np.inf, -np.inf, -np.inf, -np.inf,
    ]
    upper = [
        seed_mhz + center_bound, half_window_mhz,
        np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
    ]
    best = None
    # Local verification scans can be narrower than the historical seed list.
    # Keep the existing objective/bounds and use only feasible starting points.
    offset_seed = min(0.10, center_bound / 2.0)
    width_seeds = [width for width in (0.025, 0.05, 0.10, 0.25, 0.45)
                   if lower[1] < width < upper[1]]
    if not width_seeds:
        width_seeds = [(lower[1] + upper[1]) / 2.0]
    for center_offset in (-offset_seed, 0.0, offset_seed):
        for half_width in width_seeds:
            center = seed_mhz + center_offset
            line = 1.0 / (1.0 + ((x - center) / half_width) ** 2)
            line_centered = line - np.mean(line)
            denominator = float(line_centered @ line_centered)
            i_amp = float(line_centered @ (i_values - np.mean(i_values)) / denominator)
            q_amp = float(line_centered @ (q_values - np.mean(q_values)) / denominator)
            initial = [
                center, half_width, i_amp, i_base, i_slope,
                q_amp, q_base, q_slope,
            ]
            result = least_squares(
                residual, initial, bounds=(lower, upper), loss="soft_l1",
                f_scale=1.5, max_nfev=20000,
            )
            if best is None or result.cost < best.cost:
                best = result

    center, half_width, i_amp, _, _, q_amp, _, _ = best.x
    fitted_i, fitted_q = model(best.x)
    residual_rms = float(np.sqrt(np.mean(residual(best.x) ** 2)))
    vector_snr = float(np.hypot(i_amp / i_noise, q_amp / q_noise))
    return {
        "center_MHz": float(center),
        "fwhm_MHz": float(2.0 * half_width),
        "vector_snr": vector_snr,
        "residual_rms_sigma": residual_rms,
        "frequency": x,
        "I": i_values,
        "Q": q_values,
        "I_fit": fitted_i,
        "Q_fit": fitted_q,
    }


def gain_from_h5(group):
    text = decode_text(group["Syst Config"])
    matches = re.findall(r"['\"]qubit_gain_ge['\"]:\s*([-+0-9.eE]+)", text)
    if not matches:
        return float("nan")
    return float(matches[-1])


def resolve_paths(patterns):
    paths = []
    for pattern in patterns:
        matches = [Path(path) for path in glob.glob(pattern, recursive=True)]
        if matches:
            paths.extend(matches)
        else:
            path = Path(pattern)
            if path.exists():
                paths.append(path)
    return sorted(set(paths), key=lambda path: path.stat().st_mtime)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="HDF5 paths or quoted glob patterns")
    parser.add_argument("--group", default="Q5")
    parser.add_argument("--seed-mhz", type=float, required=True)
    parser.add_argument("--half-window-mhz", type=float, default=0.8)
    parser.add_argument(
        "--population-metrics", action="store_true",
        help="also report absolute SSF population peak, baseline, excursion, and robust SNR",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    paths = resolve_paths(args.paths)
    if not paths:
        raise FileNotFoundError("No HDF5 files matched")
    columns = "gain\tcenter_MHz\tFWHM_MHz\tvector_SNR\tresidual_sigma"
    if args.population_metrics:
        columns += "\tpop_baseline\tpop_peak\tpop_excursion\tpop_SNR\tpop_smooth_points"
    print(columns + "\tpath")
    for path in paths:
        with h5py.File(path, "r") as handle:
            group = handle[args.group]
            frequency = decode_numeric(group["Frequencies"])
            i_data = decode_numeric(group["I"])
            q_data = decode_numeric(group["Q"])
            length = min(frequency.size, i_data.size, q_data.size)
            result = joint_fit(
                frequency[:length], i_data[:length], q_data[:length],
                args.seed_mhz, args.half_window_mhz,
            )
            gain = gain_from_h5(group)
            metrics = None
            if args.population_metrics:
                if "Population" in group:
                    population = decode_numeric(group["Population"])
                    length = min(length, population.size)
                    metrics = population_metrics(
                        frequency[:length], population[:length],
                        args.seed_mhz, args.half_window_mhz,
                    )
        line = (
            f"{gain:.9g}\t{result['center_MHz']:.9f}\t{result['fwhm_MHz']:.9f}\t"
            f"{result['vector_snr']:.3f}\t{result['residual_rms_sigma']:.3f}"
        )
        if metrics is not None:
            line += (
                f"\t{metrics['baseline']:.6f}\t{metrics['peak']:.6f}"
                f"\t{metrics['excursion']:.6f}\t{metrics['snr']:.3f}"
                f"\t{metrics['smooth_points']}"
            )
        elif args.population_metrics:
            line += "\tNA\tNA\tNA\tNA\tNA"
        print(line + f"\t{path}")


if __name__ == "__main__":
    main()
