"""
Standalone QZE nbar/T1 analysis script for a collaborator.

This file intentionally does not import anything from the rest of the repo. It
only needs the saved HDF5 data folders and common scientific Python packages:

    pip install numpy scipy h5py matplotlib

Typical command-line use:

    python data_loading_and_analysis_scripts_for_le.py \
        --data-root "M:/_Data/20250822 - Olivia/bob_run_started_Feb_11/squill/2d_more_stats/all_qubits" \
        --qubit 5 \
        --rounds 0:47 \
        --out analysis_for_le_outputs

The qubit index is zero-based, matching the folder names in this data set:
`--qubit 5` reads folders like `qubit_5round0` and HDF5 group `Q6`.

The script returns/writes data in tidy row formats:

    qspec_points.csv:
        one row per spectroscopy sample.
        columns: round, qubit_index, gain, frequency_mhz, population, date_iso,
        source_file, dataset_index.

    t1_points.csv:
        one row per T1 sample.
        columns: round, qubit_index, gain, delay_us, population, date_iso,
        source_file, dataset_index.

    qspec_nbar_summary.csv:
        one row per spectroscopy gain after center extraction and nbar
        calibration.
        columns: round, qubit_index, gain, nbar, center_mhz, fwhm_mhz,
        chi_mhz, center_vs_gain2_slope_mhz_per_gain2, center_vs_gain2_intercept_mhz.

    t1_gamma_summary.csv:
        one row per T1 fit.
        columns: round, qubit_index, gain, nbar, T1_us, T1_err_us,
        gamma_1_per_ms, gamma_err_1_per_ms, fit_ok, nbar_source.

The importable entry point is:

    results = run_analysis(AnalysisConfig(data_root=..., qubit_index=5))

`results` is a dictionary with `qspec_points`, `t1_points`,
`qspec_summary`, `t1_summary`, `nbar_by_round`, and `plot_files`.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import math
import os
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import curve_fit, least_squares

try:
    import h5py
except ImportError:  # The script can still show --help without h5py.
    h5py = None


DEFAULT_DATA_ROOT = Path(
    "M:/_Data/20250822 - Olivia/bob_run_started_Feb_11/squill/2d_more_stats/all_qubits"
)
DEFAULT_OUT_DIR = Path("M:/_Data/20250822 - Olivia/bob_run_started_Feb_11/squill/2d_more_stats/all_qubits/analysis_for_le_outputs")
DEFAULT_QUBIT_INDEX = 5
DEFAULT_ROUNDS = "0:47"
DEFAULT_CHI_MHZ = -0.137

NUMBER_RE = re.compile(r"[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][-+]?\d+)?")
TIMESTAMP_RE = re.compile(
    r"(\d{4})[-_\.]?(\d{2})[-_\.]?(\d{2})[ Tt_-]?(\d{2})[-_\.]?(\d{2})[-_\.]?(\d{2})"
)


@dataclass
class AnalysisConfig:
    data_root: Path = DEFAULT_DATA_ROOT
    out_dir: Path = DEFAULT_OUT_DIR
    qubit_index: int = DEFAULT_QUBIT_INDEX
    rounds: str | list[int] = DEFAULT_ROUNDS
    chi_mhz: float = DEFAULT_CHI_MHZ
    qspec_subdir: str = "QSpec_zeno"
    t1_subdir: str = "T1_ge_zeno"
    make_plots: bool = True
    make_qspec_heatmaps: bool = True
    make_t1_heatmaps: bool = True
    make_per_round_summary_plots: bool = True
    min_qspec_points: int = 5
    min_t1_points: int = 4
    dpi: int = 200
    gain_match_atol: float = 1e-8
    gain_match_rtol: float = 1e-6


DATA_FORMAT_DESCRIPTION = """Data format written by data_loading_and_analysis_scripts_for_le.py

All CSV files are tidy tables: each row is one observation or one fitted result.

qspec_points.csv
  Raw/normalized qubit spectroscopy samples. `frequency_mhz` is the spectroscopy
  sweep axis. `population` is the IQ-normalized qubit population when calibration
  shots are present; otherwise it is sqrt(I^2 + Q^2).

t1_points.csv
  Raw/normalized T1 samples. `delay_us` is the T1 delay axis. `population` uses
  the same IQ normalization convention as qspec_points.csv.

qspec_nbar_summary.csv
  Spectroscopy-derived nbar calibration. For each round and gain, the script
  averages qspec_points at the same frequency, fits a Lorentzian-like peak
  center, then fits center_mhz versus gain^2 within that round. nbar is computed
  from that fitted line as (fitted_center_mhz - intercept_mhz) / (2 * chi_mhz).
  The default chi_mhz is -0.137 to match the original analysis script.

t1_gamma_summary.csv
  T1 fit results. For each round and gain, the script averages t1_points at the
  same delay, fits A * exp(-(t - t0) / T1_us) + c, and computes
  gamma_1_per_ms = 1000 / T1_us. nbar is matched from qspec_nbar_summary by
  round and gain; if an exact gain match is not found but the gain lies inside
  the spectroscopy gain range, nbar is linearly interpolated and nbar_source is
  set to `interpolated`.

results_summary.json
  Small machine-readable summary containing nbar_by_round, qspec_summary,
  t1_summary, and plot file paths. The raw point tables are kept as CSV because
  they can be large.
"""


def require_h5py() -> None:
    if h5py is None:
        raise ImportError(
            "h5py is required to read the saved HDF5 files. Install with: "
            "pip install h5py"
        )


def parse_rounds(rounds: str | list[int], data_root: Path, qubit_index: int) -> list[int]:
    if isinstance(rounds, list):
        return [int(r) for r in rounds]

    text = str(rounds).strip().lower()
    if text == "auto":
        found = []
        for path in data_root.glob(f"qubit_{qubit_index}round*"):
            match = re.search(r"round(\d+)$", path.name)
            if match:
                found.append(int(match.group(1)))
        return sorted(set(found))

    out: list[int] = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            parts = [p.strip() for p in chunk.split(":")]
            if len(parts) not in (2, 3):
                raise ValueError(f"Bad round range: {chunk!r}")
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            out.extend(range(start, stop, step))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def sort_key_for_file(path: Path) -> tuple[dt.datetime, str]:
    match = TIMESTAMP_RE.search(path.name)
    if not match:
        return dt.datetime.min, path.name
    year, month, day, hour, minute, second = map(int, match.groups())
    return dt.datetime(year, month, day, hour, minute, second), path.name


def discover_h5_files(data_root: Path, qubit_index: int, rounds: list[int], subdir: str) -> list[tuple[int, Path]]:
    files: list[tuple[int, Path]] = []
    for round_id in rounds:
        folder = data_root / f"qubit_{qubit_index}round{round_id}" / "study_data" / "Data_h5" / subdir
        if not folder.exists():
            continue
        for h5_file in sorted(folder.glob("*.h5"), key=sort_key_for_file):
            files.append((round_id, h5_file))
    return files


def decode_if_needed(value: Any) -> Any:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        return value.item()
    return value


def dataset_entries(group: Any, name: str, expected_records: int | None = None) -> list[Any]:
    if name not in group:
        return []

    raw = group[name][()]
    if isinstance(raw, np.ndarray):
        if raw.ndim == 0:
            return [decode_if_needed(raw.item())]

        if raw.dtype.kind in ("S", "U", "O"):
            return [decode_if_needed(x) for x in raw.ravel().tolist()]

        if expected_records is not None:
            if raw.shape[0] == expected_records:
                return [raw[i] for i in range(expected_records)]
            return [raw]

        return [decode_if_needed(x) for x in raw.ravel().tolist()]

    return [decode_if_needed(raw)]


def entry_at(group: Any, name: str, index: int, expected_records: int) -> Any:
    entries = dataset_entries(group, name, expected_records=expected_records)
    if not entries:
        return None
    if len(entries) == expected_records and index < len(entries):
        return entries[index]
    if len(entries) == 1:
        return entries[0]
    if index < len(entries):
        return entries[index]
    return entries[-1]


def parse_numeric_vector(value: Any) -> list[float]:
    value = decode_if_needed(value)
    if value is None:
        return []

    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("S", "U", "O"):
            numbers: list[float] = []
            for item in value.ravel().tolist():
                numbers.extend(parse_numeric_vector(item))
            return numbers
        arr = np.asarray(value, dtype=float).ravel()
        return [float(x) for x in arr if np.isfinite(x)]

    if isinstance(value, (list, tuple)):
        numbers = []
        for item in value:
            numbers.extend(parse_numeric_vector(item))
        return numbers

    if isinstance(value, (int, float, np.integer, np.floating)):
        val = float(value)
        return [val] if np.isfinite(val) else []

    text = str(value)
    text = re.sub(r"\bnp\.(?:float|int)\d*\(", "(", text)
    text = re.sub(r"\b(?:float|int)\(", "(", text)
    return [float(match.group(0)) for match in NUMBER_RE.finditer(text)]


def nested_from_python(obj: Any) -> list[list[float]]:
    if isinstance(obj, np.ndarray):
        if obj.ndim == 0:
            return [[float(obj.item())]]
        if obj.ndim == 1:
            return [[float(x) for x in obj.ravel()]]
        return [[float(x) for x in np.asarray(row, dtype=float).ravel()] for row in obj]

    if isinstance(obj, (list, tuple)):
        if not obj:
            return []
        if any(isinstance(item, (list, tuple, np.ndarray)) for item in obj):
            rows = []
            for item in obj:
                child = nested_from_python(item)
                if child:
                    if len(child) == 1:
                        rows.append(child[0])
                    else:
                        rows.extend(child)
            return rows
        return [[float(x) for x in obj]]

    numbers = parse_numeric_vector(obj)
    return [numbers] if numbers else []


def parse_nested_numeric(value: Any) -> list[list[float]]:
    value = decode_if_needed(value)
    if value is None:
        return []

    if isinstance(value, np.ndarray) and value.dtype.kind not in ("S", "U", "O"):
        return nested_from_python(value)

    text = str(value)
    cleaned_for_literal = (
        text.replace("np.float64(", "")
        .replace("np.int64(", "")
        .replace("array(", "")
    )
    try:
        import ast

        parsed = ast.literal_eval(cleaned_for_literal)
        rows = nested_from_python(parsed)
        if rows:
            return rows
    except Exception:
        pass

    bracket_groups = re.findall(r"\[([^\[\]]+)\]", text)
    rows = []
    for group in bracket_groups:
        numbers = parse_numeric_vector(group)
        if numbers:
            rows.append(numbers)
    if rows:
        return rows

    numbers = parse_numeric_vector(text)
    return [numbers] if numbers else []


def first_nonempty_row(rows: list[list[float]]) -> np.ndarray | None:
    for row in rows:
        arr = np.asarray(row, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            return arr
    return None


def normalize_iq_rows(
    i_rows: list[list[float]],
    q_rows: list[list[float]],
    ie_rows: list[list[float]],
    qe_rows: list[list[float]],
    ig_rows: list[list[float]],
    qg_rows: list[list[float]],
) -> list[np.ndarray]:
    nrows = min(len(i_rows), len(q_rows))
    if nrows == 0:
        return []

    ie = first_nonempty_row(ie_rows)
    qe = first_nonempty_row(qe_rows)
    ig = first_nonempty_row(ig_rows)
    qg = first_nonempty_row(qg_rows)

    can_normalize = ie is not None and qe is not None and ig is not None and qg is not None
    if can_normalize:
        ncal = min(ie.size, qe.size, ig.size, qg.size)
        can_normalize = ncal > 0
    if can_normalize:
        e_center = np.mean(ie[:ncal] + 1j * qe[:ncal])
        g_center = np.mean(ig[:ncal] + 1j * qg[:ncal])
        denom = float(np.abs(e_center - g_center) ** 2)
        can_normalize = np.isfinite(denom) and denom > 0.0

    out = []
    for row_i, row_q in zip(i_rows[:nrows], q_rows[:nrows]):
        i_arr = np.asarray(row_i, dtype=float).ravel()
        q_arr = np.asarray(row_q, dtype=float).ravel()
        n = min(i_arr.size, q_arr.size)
        if n == 0:
            continue
        z = i_arr[:n] + 1j * q_arr[:n]
        if can_normalize:
            pop = np.abs(((z - g_center) * (e_center - g_center)) / denom)
        else:
            pop = np.hypot(i_arr[:n], q_arr[:n])
        out.append(np.asarray(pop, dtype=float))
    return out


def date_info(value: Any) -> tuple[float | None, str]:
    numbers = parse_numeric_vector(value)
    if not numbers:
        return None, ""
    unix_time = float(numbers[0])
    if not np.isfinite(unix_time):
        return None, ""
    try:
        iso = dt.datetime.fromtimestamp(unix_time).strftime("%Y-%m-%d %H:%M:%S")
    except Exception:
        iso = ""
    return unix_time, iso


def row_points_from_sweep(
    *,
    round_id: int,
    qubit_index: int,
    x_values: list[float],
    gains: list[float],
    population_rows: list[np.ndarray],
    x_column: str,
    source_file: Path,
    dataset_index: int,
    date_unix: float | None,
    date_iso: str,
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    x_arr = np.asarray(x_values, dtype=float).ravel()
    gains_arr = np.asarray(gains, dtype=float).ravel()
    rows = [np.asarray(row, dtype=float).ravel() for row in population_rows if len(row)]
    if not rows:
        return records

    def append_trace(gain: float, trace_x: np.ndarray, trace_y: np.ndarray) -> None:
        n = min(trace_x.size, trace_y.size)
        if n == 0:
            return
        for x_val, pop in zip(trace_x[:n], trace_y[:n]):
            if not (np.isfinite(x_val) and np.isfinite(pop) and np.isfinite(gain)):
                continue
            records.append(
                {
                    "round": int(round_id),
                    "qubit_index": int(qubit_index),
                    "gain": float(gain),
                    x_column: float(x_val),
                    "population": float(pop),
                    "date_unix": "" if date_unix is None else float(date_unix),
                    "date_iso": date_iso,
                    "source_file": str(source_file),
                    "dataset_index": int(dataset_index),
                }
            )

    if gains_arr.size == len(rows) and len(rows) > 1:
        for gain, row in zip(gains_arr, rows):
            append_trace(float(gain), x_arr, row)
        return records

    if len(rows) == 1:
        row = rows[0]
        if gains_arr.size > 1 and x_arr.size > 0 and row.size == gains_arr.size * x_arr.size:
            for i, gain in enumerate(gains_arr):
                start = i * x_arr.size
                stop = start + x_arr.size
                append_trace(float(gain), x_arr, row[start:stop])
            return records

        if gains_arr.size == row.size and x_arr.size == row.size:
            for gain, x_val, pop in zip(gains_arr, x_arr, row):
                append_trace(float(gain), np.asarray([x_val]), np.asarray([pop]))
            return records

        gain = float(gains_arr[0]) if gains_arr.size else float("nan")
        append_trace(gain, x_arr, row)
        return records

    if gains_arr.size == 1:
        same_length = all(row.size == rows[0].size for row in rows)
        if same_length:
            append_trace(float(gains_arr[0]), x_arr, np.nanmean(np.vstack(rows), axis=0))
            return records

    for idx, row in enumerate(rows):
        gain = float(gains_arr[idx]) if idx < gains_arr.size else float(gains_arr[0]) if gains_arr.size else float("nan")
        append_trace(gain, x_arr, row)
    return records


def get_qubit_group(h5_file: Any, qubit_index: int) -> Any | None:
    primary = f"Q{qubit_index + 1}"
    fallback = f"Q{qubit_index}"
    if primary in h5_file:
        return h5_file[primary]
    if fallback in h5_file:
        return h5_file[fallback]
    return None


def load_sweep_points(
    files: list[tuple[int, Path]],
    *,
    qubit_index: int,
    x_dataset: str,
    x_column: str,
) -> list[dict[str, Any]]:
    require_h5py()
    points: list[dict[str, Any]] = []
    for round_id, path in files:
        with h5py.File(path, "r") as handle:
            group = get_qubit_group(handle, qubit_index)
            if group is None:
                continue

            date_entries = dataset_entries(group, "Dates")
            expected_records = max(1, len(date_entries))
            for dataset_index in range(expected_records):
                i_rows = parse_nested_numeric(entry_at(group, "I", dataset_index, expected_records))
                q_rows = parse_nested_numeric(entry_at(group, "Q", dataset_index, expected_records))
                x_values = parse_numeric_vector(entry_at(group, x_dataset, dataset_index, expected_records))
                gains = parse_numeric_vector(entry_at(group, "Gains", dataset_index, expected_records))
                date_unix, date_iso = date_info(entry_at(group, "Dates", dataset_index, expected_records))

                population_rows = normalize_iq_rows(
                    i_rows,
                    q_rows,
                    parse_nested_numeric(entry_at(group, "ss_I_e", dataset_index, expected_records)),
                    parse_nested_numeric(entry_at(group, "ss_Q_e", dataset_index, expected_records)),
                    parse_nested_numeric(entry_at(group, "ss_I_g", dataset_index, expected_records)),
                    parse_nested_numeric(entry_at(group, "ss_Q_g", dataset_index, expected_records)),
                )

                points.extend(
                    row_points_from_sweep(
                        round_id=round_id,
                        qubit_index=qubit_index,
                        x_values=x_values,
                        gains=gains,
                        population_rows=population_rows,
                        x_column=x_column,
                        source_file=path,
                        dataset_index=dataset_index,
                        date_unix=date_unix,
                        date_iso=date_iso,
                    )
                )
    return points


def finite_xy(x_values: list[float] | np.ndarray, y_values: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x_values, dtype=float).ravel()
    y = np.asarray(y_values, dtype=float).ravel()
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def fit_qspec_center(frequency_mhz: list[float], population: list[float], min_points: int) -> tuple[float, float, bool]:
    x, y = finite_xy(frequency_mhz, population)
    if x.size < min_points:
        return float("nan"), float("nan"), False

    order = np.argsort(x)
    x = x[order]
    y = y[order]
    if np.unique(x).size < min_points:
        return float("nan"), float("nan"), False

    span = float(np.nanmax(x) - np.nanmin(x))
    if not np.isfinite(span) or span <= 0:
        return float("nan"), float("nan"), False

    idx_peak = int(np.nanargmax(y))
    xbar = float(np.nanmean(x))

    edge_count = max(2, int(0.15 * x.size))
    edge_y = np.r_[y[:edge_count], y[-edge_count:]]
    c0 = float(np.nanmedian(edge_y))
    c1 = 0.0
    amp = float(y[idx_peak] - c0)
    dx = float(np.nanmedian(np.diff(np.unique(x))))
    gamma = max(span / 20.0, dx)
    p0 = np.array([float(x[idx_peak]), gamma, amp, c0, c1], dtype=float)

    def model(params: np.ndarray) -> np.ndarray:
        f0, half_width, amplitude, baseline, slope = params
        return amplitude * (half_width**2) / ((x - f0) ** 2 + half_width**2) + baseline + slope * (x - xbar)

    def residuals(params: np.ndarray) -> np.ndarray:
        return model(params) - y

    lower = np.array([float(np.nanmin(x)), max(dx * 0.25, 1e-12), -np.inf, -np.inf, -np.inf])
    upper = np.array([float(np.nanmax(x)), span * 2.0, np.inf, np.inf, np.inf])
    try:
        result = least_squares(
            residuals,
            p0,
            bounds=(lower, upper),
            loss="soft_l1",
            f_scale=float(np.nanstd(y)) if np.nanstd(y) > 0 else 1.0,
            max_nfev=5000,
        )
        if not result.success:
            return float("nan"), float("nan"), False
        center = float(result.x[0])
        fwhm = float(2.0 * abs(result.x[1]))
        return center, fwhm, np.isfinite(center)
    except Exception:
        return float("nan"), float("nan"), False


def average_by_axis(points: list[dict[str, Any]], x_column: str) -> dict[tuple[int, float], tuple[list[float], list[float]]]:
    buckets: dict[tuple[int, float, float], list[float]] = defaultdict(list)
    for row in points:
        try:
            key = (int(row["round"]), float(row["gain"]), float(row[x_column]))
            pop = float(row["population"])
        except Exception:
            continue
        if np.isfinite(key[1]) and np.isfinite(key[2]) and np.isfinite(pop):
            buckets[key].append(pop)

    traces: dict[tuple[int, float], dict[float, float]] = defaultdict(dict)
    for (round_id, gain, x_val), pops in buckets.items():
        traces[(round_id, gain)][x_val] = float(np.nanmean(pops))

    out = {}
    for key, values in traces.items():
        xs = sorted(values.keys())
        ys = [values[x] for x in xs]
        out[key] = (xs, ys)
    return out


def calculate_nbar_from_qspec(
    qspec_points: list[dict[str, Any]],
    *,
    qubit_index: int,
    chi_mhz: float,
    min_points: int,
) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]]]:
    traces = average_by_axis(qspec_points, "frequency_mhz")
    centers_by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)

    for (round_id, gain), (freqs, pops) in sorted(traces.items()):
        center, fwhm, fit_ok = fit_qspec_center(freqs, pops, min_points=min_points)
        if fit_ok:
            centers_by_round[round_id].append(
                {
                    "round": int(round_id),
                    "qubit_index": int(qubit_index),
                    "gain": float(gain),
                    "center_mhz": float(center),
                    "fwhm_mhz": float(fwhm),
                }
            )

    nbar_by_round: dict[str, dict[str, Any]] = {}
    summary_rows: list[dict[str, Any]] = []

    for round_id, rows in sorted(centers_by_round.items()):
        rows = sorted(rows, key=lambda row: row["gain"])
        gains = np.asarray([row["gain"] for row in rows], dtype=float)
        centers = np.asarray([row["center_mhz"] for row in rows], dtype=float)
        ok = np.isfinite(gains) & np.isfinite(centers)
        if np.sum(ok) < 2:
            continue

        gain2 = gains[ok] ** 2
        centers_ok = centers[ok]
        slope, intercept = np.polyfit(gain2, centers_ok, deg=1)
        fitted_centers = slope * (gains**2) + intercept
        nbar = (fitted_centers - intercept) / (2.0 * chi_mhz)

        nbar_by_round[str(round_id)] = {
            "gains": [float(x) for x in gains],
            "nbar": [float(x) for x in nbar],
            "centers_mhz": [float(x) for x in centers],
            "fwhm_mhz": [float(row["fwhm_mhz"]) for row in rows],
            "chi_mhz": float(chi_mhz),
            "center_vs_gain2_slope_mhz_per_gain2": float(slope),
            "center_vs_gain2_intercept_mhz": float(intercept),
        }

        for row, nbar_value in zip(rows, nbar):
            summary_rows.append(
                {
                    **row,
                    "nbar": float(nbar_value),
                    "chi_mhz": float(chi_mhz),
                    "center_vs_gain2_slope_mhz_per_gain2": float(slope),
                    "center_vs_gain2_intercept_mhz": float(intercept),
                }
            )

    return nbar_by_round, summary_rows


def exp_decay_for_t1(t_shifted_us: np.ndarray, amplitude: float, t1_us: float, offset: float) -> np.ndarray:
    return amplitude * np.exp(-t_shifted_us / t1_us) + offset


def fit_t1_trace(delay_us: list[float], population: list[float], min_points: int) -> tuple[float, float, bool]:
    t, y = finite_xy(delay_us, population)
    if t.size < min_points:
        return float("nan"), float("nan"), False

    order = np.argsort(t)
    t = t[order]
    y = y[order]
    unique_t = np.unique(t)
    if unique_t.size < min_points:
        return float("nan"), float("nan"), False

    t0 = float(np.nanmin(t))
    t_shifted = t - t0
    span = float(np.nanmax(t_shifted))
    if not np.isfinite(span) or span <= 0:
        return float("nan"), float("nan"), False

    edge_count = max(1, int(0.2 * t.size))
    offset_guess = float(np.nanmedian(y[-edge_count:]))
    amplitude_guess = float(np.nanmedian(y[:edge_count]) - offset_guess)
    t1_guess = max(span / 3.0, 1e-6)

    diffs = np.diff(unique_t)
    min_dt = float(np.nanmin(diffs)) if diffs.size else span / max(1, t.size - 1)
    min_dt = min_dt if np.isfinite(min_dt) and min_dt > 0 else 1e-6

    try:
        popt, pcov = curve_fit(
            exp_decay_for_t1,
            t_shifted,
            y,
            p0=[amplitude_guess, t1_guess, offset_guess],
            bounds=([-np.inf, 0.2 * min_dt, -np.inf], [np.inf, 10.0 * span, np.inf]),
            method="trf",
            maxfev=10000,
        )
        t1_us = float(popt[1])
        t1_err_us = float(math.sqrt(pcov[1][1])) if pcov.shape == (3, 3) and pcov[1][1] >= 0 else float("nan")
        fit_ok = np.isfinite(t1_us) and t1_us > 0
        return t1_us, t1_err_us, bool(fit_ok)
    except Exception:
        return float("nan"), float("nan"), False


def nbar_for_gain(
    nbar_by_round: dict[str, dict[str, Any]],
    round_id: int,
    gain: float,
    *,
    atol: float,
    rtol: float,
) -> tuple[float, str]:
    entry = nbar_by_round.get(str(round_id))
    if not entry:
        return float("nan"), "missing_round"

    gains = np.asarray(entry.get("gains", []), dtype=float).ravel()
    nbar = np.asarray(entry.get("nbar", []), dtype=float).ravel()
    n = min(gains.size, nbar.size)
    gains = gains[:n]
    nbar = nbar[:n]
    mask = np.isfinite(gains) & np.isfinite(nbar)
    gains = gains[mask]
    nbar = nbar[mask]
    if gains.size == 0:
        return float("nan"), "missing_gain"

    nearest_idx = int(np.argmin(np.abs(gains - gain)))
    tolerance = atol + rtol * max(abs(float(gain)), abs(float(gains[nearest_idx])))
    if abs(float(gains[nearest_idx]) - float(gain)) <= tolerance:
        return float(nbar[nearest_idx]), "matched"

    order = np.argsort(gains)
    gains = gains[order]
    nbar = nbar[order]
    if gains.size >= 2 and gains[0] <= gain <= gains[-1]:
        return float(np.interp(gain, gains, nbar)), "interpolated"

    return float("nan"), "missing_gain"


def fit_t1_summary(
    t1_points: list[dict[str, Any]],
    nbar_by_round: dict[str, dict[str, Any]],
    *,
    qubit_index: int,
    min_points: int,
    gain_match_atol: float,
    gain_match_rtol: float,
) -> list[dict[str, Any]]:
    traces = average_by_axis(t1_points, "delay_us")
    summary: list[dict[str, Any]] = []

    for (round_id, gain), (delays, pops) in sorted(traces.items()):
        t1_us, t1_err_us, fit_ok = fit_t1_trace(delays, pops, min_points=min_points)
        nbar, nbar_source = nbar_for_gain(
            nbar_by_round,
            round_id,
            gain,
            atol=gain_match_atol,
            rtol=gain_match_rtol,
        )

        if fit_ok and np.isfinite(t1_us) and t1_us > 0:
            gamma = 1000.0 / t1_us
            gamma_err = 1000.0 * t1_err_us / (t1_us**2) if np.isfinite(t1_err_us) else float("nan")
        else:
            gamma = float("nan")
            gamma_err = float("nan")

        summary.append(
            {
                "round": int(round_id),
                "qubit_index": int(qubit_index),
                "gain": float(gain),
                "nbar": float(nbar),
                "T1_us": float(t1_us),
                "T1_err_us": float(t1_err_us),
                "gamma_1_per_ms": float(gamma),
                "gamma_err_1_per_ms": float(gamma_err),
                "fit_ok": bool(fit_ok),
                "nbar_source": nbar_source,
            }
        )

    return summary


def centers_to_edges(centers: np.ndarray) -> np.ndarray:
    centers = np.asarray(centers, dtype=float).ravel()
    if centers.size == 0:
        return np.asarray([])
    if centers.size == 1:
        width = max(abs(float(centers[0])) * 0.1, 1.0)
        return np.asarray([centers[0] - width / 2.0, centers[0] + width / 2.0])
    diffs = np.diff(centers)
    if not np.all(diffs > 0):
        centers = np.arange(centers.size, dtype=float)
        diffs = np.diff(centers)
    mids = (centers[:-1] + centers[1:]) / 2.0
    first = centers[0] - diffs[0] / 2.0
    last = centers[-1] + diffs[-1] / 2.0
    return np.r_[first, mids, last]


def import_pyplot(out_dir: Path):
    mpl_config = out_dir / ".matplotlib"
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config))
    mpl_config.mkdir(parents=True, exist_ok=True)
    import matplotlib

    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt
    import matplotlib.ticker as mticker

    return plt, mticker


def plot_summary_vs_nbar(
    summary: list[dict[str, Any]],
    *,
    y_key: str,
    yerr_key: str,
    ylabel: str,
    title: str,
    outfile: Path,
    dpi: int,
) -> list[Path]:
    rows = [
        row
        for row in summary
        if row.get("fit_ok")
        and np.isfinite(float(row.get("nbar", float("nan"))))
        and np.isfinite(float(row.get(y_key, float("nan"))))
    ]
    if not rows:
        return []

    plt, _ = import_pyplot(outfile.parent)
    x = np.asarray([float(row["nbar"]) for row in rows], dtype=float)
    y = np.asarray([float(row[y_key]) for row in rows], dtype=float)
    yerr = np.asarray([float(row.get(yerr_key, float("nan"))) for row in rows], dtype=float)
    rounds = np.asarray([int(row["round"]) for row in rows], dtype=float)

    fig, ax = plt.subplots(figsize=(6.6, 4.4))
    finite_err = np.isfinite(yerr) & (yerr >= 0)
    if np.any(finite_err):
        ax.errorbar(x[finite_err], y[finite_err], yerr=yerr[finite_err], fmt="none", ecolor="0.7", alpha=0.8, capsize=2)

    scatter = ax.scatter(x, y, c=rounds, s=28, edgecolors="k", linewidths=0.25)
    cbar = fig.colorbar(scatter, ax=ax, pad=0.02)
    cbar.set_label("Round")
    ax.set_xlabel(r"Photon number $\bar{n}$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    outfile.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(outfile, dpi=dpi)
    plt.close(fig)
    return [outfile]


def plot_per_round_summary(
    summary: list[dict[str, Any]],
    *,
    y_key: str,
    yerr_key: str,
    ylabel: str,
    title_prefix: str,
    filename_prefix: str,
    out_dir: Path,
    dpi: int,
) -> list[Path]:
    rows_by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in summary:
        if row.get("fit_ok") and np.isfinite(float(row.get("nbar", float("nan")))):
            rows_by_round[int(row["round"])].append(row)

    if not rows_by_round:
        return []

    plt, _ = import_pyplot(out_dir)
    plot_files = []
    for round_id, rows in sorted(rows_by_round.items()):
        rows = sorted(rows, key=lambda row: float(row["nbar"]))
        x = np.asarray([float(row["nbar"]) for row in rows], dtype=float)
        y = np.asarray([float(row[y_key]) for row in rows], dtype=float)
        yerr = np.asarray([float(row.get(yerr_key, float("nan"))) for row in rows], dtype=float)
        finite_err = np.isfinite(yerr) & (yerr >= 0)

        fig, ax = plt.subplots(figsize=(6.2, 4.0))
        if np.any(finite_err):
            ax.errorbar(x[finite_err], y[finite_err], yerr=yerr[finite_err], fmt="none", ecolor="0.65", capsize=2)
        ax.plot(x, y, "o-", linewidth=1.2, markersize=4)
        ax.set_xlabel(r"Photon number $\bar{n}$")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{title_prefix}, round {round_id}")
        ax.grid(True, alpha=0.25)
        fig.tight_layout()

        outfile = out_dir / f"{filename_prefix}_round{round_id}.png"
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)
        plot_files.append(outfile)
    return plot_files


def plot_qspec_heatmaps(
    qspec_points: list[dict[str, Any]],
    nbar_by_round: dict[str, dict[str, Any]],
    *,
    qubit_index: int,
    out_dir: Path,
    dpi: int,
) -> list[Path]:
    if not qspec_points:
        return []

    plt, mticker = import_pyplot(out_dir)
    plot_files: list[Path] = []

    all_pop = np.asarray([float(row["population"]) for row in qspec_points], dtype=float)
    finite_pop = all_pop[np.isfinite(all_pop)]
    if finite_pop.size == 0:
        return []
    vmin = float(np.nanmin(finite_pop))
    vmax = float(np.nanmax(finite_pop))

    points_by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in qspec_points:
        points_by_round[int(row["round"])].append(row)

    for round_id, rows in sorted(points_by_round.items()):
        entry = nbar_by_round.get(str(round_id))
        if not entry:
            continue

        gains = np.asarray(entry["gains"], dtype=float)
        nbar = np.asarray(entry["nbar"], dtype=float)
        n = min(gains.size, nbar.size)
        if n == 0:
            continue
        gain_to_nbar = {float(g): float(nb) for g, nb in zip(gains[:n], nbar[:n])}

        bucket: dict[tuple[float, float], list[float]] = defaultdict(list)
        for row in rows:
            gain = float(row["gain"])
            freq = float(row["frequency_mhz"])
            pop = float(row["population"])
            if gain in gain_to_nbar and np.isfinite(freq) and np.isfinite(pop):
                bucket[(gain, freq)].append(pop)

        gains_r = sorted({gain for gain, _ in bucket})
        freqs_r = sorted({freq for _, freq in bucket})
        if len(gains_r) == 0 or len(freqs_r) == 0:
            continue

        x_vals = np.asarray([gain_to_nbar[g] for g in gains_r], dtype=float)
        order = np.argsort(x_vals)
        x_vals = x_vals[order]
        gains_sorted = [gains_r[i] for i in order]
        freqs_arr = np.asarray(freqs_r, dtype=float)

        c_grid = np.full((len(freqs_r), len(gains_sorted)), np.nan, dtype=float)
        f_index = {freq: idx for idx, freq in enumerate(freqs_r)}
        g_index = {gain: idx for idx, gain in enumerate(gains_sorted)}
        for (gain, freq), vals in bucket.items():
            if gain in g_index:
                c_grid[f_index[freq], g_index[gain]] = float(np.nanmean(vals))

        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        mesh = ax.pcolormesh(
            centers_to_edges(x_vals),
            centers_to_edges(freqs_arr),
            c_grid,
            shading="flat",
            vmin=vmin,
            vmax=vmax,
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label("Qubit population")

        ax.set_title(f"Qubit {qubit_index}: qspec vs nbar, round {round_id}")
        ax.set_xlabel(r"Photon number $\bar{n}$")
        ax.set_ylabel("Frequency (MHz)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
        fig.tight_layout()

        outfile = out_dir / f"qspec_heatmap_q{qubit_index}_round{round_id}_nbar.png"
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)
        plot_files.append(outfile)

    return plot_files


def plot_t1_heatmaps(
    t1_points: list[dict[str, Any]],
    nbar_by_round: dict[str, dict[str, Any]],
    *,
    qubit_index: int,
    out_dir: Path,
    dpi: int,
) -> list[Path]:
    if not t1_points:
        return []

    plt, mticker = import_pyplot(out_dir)
    plot_files: list[Path] = []

    all_pop = np.asarray([float(row["population"]) for row in t1_points], dtype=float)
    finite_pop = all_pop[np.isfinite(all_pop)]
    if finite_pop.size == 0:
        return []
    vmin = float(np.nanmin(finite_pop))
    vmax = float(np.nanmax(finite_pop))

    points_by_round: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in t1_points:
        points_by_round[int(row["round"])].append(row)

    for round_id, rows in sorted(points_by_round.items()):
        entry = nbar_by_round.get(str(round_id))
        if not entry:
            continue

        gains = np.asarray(entry["gains"], dtype=float)
        nbar = np.asarray(entry["nbar"], dtype=float)
        n = min(gains.size, nbar.size)
        if n == 0:
            continue
        gain_to_nbar = {float(g): float(nb) for g, nb in zip(gains[:n], nbar[:n])}

        bucket: dict[tuple[float, float], list[float]] = defaultdict(list)
        for row in rows:
            gain = float(row["gain"])
            delay = float(row["delay_us"])
            pop = float(row["population"])
            if gain in gain_to_nbar and np.isfinite(delay) and np.isfinite(pop):
                bucket[(gain, delay)].append(pop)

        gains_r = sorted({gain for gain, _ in bucket})
        delays_r = sorted({delay for _, delay in bucket})
        if len(gains_r) == 0 or len(delays_r) == 0:
            continue

        x_vals = np.asarray([gain_to_nbar[g] for g in gains_r], dtype=float)
        order = np.argsort(x_vals)
        x_vals = x_vals[order]
        gains_sorted = [gains_r[i] for i in order]
        delays_arr = np.asarray(delays_r, dtype=float)

        c_grid = np.full((len(delays_r), len(gains_sorted)), np.nan, dtype=float)
        d_index = {delay: idx for idx, delay in enumerate(delays_r)}
        g_index = {gain: idx for idx, gain in enumerate(gains_sorted)}
        for (gain, delay), vals in bucket.items():
            if gain in g_index:
                c_grid[d_index[delay], g_index[gain]] = float(np.nanmean(vals))

        fig, ax = plt.subplots(figsize=(6.8, 4.6))
        mesh = ax.pcolormesh(
            centers_to_edges(x_vals),
            centers_to_edges(delays_arr),
            c_grid,
            shading="flat",
            vmin=vmin,
            vmax=vmax,
        )
        cbar = fig.colorbar(mesh, ax=ax, pad=0.02)
        cbar.set_label("Qubit population")

        ax.set_title(f"Qubit {qubit_index}: T1 decay vs nbar, round {round_id}")
        ax.set_xlabel(r"Photon number $\bar{n}$")
        ax.set_ylabel("Delay time (us)")
        ax.xaxis.set_major_locator(mticker.MaxNLocator(nbins=7))
        fig.tight_layout()

        outfile = out_dir / f"t1_heatmap_q{qubit_index}_round{round_id}_nbar.png"
        outfile.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(outfile, dpi=dpi)
        plt.close(fig)
        plot_files.append(outfile)

    return plot_files


def csv_value(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_value(row.get(key, "")) for key in fieldnames})


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.integer, np.floating)):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def write_outputs(results: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    write_csv(
        out_dir / "qspec_points.csv",
        results["qspec_points"],
        ["round", "qubit_index", "gain", "frequency_mhz", "population", "date_unix", "date_iso", "source_file", "dataset_index"],
    )
    write_csv(
        out_dir / "t1_points.csv",
        results["t1_points"],
        ["round", "qubit_index", "gain", "delay_us", "population", "date_unix", "date_iso", "source_file", "dataset_index"],
    )
    write_csv(
        out_dir / "qspec_nbar_summary.csv",
        results["qspec_summary"],
        [
            "round",
            "qubit_index",
            "gain",
            "nbar",
            "center_mhz",
            "fwhm_mhz",
            "chi_mhz",
            "center_vs_gain2_slope_mhz_per_gain2",
            "center_vs_gain2_intercept_mhz",
        ],
    )
    write_csv(
        out_dir / "t1_gamma_summary.csv",
        results["t1_summary"],
        [
            "round",
            "qubit_index",
            "gain",
            "nbar",
            "T1_us",
            "T1_err_us",
            "gamma_1_per_ms",
            "gamma_err_1_per_ms",
            "fit_ok",
            "nbar_source",
        ],
    )

    summary_json = {
        "data_format": DATA_FORMAT_DESCRIPTION,
        "nbar_by_round": results["nbar_by_round"],
        "qspec_summary": results["qspec_summary"],
        "t1_summary": results["t1_summary"],
        "plot_files": [str(path) for path in results["plot_files"]],
    }
    with (out_dir / "results_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(summary_json), handle, indent=2)

    (out_dir / "DATA_FORMAT_README.txt").write_text(DATA_FORMAT_DESCRIPTION, encoding="utf-8")


def make_all_plots(
    qspec_points: list[dict[str, Any]],
    t1_points: list[dict[str, Any]],
    t1_summary: list[dict[str, Any]],
    nbar_by_round: dict[str, dict[str, Any]],
    *,
    config: AnalysisConfig,
) -> list[Path]:
    if not config.make_plots:
        return []

    plot_dir = config.out_dir / "plots"
    plot_files: list[Path] = []

    if config.make_per_round_summary_plots:
        round_dir = plot_dir / "per_round_t1_gamma"
        plot_files.extend(
            plot_per_round_summary(
                t1_summary,
                y_key="T1_us",
                yerr_key="T1_err_us",
                ylabel="T1 (us)",
                title_prefix=f"Qubit {config.qubit_index}: T1 vs nbar",
                filename_prefix=f"T1_vs_nbar_q{config.qubit_index}",
                out_dir=round_dir,
                dpi=config.dpi,
            )
        )
        plot_files.extend(
            plot_per_round_summary(
                t1_summary,
                y_key="gamma_1_per_ms",
                yerr_key="gamma_err_1_per_ms",
                ylabel="Gamma = 1/T1 (1/ms)",
                title_prefix=f"Qubit {config.qubit_index}: Gamma vs nbar",
                filename_prefix=f"Gamma_vs_nbar_q{config.qubit_index}",
                out_dir=round_dir,
                dpi=config.dpi,
            )
        )

    if config.make_qspec_heatmaps:
        plot_files.extend(
            plot_qspec_heatmaps(
                qspec_points,
                nbar_by_round,
                qubit_index=config.qubit_index,
                out_dir=plot_dir / "qspec_heatmaps_vs_nbar",
                dpi=config.dpi,
            )
        )

    if config.make_t1_heatmaps:
        plot_files.extend(
            plot_t1_heatmaps(
                t1_points,
                nbar_by_round,
                qubit_index=config.qubit_index,
                out_dir=plot_dir / "t1_heatmaps_vs_nbar",
                dpi=config.dpi,
            )
        )

    return plot_files


def run_analysis(config: AnalysisConfig) -> dict[str, Any]:
    config.data_root = Path(config.data_root)
    config.out_dir = Path(config.out_dir)
    rounds = parse_rounds(config.rounds, config.data_root, config.qubit_index)

    qspec_files = discover_h5_files(config.data_root, config.qubit_index, rounds, config.qspec_subdir)
    t1_files = discover_h5_files(config.data_root, config.qubit_index, rounds, config.t1_subdir)

    qspec_points = load_sweep_points(
        qspec_files,
        qubit_index=config.qubit_index,
        x_dataset="Frequencies",
        x_column="frequency_mhz",
    )
    t1_points = load_sweep_points(
        t1_files,
        qubit_index=config.qubit_index,
        x_dataset="Delay Times",
        x_column="delay_us",
    )

    nbar_by_round, qspec_summary = calculate_nbar_from_qspec(
        qspec_points,
        qubit_index=config.qubit_index,
        chi_mhz=config.chi_mhz,
        min_points=config.min_qspec_points,
    )
    t1_summary = fit_t1_summary(
        t1_points,
        nbar_by_round,
        qubit_index=config.qubit_index,
        min_points=config.min_t1_points,
        gain_match_atol=config.gain_match_atol,
        gain_match_rtol=config.gain_match_rtol,
    )

    results = {
        "config": config,
        "rounds": rounds,
        "qspec_files": [path for _, path in qspec_files],
        "t1_files": [path for _, path in t1_files],
        "qspec_points": qspec_points,
        "t1_points": t1_points,
        "nbar_by_round": nbar_by_round,
        "qspec_summary": qspec_summary,
        "t1_summary": t1_summary,
        "plot_files": [],
        "data_format": DATA_FORMAT_DESCRIPTION,
    }
    results["plot_files"] = make_all_plots(
        qspec_points,
        t1_points,
        t1_summary,
        nbar_by_round,
        config=config,
    )
    write_outputs(results, config.out_dir)
    return results


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Load saved QZE HDF5 data, compute nbar, fit T1, and make T1/Gamma/qspec plots vs nbar."
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT, help="Folder containing qubit_<q>round<N> folders.")
    parser.add_argument("--out", dest="out_dir", type=Path, default=DEFAULT_OUT_DIR, help="Output folder for CSV/JSON/plots.")
    parser.add_argument("--qubit", dest="qubit_index", type=int, default=DEFAULT_QUBIT_INDEX, help="Zero-based qubit index.")
    parser.add_argument("--rounds", default=DEFAULT_ROUNDS, help="Rounds to load, e.g. 0:47, 0,1,2, or auto.")
    parser.add_argument("--chi-mhz", type=float, default=DEFAULT_CHI_MHZ, help="chi/2pi in MHz for nbar extraction.")
    parser.add_argument("--qspec-subdir", default="QSpec_zeno", help="Data_h5 subfolder for qspec data.")
    parser.add_argument("--t1-subdir", default="T1_ge_zeno", help="Data_h5 subfolder for T1 data.")
    parser.add_argument("--dpi", type=int, default=200, help="Saved plot DPI.")
    parser.add_argument("--no-plots", action="store_true", help="Only write CSV/JSON; do not make plots.")
    parser.add_argument("--no-qspec-heatmaps", action="store_true", help="Skip qubit spectroscopy heatmaps vs nbar.")
    parser.add_argument("--no-t1-heatmaps", action="store_true", help="Skip T1 heatmaps vs nbar.")
    parser.add_argument("--no-per-round-summary-plots", action="store_true", help="Skip per-round T1/Gamma summary plots.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config = AnalysisConfig(
        data_root=args.data_root,
        out_dir=args.out_dir,
        qubit_index=args.qubit_index,
        rounds=args.rounds,
        chi_mhz=args.chi_mhz,
        qspec_subdir=args.qspec_subdir,
        t1_subdir=args.t1_subdir,
        make_plots=not args.no_plots,
        make_qspec_heatmaps=not args.no_qspec_heatmaps,
        make_t1_heatmaps=not args.no_t1_heatmaps,
        make_per_round_summary_plots=not args.no_per_round_summary_plots,
        dpi=args.dpi,
    )
    results = run_analysis(config)

    print(f"Loaded {len(results['qspec_files'])} qspec HDF5 files.")
    print(f"Loaded {len(results['t1_files'])} T1 HDF5 files.")
    print(f"Wrote {len(results['qspec_points'])} qspec point rows.")
    print(f"Wrote {len(results['t1_points'])} T1 point rows.")
    print(f"Wrote {len(results['qspec_summary'])} qspec/nbar summary rows.")
    print(f"Wrote {len(results['t1_summary'])} T1/Gamma summary rows.")
    print(f"Wrote {len(results['plot_files'])} plot files.")
    print(f"Output folder: {config.out_dir}")


if __name__ == "__main__":
    main()
