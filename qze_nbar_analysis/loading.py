from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import numpy as np
import h5py

NUMBER_RE = re.compile(r"[-+]?(?:(?:\d+\.\d*)|(?:\.\d+)|(?:\d+))(?:[eE][-+]?\d+)?")
TIMESTAMP_RE = re.compile(
    r"(\d{4})[-_\.]?(\d{2})[-_\.]?(\d{2})[ Tt_-]?(\d{2})[-_\.]?(\d{2})[-_\.]?(\d{2})"
)


def parse_rounds(rounds, data_root, qubit_index):
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
    out = []
    for chunk in text.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if ":" in chunk:
            parts = [p.strip() for p in chunk.split(":")]
            start = int(parts[0])
            stop = int(parts[1])
            step = int(parts[2]) if len(parts) == 3 and parts[2] else 1
            out.extend(range(start, stop, step))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def sort_key_for_file(path):
    match = TIMESTAMP_RE.search(path.name)
    if not match:
        return dt.datetime.min, path.name
    y, mo, d, h, mi, s = map(int, match.groups())
    return dt.datetime(y, mo, d, h, mi, s), path.name


def discover_h5_files(data_root, qubit_index, rounds, subdir):
    files = []
    for round_id in rounds:
        folder = data_root / f"qubit_{qubit_index}round{round_id}" / "study_data" / "Data_h5" / subdir
        if not folder.exists():
            continue
        for h5_file in sorted(folder.glob("*.h5"), key=sort_key_for_file):
            files.append((round_id, h5_file))
    return files


def decode_if_needed(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="ignore")
    if isinstance(value, np.bytes_):
        return bytes(value).decode("utf-8", errors="ignore")
    if isinstance(value, np.generic):
        return value.item()
    return value


def dataset_entries(group, name, expected_records=None):
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


def entry_at(group, name, index, expected_records):
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


def parse_numeric_vector(value):
    value = decode_if_needed(value)
    if value is None:
        return []
    if isinstance(value, np.ndarray):
        if value.dtype.kind in ("S", "U", "O"):
            numbers = []
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
    return [float(m.group(0)) for m in NUMBER_RE.finditer(text)]


def nested_from_python(obj):
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


def parse_nested_numeric(value):
    value = decode_if_needed(value)
    if value is None:
        return []
    if isinstance(value, np.ndarray) and value.dtype.kind not in ("S", "U", "O"):
        return nested_from_python(value)
    text = str(value)
    cleaned = text.replace("np.float64(", "").replace("np.int64(", "").replace("array(", "")
    try:
        import ast
        parsed = ast.literal_eval(cleaned)
        rows = nested_from_python(parsed)
        if rows:
            return rows
    except Exception:
        pass
    bracket_groups = re.findall(r"\[([^\[\]]+)\]", text)
    rows = []
    for grp in bracket_groups:
        numbers = parse_numeric_vector(grp)
        if numbers:
            rows.append(numbers)
    if rows:
        return rows
    numbers = parse_numeric_vector(text)
    return [numbers] if numbers else []


def first_nonempty_row(rows):
    for row in rows:
        arr = np.asarray(row, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            return arr
    return None


def normalize_iq_rows(i_rows, q_rows, ie_rows, qe_rows, ig_rows, qg_rows):
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


def date_info(value):
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


def row_points_from_sweep(round_id, qubit_index, x_values, gains, population_rows,
                          x_column, source_file, dataset_index, date_unix, date_iso):
    records = []
    x_arr = np.asarray(x_values, dtype=float).ravel()
    gains_arr = np.asarray(gains, dtype=float).ravel()
    rows = [np.asarray(row, dtype=float).ravel() for row in population_rows if len(row)]
    if not rows:
        return records

    def append_trace(gain, trace_x, trace_y):
        n = min(trace_x.size, trace_y.size)
        for x_val, pop in zip(trace_x[:n], trace_y[:n]):
            if not (np.isfinite(x_val) and np.isfinite(pop) and np.isfinite(gain)):
                continue
            records.append({
                "round": int(round_id),
                "qubit_index": int(qubit_index),
                "gain": float(gain),
                x_column: float(x_val),
                "population": float(pop),
                "date_unix": "" if date_unix is None else float(date_unix),
                "date_iso": date_iso,
                "source_file": str(source_file),
                "dataset_index": int(dataset_index),
            })

    # 2d sweep saved as one long row of n_gain * n_x values
    if len(rows) == 1 and gains_arr.size > 1 and x_arr.size > 0 and rows[0].size == gains_arr.size * x_arr.size:
        row = rows[0]
        for i, gain in enumerate(gains_arr):
            append_trace(float(gain), x_arr, row[i * x_arr.size:(i + 1) * x_arr.size])
        return records

    if gains_arr.size == len(rows) and len(rows) > 1:
        for gain, row in zip(gains_arr, rows):
            append_trace(float(gain), x_arr, row)
        return records

    if len(rows) == 1:
        row = rows[0]
        if gains_arr.size == row.size and x_arr.size == row.size:
            for gain, x_val, pop in zip(gains_arr, x_arr, row):
                append_trace(float(gain), np.asarray([x_val]), np.asarray([pop]))
            return records
        gain = float(gains_arr[0]) if gains_arr.size else float("nan")
        append_trace(gain, x_arr, row)
        return records

    if gains_arr.size == 1 and all(row.size == rows[0].size for row in rows):
        append_trace(float(gains_arr[0]), x_arr, np.nanmean(np.vstack(rows), axis=0))
        return records

    for idx, row in enumerate(rows):
        gain = float(gains_arr[idx]) if idx < gains_arr.size else (float(gains_arr[0]) if gains_arr.size else float("nan"))
        append_trace(gain, x_arr, row)
    return records


def get_qubit_group(h5_file, qubit_index):
    primary = f"Q{qubit_index + 1}"
    fallback = f"Q{qubit_index}"
    if primary in h5_file:
        return h5_file[primary]
    if fallback in h5_file:
        return h5_file[fallback]
    return None


def load_sweep_points(files, qubit_index, x_dataset, x_column):
    points = []
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
                    i_rows, q_rows,
                    parse_nested_numeric(entry_at(group, "ss_I_e", dataset_index, expected_records)),
                    parse_nested_numeric(entry_at(group, "ss_Q_e", dataset_index, expected_records)),
                    parse_nested_numeric(entry_at(group, "ss_I_g", dataset_index, expected_records)),
                    parse_nested_numeric(entry_at(group, "ss_Q_g", dataset_index, expected_records)),
                )
                points.extend(row_points_from_sweep(
                    round_id=round_id, qubit_index=qubit_index, x_values=x_values,
                    gains=gains, population_rows=population_rows, x_column=x_column,
                    source_file=path, dataset_index=dataset_index,
                    date_unix=date_unix, date_iso=date_iso,
                ))
    return points
