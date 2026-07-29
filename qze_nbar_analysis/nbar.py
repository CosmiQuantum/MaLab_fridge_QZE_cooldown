from __future__ import annotations

from collections import defaultdict

import numpy as np
from scipy.optimize import least_squares


def finite_xy(x_values, y_values):
    x = np.asarray(x_values, dtype=float).ravel()
    y = np.asarray(y_values, dtype=float).ravel()
    n = min(x.size, y.size)
    x = x[:n]
    y = y[:n]
    mask = np.isfinite(x) & np.isfinite(y)
    return x[mask], y[mask]


def fit_qspec_center(frequency_mhz, population, min_points):
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
    amp = float(y[idx_peak] - c0)
    dx = float(np.nanmedian(np.diff(np.unique(x))))
    gamma = max(span / 20.0, dx)
    p0 = np.array([float(x[idx_peak]), gamma, amp, c0, 0.0], dtype=float)

    def model(params):
        f0, half_width, amplitude, baseline, slope = params
        return amplitude * (half_width ** 2) / ((x - f0) ** 2 + half_width ** 2) + baseline + slope * (x - xbar)

    def residuals(params):
        return model(params) - y

    lower = np.array([float(np.nanmin(x)), max(dx * 0.25, 1e-12), -np.inf, -np.inf, -np.inf])
    upper = np.array([float(np.nanmax(x)), span * 2.0, np.inf, np.inf, np.inf])
    try:
        result = least_squares(
            residuals, p0, bounds=(lower, upper), loss="soft_l1",
            f_scale=float(np.nanstd(y)) if np.nanstd(y) > 0 else 1.0, max_nfev=5000,
        )
        if not result.success:
            return float("nan"), float("nan"), False
        center = float(result.x[0])
        fwhm = float(2.0 * abs(result.x[1]))
        return center, fwhm, np.isfinite(center)
    except Exception:
        return float("nan"), float("nan"), False


def average_by_axis(points, x_column):
    buckets = defaultdict(list)
    for row in points:
        try:
            key = (int(row["round"]), float(row["gain"]), float(row[x_column]))
            pop = float(row["population"])
        except Exception:
            continue
        if np.isfinite(key[1]) and np.isfinite(key[2]) and np.isfinite(pop):
            buckets[key].append(pop)
    traces = defaultdict(dict)
    for (round_id, gain, x_val), pops in buckets.items():
        traces[(round_id, gain)][x_val] = float(np.nanmean(pops))
    out = {}
    for key, values in traces.items():
        xs = sorted(values.keys())
        ys = [values[x] for x in xs]
        out[key] = (xs, ys)
    return out


def average_over_rounds(points, x_column):
    # pool all rounds: mean population per (gain, x) across rounds.
    # returns gain -> (x_array, mean_pop, n_rounds)
    per = average_by_axis(points, x_column)
    by_gain = defaultdict(list)
    for (round_id, gain), (xs, ys) in per.items():
        by_gain[round(float(gain), 6)].append((np.asarray(xs, float), np.asarray(ys, float)))
    out = {}
    for gain, entries in by_gain.items():
        x0 = entries[0][0]
        stack = np.vstack([ys for xs, ys in entries if ys.size == x0.size])
        out[gain] = (x0, stack.mean(axis=0), len(stack))
    return out


def calculate_nbar_from_qspec(qspec_points, qubit_index, chi_mhz, min_points):
    # fit qspec center per (round, gain), then within a round fit center vs gain^2.
    # nbar = slope * gain^2 / (2 chi)
    traces = average_by_axis(qspec_points, "frequency_mhz")
    centers_by_round = defaultdict(list)
    for (round_id, gain), (freqs, pops) in sorted(traces.items()):
        center, fwhm, fit_ok = fit_qspec_center(freqs, pops, min_points=min_points)
        if fit_ok:
            centers_by_round[round_id].append(
                {"round": int(round_id), "gain": float(gain),
                 "center_mhz": float(center), "fwhm_mhz": float(fwhm)}
            )

    nbar_by_round = {}
    for round_id, rows in sorted(centers_by_round.items()):
        rows = sorted(rows, key=lambda row: row["gain"])
        gains = np.asarray([row["gain"] for row in rows], dtype=float)
        centers = np.asarray([row["center_mhz"] for row in rows], dtype=float)
        ok = np.isfinite(gains) & np.isfinite(centers)
        if np.sum(ok) < 2:
            continue
        slope, intercept = np.polyfit(gains[ok] ** 2, centers[ok], deg=1)
        fitted_centers = slope * (gains ** 2) + intercept
        nbar = (fitted_centers - intercept) / (2.0 * chi_mhz)
        nbar_by_round[str(round_id)] = {
            "gains": [float(x) for x in gains],
            "nbar": [float(x) for x in nbar],
            "centers_mhz": [float(x) for x in centers],
            "chi_mhz": float(chi_mhz),
        }
    return nbar_by_round


def gain_to_mean_nbar(nbar_by_round, round_to=6):
    # average nbar across all rounds at each gain
    gain_to_nbars = defaultdict(list)
    for entry in nbar_by_round.values():
        for g_val, nb_val in zip(entry["gains"], entry["nbar"]):
            if np.isfinite(g_val) and np.isfinite(nb_val):
                gain_to_nbars[round(float(g_val), round_to)].append(nb_val)
    return {g: float(np.mean(v)) for g, v in gain_to_nbars.items()}


def nbar_lookup(gain_val, mapping, atol=1e-3):
    gain_val = float(gain_val)
    best_key = None
    best_diff = atol
    for g_key, nb in mapping.items():
        diff = abs(gain_val - g_key)
        if diff < best_diff:
            best_diff = diff
            best_key = g_key
    return mapping[best_key] if best_key is not None else np.nan
