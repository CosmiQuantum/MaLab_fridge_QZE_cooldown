"""Produce a frequency-offset series from the canonical overnight Ramsey batches."""
import argparse
import csv
import datetime as dt
import os
from pathlib import Path
import re
import time

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


def vector(group, key):
    data = group[key][()]
    if data.dtype.kind in "SO":
        raw = data.reshape(-1)[0].decode()
        if "..." in raw:
            raise ValueError("Truncated HDF5 array")
        return np.fromstring(raw.strip("[]").replace(",", " "), sep=" ")
    return data.astype(float).ravel()


def setting(group, key):
    # Data_H5's legacy writer stores the system config as a dictionary repr.
    raw = group["Syst Config"][0].decode()
    return float(re.search(r"'" + key + r"':\s*([-+.0-9eE]+)", raw).group(1))


def fringe(t, offset, transient, c, s, tau, f):
    angle = 2 * np.pi * f * t
    return offset + np.exp(-t / tau) * (transient + c*np.cos(angle) + s*np.sin(angle))


def estimate(t, y, detuning):
    span = np.ptp(t)
    step = np.median(np.diff(t))
    spectrum = abs(np.fft.rfft((y - np.mean(y)) * np.hanning(len(y)), 4*len(y)))
    freqs = np.fft.rfftfreq(4*len(y), step)
    bins = (freqs > 1/span) & (freqs < 0.5/step)
    peak = freqs[bins][np.argmax(spectrum[bins])]
    candidates = []
    for seed in (peak, detuning):
        try:
            p, cov = curve_fit(fringe, t, y, p0=[np.mean(y), 0, np.std(y), 0, span/3, seed],
                               bounds=([-np.inf]*4 + [step, 0], [np.inf]*4 + [20*span, 0.5/step]),
                               maxfev=4000)
            rms = np.sqrt(np.mean((fringe(t, *p) - y)**2))
            candidates.append((rms, p, cov))
        except (RuntimeError, ValueError):
            continue
    if not candidates:
        return np.nan, np.inf, 0.
    rms, p, cov = min(candidates, key=lambda result: result[0])
    return p[-1], np.sqrt(cov[-1, -1]), np.hypot(p[2], p[3])/max(rms, 1e-12)


def process(path, pending, rows):
    with h5py.File(path) as h5:
        traces = [(key, vector(g, "Delay Times"), vector(g, "Population"),
                   float(g["Dates"][0]), setting(g, "programmed_detuning_MHz"),
                   setting(g, "startup_carrier_MHz")) for key, g in h5.items()]
    for key, t, y, epoch, detuning, carrier in traces:
        queue = pending.setdefault(key, [])
        queue.append((y, epoch))
        avg = np.mean([item[0] for item in queue], axis=0)
        f, err, snr = estimate(t, avg, detuning)
        # Independent interleaved delay subsets must identify the same line.
        halves = [estimate(t[k::2], avg[k::2], detuning)[0] for k in (0, 1)]
        split = abs(halves[0] - halves[1])
        valid = (snr >= 3 and err <= 0.025 and split <= max(0.025, 4*err)
                 and 0 < f < 1/(8*np.median(np.diff(t))))
        offset = detuning - f if valid else np.nan
        rows.append(dict(row="M"+key[1:], epoch=np.mean([item[1] for item in queue]),
                         averaged_traces=len(queue), carrier_MHz=carrier,
                         fringe_MHz=f, uncertainty_MHz=err, amplitude_over_residual=snr,
                         split_difference_MHz=split, offset_MHz=offset,
                         qubit_MHz=carrier+offset, source=path.name))
        if valid or len(queue) >= 4:
            queue.clear()


def publish(folder, rows):
    with (folder/"ramsey_offsets.tmp").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(folder/"ramsey_offsets.tmp", folder/"ramsey_offsets.csv")
    fig, ax = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for row in sorted({entry["row"] for entry in rows}):
        points = [entry for entry in rows if entry["row"] == row]
        ax.errorbar([dt.datetime.fromtimestamp(entry["epoch"]) for entry in points],
                    [entry["offset_MHz"] for entry in points],
                    yerr=[entry["uncertainty_MHz"] if np.isfinite(entry["offset_MHz"]) else 0 for entry in points],
                    fmt=".-", label=row)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M"))
    ax.set(xlabel="Local time", ylabel="Frequency offset from startup carrier (MHz)")
    ax.legend(ncol=6)
    fig.savefig(folder/"ramsey_offsets.tmp.png", dpi=150)
    plt.close(fig)
    os.replace(folder/"ramsey_offsets.tmp.png", folder/"ramsey_offsets.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("run", type=Path)
    parser.add_argument("--until", type=dt.datetime.fromisoformat, required=True)
    args = parser.parse_args()
    processed, pending, rows = set(), {}, []
    while True:
        for path in sorted((args.run/"study_data"/"Data_h5"/"t2_ge").glob("*.h5")):
            if path in processed:
                continue
            try:
                process(path, pending, rows)
            except (OSError, KeyError):
                break  # Writer may still have the batch open.
            publish(args.run/"documentation", rows)
            processed.add(path)
        if dt.datetime.now() >= args.until:
            break
        time.sleep(30)
