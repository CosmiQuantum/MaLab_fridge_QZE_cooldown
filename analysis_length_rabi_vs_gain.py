"""
Re-fit and re-plot a length-Rabi-vs-gain dataset that was saved by
longitudinal_qze_005_ge_length_rabi_vs_gain.py.

It loads the saved I/Q data from the 'Rabi_chevron' h5 file, re-fits a cosine to the
length-Rabi oscillation at each qubit-drive gain, prints the Rabi frequency per gain,
and saves the same two plots the live script makes (the magnitude heatmap + Rabi
frequency vs gain, and the per-gain fit overlays) into the documentation folder.

The fitting here is more robust than the live script so every gain gets a good fit:
  * each gain's (I, Q) trace is projected onto its principal axis, so we always fit the
    maximum-contrast 1D signal regardless of how the signal is split between I and Q
    (this is what fixes the gain whose oscillation lived in Q instead of I), and
  * the cosine frequency is found with an FFT-seeded multi-start search, keeping the fit
    with the smallest residual, so the optimiser can't get stuck on a low-frequency alias.
"""

import os
import glob
import datetime

import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

from expt_config import expt_cfg
from section_008_save_data_to_h5 import Data_H5

# ----------------------------------------------------------------------------------------
# Which dataset to load  (match longitudinal_qze_005_ge_length_rabi_vs_gain.py)
# ----------------------------------------------------------------------------------------
run_name = 'bob_run_started_Feb_11'
device_name = 'squill'
study = 'length_rabi_chevron2'
sub_study = 'all_qubits'
QubitIndex = 4                 # 0-indexed -> this is Q5
signal = 'None'               # 'I', 'Q', or 'None' (None -> use principal-axis projection)

# Optionally point straight at a specific .h5 file. If left as None we auto-find the most
# recent Rabi_chevron file for the dataset above.
h5_file_override = None

# ----------------------------------------------------------------------------------------
# Build the paths exactly the way the acquisition script does
# ----------------------------------------------------------------------------------------
data_set = f'qubit_{QubitIndex}round0'
dataSetFolder = os.path.join(f"M:/_Data/20250822 - Olivia/{run_name}/{device_name}/",
                             study, sub_study, data_set)
optimizationFolder = os.path.join(dataSetFolder, 'optimization')
studyDocumentationFolder = os.path.join(dataSetFolder, 'documentation')

expt_name = 'length_rabi_vs_gain'


def find_latest_h5():
    if h5_file_override is not None:
        return h5_file_override
    search_dir = os.path.join(optimizationFolder, 'Data_h5', 'Rabi_chevron')
    files = glob.glob(os.path.join(search_dir, '*.h5'))
    if not files:
        raise FileNotFoundError(f"No Rabi_chevron .h5 files found in {search_dir}")
    return max(files, key=os.path.getmtime)


def process_h5_data(data):
    """Parse a numpy-array-repr byte/str (how I/Q/Gains were serialized) into a flat
    list of floats. Same parser as analysis_004_pi_amp_vs_time_plots.py -- the character
    filter drops every bracket (including the internal ones in a 2D array repr), so it
    handles the (n_gains, n_lengths) I/Q arrays as well as 1D Gains."""
    if isinstance(data, bytes):
        data_str = data.decode()
    elif isinstance(data, str):
        data_str = data
    else:
        raise ValueError("Unsupported data type. Data should be bytes or string.")

    cleaned = ''.join(c for c in data_str if c.isdigit() or c in ['-', '.', ' ', 'e'])
    return [float(x) for x in cleaned.split() if x]


def cosine(x, a, b, c, d):
    return a * np.cos(2. * np.pi * b * x - c * 2 * np.pi) + d


def estimate_freq(x, y):
    """Dominant oscillation frequency of y(x) via rFFT (cycles per unit x)."""
    n = len(x)
    dt = (x[-1] - x[0]) / (n - 1)
    yf = np.abs(np.fft.rfft(y - np.mean(y)))
    xf = np.fft.rfftfreq(n, d=dt)
    if len(yf) <= 1:
        return 1.0 / (x[-1] - x[0])
    return xf[np.argmax(yf[1:]) + 1]      # skip the DC bin


def principal_axis_signal(I_row, Q_row):
    """Project (I, Q) onto its principal axis -> the 1D signal with the most contrast.
    This makes the fit independent of how the oscillation is split between I and Q."""
    pts = np.vstack([I_row - np.mean(I_row), Q_row - np.mean(Q_row)])  # 2 x N
    cov = pts @ pts.T
    _, vecs = np.linalg.eigh(cov)            # eigenvalues ascending
    principal = vecs[:, -1]                   # largest-variance direction
    return principal @ pts                     # 1 x N projected signal


def robust_cosine_fit(x, y):
    """Fit cosine(x) with an FFT-seeded multi-start frequency search.
    Returns (popt, fit_curve, rabi_freq). Always returns a result."""
    dt = (x[-1] - x[0]) / (len(x) - 1)
    nyquist = 0.5 / dt
    f_min = 0.5 / (x[-1] - x[0])

    a_guess = (np.max(y) - np.min(y)) / 2
    d_guess = np.mean(y)
    f_fft = estimate_freq(x, y)

    # candidate starting frequencies: the FFT peak, a few harmonics, and a coarse sweep
    candidates = [f_fft, f_fft / 2.0, 2.0 * f_fft]
    candidates += list(np.linspace(f_min, nyquist, 20))
    candidates = [min(max(f, f_min), nyquist) for f in candidates]

    lower = [0, f_min, -1, -np.inf]
    upper = [np.inf, nyquist, 1, np.inf]

    best = None
    best_res = np.inf
    for f0 in candidates:
        try:
            popt, _ = curve_fit(cosine, x, y, maxfev=100000,
                                p0=[a_guess, f0, 0, d_guess], bounds=(lower, upper))
            res = np.sum((cosine(x, *popt) - y) ** 2)
            if res < best_res:
                best_res = res
                best = popt
        except Exception:
            continue

    if best is None:
        # extremely unlikely; fall back to the raw FFT estimate with no curve fit
        best = [a_guess, min(max(f_fft, f_min), nyquist), 0, d_guess]
    return best, cosine(x, *best), abs(best[1])


def main():
    h5_file = find_latest_h5()
    print(f"Loading: {h5_file}")

    # Load the same way the other analysis_* scripts do: Data_H5.load_from_h5 returns,
    # for each key, [group[key][()]] * save_r, and the byte-strings get .decode()'d before
    # process_h5_data. The Rabi_chevron file uses the same keys as 'Rabi', so we reuse that
    # mapping (scaling=True maps I, Q, Gains, Fit, ss_*, and shots).
    save_round = int(h5_file.split('Num_per_batch')[-1].split('.')[0].split('_')[0])
    H5_class_instance = Data_H5(h5_file)
    load_data = H5_class_instance.load_from_h5(data_type='Rabi', save_r=save_round, scaling=True)
    rabi = load_data['Rabi'][QubitIndex]

    dataset = 0  # only one round saved per Rabi_chevron file
    I_flat = np.array(process_h5_data(rabi.get('I', [])[0][dataset].decode()))
    Q_flat = np.array(process_h5_data(rabi.get('Q', [])[0][dataset].decode()))
    gains = np.array(process_h5_data(rabi.get('Gains', [])[0][dataset].decode()))

    n_gains = len(gains)
    if n_gains == 0 or I_flat.size % n_gains != 0:
        raise ValueError(f"Can't reshape I (size {I_flat.size}) into {n_gains} gains.")
    n_lengths = I_flat.size // n_gains
    I = I_flat.reshape(n_gains, n_lengths)
    Q = Q_flat.reshape(n_gains, n_lengths)

    # length axis isn't saved -> rebuild it from the experiment config
    cfg = expt_cfg[expt_name]
    lengths = np.linspace(cfg['start'], cfg['stop'], n_lengths)

    print(f"n_gains = {n_gains}, n_lengths = {n_lengths}")

    # ---- fit each gain ----
    rabi_freqs = []
    fits = []
    fit_traces = []      # the 1D signal that was actually fit (for the overlay plot)
    print(f"\n--- Rabi frequency vs gain (Q{QubitIndex + 1}) ---")
    for idx in range(n_gains):
        if 'Q' in signal and 'None' not in signal:
            trace = Q[idx]
        elif 'I' in signal and 'None' not in signal:
            trace = I[idx]
        else:
            trace = principal_axis_signal(I[idx], Q[idx])   # robust default
        _, fit_curve, rabi_freq = robust_cosine_fit(lengths, trace)
        rabi_freqs.append(rabi_freq)
        fits.append(fit_curve)
        fit_traces.append(trace)
        print(f"  gain = {gains[idx]:.4f}  ->  Rabi frequency = {rabi_freq:.4f} MHz")
    print("-------------------------------------------\n")

    rabi_freqs = np.array(rabi_freqs)
    fits = np.array(fits)
    fit_traces = np.array(fit_traces)

    # ---- figure 1: magnitude heatmap + Rabi frequency vs gain ----
    plt.rcParams.update({'font.size': 18})
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    mag = np.sqrt(I ** 2 + Q ** 2)
    extent = [lengths[0], lengths[-1], gains[0], gains[-1]]
    im = ax1.imshow(mag, aspect='auto', origin='lower', extent=extent)
    ax1.set_xlabel("Qubit drive pulse length (us)", fontsize=18)
    ax1.set_ylabel("Qubit drive gain (a.u.)", fontsize=18)
    ax1.set_title(f"Rabi magnitude Q{QubitIndex + 1}", fontsize=18)
    fig.colorbar(im, ax=ax1)

    ax2.plot(gains, rabi_freqs, 'o-', linewidth=2)
    ax2.set_xlabel("Qubit drive gain (a.u.)", fontsize=18)
    ax2.set_ylabel("Rabi frequency (MHz)", fontsize=18)
    ax2.tick_params(axis='both', which='major', labelsize=16)
    plt.tight_layout()

    # ---- figure 2: per-gain fit overlays ----
    ncols = 4
    nrows = int(np.ceil(n_gains / ncols))
    fig2, axes = plt.subplots(nrows, ncols, figsize=(4 * ncols, 2.6 * nrows), squeeze=False)
    for idx in range(nrows * ncols):
        ax = axes[idx // ncols][idx % ncols]
        if idx < n_gains:
            ax.plot(lengths, fit_traces[idx], '.', markersize=4, label="data")
            ax.plot(lengths, fits[idx], '-', color='red', linewidth=1.5, label="fit")
            ax.set_title(f"gain={gains[idx]:.3f}, f={rabi_freqs[idx]:.3f} MHz", fontsize=10)
            ax.tick_params(axis='both', which='major', labelsize=8)
        else:
            ax.axis('off')
    fig2.text(0.5, 0.995, f"Length-Rabi fits per gain  Q{QubitIndex + 1} (refit)",
              ha='center', va='top', fontsize=14)
    fig2.tight_layout(rect=[0, 0, 1, 0.98])

    # ---- save to the documentation folder ----
    out_dir = os.path.join(studyDocumentationFolder, expt_name)
    os.makedirs(out_dir, exist_ok=True)
    stamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = f"R_0_Q_{QubitIndex + 1}_{stamp}_{expt_name}_refit"
    f1 = os.path.join(out_dir, f"{base}_vs_gain_q{QubitIndex + 1}.png")
    f2 = os.path.join(out_dir, f"{base}_per_gain_fits_q{QubitIndex + 1}.png")
    fig.savefig(f1, dpi=100, bbox_inches='tight')
    fig2.savefig(f2, dpi=100, bbox_inches='tight')
    plt.close(fig)
    plt.close(fig2)
    print(f"Saved:\n  {f1}\n  {f2}")


if __name__ == '__main__':
    main()
