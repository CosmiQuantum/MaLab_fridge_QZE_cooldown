"""
analysis_ckp_nbar.py — CKP Photon Number (n̄) Calibration Analysis
=================================================================
Based on: Sank et al., Phys. Rev. Applied 23, 024055 (2025)
          "System Characterization of Dispersive Readout in
           Superconducting Qubits"  (arXiv:2402.00413)

CKP Protocol Summary
---------------------
The CKP (Chi-Kappa-Power) protocol drives the resonator at variable
frequency (f_d) and gain while performing qubit spectroscopy. The AC
Stark effect shifts the qubit frequency by:

    ω_q'(f_d) = ω_q + 2χ · n̄(f_d)

where the steady-state photon number follows a Lorentzian:

    n̄(f_d) = ε² / [(f_d − ω_r,state)² + (κ/2)²]

The |g⟩ and |e⟩ branches correspond to the qubit-state-dependent
resonator frequencies:  ω_r,g = ω_r + χ,  ω_r,e = ω_r − χ.

Extracted Parameters
--------------------
  • χ  = half the splitting between |g⟩ and |e⟩ branch centers
  • κ  = FWHM of each Lorentzian branch
  • n̄  = |Δf_q| / (2|χ|)   for each (gain, res_drive_freq)

Outputs
-------
  1. Scatter plot  — n̄ vs. resonator gain at on-resonance freq
  2. Heatmap       — n̄ vs. (gain, resonator drive frequency)
  3. Diagnostic    — Branch extraction with Lorentzian fits

Usage
-----
  Run from the lab machine where the H5 data lives:
      python analysis_ckp_nbar.py
"""

import numpy as np
import h5py
import os
import glob
import re
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt

# ═══════════════════════════════════════════════════════════════
#  USER CONFIGURATION — edit paths and parameters here
# ═══════════════════════════════════════════════════════════════

# Path to the directory containing the CKP h5 file
H5_DIR = (r"M:\_Data\20250822 - Olivia\bob_run_started_Feb_11\squill"
          r"\ckp_nbar_calibration\q5\2026-04-16_20-47-19"
          r"\study_data\Data_h5\ckp_calibration")

QUBIT_GROUP = "Q6"  # QubitIndex=5 → h5 group "Q6"

# ----- System parameters -----
RES_FREQ_ON_RESONANCE = 7287.570   # MHz, bare resonator frequency
BARE_QUBIT_FREQ       = 3095.192   # MHz, bare qubit g-e frequency

# ----- Config values for sweep reconstruction (fallbacks) -----
CFG_GAIN_START     = 0.0
CFG_GAIN_END       = 0.15
CFG_GAIN_STEPS     = 20

CFG_RES_FREQ_START = 7287.57 + 1   # 7288.57 MHz
CFG_RES_FREQ_STOP  = 7287.57 - 1   # 7286.57 MHz
CFG_RES_FREQ_STEPS = 50

CFG_QU_FREQ_OFFSET_START = -75     # MHz from bare qubit freq
CFG_QU_FREQ_OFFSET_END   = 5      # MHz from bare qubit freq
CFG_QU_FREQ_STEPS         = 100

# Known χ from previous measurement (for comparison only)
CHI_CONFIG = -0.234 / 2  # MHz  → −0.117 MHz

# ----- Output -----
SAVE_FIGS  = True
SHOW_FIGS  = True
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


# ═══════════════════════════════════════════════════════════════
#  H5 LOADING UTILITIES
# ═══════════════════════════════════════════════════════════════

def find_h5_file(directory):
    """Find the most-recent .h5 file in *directory*."""
    h5_files = sorted(glob.glob(os.path.join(directory, "*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files in:\n  {directory}")
    if len(h5_files) > 1:
        print(f"  Found {len(h5_files)} h5 files — using latest: "
              f"{os.path.basename(h5_files[-1])}")
    return h5_files[-1]


def _bytes_to_str(raw):
    """Decode an h5 byte-string to Python str."""
    if isinstance(raw, np.ndarray):
        raw = raw.flat[0]
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def parse_1d_array(raw):
    """
    Parse a 1-D numpy array stored as a space-separated string
    (the result of str(np.array([...]))).
    """
    s = _bytes_to_str(raw).strip()
    inner = s.strip("[]").strip()
    inner = re.sub(r"\s+", " ", inner)         # normalise whitespace
    return np.array([float(v) for v in inner.split() if v])


def parse_nested_array(raw, n_outer=None, n_mid=None, n_inner=None):
    """
    Parse a nested list-of-lists-of-numpy-arrays that was saved via
    ``str(nested_list)`` → ``'[[array([...]), ...], ...]'``.

    Returns an np.float64 array with the recovered shape.
    """
    s = _bytes_to_str(raw).strip()

    # --- Method 1: replace array() notation → plain list, then eval ----
    cleaned = s.replace("array(", "").replace(")", "")
    cleaned = re.sub(r"\s+", " ", cleaned)     # collapse whitespace
    try:
        data = eval(cleaned, {"__builtins__": {},
                              "nan": float("nan"), "inf": float("inf")})
        return np.asarray(data, dtype=np.float64)
    except Exception as e1:
        print(f"    parse_nested (eval-cleaned) failed: {e1}")

    # --- Method 2: eval with array=np.array (handles repr output) ------
    try:
        data = eval(s, {"__builtins__": {},
                        "array": np.array,
                        "nan": float("nan"), "inf": float("inf"),
                        "float64": np.float64})
        return np.asarray(data, dtype=np.float64)
    except Exception as e2:
        print(f"    parse_nested (eval-array) failed: {e2}")

    # --- Method 3: regex extraction (requires known shape) -------------
    if n_outer and n_mid and n_inner:
        try:
            blocks = re.findall(r"\[([\d\s.eE+\-,]+?)\]", s)
            arrays = []
            for blk in blocks:
                vals = [float(x) for x in re.split(r"[,\s]+", blk.strip()) if x]
                if len(vals) == n_inner:
                    arrays.append(vals)
            if len(arrays) == n_outer * n_mid:
                return np.array(arrays, dtype=np.float64).reshape(
                    n_outer, n_mid, n_inner)
        except Exception as e3:
            print(f"    parse_nested (regex) failed: {e3}")

    raise ValueError("Could not parse nested array from H5. "
                     f"First 200 chars: {s[:200]}")


# ═══════════════════════════════════════════════════════════════
#  PHYSICS / ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def iq_distance_from_baseline(I_row, Q_row, n_baseline=10):
    """
    Compute IQ Euclidean distance from the off-resonance baseline
    for a single spectroscopy sweep row.

    The baseline is estimated from the *high-frequency* end of the
    qubit-probe sweep (far above the expected qubit resonance).
    """
    I_base = np.mean(I_row[-n_baseline:])
    Q_base = np.mean(Q_row[-n_baseline:])
    return np.sqrt((I_row - I_base)**2 + (Q_row - Q_base)**2)


def find_peak_center(dist, freq_sweep):
    """
    Locate the spectroscopic peak via argmax + sub-bin quadratic
    interpolation.
    """
    x = np.asarray(freq_sweep, dtype=float)
    idx = int(np.argmax(dist))

    if 1 <= idx <= len(x) - 2:
        xs = x[idx - 1 : idx + 2]
        ys = dist[idx - 1 : idx + 2]
        try:
            a, b, _ = np.polyfit(xs, ys, 2)
            if abs(a) > 1e-20:
                xv = -b / (2 * a)
                if xs[0] <= xv <= xs[-1]:
                    return xv
        except Exception:
            pass
    return x[idx]


def extract_all_branch_centers(I, Q, qu_freq_sweep, n_baseline=10):
    """
    Extract spectroscopy peak centers for a 2-D slice
    ``[n_res_freqs, n_qubit_freqs]`` at a fixed gain.

    Returns 1-D array of centres, length = n_res_freqs.
    """
    n_res = I.shape[0]
    centers = np.empty(n_res)

    for ri in range(n_res):
        dist = iq_distance_from_baseline(I[ri], Q[ri], n_baseline)
        centers[ri] = find_peak_center(dist, qu_freq_sweep)

    return centers


def lorentzian_dip(x, x0, depth, hwhm, offset):
    """
    Lorentzian dip:
        f(x) = offset − depth / [1 + ((x − x₀) / hwhm)²]

    Physical mapping (CKP branches):
      x0     ↔ state-dependent resonator frequency  ω_r ± χ
      depth  ↔ 2|χ| × n̄_peak
      hwhm   ↔ κ / 2  (half the resonator linewidth)
      offset ↔ bare qubit frequency
    """
    return offset - depth / (1.0 + ((x - x0) / hwhm)**2)


def fit_branch(res_freq_sweep, centers, bare_freq_guess):
    """
    Fit the qubit-resonance-vs-drive-frequency curve to a Lorentzian
    dip.  Returns dict of fit parameters or *None* on failure.
    """
    x = np.asarray(res_freq_sweep, dtype=float)
    y = np.asarray(centers, dtype=float)

    depth0 = np.max(y) - np.min(y)
    if depth0 < 0.01:                        # no significant dip
        return None

    x0_guess   = x[np.argmin(y)]
    hwhm0      = np.ptp(x) / 20
    offset0    = bare_freq_guess

    p0 = [x0_guess, depth0, abs(hwhm0), offset0]
    lb = [x.min() - 2,  0.0,   0.001,   y.min() - 5]
    ub = [x.max() + 2,  300.0, np.ptp(x), y.max() + 5]

    try:
        popt, pcov = curve_fit(lorentzian_dip, x, y, p0=p0,
                               bounds=(lb, ub), maxfev=30000)
        perr  = np.sqrt(np.diag(pcov))
        fit_y = lorentzian_dip(x, *popt)

        return dict(
            x0     = popt[0],
            depth  = popt[1],
            hwhm   = abs(popt[2]),
            offset = popt[3],
            kappa  = 2 * abs(popt[2]),
            popt   = popt,
            perr   = perr,
            fit_y  = fit_y,
        )
    except Exception as e:
        print(f"    Lorentzian fit failed: {e}")
        return None


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 62)
    print("  CKP n̄ Calibration Analysis")
    print("  Sank et al., Phys. Rev. Applied 23, 024055 (2025)")
    print("=" * 62)

    # ──────────────────────────────────────────────────────────
    #  1.  Load H5 data
    # ──────────────────────────────────────────────────────────
    h5_path = find_h5_file(H5_DIR)
    print(f"\n[1] Loading: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        grp = f[QUBIT_GROUP]
        print(f"    Datasets in {QUBIT_GROUP}: {list(grp.keys())}")

        # ---- IQ data (nested 3-D arrays stored as strings) ----
        print("    Parsing I_g …", end=" ", flush=True)
        I_g = parse_nested_array(grp["I_g"][()],
                                 CFG_GAIN_STEPS, CFG_RES_FREQ_STEPS,
                                 CFG_QU_FREQ_STEPS)
        print(f"shape {I_g.shape}")

        print("    Parsing Q_g …", end=" ", flush=True)
        Q_g = parse_nested_array(grp["Q_g"][()],
                                 CFG_GAIN_STEPS, CFG_RES_FREQ_STEPS,
                                 CFG_QU_FREQ_STEPS)
        print(f"shape {Q_g.shape}")

        print("    Parsing I_e …", end=" ", flush=True)
        I_e = parse_nested_array(grp["I_e"][()],
                                 CFG_GAIN_STEPS, CFG_RES_FREQ_STEPS,
                                 CFG_QU_FREQ_STEPS)
        print(f"shape {I_e.shape}")

        print("    Parsing Q_e …", end=" ", flush=True)
        Q_e = parse_nested_array(grp["Q_e"][()],
                                 CFG_GAIN_STEPS, CFG_RES_FREQ_STEPS,
                                 CFG_QU_FREQ_STEPS)
        print(f"shape {Q_e.shape}")

        # ---- Sweep arrays ----
        try:
            print("    Parsing gain sweep …", end=" ", flush=True)
            gain_sweep = parse_1d_array(grp["Res Gain Sweep"][()])
            print(f"range [{gain_sweep[0]:.4f}, {gain_sweep[-1]:.4f}]")
        except Exception:
            print("FAILED — using config fallback")
            gain_sweep = np.linspace(CFG_GAIN_START, CFG_GAIN_END,
                                     CFG_GAIN_STEPS)

        try:
            print("    Parsing qubit freq sweep …", end=" ", flush=True)
            qu_freq_sweep = parse_1d_array(grp["Qu Frequency Sweep"][()])
            print(f"range [{qu_freq_sweep[0]:.3f}, {qu_freq_sweep[-1]:.3f}] MHz")
        except Exception:
            print("FAILED — using config fallback")
            qu_freq_sweep = np.linspace(
                BARE_QUBIT_FREQ + CFG_QU_FREQ_OFFSET_START,
                BARE_QUBIT_FREQ + CFG_QU_FREQ_OFFSET_END,
                CFG_QU_FREQ_STEPS)

    # Reconstruct res_freq_sweep from config (most reliable)
    n_gains, n_res, n_qf = I_g.shape
    res_freq_sweep = np.linspace(CFG_RES_FREQ_START, CFG_RES_FREQ_STOP, n_res)

    print(f"\n    Dimensions : {n_gains} gains × {n_res} res_freqs "
          f"× {n_qf} qubit_freqs")
    print(f"    Gain       : {gain_sweep[0]:.4f} – {gain_sweep[-1]:.4f}")
    print(f"    Res freq   : {res_freq_sweep[0]:.4f} – "
          f"{res_freq_sweep[-1]:.4f} MHz")
    print(f"    Qubit freq : {qu_freq_sweep[0]:.3f} – "
          f"{qu_freq_sweep[-1]:.3f} MHz")

    # ──────────────────────────────────────────────────────────
    #  2.  Extract branch centres for every gain
    # ──────────────────────────────────────────────────────────
    print("\n[2] Extracting branch centres …")

    g_centers = np.zeros((n_gains, n_res))   # |g⟩ prep
    e_centers = np.zeros((n_gains, n_res))   # |e⟩ prep

    for gi in range(n_gains):
        g_centers[gi] = extract_all_branch_centers(
            I_g[gi], Q_g[gi], qu_freq_sweep)
        e_centers[gi] = extract_all_branch_centers(
            I_e[gi], Q_e[gi], qu_freq_sweep)

    print("    Done.")

    # ──────────────────────────────────────────────────────────
    #  3.  Fit Lorentzians → χ, κ
    # ──────────────────────────────────────────────────────────
    print("\n[3] Fitting Lorentzians for χ and κ …")

    # Search middle–high gains for best fit (low gain has weak signal)
    best_fit_g, best_fit_e, best_gi = None, None, n_gains // 2

    for trial in range(max(2, n_gains // 4),
                       min(n_gains, 3 * n_gains // 4 + 1)):
        fg = fit_branch(res_freq_sweep, g_centers[trial], BARE_QUBIT_FREQ)
        fe = fit_branch(res_freq_sweep, e_centers[trial], BARE_QUBIT_FREQ)
        if fg is not None and fe is not None:
            if best_fit_g is None or fg["depth"] > best_fit_g["depth"]:
                best_fit_g, best_fit_e, best_gi = fg, fe, trial

    if best_fit_g is not None and best_fit_e is not None:
        chi   = abs(best_fit_g["x0"] - best_fit_e["x0"]) / 2
        kappa = (best_fit_g["kappa"] + best_fit_e["kappa"]) / 2
        ref_gain = gain_sweep[best_gi]

        print(f"    Reference gain index : {best_gi}  "
              f"(gain = {ref_gain:.4f})")
        print(f"    |g⟩ branch center    : {best_fit_g['x0']:.4f} MHz")
        print(f"    |e⟩ branch center    : {best_fit_e['x0']:.4f} MHz")
        print(f"    2χ (splitting)       : {2*chi:.4f} MHz")
        print(f"    χ                    : {chi:.4f} MHz  "
              f"({chi*1e3:.2f} kHz)")
        print(f"    κ                    : {kappa:.4f} MHz  "
              f"({kappa*1e3:.2f} kHz)")
        print(f"    χ from config (ref)  : {abs(CHI_CONFIG):.4f} MHz")
    else:
        print("    ⚠  Lorentzian fit unsuccessful — using config χ")
        chi   = abs(CHI_CONFIG)
        kappa = 0.5                         # default guess [MHz]
        best_fit_g = best_fit_e = None

    # ──────────────────────────────────────────────────────────
    #  4.  Compute n̄ for every (gain, res_freq)
    # ──────────────────────────────────────────────────────────
    print("\n[4] Computing n̄ …")

    # n̄ = |Δf_q| / (2|χ|)
    delta_fq_g = g_centers - BARE_QUBIT_FREQ          # typically ≤ 0
    delta_fq_e = e_centers - BARE_QUBIT_FREQ

    nbar_g = np.abs(delta_fq_g) / (2 * chi)
    nbar_e = np.abs(delta_fq_e) / (2 * chi)

    # ── on-resonance slice ──
    on_res_idx = int(np.argmin(np.abs(res_freq_sweep - RES_FREQ_ON_RESONANCE)))
    print(f"    On-resonance index   : {on_res_idx}  "
          f"(f_d = {res_freq_sweep[on_res_idx]:.4f} MHz)")

    nbar_on_g = nbar_g[:, on_res_idx]
    nbar_on_e = nbar_e[:, on_res_idx]

    # ── fit  n̄ = a·gain² + b  ──
    def quad(g, a, b):
        return a * g**2 + b

    try:
        popt_q, _ = curve_fit(quad, gain_sweep, nbar_on_g, p0=[100, 0])
        fit_g_fine = np.linspace(gain_sweep[0], gain_sweep[-1], 300)
        fit_nbar   = quad(fit_g_fine, *popt_q)
        print(f"    Fit: n̄ ≈ {popt_q[0]:.1f} × g² + {popt_q[1]:.3f}")
    except Exception as e:
        print(f"    Quadratic fit failed: {e}")
        popt_q = None

    # ──────────────────────────────────────────────────────────
    #  5.  Summary table
    # ──────────────────────────────────────────────────────────
    print("\n" + "─" * 62)
    print(f"  {'Gain':>8s}   {'n̄ (|g⟩, on-res)':>16s}   "
          f"{'n̄ (|e⟩, on-res)':>16s}")
    print(f"  {'─'*8}   {'─'*16}   {'─'*16}")
    for i, g in enumerate(gain_sweep):
        print(f"  {g:8.4f}   {nbar_on_g[i]:16.4f}   {nbar_on_e[i]:16.4f}")
    print("─" * 62)

    # ══════════════════════════════════════════════════════════
    #  PLOTS
    # ══════════════════════════════════════════════════════════
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "figure.dpi": 120,
        "savefig.dpi": 200,
    })

    # ── Plot 1: n̄ vs gain — on-resonance (scatter) ───────────
    fig1, ax1 = plt.subplots(figsize=(7, 5))

    ax1.scatter(gain_sweep, nbar_on_g,
                s=80, c="#1976D2", edgecolors="#0D47A1", linewidth=1.2,
                zorder=5, label=r"$|0\rangle$ branch")
    ax1.scatter(gain_sweep, nbar_on_e,
                s=55, c="#E53935", edgecolors="#B71C1C", linewidth=1.2,
                marker="s", zorder=5, label=r"$|1\rangle$ branch")

    if popt_q is not None:
        ax1.plot(fit_g_fine, fit_nbar, "--", color="#0D47A1", lw=1.5,
                 alpha=0.55,
                 label=(rf"fit:  $\bar{{n}} = {popt_q[0]:.0f}\,g^2"
                        rf" {'+' if popt_q[1] >= 0 else ''}"
                        rf"{popt_q[1]:.2f}$"))

    ax1.set_xlabel("Resonator drive gain  [DAC units]")
    ax1.set_ylabel(r"$\bar{n}$  (photon number)")
    ax1.set_title(
        rf"On-resonance  $\bar{{n}}$  vs. gain   "
        rf"($f_d$ = {RES_FREQ_ON_RESONANCE} MHz)")
    ax1.legend(fontsize=10, framealpha=0.9)
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(left=-0.002)
    ax1.set_ylim(bottom=0)

    # annotate extracted parameters
    txt = (rf"$\chi$ = {chi:.4f} MHz" "\n"
           rf"$\kappa$ = {kappa:.4f} MHz")
    ax1.text(0.03, 0.97, txt, transform=ax1.transAxes,
             fontsize=10, va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))

    fig1.tight_layout()
    if SAVE_FIGS:
        p = os.path.join(OUTPUT_DIR, "ckp_nbar_vs_gain_on_resonance.png")
        fig1.savefig(p, bbox_inches="tight")
        print(f"\n  Saved → {p}")

    # ── Plot 2: n̄ heatmap (gain × res_freq) ──────────────────
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))

    # Sort res_freq for increasing y-axis
    sort_idx = np.argsort(res_freq_sweep)
    rf_sorted = res_freq_sweep[sort_idx]
    nbar_sorted = nbar_g[:, sort_idx].T      # shape [n_res, n_gains]

    im = ax2.pcolormesh(gain_sweep, rf_sorted, nbar_sorted,
                        shading="nearest", cmap="inferno")
    cbar = fig2.colorbar(im, ax=ax2, pad=0.02)
    cbar.set_label(r"$\bar{n}$  (photon number)", fontsize=12)

    # mark on-resonance and branch centres
    ax2.axhline(RES_FREQ_ON_RESONANCE, color="white", ls="--", lw=1.4,
                alpha=0.85, label=f"bare resonator ({RES_FREQ_ON_RESONANCE} MHz)")
    if best_fit_g is not None:
        ax2.axhline(best_fit_g["x0"], color="cyan", ls=":", lw=1.2,
                    alpha=0.7, label=rf"$\omega_{{r,|0\rangle}}$ = "
                                     f"{best_fit_g['x0']:.3f} MHz")
        ax2.axhline(best_fit_e["x0"], color="#FF8A65", ls=":", lw=1.2,
                    alpha=0.7, label=rf"$\omega_{{r,|1\rangle}}$ = "
                                     f"{best_fit_e['x0']:.3f} MHz")

    ax2.set_xlabel("Resonator drive gain  [DAC units]")
    ax2.set_ylabel("Resonator drive frequency  [MHz]")
    ax2.set_title(r"$\bar{n}$  across gain and drive frequency  "
                  r"($|0\rangle$ branch)")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.85)

    fig2.tight_layout()
    if SAVE_FIGS:
        p = os.path.join(OUTPUT_DIR, "ckp_nbar_heatmap_gain_vs_resfreq.png")
        fig2.savefig(p, bbox_inches="tight")
        print(f"  Saved → {p}")

    # ── Plot 3: diagnostic — branch extraction at ref gain ────
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5),
                               gridspec_kw={"width_ratios": [1.4, 1]})

    # Left panel: branch centres with Lorentzian fits
    ax = axes3[0]
    ax.plot(res_freq_sweep, g_centers[best_gi], "o", ms=5,
            color="#1976D2", label=r"$|0\rangle$ branch data")
    ax.plot(res_freq_sweep, e_centers[best_gi], "s", ms=5,
            color="#E53935", label=r"$|1\rangle$ branch data")

    if best_fit_g is not None:
        ax.plot(res_freq_sweep, best_fit_g["fit_y"], "-",
                color="#0D47A1", lw=2, label="Lorentzian fit")
    if best_fit_e is not None:
        ax.plot(res_freq_sweep, best_fit_e["fit_y"], "-",
                color="#B71C1C", lw=2)

    ax.axhline(BARE_QUBIT_FREQ, color="gray", ls=":", lw=1, alpha=0.6,
               label=rf"bare $\omega_q$ = {BARE_QUBIT_FREQ} MHz")

    ax.set_xlabel("Resonator drive frequency  [MHz]")
    ax.set_ylabel("Qubit resonance frequency  [MHz]")
    ax.set_title(f"Branch extraction   "
                 f"(gain = {gain_sweep[best_gi]:.4f})")
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.25)

    # Right panel: parameter summary
    ax = axes3[1]
    ax.axis("off")

    lines = [
        r"$\bf{Extracted\ Parameters}$",
        "",
        rf"$\chi$  =  {chi:.4f} MHz  ({chi*1e3:.1f} kHz)",
        rf"$\kappa$  =  {kappa:.4f} MHz  ({kappa*1e3:.1f} kHz)",
        "",
        rf"$2\chi$ (splitting)  =  {2*chi:.4f} MHz",
        "",
        f"On-res freq  =  {RES_FREQ_ON_RESONANCE} MHz",
        f"Bare qubit   =  {BARE_QUBIT_FREQ} MHz",
        "",
    ]

    if best_fit_g is not None:
        lines += [
            rf"$\omega_{{r,|0\rangle}}$  =  {best_fit_g['x0']:.4f} MHz",
            rf"$\omega_{{r,|1\rangle}}$  =  {best_fit_e['x0']:.4f} MHz",
            "",
            rf"$n_{{peak}}$ at ref gain  =  "
            rf"{best_fit_g['depth'] / (2*chi):.1f}  photons",
        ]

    if popt_q is not None:
        lines += [
            "",
            rf"$\bar{{n}}(g) \approx {popt_q[0]:.0f}\,g^2"
            rf" {'+'  if popt_q[1]>=0 else ''}{popt_q[1]:.2f}$",
        ]

    lines += ["", f"Config χ (ref.)  =  {abs(CHI_CONFIG):.4f} MHz"]

    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=11, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow",
                      ec="gray", alpha=0.9))

    fig3.tight_layout()
    if SAVE_FIGS:
        p = os.path.join(OUTPUT_DIR, "ckp_chi_kappa_extraction.png")
        fig3.savefig(p, bbox_inches="tight")
        print(f"  Saved → {p}")

    if SHOW_FIGS:
        plt.show()

    # ──────────────────────────────────────────────────────────
    #  6.  Save results to .npz for downstream use
    # ──────────────────────────────────────────────────────────
    npz_path = os.path.join(OUTPUT_DIR, "ckp_nbar_results.npz")
    np.savez(npz_path,
             chi=chi,
             kappa=kappa,
             gain_sweep=gain_sweep,
             res_freq_sweep=res_freq_sweep,
             qu_freq_sweep=qu_freq_sweep,
             nbar_g=nbar_g,
             nbar_e=nbar_e,
             g_centers=g_centers,
             e_centers=e_centers,
             nbar_on_resonance_g=nbar_on_g,
             nbar_on_resonance_e=nbar_on_e)
    print(f"\n  Results saved → {npz_path}")

    print("\n✓  Analysis complete.")
    return dict(chi=chi, kappa=kappa, nbar_g=nbar_g, nbar_e=nbar_e,
                gain_sweep=gain_sweep, res_freq_sweep=res_freq_sweep)


# ═══════════════════════════════════════════════════════════════
if __name__ == "__main__":
    results = main()
