"""
analysis_ckp_nbar_paper_style.py — CKP analysis using the paper-style fit
========================================================================

Based on:
    Sank et al., Phys. Rev. Applied 23, 024055 (2025)
    "System Characterization of Dispersive Readout in Superconducting Qubits"

Main goal
---------
This script implements the CKP fit in the style described in the paper:

    For a given resonator drive power, fit the two CKP branches together
    with one five-parameter pair-of-Lorentzians model:

        omega_q0
        omega_r,m
        |chi|
        kappa
        |A|^2

Your dataset has many resonator-drive gains, so this script repeats that
five-parameter fit independently for each gain.

Important difference from your older script
-------------------------------------------
The older script used one global nonlinear fit across all gains with shared
omega_q0, omega_r,m, chi, and kappa, plus one |A|^2 per gain.

This script does NOT do that. It fits each gain independently, which is the
closest match to the paper's CKP fitting procedure for a multi-gain dataset.

Sign convention
---------------
The paper uses signed chi < 0, with

    omega_r,|1> - omega_r,|0> = 2 chi < 0.

For numerical stability and clarity, this code fits

    chi_mag = |chi| > 0

and uses

    omega_r,|0> = omega_r,m + chi_mag
    omega_r,|1> = omega_r,m - chi_mag

The corresponding signed paper chi is

    chi_paper_signed = -chi_mag.
"""

import glob
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit


# ═══════════════════════════════════════════════════════════════
#  USER CONFIGURATION
# ═══════════════════════════════════════════════════════════════

date = "2026-04-20_11-32-25"

# Path to the directory containing the CKP h5 file.
H5_DIR = (
    r"M:\_Data\20250822 - Olivia\bob_run_started_Aug_3_2026\squill"
    rf"\ckp_nbar_calibration\q5\{date}"
    r"\study_data\Data_h5\ckp_calibration"
)

QUBIT_GROUP = "Q6"  # QubitIndex=5 -> h5 group "Q6"

# System reference values.
# This is only used as a reference line / optional diagnostic. The paper-style
# extraction below uses fitted branch resonances, not this manual value.
RES_FREQ_ON_RESONANCE = 7287.570  # MHz
BARE_QUBIT_FREQ = 3095.45         # MHz

# Fallback sweep values, used only if the values cannot be read from the H5.
CFG_GAIN_START = 0.0
CFG_GAIN_END = 0.15
CFG_GAIN_STEPS = 20

CFG_RES_FREQ_START = 7287.57 + 1
CFG_RES_FREQ_STOP = 7287.57 - 1
CFG_RES_FREQ_STEPS = 50

CFG_QU_FREQ_OFFSET_START = -75
CFG_QU_FREQ_OFFSET_END = 5
CFG_QU_FREQ_STEPS = 100

# Known chi from previous measurement, for initial guesses and comparison only.
CHI_CONFIG = -0.234 / 2  # MHz -> -0.117 MHz

# Output options.
SAVE_FIGS = True
SHOW_FIGS = True

run_name = "bob_run_started_Aug_3_2026"
device_name = "squill"
study = "ckp_nbar_calibration"
sub_study = "q5"
data_set = date


# ═══════════════════════════════════════════════════════════════
#  OUTPUT DIRECTORY SETUP
# ═══════════════════════════════════════════════════════════════

base_dir = f"M:/_Data/20250822 - Olivia/{run_name}"
device_dir = os.path.join(base_dir, device_name)
study_dir = os.path.join(device_dir, study)
sub_study_dir = os.path.join(study_dir, sub_study)
data_set_dir = os.path.join(sub_study_dir, data_set)

optimization_dir = os.path.join(data_set_dir, "optimization")
study_data_dir = os.path.join(data_set_dir, "study_data")
documentation_dir = os.path.join(data_set_dir, "documentation")

for path in [
    base_dir,
    device_dir,
    study_dir,
    sub_study_dir,
    data_set_dir,
    optimization_dir,
    study_data_dir,
    documentation_dir,
]:
    os.makedirs(path, exist_ok=True)

OUTPUT_DIR = documentation_dir


# ═══════════════════════════════════════════════════════════════
#  H5 LOADING UTILITIES
# ═══════════════════════════════════════════════════════════════

def find_h5_file(directory):
    """Find the most recent .h5 file in directory."""
    h5_files = sorted(glob.glob(os.path.join(directory, "*.h5")))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in:\n  {directory}")

    if len(h5_files) > 1:
        print(
            f"  Found {len(h5_files)} h5 files. "
            f"Using latest: {os.path.basename(h5_files[-1])}"
        )

    return h5_files[-1]


def _bytes_to_str(raw):
    """Decode an H5 byte-string or scalar-like dataset into a Python string."""
    if isinstance(raw, np.ndarray):
        raw = raw.flat[0]
    if isinstance(raw, bytes):
        return raw.decode("utf-8")
    return str(raw)


def parse_1d_array(raw):
    """Parse a 1D array saved as a string-like H5 dataset."""
    s = _bytes_to_str(raw).strip()
    inner = s.strip("[]").strip()
    inner = re.sub(r"\s+", " ", inner)
    return np.array([float(v) for v in inner.split() if v], dtype=float)


def parse_nested_array(raw, n_outer=None, n_mid=None, n_inner=None):
    """
    Parse nested arrays saved as strings in H5.

    This supports the common forms:
        [[...], [...]]
        array([...])
        stringified numpy arrays
    """
    s = _bytes_to_str(raw).strip()

    cleaned = s.replace("array(", "").replace(")", "")
    cleaned = re.sub(r"\s+", " ", cleaned)

    try:
        data = eval(
            cleaned,
            {
                "__builtins__": {},
                "nan": float("nan"),
                "inf": float("inf"),
            },
        )
        return np.asarray(data, dtype=np.float64)
    except Exception as e1:
        print(f"    parse_nested eval-cleaned failed: {e1}")

    try:
        data = eval(
            s,
            {
                "__builtins__": {},
                "array": np.array,
                "nan": float("nan"),
                "inf": float("inf"),
                "float64": np.float64,
            },
        )
        return np.asarray(data, dtype=np.float64)
    except Exception as e2:
        print(f"    parse_nested eval-array failed: {e2}")

    if n_outer and n_mid and n_inner:
        try:
            blocks = re.findall(r"\[([\d\s.eE+\-,]+?)\]", s)
            arrays = []
            for blk in blocks:
                vals = [
                    float(x)
                    for x in re.split(r"[,\s]+", blk.strip())
                    if x
                ]
                if len(vals) == n_inner:
                    arrays.append(vals)

            if len(arrays) == n_outer * n_mid:
                return np.array(arrays, dtype=np.float64).reshape(
                    n_outer,
                    n_mid,
                    n_inner,
                )
        except Exception as e3:
            print(f"    parse_nested regex failed: {e3}")

    raise ValueError(
        "Could not parse nested array from H5. "
        f"First 200 chars: {s[:200]}"
    )


# ═══════════════════════════════════════════════════════════════
#  CENTER EXTRACTION FROM IQ DATA
# ═══════════════════════════════════════════════════════════════

def iq_distance_from_baseline(I_row, Q_row, n_baseline=10):
    """
    Compute IQ Euclidean distance from an off-resonance baseline.

    This is not exactly the paper's raw flip-probability observable.
    It is your available proxy for extracting the qubit spectroscopy center
    from each vertical slice.
    """
    I_base = np.mean(I_row[-n_baseline:])
    Q_base = np.mean(Q_row[-n_baseline:])
    return np.sqrt((I_row - I_base) ** 2 + (Q_row - Q_base) ** 2)


def find_peak_center(dist, freq_sweep):
    """
    Locate the spectroscopy peak via argmax plus sub-bin quadratic interpolation.
    """
    x = np.asarray(freq_sweep, dtype=float)
    y = np.asarray(dist, dtype=float)

    idx = int(np.argmax(y))

    if 1 <= idx <= len(x) - 2:
        xs = x[idx - 1: idx + 2]
        ys = y[idx - 1: idx + 2]

        try:
            a, b, _ = np.polyfit(xs, ys, 2)
            if abs(a) > 1e-20:
                xv = -b / (2.0 * a)
                if xs[0] <= xv <= xs[-1]:
                    return float(xv)
        except Exception:
            pass

    return float(x[idx])


def extract_all_branch_centers(I, Q, qu_freq_sweep, n_baseline=10):
    """
    Extract one qubit-resonance estimate for each resonator-drive frequency.
    """
    n_res = I.shape[0]
    centers = np.empty(n_res, dtype=float)

    for ri in range(n_res):
        dist = iq_distance_from_baseline(I[ri], Q[ri], n_baseline)
        centers[ri] = find_peak_center(dist, qu_freq_sweep)

    return centers


# ═══════════════════════════════════════════════════════════════
#  PAPER-STYLE CKP MODEL
# ═══════════════════════════════════════════════════════════════

def ckp_paper_branch_model(x, omega_q0, omega_rm, chi_mag, kappa, A2, branch):
    """
    Paper-style CKP branch model, Eq. (7), using chi_mag = |chi| > 0.

    For branch "g", corresponding to |0> preparation:

        omega_r,|0> = omega_rm + chi_mag

    For branch "e", corresponding to |1> preparation:

        omega_r,|1> = omega_rm - chi_mag

    The qubit transition frequency shifts downward in this convention:

        omega_d,q* = omega_q0 - 2 chi_mag nbar
    """
    x = np.asarray(x, dtype=float)

    if branch == "g":
        center = omega_rm + chi_mag
    elif branch == "e":
        center = omega_rm - chi_mag
    else:
        raise ValueError("branch must be 'g' or 'e'")

    denom = (x - center) ** 2 + (kappa / 2.0) ** 2
    return omega_q0 - (2.0 * chi_mag * kappa * A2) / denom


def ckp_paper_pair_model(xdata, omega_q0, omega_rm, chi_mag, kappa, A2):
    """
    Five-parameter pair-of-Lorentzians CKP model.

    Fit parameters:
        omega_q0
        omega_rm
        chi_mag
        kappa
        A2

    xdata[0] = resonator drive frequency
    xdata[1] = branch flag, +1 for |0>, -1 for |1>
    """
    x = np.asarray(xdata[0], dtype=float)
    branch_flag = np.asarray(xdata[1], dtype=float)

    y = np.empty_like(x, dtype=float)

    mask_g = branch_flag > 0
    mask_e = branch_flag < 0

    y[mask_g] = ckp_paper_branch_model(
        x[mask_g],
        omega_q0,
        omega_rm,
        chi_mag,
        kappa,
        A2,
        branch="g",
    )

    y[mask_e] = ckp_paper_branch_model(
        x[mask_e],
        omega_q0,
        omega_rm,
        chi_mag,
        kappa,
        A2,
        branch="e",
    )

    return y


def build_paper_pair_fit_arrays(res_freq_sweep, g_centers_one_gain, e_centers_one_gain):
    """
    Build xdata and ydata for one gain setting.

    Both branches are fitted together with one five-parameter model.
    """
    x_g = np.asarray(res_freq_sweep, dtype=float)
    x_e = np.asarray(res_freq_sweep, dtype=float)

    branch_g = np.ones_like(x_g)
    branch_e = -np.ones_like(x_e)

    xdata = np.vstack([
        np.r_[x_g, x_e],
        np.r_[branch_g, branch_e],
    ])

    ydata = np.r_[
        np.asarray(g_centers_one_gain, dtype=float),
        np.asarray(e_centers_one_gain, dtype=float),
    ]

    return xdata, ydata


def estimate_paper_initial_params(res_freq_sweep, g_centers_one_gain, e_centers_one_gain):
    """
    Initial guesses for one five-parameter CKP fit.
    """
    x = np.asarray(res_freq_sweep, dtype=float)
    yg = np.asarray(g_centers_one_gain, dtype=float)
    ye = np.asarray(e_centers_one_gain, dtype=float)

    # Approximate Lorentzian minima give approximate branch resonator centers.
    x0_g = x[int(np.argmin(yg))]
    x0_e = x[int(np.argmin(ye))]

    omega_rm0 = 0.5 * (x0_g + x0_e)

    chi_mag0 = max(
        abs(x0_g - x0_e) / 2.0,
        abs(CHI_CONFIG),
        0.005,
    )

    # Use edge values as a crude estimate of the unshifted qubit frequency.
    edge_vals = np.r_[yg[0], yg[-1], ye[0], ye[-1]]
    omega_q0_0 = float(np.median(edge_vals))

    # Crude linewidth guess.
    kappa0 = max(np.ptp(x) / 8.0, 0.01)

    # At resonance:
    #   depth = 2 chi_mag kappa A2 / (kappa/2)^2
    #         = 8 chi_mag A2 / kappa
    # so:
    #   A2 = depth * kappa / (8 chi_mag)
    depth_g = max(omega_q0_0 - np.min(yg), 0.0)
    depth_e = max(omega_q0_0 - np.min(ye), 0.0)
    depth0 = max(0.5 * (depth_g + depth_e), 1e-6)

    A2_0 = max(depth0 * kappa0 / (8.0 * chi_mag0), 1e-12)

    return omega_q0_0, omega_rm0, chi_mag0, kappa0, A2_0


def fit_ckp_paper_style_one_gain(res_freq_sweep, g_centers_one_gain, e_centers_one_gain):
    """
    Fit one gain setting using the paper-style five-parameter CKP model.
    """
    xdata, ydata = build_paper_pair_fit_arrays(
        res_freq_sweep,
        g_centers_one_gain,
        e_centers_one_gain,
    )

    p0 = estimate_paper_initial_params(
        res_freq_sweep,
        g_centers_one_gain,
        e_centers_one_gain,
    )

    x_min = float(np.min(res_freq_sweep))
    x_max = float(np.max(res_freq_sweep))
    x_span = x_max - x_min

    y_min = float(np.min(ydata))
    y_max = float(np.max(ydata))

    lower = [
        y_min - 20.0,  # omega_q0
        x_min - 2.0,   # omega_rm
        0.001,         # chi_mag
        0.001,         # kappa
        0.0,           # A2
    ]

    upper = [
        y_max + 20.0,  # omega_q0
        x_max + 2.0,   # omega_rm
        x_span,        # chi_mag
        x_span,        # kappa
        np.inf,        # A2
    ]

    popt, pcov = curve_fit(
        ckp_paper_pair_model,
        xdata,
        ydata,
        p0=p0,
        bounds=(lower, upper),
        maxfev=200000,
    )

    perr = np.sqrt(np.diag(pcov))
    yfit = ckp_paper_pair_model(xdata, *popt)
    residuals = ydata - yfit
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    omega_q0, omega_rm, chi_mag, kappa, A2 = popt

    model_g = ckp_paper_branch_model(
        res_freq_sweep,
        omega_q0,
        omega_rm,
        chi_mag,
        kappa,
        A2,
        branch="g",
    )

    model_e = ckp_paper_branch_model(
        res_freq_sweep,
        omega_q0,
        omega_rm,
        chi_mag,
        kappa,
        A2,
        branch="e",
    )

    center_g = omega_rm + chi_mag
    center_e = omega_rm - chi_mag

    nbar_g = kappa * A2 / (
        (res_freq_sweep - center_g) ** 2 + (kappa / 2.0) ** 2
    )

    nbar_e = kappa * A2 / (
        (res_freq_sweep - center_e) ** 2 + (kappa / 2.0) ** 2
    )

    return {
        "popt": popt,
        "perr": perr,
        "pcov": pcov,
        "omega_q0": omega_q0,
        "omega_rm": omega_rm,
        "chi_mag": chi_mag,
        "chi_paper_signed": -chi_mag,
        "kappa": kappa,
        "A2": A2,
        "omega_r_g": center_g,
        "omega_r_e": center_e,
        "model_g": model_g,
        "model_e": model_e,
        "nbar_g": nbar_g,
        "nbar_e": nbar_e,
        "yfit": yfit,
        "residuals": residuals,
        "rmse": rmse,
    }


def fit_ckp_paper_style_all_gains(res_freq_sweep, gain_sweep, g_centers, e_centers):
    """
    Repeat the paper-style five-parameter CKP fit independently for every gain.
    """
    fits = []

    for gi, gain in enumerate(gain_sweep):
        try:
            fit = fit_ckp_paper_style_one_gain(
                res_freq_sweep,
                g_centers[gi],
                e_centers[gi],
            )

            fit["gain"] = float(gain)
            fit["gain_index"] = int(gi)
            fits.append(fit)

            print(
                f"    gain {gain:.4f}: "
                f"|chi| = {fit['chi_mag']:.5f} MHz, "
                f"kappa = {fit['kappa']:.5f} MHz, "
                f"A2 = {fit['A2']:.6g}, "
                f"RMSE = {fit['rmse']:.4f} MHz"
            )

        except Exception as exc:
            print(f"    gain {gain:.4f}: fit failed: {exc}")
            fits.append({
                "gain": float(gain),
                "gain_index": int(gi),
                "failed": True,
                "error": str(exc),
            })

    return fits


# ═══════════════════════════════════════════════════════════════
#  PLOTTING HELPERS
# ═══════════════════════════════════════════════════════════════

def quadratic(g, a, b):
    """Simple gain-to-nbar diagnostic fit."""
    return a * g ** 2 + b


def safe_nanmax(row):
    """nan-safe max for rows that may contain failed fits."""
    row = np.asarray(row, dtype=float)
    if np.any(np.isfinite(row)):
        return float(np.nanmax(row))
    return np.nan


def savefig_if_requested(fig, filename):
    """Save figure if SAVE_FIGS is true."""
    if SAVE_FIGS:
        path = os.path.join(OUTPUT_DIR, filename)
        fig.savefig(path, bbox_inches="tight")
        print(f"  Saved -> {path}")


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 76)
    print("  CKP nbar Calibration Analysis — paper-style five-parameter fits")
    print("  Sank et al., Phys. Rev. Applied 23, 024055 (2025)")
    print("=" * 76)

    # 1. Load H5 data.
    h5_path = find_h5_file(H5_DIR)
    print(f"\n[1] Loading: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        grp = f[QUBIT_GROUP]
        print(f"    Datasets in {QUBIT_GROUP}: {list(grp.keys())}")

        print("    Parsing I_g ...", end=" ", flush=True)
        I_g = parse_nested_array(
            grp["I_g"][()],
            CFG_GAIN_STEPS,
            CFG_RES_FREQ_STEPS,
            CFG_QU_FREQ_STEPS,
        )
        print(f"shape {I_g.shape}")

        print("    Parsing Q_g ...", end=" ", flush=True)
        Q_g = parse_nested_array(
            grp["Q_g"][()],
            CFG_GAIN_STEPS,
            CFG_RES_FREQ_STEPS,
            CFG_QU_FREQ_STEPS,
        )
        print(f"shape {Q_g.shape}")

        print("    Parsing I_e ...", end=" ", flush=True)
        I_e = parse_nested_array(
            grp["I_e"][()],
            CFG_GAIN_STEPS,
            CFG_RES_FREQ_STEPS,
            CFG_QU_FREQ_STEPS,
        )
        print(f"shape {I_e.shape}")

        print("    Parsing Q_e ...", end=" ", flush=True)
        Q_e = parse_nested_array(
            grp["Q_e"][()],
            CFG_GAIN_STEPS,
            CFG_RES_FREQ_STEPS,
            CFG_QU_FREQ_STEPS,
        )
        print(f"shape {Q_e.shape}")

        try:
            print("    Parsing gain sweep ...", end=" ", flush=True)
            gain_sweep = parse_1d_array(grp["Res Gain Sweep"][()])
            print(f"range [{gain_sweep[0]:.4f}, {gain_sweep[-1]:.4f}]")
        except Exception as exc:
            print(f"FAILED: {exc}. Using config fallback.")
            gain_sweep = np.linspace(
                CFG_GAIN_START,
                CFG_GAIN_END,
                CFG_GAIN_STEPS,
            )

        try:
            print("    Parsing qubit freq sweep ...", end=" ", flush=True)
            qu_freq_sweep = parse_1d_array(grp["Qu Frequency Sweep"][()])
            print(
                f"range [{qu_freq_sweep[0]:.3f}, "
                f"{qu_freq_sweep[-1]:.3f}] MHz"
            )
        except Exception as exc:
            print(f"FAILED: {exc}. Using config fallback.")
            qu_freq_sweep = np.linspace(
                BARE_QUBIT_FREQ + CFG_QU_FREQ_OFFSET_START,
                BARE_QUBIT_FREQ + CFG_QU_FREQ_OFFSET_END,
                CFG_QU_FREQ_STEPS,
            )

    n_gains, n_res, n_qf = I_g.shape
    res_freq_sweep = np.linspace(CFG_RES_FREQ_START, CFG_RES_FREQ_STOP, n_res)

    if len(gain_sweep) != n_gains:
        print(
            "    WARNING: gain_sweep length does not match data shape. "
            "Using evenly spaced fallback gain sweep."
        )
        gain_sweep = np.linspace(CFG_GAIN_START, CFG_GAIN_END, n_gains)

    print(f"\n    Dimensions : {n_gains} gains x {n_res} res_freqs x {n_qf} qubit_freqs")
    print(f"    Gain       : {gain_sweep[0]:.4f} to {gain_sweep[-1]:.4f}")
    print(f"    Res freq   : {res_freq_sweep[0]:.4f} to {res_freq_sweep[-1]:.4f} MHz")
    print(f"    Qubit freq : {qu_freq_sweep[0]:.3f} to {qu_freq_sweep[-1]:.3f} MHz")

    # 2. Extract branch centers.
    print("\n[2] Extracting branch centers from IQ slices ...")

    g_centers = np.zeros((n_gains, n_res), dtype=float)
    e_centers = np.zeros((n_gains, n_res), dtype=float)

    for gi in range(n_gains):
        g_centers[gi] = extract_all_branch_centers(
            I_g[gi],
            Q_g[gi],
            qu_freq_sweep,
        )

        e_centers[gi] = extract_all_branch_centers(
            I_e[gi],
            Q_e[gi],
            qu_freq_sweep,
        )

    print("    Done.")

    # 3. Fit each gain independently using paper-style five-parameter CKP.
    print("\n[3] Fitting paper-style five-parameter CKP model independently at each gain ...")

    fits = fit_ckp_paper_style_all_gains(
        res_freq_sweep,
        gain_sweep,
        g_centers,
        e_centers,
    )

    valid_fits = [f for f in fits if not f.get("failed", False)]

    if not valid_fits:
        raise RuntimeError("All paper-style CKP fits failed.")

    # Choose a representative fit.
    # Since the paper uses one power level, and your dataset has many gains,
    # this selects the largest fitted A2 as the reference diagnostic.
    ref_fit = max(valid_fits, key=lambda f: f["A2"])

    best_gi = ref_fit["gain_index"]
    ref_gain = ref_fit["gain"]

    omega_q0_ref = ref_fit["omega_q0"]
    omega_rm_ref = ref_fit["omega_rm"]
    chi_ref = ref_fit["chi_mag"]
    kappa_ref = ref_fit["kappa"]
    A2_ref = ref_fit["A2"]
    center_g_ref = ref_fit["omega_r_g"]
    center_e_ref = ref_fit["omega_r_e"]

    print("\n    Reference paper-style CKP fit")
    print(f"    Gain index             : {best_gi}")
    print(f"    Gain                   : {ref_gain:.4f}")
    print(f"    RMSE                   : {ref_fit['rmse']:.4f} MHz")
    print(f"    omega_q0               : {omega_q0_ref:.4f} MHz")
    print(f"    omega_rm               : {omega_rm_ref:.4f} MHz")
    print(f"    omega_r,|0>            : {center_g_ref:.4f} MHz")
    print(f"    omega_r,|1>            : {center_e_ref:.4f} MHz")
    print(f"    2|chi|                 : {2.0 * chi_ref:.4f} MHz")
    print(f"    chi paper signed       : {-chi_ref:.4f} MHz")
    print(f"    kappa                  : {kappa_ref:.4f} MHz")
    print(f"    |A|^2                  : {A2_ref:.6g}")

    # 4. Collect results into arrays.
    print("\n[4] Computing nbar from Eq. (6) for each independent fit ...")

    nbar_g = np.full((n_gains, n_res), np.nan)
    nbar_e = np.full((n_gains, n_res), np.nan)
    model_g = np.full((n_gains, n_res), np.nan)
    model_e = np.full((n_gains, n_res), np.nan)

    omega_q0_by_gain = np.full(n_gains, np.nan)
    omega_rm_by_gain = np.full(n_gains, np.nan)
    chi_mag_by_gain = np.full(n_gains, np.nan)
    kappa_by_gain = np.full(n_gains, np.nan)
    A2_by_gain = np.full(n_gains, np.nan)
    omega_r_g_by_gain = np.full(n_gains, np.nan)
    omega_r_e_by_gain = np.full(n_gains, np.nan)
    rmse_by_gain = np.full(n_gains, np.nan)

    for fit in valid_fits:
        gi = fit["gain_index"]

        nbar_g[gi] = fit["nbar_g"]
        nbar_e[gi] = fit["nbar_e"]
        model_g[gi] = fit["model_g"]
        model_e[gi] = fit["model_e"]

        omega_q0_by_gain[gi] = fit["omega_q0"]
        omega_rm_by_gain[gi] = fit["omega_rm"]
        chi_mag_by_gain[gi] = fit["chi_mag"]
        kappa_by_gain[gi] = fit["kappa"]
        A2_by_gain[gi] = fit["A2"]
        omega_r_g_by_gain[gi] = fit["omega_r_g"]
        omega_r_e_by_gain[gi] = fit["omega_r_e"]
        rmse_by_gain[gi] = fit["rmse"]

    nbar_peak_g = np.array([safe_nanmax(nbar_g[gi]) for gi in range(n_gains)])
    nbar_peak_e = np.array([safe_nanmax(nbar_e[gi]) for gi in range(n_gains)])
    nbar_peak_avg = 0.5 * (nbar_peak_g + nbar_peak_e)

    idx_bare = int(np.argmin(np.abs(res_freq_sweep - RES_FREQ_ON_RESONANCE)))
    nbar_bare_g = nbar_g[:, idx_bare]
    nbar_bare_e = nbar_e[:, idx_bare]
    nbar_bare_avg = 0.5 * (nbar_bare_g + nbar_bare_e)

    print(
        f"    Bare/reference frequency index: {idx_bare} "
        f"(f_d = {res_freq_sweep[idx_bare]:.4f} MHz)"
    )

    # Stark-shift back-out diagnostic using the reference fit parameters.
    # This is not the main paper-style output. It is just a sanity check.
    nbar_stark_g_ref = np.maximum(omega_q0_ref - g_centers, 0.0) / (2.0 * chi_ref)
    nbar_stark_e_ref = np.maximum(omega_q0_ref - e_centers, 0.0) / (2.0 * chi_ref)

    # 5. Print summary table.
    print("\n" + "-" * 116)
    print(
        f"  {'Gain':>8s}   {'|chi| MHz':>10s}   {'kappa MHz':>10s}   "
        f"{'|A|^2':>12s}   {'peak nbar |0>':>14s}   "
        f"{'peak nbar |1>':>14s}   {'RMSE MHz':>10s}"
    )
    print("-" * 116)

    for i, gain in enumerate(gain_sweep):
        print(
            f"  {gain:8.4f}   "
            f"{chi_mag_by_gain[i]:10.5f}   "
            f"{kappa_by_gain[i]:10.5f}   "
            f"{A2_by_gain[i]:12.6g}   "
            f"{nbar_peak_g[i]:14.4f}   "
            f"{nbar_peak_e[i]:14.4f}   "
            f"{rmse_by_gain[i]:10.4f}"
        )

    print("-" * 116)

    # 6. Plots.
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "figure.dpi": 120,
        "savefig.dpi": 200,
    })

    # Plot 1: peak nbar vs gain.
    fig1, ax1 = plt.subplots(figsize=(8.8, 5.0))

    ax1.scatter(gain_sweep, nbar_peak_g, s=70, label=r"$|0\rangle$ branch peak")
    ax1.scatter(gain_sweep, nbar_peak_e, s=55, marker="s", label=r"$|1\rangle$ branch peak")
    ax1.plot(gain_sweep, nbar_peak_avg, "--", alpha=0.7, label="branch average")

    good = np.isfinite(nbar_peak_avg)

    if np.count_nonzero(good) >= 3:
        try:
            popt_q, _ = curve_fit(
                quadratic,
                gain_sweep[good],
                nbar_peak_avg[good],
                p0=[100.0, 0.0],
            )

            fit_g_fine = np.linspace(
                np.nanmin(gain_sweep),
                np.nanmax(gain_sweep),
                300,
            )

            fit_nbar = quadratic(fit_g_fine, *popt_q)

            ax1.plot(
                fit_g_fine,
                fit_nbar,
                ":",
                lw=2,
                label=(
                    rf"fit: $\bar{{n}} = {popt_q[0]:.0f}\,g^2"
                    rf" {'+' if popt_q[1] >= 0 else ''}{popt_q[1]:.2f}$"
                ),
            )

            print(
                f"    Peak nbar quadratic diagnostic: "
                f"nbar ≈ {popt_q[0]:.1f} g^2 + {popt_q[1]:.3f}"
            )

        except Exception as exc:
            print(f"    Peak nbar quadratic fit failed: {exc}")

    txt = (
        "Reference five-parameter CKP fit\n"
        rf"gain = {ref_gain:.4f}" "\n"
        rf"$|\chi|$ = {chi_ref:.4f} MHz" "\n"
        rf"$\kappa$ = {kappa_ref:.4f} MHz" "\n"
        rf"$\omega_{{r,m}}$ = {omega_rm_ref:.4f} MHz" "\n"
        rf"$\omega_{{q,0}}$ = {omega_q0_ref:.4f} MHz"
    )

    ax1.text(
        0.03,
        0.97,
        txt,
        transform=ax1.transAxes,
        va="top",
        bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85),
    )

    ax1.set_xlabel("Resonator drive gain [DAC units]")
    ax1.set_ylabel(r"peak $\bar{n}$")
    ax1.set_title(r"Paper-style CKP: peak $\bar{n}$ from independent five-parameter fits")
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(left=-0.002)
    ax1.set_ylim(bottom=0)
    ax1.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=10,
        framealpha=0.9,
    )

    fig1.tight_layout(rect=[0, 0, 0.80, 1])
    savefig_if_requested(fig1, "ckp_paperstyle_peak_nbar_vs_gain.png")

    # Plot 1b: nbar sampled at manually supplied bare/reference frequency.
    fig1b, ax1b = plt.subplots(figsize=(8.8, 5.0))

    ax1b.scatter(
        gain_sweep,
        nbar_bare_g,
        s=70,
        label=rf"$|0\rangle$ at {RES_FREQ_ON_RESONANCE:.3f} MHz",
    )

    ax1b.scatter(
        gain_sweep,
        nbar_bare_e,
        s=55,
        marker="s",
        label=rf"$|1\rangle$ at {RES_FREQ_ON_RESONANCE:.3f} MHz",
    )

    ax1b.plot(gain_sweep, nbar_bare_avg, "--", alpha=0.7, label="branch average")

    ax1b.set_xlabel("Resonator drive gain [DAC units]")
    ax1b.set_ylabel(r"$\bar{n}$")
    ax1b.set_title(
        rf"Paper-style CKP: $\bar{{n}}$ sampled at reference frequency "
        rf"$f_d$ = {RES_FREQ_ON_RESONANCE:.3f} MHz"
    )
    ax1b.grid(True, alpha=0.25)
    ax1b.set_xlim(left=-0.002)
    ax1b.set_ylim(bottom=0)
    ax1b.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1.0),
        borderaxespad=0.0,
        fontsize=10,
        framealpha=0.9,
    )

    fig1b.tight_layout(rect=[0, 0, 0.80, 1])
    savefig_if_requested(fig1b, "ckp_paperstyle_referencefreq_nbar_vs_gain.png")

    # Plot 2: heatmap for |0> branch.
    sort_idx = np.argsort(res_freq_sweep)
    rf_sorted = res_freq_sweep[sort_idx]

    fig2, ax2 = plt.subplots(figsize=(9, 5.5))

    nbar_g_sorted = nbar_g[:, sort_idx].T

    im = ax2.pcolormesh(
        gain_sweep,
        rf_sorted,
        nbar_g_sorted,
        shading="nearest",
        cmap="inferno",
    )

    cbar = fig2.colorbar(im, ax=ax2, pad=0.02)
    cbar.set_label(r"$\bar{n}$ (photon number)", fontsize=12)

    ax2.axhline(
        RES_FREQ_ON_RESONANCE,
        ls="--",
        lw=1.4,
        alpha=0.85,
        label=f"reference freq ({RES_FREQ_ON_RESONANCE:.3f} MHz)",
    )

    ax2.plot(
        gain_sweep,
        omega_r_g_by_gain,
        ".",
        ms=5,
        label=rf"fitted $\omega_{{r,|0\rangle}}$",
    )

    ax2.set_xlabel("Resonator drive gain [DAC units]")
    ax2.set_ylabel("Resonator drive frequency [MHz]")
    ax2.set_title(r"Paper-style CKP $\bar{n}$ heatmap, $|0\rangle$ branch")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.85)

    fig2.tight_layout()
    savefig_if_requested(fig2, "ckp_paperstyle_nbar_heatmap_0branch.png")

    # Plot 2b: heatmap for |1> branch.
    fig2b, ax2b = plt.subplots(figsize=(9, 5.5))

    nbar_e_sorted = nbar_e[:, sort_idx].T

    im = ax2b.pcolormesh(
        gain_sweep,
        rf_sorted,
        nbar_e_sorted,
        shading="nearest",
        cmap="inferno",
    )

    cbar = fig2b.colorbar(im, ax=ax2b, pad=0.02)
    cbar.set_label(r"$\bar{n}$ (photon number)", fontsize=12)

    ax2b.axhline(
        RES_FREQ_ON_RESONANCE,
        ls="--",
        lw=1.4,
        alpha=0.85,
        label=f"reference freq ({RES_FREQ_ON_RESONANCE:.3f} MHz)",
    )

    ax2b.plot(
        gain_sweep,
        omega_r_e_by_gain,
        ".",
        ms=5,
        label=rf"fitted $\omega_{{r,|1\rangle}}$",
    )

    ax2b.set_xlabel("Resonator drive gain [DAC units]")
    ax2b.set_ylabel("Resonator drive frequency [MHz]")
    ax2b.set_title(r"Paper-style CKP $\bar{n}$ heatmap, $|1\rangle$ branch")
    ax2b.legend(loc="upper left", fontsize=9, framealpha=0.85)

    fig2b.tight_layout()
    savefig_if_requested(fig2b, "ckp_paperstyle_nbar_heatmap_1branch.png")

    # Plot 3: diagnostic fit at reference gain.
    fig3, axes3 = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        gridspec_kw={"width_ratios": [1.5, 1.0]},
    )

    ax = axes3[0]

    ax.plot(
        res_freq_sweep,
        g_centers[best_gi],
        "o",
        ms=5,
        label=r"$|0\rangle$ extracted centers",
    )

    ax.plot(
        res_freq_sweep,
        e_centers[best_gi],
        "s",
        ms=5,
        label=r"$|1\rangle$ extracted centers",
    )

    ax.plot(
        res_freq_sweep,
        ref_fit["model_g"],
        "-",
        lw=2,
        label=r"five-parameter fit: $|0\rangle$",
    )

    ax.plot(
        res_freq_sweep,
        ref_fit["model_e"],
        "-",
        lw=2,
        label=r"five-parameter fit: $|1\rangle$",
    )

    ax.axhline(
        omega_q0_ref,
        ls=":",
        lw=1,
        alpha=0.6,
        label=rf"fitted $\omega_q$ = {omega_q0_ref:.3f} MHz",
    )

    ax.axvline(
        center_g_ref,
        ls=":",
        lw=1,
        alpha=0.7,
        label=rf"$\omega_{{r,|0\rangle}}$ = {center_g_ref:.3f} MHz",
    )

    ax.axvline(
        center_e_ref,
        ls=":",
        lw=1,
        alpha=0.7,
        label=rf"$\omega_{{r,|1\rangle}}$ = {center_e_ref:.3f} MHz",
    )

    ax.set_xlabel("Resonator drive frequency [MHz]")
    ax.set_ylabel("Qubit resonance frequency [MHz]")
    ax.set_title(f"Paper-style CKP fit at reference gain = {ref_gain:.4f}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8, loc="best")

    ax = axes3[1]
    ax.axis("off")

    lines = [
        r"$\bf{Paper\ style\ five\ parameter\ CKP\ fit}$",
        "",
        rf"gain index = {best_gi}",
        rf"gain = {ref_gain:.4f}",
        "",
        rf"$\omega_{{q,0}}$ = {omega_q0_ref:.4f} MHz",
        rf"$\omega_{{r,m}}$ = {omega_rm_ref:.4f} MHz",
        rf"$|\chi|$ = {chi_ref:.4f} MHz ({chi_ref * 1e3:.1f} kHz)",
        rf"signed paper $\chi$ = {-chi_ref:.4f} MHz",
        rf"$\kappa$ = {kappa_ref:.4f} MHz ({kappa_ref * 1e3:.1f} kHz)",
        rf"$|A|^2$ = {A2_ref:.6g}",
        "",
        rf"$\omega_{{r,|0\rangle}}$ = {center_g_ref:.4f} MHz",
        rf"$\omega_{{r,|1\rangle}}$ = {center_e_ref:.4f} MHz",
        rf"RMSE = {ref_fit['rmse']:.4f} MHz",
        "",
        rf"$\bar{{n}}_{{peak,|0\rangle}}$ = {np.nanmax(ref_fit['nbar_g']):.2f}",
        rf"$\bar{{n}}_{{peak,|1\rangle}}$ = {np.nanmax(ref_fit['nbar_e']):.2f}",
        "",
        rf"Config $|\chi|$ reference = {abs(CHI_CONFIG):.4f} MHz",
    ]

    ax.text(
        0.05,
        0.95,
        "\n".join(lines),
        transform=ax.transAxes,
        fontsize=11,
        va="top",
        fontfamily="monospace",
        bbox=dict(
            boxstyle="round,pad=0.5",
            fc="lightyellow",
            ec="gray",
            alpha=0.9,
        ),
    )

    fig3.tight_layout()
    savefig_if_requested(fig3, "ckp_paperstyle_reference_fit_diagnostic.png")

    # Plot 4: fitted parameters versus gain.
    fig4, ax4 = plt.subplots(figsize=(8.5, 5.0))

    ax4.plot(gain_sweep, chi_mag_by_gain, "o-", label=r"$|\chi|$")
    ax4.plot(gain_sweep, kappa_by_gain, "s-", label=r"$\kappa$")

    ax4.axhline(
        abs(CHI_CONFIG),
        ls=":",
        lw=1.2,
        alpha=0.8,
        label=rf"config $|\chi|$ = {abs(CHI_CONFIG):.4f} MHz",
    )

    ax4.set_xlabel("Resonator drive gain [DAC units]")
    ax4.set_ylabel("Frequency [MHz]")
    ax4.set_title(r"Paper-style CKP fitted $|\chi|$ and $\kappa$ versus gain")
    ax4.grid(True, alpha=0.25)
    ax4.legend(fontsize=9)

    fig4.tight_layout()
    savefig_if_requested(fig4, "ckp_paperstyle_chi_kappa_vs_gain.png")

    # Plot 5: RMSE versus gain.
    fig5, ax5 = plt.subplots(figsize=(8.5, 5.0))

    ax5.plot(gain_sweep, rmse_by_gain, "o-")
    ax5.set_xlabel("Resonator drive gain [DAC units]")
    ax5.set_ylabel("Fit RMSE [MHz]")
    ax5.set_title("Paper-style CKP fit residual versus gain")
    ax5.grid(True, alpha=0.25)

    fig5.tight_layout()
    savefig_if_requested(fig5, "ckp_paperstyle_rmse_vs_gain.png")

    # Plot 6: cross-check at reference frequency.
    fig6, ax6 = plt.subplots(figsize=(8.5, 5.0))

    ax6.plot(
        gain_sweep,
        nbar_stark_g_ref[:, idx_bare],
        "o",
        ms=5,
        label=r"Stark back-out, $|0\rangle$",
    )

    ax6.plot(
        gain_sweep,
        nbar_bare_g,
        "-",
        lw=2,
        label=r"Eq. (6), $|0\rangle$",
    )

    ax6.plot(
        gain_sweep,
        nbar_stark_e_ref[:, idx_bare],
        "s",
        ms=5,
        label=r"Stark back-out, $|1\rangle$",
    )

    ax6.plot(
        gain_sweep,
        nbar_bare_e,
        "-",
        lw=2,
        label=r"Eq. (6), $|1\rangle$",
    )

    ax6.set_xlabel("Resonator drive gain [DAC units]")
    ax6.set_ylabel(r"$\bar{n}$")
    ax6.set_title(
        rf"Cross-check at reference frequency "
        rf"$f_d$ = {RES_FREQ_ON_RESONANCE:.3f} MHz"
    )
    ax6.grid(True, alpha=0.25)
    ax6.legend(fontsize=9)

    fig6.tight_layout()
    savefig_if_requested(fig6, "ckp_paperstyle_nbar_crosscheck_referencefreq.png")

    if SHOW_FIGS:
        plt.show()

    # 7. Save results.
    npz_path = os.path.join(OUTPUT_DIR, "ckp_paperstyle_results.npz")

    np.savez(
        npz_path,

        gain_sweep=gain_sweep,
        res_freq_sweep=res_freq_sweep,
        qu_freq_sweep=qu_freq_sweep,

        omega_q0_by_gain=omega_q0_by_gain,
        omega_rm_by_gain=omega_rm_by_gain,
        chi_mag_by_gain=chi_mag_by_gain,
        chi_paper_signed_by_gain=-chi_mag_by_gain,
        kappa_by_gain=kappa_by_gain,
        A2_by_gain=A2_by_gain,
        omega_r_g_by_gain=omega_r_g_by_gain,
        omega_r_e_by_gain=omega_r_e_by_gain,
        rmse_by_gain=rmse_by_gain,

        nbar_g=nbar_g,
        nbar_e=nbar_e,
        nbar_peak_g=nbar_peak_g,
        nbar_peak_e=nbar_peak_e,
        nbar_bare_g=nbar_bare_g,
        nbar_bare_e=nbar_bare_e,

        g_centers=g_centers,
        e_centers=e_centers,
        model_g=model_g,
        model_e=model_e,

        reference_gain_index=best_gi,
        reference_gain=ref_gain,
        reference_omega_q0=omega_q0_ref,
        reference_omega_rm=omega_rm_ref,
        reference_chi_mag=chi_ref,
        reference_chi_paper_signed=-chi_ref,
        reference_kappa=kappa_ref,
        reference_A2=A2_ref,
        reference_omega_r_g=center_g_ref,
        reference_omega_r_e=center_e_ref,
        reference_rmse=ref_fit["rmse"],
    )

    print(f"\n  Paper-style results saved -> {npz_path}")

    print("\nAnalysis complete.")

    return {
        "fits": fits,
        "valid_fits": valid_fits,
        "gain_sweep": gain_sweep,
        "res_freq_sweep": res_freq_sweep,
        "qu_freq_sweep": qu_freq_sweep,
        "g_centers": g_centers,
        "e_centers": e_centers,
        "nbar_g": nbar_g,
        "nbar_e": nbar_e,
        "nbar_peak_g": nbar_peak_g,
        "nbar_peak_e": nbar_peak_e,
        "omega_q0_by_gain": omega_q0_by_gain,
        "omega_rm_by_gain": omega_rm_by_gain,
        "chi_mag_by_gain": chi_mag_by_gain,
        "kappa_by_gain": kappa_by_gain,
        "A2_by_gain": A2_by_gain,
        "reference_fit": ref_fit,
    }


if __name__ == "__main__":
    results = main()