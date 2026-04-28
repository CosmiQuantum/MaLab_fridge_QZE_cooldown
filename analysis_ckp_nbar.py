"""
analysis_ckp_nbar_full_ckp.py — CKP analysis using the paper's coupled model
===============================================================================
Based on: Sank et al., Phys. Rev. Applied 23, 024055 (2025)
          "System Characterization of Dispersive Readout in
           Superconducting Qubits"

What changed relative to the earlier script
-------------------------------------------
1. The two CKP branches are fit *simultaneously* with a shared model instead of
   fitting two independent Lorentzians and subtracting the centers afterward.
2. All gains are fit in one global nonlinear fit with shared physical parameters
   (omega_q0, omega_r,m, chi, kappa) and one drive-power parameter |A|^2 for
   each gain setting.
3. Photon number is computed from Eq. (6) of the paper using the fitted |A|^2
   and kappa, rather than only from |Δf_q|/(2|chi|).

Model summary
-------------
The paper's CKP model [Eq. (7)] is

    omega_d,q^*(omega_d,r)
      = omega_q,0 + 2 chi * kappa |A|^2 /
        [ (omega_d,r - (omega_r,m ∓ chi))^2 + (kappa/2)^2 ]

with upper/lower signs for the qubit prepared in |0> / |1> and chi < 0 in the
paper's convention. In this script we fit chi_mag = |chi| > 0 and write the
same model as a downward Lorentzian pair:

    y_|0>(x) = omega_q0 - 2 chi_mag kappa A2 /
               [ (x - (omega_rm + chi_mag))^2 + (kappa/2)^2 ]
    y_|1>(x) = omega_q0 - 2 chi_mag kappa A2 /
               [ (x - (omega_rm - chi_mag))^2 + (kappa/2)^2 ]

where A2 = |A|^2.
"""

import glob
import os
import re

import h5py
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# ═══════════════════════════════════════════════════════════════
#  USER CONFIGURATION — edit paths and parameters here
# ═══════════════════════════════════════════════════════════════

# Path to the directory containing the CKP h5 file
H5_DIR = (r"M:\_Data\20250822 - Olivia\bob_run_started_Feb_11\squill"
          r"\ckp_nbar_calibration\q5\2026-04-22_14-32-53"
          r"\study_data\Data_h5\ckp_calibration")

QUBIT_GROUP = "Q6"  # QubitIndex=5 → h5 group "Q6"

# ----- System parameters -----
RES_FREQ_ON_RESONANCE = 7287.570   # MHz, bare resonator frequency
BARE_QUBIT_FREQ       = 3095.45    # MHz, bare qubit g-e frequency

# ----- Config values for sweep reconstruction (fallbacks) -----
CFG_GAIN_START     = 0.0
CFG_GAIN_END       = 0.15
CFG_GAIN_STEPS     = 20

CFG_RES_FREQ_START = 7287.57 + 1   # 7288.57 MHz
CFG_RES_FREQ_STOP  = 7287.57 - 1   # 7286.57 MHz
CFG_RES_FREQ_STEPS = 50

CFG_QU_FREQ_OFFSET_START = -75     # MHz from bare qubit freq
CFG_QU_FREQ_OFFSET_END   = 5       # MHz from bare qubit freq
CFG_QU_FREQ_STEPS        = 100

# Known χ from previous measurement (for comparison only)
CHI_CONFIG = -0.234 / 2  # MHz  → −0.117 MHz

# ----- Output -----
SAVE_FIGS = True
SHOW_FIGS = True
run_name = 'bob_run_started_Feb_11'
device_name = 'squill'
substudy_txt_notes = 'ckp full-model fit'

study = 'ckp_nbar_calibration'
sub_study = 'q5'
data_set = '2026-04-22_14-32-53'

run_flags = {
    "tof": False,
    "res_spec": True,
    "q_spec": True,
    "ss": True,
    "rabi": True,
    "t1": True,
}

if not os.path.exists(f"M:/_Data/20250822 - Olivia/{run_name}/"):
    os.makedirs(f"M:/_Data/20250822 - Olivia/{run_name}/")
if not os.path.exists(f"M:/_Data/20250822 - Olivia/{run_name}/{device_name}/"):
    os.makedirs(f"M:/_Data/20250822 - Olivia/{run_name}/{device_name}/")
studyFolder = os.path.join(f"M:/_Data/20250822 - Olivia/{run_name}/{device_name}/", study)
if not os.path.exists(studyFolder):
    os.makedirs(studyFolder)
subStudyFolder = os.path.join(studyFolder, sub_study)
if not os.path.exists(subStudyFolder):
    os.makedirs(subStudyFolder)

dataSetFolder = os.path.join(subStudyFolder, data_set)
optimizationFolder = os.path.join(dataSetFolder, 'optimization')
studyFolder = os.path.join(dataSetFolder, 'study_data')
studyDocumentationFolder = os.path.join(dataSetFolder, 'documentation')
subStudyDataFolder = os.path.join(dataSetFolder, 'study_data')
if not os.path.exists(studyDocumentationFolder):
    os.makedirs(studyDocumentationFolder)
if not os.path.exists(optimizationFolder):
    os.makedirs(optimizationFolder)
if not os.path.exists(subStudyDataFolder):
    os.makedirs(subStudyDataFolder)

OUTPUT_DIR = studyDocumentationFolder

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
    s = _bytes_to_str(raw).strip()
    inner = s.strip("[]").strip()
    inner = re.sub(r"\s+", " ", inner)
    return np.array([float(v) for v in inner.split() if v])


def parse_nested_array(raw, n_outer=None, n_mid=None, n_inner=None):
    s = _bytes_to_str(raw).strip()

    cleaned = s.replace("array(", "").replace(")", "")
    cleaned = re.sub(r"\s+", " ", cleaned)
    try:
        data = eval(cleaned, {"__builtins__": {}, "nan": float("nan"), "inf": float("inf")})
        return np.asarray(data, dtype=np.float64)
    except Exception as e1:
        print(f"    parse_nested (eval-cleaned) failed: {e1}")

    try:
        data = eval(s, {
            "__builtins__": {},
            "array": np.array,
            "nan": float("nan"),
            "inf": float("inf"),
            "float64": np.float64,
        })
        return np.asarray(data, dtype=np.float64)
    except Exception as e2:
        print(f"    parse_nested (eval-array) failed: {e2}")

    if n_outer and n_mid and n_inner:
        try:
            blocks = re.findall(r"\[([\d\s.eE+\-,]+?)\]", s)
            arrays = []
            for blk in blocks:
                vals = [float(x) for x in re.split(r"[,\s]+", blk.strip()) if x]
                if len(vals) == n_inner:
                    arrays.append(vals)
            if len(arrays) == n_outer * n_mid:
                return np.array(arrays, dtype=np.float64).reshape(n_outer, n_mid, n_inner)
        except Exception as e3:
            print(f"    parse_nested (regex) failed: {e3}")

    raise ValueError("Could not parse nested array from H5. "
                     f"First 200 chars: {s[:200]}")


# ═══════════════════════════════════════════════════════════════
#  PHYSICS / ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def iq_distance_from_baseline(I_row, Q_row, n_baseline=10):
    """Compute IQ Euclidean distance from the off-resonance baseline."""
    I_base = np.mean(I_row[-n_baseline:])
    Q_base = np.mean(Q_row[-n_baseline:])
    return np.sqrt((I_row - I_base) ** 2 + (Q_row - Q_base) ** 2)



def find_peak_center(dist, freq_sweep):
    """Locate the spectroscopy peak via argmax + sub-bin quadratic interpolation."""
    x = np.asarray(freq_sweep, dtype=float)
    idx = int(np.argmax(dist))

    if 1 <= idx <= len(x) - 2:
        xs = x[idx - 1: idx + 2]
        ys = dist[idx - 1: idx + 2]
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
    """Extract one qubit-resonance estimate for each resonator-drive frequency."""
    n_res = I.shape[0]
    centers = np.empty(n_res)
    for ri in range(n_res):
        dist = iq_distance_from_baseline(I[ri], Q[ri], n_baseline)
        centers[ri] = find_peak_center(dist, qu_freq_sweep)
    return centers



def ckp_center_model(x, omega_q0, omega_rm, chi_mag, kappa, A2, branch_sign):
    """
    Coupled CKP center model based on Eq. (7) of the paper.

    branch_sign:
        +1 for |0> preparation  -> center = omega_rm + chi_mag
        -1 for |1> preparation  -> center = omega_rm - chi_mag
    """
    denom_center = omega_rm + branch_sign * chi_mag
    denom = (x - denom_center) ** 2 + (kappa / 2.0) ** 2
    numerator = 2.0 * chi_mag * kappa * A2
    return omega_q0 - numerator / denom



def build_global_fit_arrays(res_freq_sweep, g_centers, e_centers):
    """Build concatenated x/y arrays for a shared fit over all gains and both branches."""
    n_gains, n_res = g_centers.shape
    rows = []
    yvals = []

    for gi in range(n_gains):
        for ri, x in enumerate(res_freq_sweep):
            rows.append((x, gi, +1))
            yvals.append(g_centers[gi, ri])
        for ri, x in enumerate(res_freq_sweep):
            rows.append((x, gi, -1))
            yvals.append(e_centers[gi, ri])

    xdata = np.asarray(rows, dtype=float).T  # shape (3, N)
    ydata = np.asarray(yvals, dtype=float)
    return xdata, ydata



def ckp_global_model(xdata, omega_q0, omega_rm, chi_mag, kappa, *A2_by_gain):
    """Global CKP model with shared omega_q0/omega_rm/chi/kappa and one A2 per gain."""
    x = xdata[0]
    gain_idx = xdata[1].astype(int)
    branch_sign = xdata[2]

    A2_by_gain = np.asarray(A2_by_gain, dtype=float)
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):
        out[i] = ckp_center_model(
            x[i],
            omega_q0=omega_q0,
            omega_rm=omega_rm,
            chi_mag=chi_mag,
            kappa=kappa,
            A2=A2_by_gain[gain_idx[i]],
            branch_sign=branch_sign[i],
        )
    return out



def estimate_initial_params(res_freq_sweep, gain_sweep, g_centers, e_centers):
    """Reasonable start values for the global nonlinear fit."""
    avg_g = np.mean(g_centers, axis=0)
    avg_e = np.mean(e_centers, axis=0)

    x0_g = res_freq_sweep[np.argmin(avg_g)]
    x0_e = res_freq_sweep[np.argmin(avg_e)]
    omega_rm0 = 0.5 * (x0_g + x0_e)
    chi0 = max(abs(x0_g - x0_e) / 2.0, max(abs(CHI_CONFIG), 0.01))
    omega_q0 = np.median(np.r_[g_centers[:, [0, -1]].ravel(), e_centers[:, [0, -1]].ravel()])
    kappa0 = max(np.ptp(res_freq_sweep) / 6.0, 0.02)

    A2_guesses = []
    for gi in range(len(gain_sweep)):
        depth_g = max(omega_q0 - np.min(g_centers[gi]), 0.0)
        depth_e = max(omega_q0 - np.min(e_centers[gi]), 0.0)
        depth = max(0.5 * (depth_g + depth_e), 1e-6)
        A2_est = depth * (kappa0 / 2.0) / (8.0 * chi0)
        A2_guesses.append(max(A2_est, 1e-8))

    return omega_q0, omega_rm0, chi0, kappa0, np.asarray(A2_guesses)



def fit_ckp_global(res_freq_sweep, gain_sweep, g_centers, e_centers):
    """Fit the coupled CKP model to all gains and both branches simultaneously."""
    n_gains = len(gain_sweep)
    xdata, ydata = build_global_fit_arrays(res_freq_sweep, g_centers, e_centers)

    omega_q0_0, omega_rm0, chi0, kappa0, A2_0 = estimate_initial_params(
        res_freq_sweep, gain_sweep, g_centers, e_centers
    )

    p0 = np.r_[omega_q0_0, omega_rm0, chi0, kappa0, A2_0]

    y_min = float(np.min(ydata))
    y_max = float(np.max(ydata))
    x_min = float(np.min(res_freq_sweep))
    x_max = float(np.max(res_freq_sweep))
    x_span = x_max - x_min

    lb = np.r_[y_min - 10.0, x_min - 2.0, 0.001, 0.001, np.zeros(n_gains)]
    ub = np.r_[y_max + 10.0, x_max + 2.0, x_span, x_span, np.full(n_gains, np.inf)]

    popt, pcov = curve_fit(
        ckp_global_model,
        xdata,
        ydata,
        p0=p0,
        bounds=(lb, ub),
        maxfev=200000,
    )

    perr = np.sqrt(np.diag(pcov))
    yfit = ckp_global_model(xdata, *popt)
    residuals = ydata - yfit
    rmse = float(np.sqrt(np.mean(residuals ** 2)))

    omega_q0, omega_rm, chi_mag, kappa = popt[:4]
    A2_by_gain = np.asarray(popt[4:])

    model_g = np.vstack([
        ckp_center_model(res_freq_sweep, omega_q0, omega_rm, chi_mag, kappa, A2_by_gain[gi], +1)
        for gi in range(n_gains)
    ])
    model_e = np.vstack([
        ckp_center_model(res_freq_sweep, omega_q0, omega_rm, chi_mag, kappa, A2_by_gain[gi], -1)
        for gi in range(n_gains)
    ])

    rms_by_gain = np.sqrt(np.mean((g_centers - model_g) ** 2 + (e_centers - model_e) ** 2, axis=1) / 2.0)
    depth_by_gain = omega_q0 - 0.5 * (
        np.min(model_g, axis=1) + np.min(model_e, axis=1)
    )

    return {
        "popt": popt,
        "perr": perr,
        "pcov": pcov,
        "omega_q0": omega_q0,
        "omega_rm": omega_rm,
        "chi": chi_mag,
        "kappa": kappa,
        "A2_by_gain": A2_by_gain,
        "yfit": yfit,
        "residuals": residuals,
        "rmse": rmse,
        "model_g": model_g,
        "model_e": model_e,
        "rms_by_gain": rms_by_gain,
        "depth_by_gain": depth_by_gain,
    }



def photon_number_from_fit(res_freq_sweep, omega_rm, chi_mag, kappa, A2_by_gain):
    """Eq. (6): nbar = kappa |A|^2 / [Delta^2 + (kappa/2)^2]."""
    n_gains = len(A2_by_gain)
    n_res = len(res_freq_sweep)
    nbar_g = np.zeros((n_gains, n_res), dtype=float)
    nbar_e = np.zeros((n_gains, n_res), dtype=float)

    center_g = omega_rm + chi_mag
    center_e = omega_rm - chi_mag

    for gi, A2 in enumerate(A2_by_gain):
        nbar_g[gi] = kappa * A2 / ((res_freq_sweep - center_g) ** 2 + (kappa / 2.0) ** 2)
        nbar_e[gi] = kappa * A2 / ((res_freq_sweep - center_e) ** 2 + (kappa / 2.0) ** 2)

    return nbar_g, nbar_e


# ═══════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("  CKP n̄ Calibration Analysis — full coupled CKP model")
    print("  Sank et al., Phys. Rev. Applied 23, 024055 (2025)")
    print("=" * 70)

    # 1. Load H5 data
    h5_path = find_h5_file(H5_DIR)
    print(f"\n[1] Loading: {h5_path}")

    with h5py.File(h5_path, "r") as f:
        grp = f[QUBIT_GROUP]
        print(f"    Datasets in {QUBIT_GROUP}: {list(grp.keys())}")

        print("    Parsing I_g …", end=" ", flush=True)
        I_g = parse_nested_array(grp["I_g"][()], CFG_GAIN_STEPS, CFG_RES_FREQ_STEPS, CFG_QU_FREQ_STEPS)
        print(f"shape {I_g.shape}")

        print("    Parsing Q_g …", end=" ", flush=True)
        Q_g = parse_nested_array(grp["Q_g"][()], CFG_GAIN_STEPS, CFG_RES_FREQ_STEPS, CFG_QU_FREQ_STEPS)
        print(f"shape {Q_g.shape}")

        print("    Parsing I_e …", end=" ", flush=True)
        I_e = parse_nested_array(grp["I_e"][()], CFG_GAIN_STEPS, CFG_RES_FREQ_STEPS, CFG_QU_FREQ_STEPS)
        print(f"shape {I_e.shape}")

        print("    Parsing Q_e …", end=" ", flush=True)
        Q_e = parse_nested_array(grp["Q_e"][()], CFG_GAIN_STEPS, CFG_RES_FREQ_STEPS, CFG_QU_FREQ_STEPS)
        print(f"shape {Q_e.shape}")

        try:
            print("    Parsing gain sweep …", end=" ", flush=True)
            gain_sweep = parse_1d_array(grp["Res Gain Sweep"][()])
            print(f"range [{gain_sweep[0]:.4f}, {gain_sweep[-1]:.4f}]")
        except Exception:
            print("FAILED — using config fallback")
            gain_sweep = np.linspace(CFG_GAIN_START, CFG_GAIN_END, CFG_GAIN_STEPS)

        try:
            print("    Parsing qubit freq sweep …", end=" ", flush=True)
            qu_freq_sweep = parse_1d_array(grp["Qu Frequency Sweep"][()])
            print(f"range [{qu_freq_sweep[0]:.3f}, {qu_freq_sweep[-1]:.3f}] MHz")
        except Exception:
            print("FAILED — using config fallback")
            qu_freq_sweep = np.linspace(
                BARE_QUBIT_FREQ + CFG_QU_FREQ_OFFSET_START,
                BARE_QUBIT_FREQ + CFG_QU_FREQ_OFFSET_END,
                CFG_QU_FREQ_STEPS,
            )

    n_gains, n_res, n_qf = I_g.shape
    res_freq_sweep = np.linspace(CFG_RES_FREQ_START, CFG_RES_FREQ_STOP, n_res)

    print(f"\n    Dimensions : {n_gains} gains × {n_res} res_freqs × {n_qf} qubit_freqs")
    print(f"    Gain       : {gain_sweep[0]:.4f} – {gain_sweep[-1]:.4f}")
    print(f"    Res freq   : {res_freq_sweep[0]:.4f} – {res_freq_sweep[-1]:.4f} MHz")
    print(f"    Qubit freq : {qu_freq_sweep[0]:.3f} – {qu_freq_sweep[-1]:.3f} MHz")

    # 2. Extract branch centers for every gain
    print("\n[2] Extracting branch centres …")
    g_centers = np.zeros((n_gains, n_res))
    e_centers = np.zeros((n_gains, n_res))

    for gi in range(n_gains):
        g_centers[gi] = extract_all_branch_centers(I_g[gi], Q_g[gi], qu_freq_sweep)
        e_centers[gi] = extract_all_branch_centers(I_e[gi], Q_e[gi], qu_freq_sweep)
    print("    Done.")

    # 3. Global coupled CKP fit
    print("\n[3] Fitting the coupled CKP model across all gains …")
    fit = fit_ckp_global(res_freq_sweep, gain_sweep, g_centers, e_centers)

    chi = fit["chi"]
    kappa = fit["kappa"]
    omega_q0 = fit["omega_q0"]
    omega_rm = fit["omega_rm"]
    A2_by_gain = fit["A2_by_gain"]

    best_gi = int(np.argmax(fit["depth_by_gain"]))
    ref_gain = gain_sweep[best_gi]

    center_g = omega_rm + chi
    center_e = omega_rm - chi

    print(f"    Global fit RMSE        : {fit['rmse']:.4f} MHz")
    print(f"    ω_q,0                 : {omega_q0:.4f} MHz")
    print(f"    ω_r,m                 : {omega_rm:.4f} MHz")
    print(f"    |0> branch center     : {center_g:.4f} MHz")
    print(f"    |1> branch center     : {center_e:.4f} MHz")
    print(f"    2χ (splitting)        : {2*chi:.4f} MHz")
    print(f"    χ = |χ_paper|         : {chi:.4f} MHz ({chi*1e3:.2f} kHz)")
    print(f"    κ                     : {kappa:.4f} MHz ({kappa*1e3:.2f} kHz)")
    print(f"    χ from config (ref)   : {abs(CHI_CONFIG):.4f} MHz")
    print(f"    Reference gain index  : {best_gi} (gain = {ref_gain:.4f})")

    # 4. Compute nbar using Eq. (6)
    print("\n[4] Computing n̄ from Eq. (6) using fitted |A|² and κ …")
    nbar_g, nbar_e = photon_number_from_fit(res_freq_sweep, omega_rm, chi, kappa, A2_by_gain)

    # Useful cross-check: Stark shift inferred nbar from the extracted centers.
    nbar_stark_g = np.maximum(omega_q0 - g_centers, 0.0) / (2.0 * chi)
    nbar_stark_e = np.maximum(omega_q0 - e_centers, 0.0) / (2.0 * chi)

    on_res_idx = int(np.argmin(np.abs(res_freq_sweep - RES_FREQ_ON_RESONANCE)))
    print(f"    On-resonance index     : {on_res_idx} (f_d = {res_freq_sweep[on_res_idx]:.4f} MHz)")

    nbar_on_g = nbar_g[:, on_res_idx]
    nbar_on_e = nbar_e[:, on_res_idx]
    nbar_on_avg = 0.5 * (nbar_on_g + nbar_on_e)

    def quad(g, a, b):
        return a * g**2 + b

    try:
        popt_q, _ = curve_fit(quad, gain_sweep, nbar_on_avg, p0=[100.0, 0.0])
        fit_g_fine = np.linspace(gain_sweep[0], gain_sweep[-1], 300)
        fit_nbar = quad(fit_g_fine, *popt_q)
        print(f"    Fit: n̄(on-res) ≈ {popt_q[0]:.1f} × g² + {popt_q[1]:.3f}")
    except Exception as e:
        print(f"    Quadratic fit failed: {e}")
        popt_q = None
        fit_g_fine = None
        fit_nbar = None

    # 5. Summary table
    print("\n" + "─" * 84)
    print(f"  {'Gain':>8s}   {'|A|² fit':>12s}   {'n̄ (|0>, on-res)':>18s}   {'n̄ (|1>, on-res)':>18s}")
    print(f"  {'─'*8}   {'─'*12}   {'─'*18}   {'─'*18}")
    for i, g in enumerate(gain_sweep):
        print(f"  {g:8.4f}   {A2_by_gain[i]:12.6g}   {nbar_on_g[i]:18.4f}   {nbar_on_e[i]:18.4f}")
    print("─" * 84)

    # 6. Plots
    plt.rcParams.update({
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 13,
        "figure.dpi": 120,
        "savefig.dpi": 200,
    })

    # Plot 1: nbar vs gain at on-resonance
    # Make the canvas a bit wider and place the legend outside the axes
    # so it does not overlap the parameter text box.
    fig1, ax1 = plt.subplots(figsize=(8.8, 5.0))
    ax1.scatter(gain_sweep, nbar_on_g, s=70, label=r"$|0\rangle$ branch")
    ax1.scatter(gain_sweep, nbar_on_e, s=55, marker="s", label=r"$|1\rangle$ branch")
    ax1.plot(gain_sweep, nbar_on_avg, "--", alpha=0.7, label="branch average")

    if popt_q is not None:
        ax1.plot(fit_g_fine, fit_nbar, ":", lw=2,
                 label=(rf"fit: $\bar{{n}} = {popt_q[0]:.0f}\,g^2"
                        rf" {'+' if popt_q[1] >= 0 else ''}{popt_q[1]:.2f}$"))

    txt = (
        rf"$\chi$ = {chi:.4f} MHz" "\n"
        rf"$\kappa$ = {kappa:.4f} MHz" "\n"
        rf"$\omega_{{r,m}}$ = {omega_rm:.4f} MHz" "\n"
        rf"$\omega_{{q,0}}$ = {omega_q0:.4f} MHz"
    )
    ax1.text(0.03, 0.97, txt, transform=ax1.transAxes, va="top",
             bbox=dict(boxstyle="round,pad=0.4", fc="white", alpha=0.85))
    ax1.set_xlabel("Resonator drive gain [DAC units]")
    ax1.set_ylabel(r"$\bar{n}$ (photon number)")
    ax1.set_title(rf"On-resonance $\bar{{n}}$ vs. gain ($f_d$ = {RES_FREQ_ON_RESONANCE} MHz)")
    ax1.grid(True, alpha=0.25)
    ax1.set_xlim(left=-0.002)
    ax1.set_ylim(bottom=0)
    ax1.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0),
               borderaxespad=0.0, fontsize=10, framealpha=0.9)
    fig1.tight_layout(rect=[0, 0, 0.80, 1])
    if SAVE_FIGS:
        p = os.path.join(OUTPUT_DIR, "ckp_fullmodel_nbar_vs_gain_on_resonance.png")
        fig1.savefig(p, bbox_inches="tight")
        print(f"\n  Saved → {p}")

    # Plot 2: nbar heatmap using Eq. (6)
    fig2, ax2 = plt.subplots(figsize=(9, 5.5))
    sort_idx = np.argsort(res_freq_sweep)
    rf_sorted = res_freq_sweep[sort_idx]
    nbar_sorted = nbar_g[:, sort_idx].T
    im = ax2.pcolormesh(gain_sweep, rf_sorted, nbar_sorted, shading="nearest", cmap="inferno")
    cbar = fig2.colorbar(im, ax=ax2, pad=0.02)
    cbar.set_label(r"$\bar{n}$ (photon number)", fontsize=12)
    ax2.axhline(RES_FREQ_ON_RESONANCE, ls="--", lw=1.4, alpha=0.85,
                label=f"bare resonator ({RES_FREQ_ON_RESONANCE} MHz)")
    ax2.axhline(center_g, ls=":", lw=1.2, alpha=0.8,
                label=rf"$\omega_{{r,|0\rangle}}$ = {center_g:.3f} MHz")
    ax2.axhline(center_e, ls=":", lw=1.2, alpha=0.8,
                label=rf"$\omega_{{r,|1\rangle}}$ = {center_e:.3f} MHz")
    ax2.set_xlabel("Resonator drive gain [DAC units]")
    ax2.set_ylabel("Resonator drive frequency [MHz]")
    ax2.set_title(r"$\bar{n}$ across gain and drive frequency from Eq. (6) ($|0\rangle$ branch)")
    ax2.legend(loc="upper left", fontsize=9, framealpha=0.85)
    fig2.tight_layout()
    if SAVE_FIGS:
        p = os.path.join(OUTPUT_DIR, "ckp_fullmodel_nbar_heatmap_gain_vs_resfreq.png")
        fig2.savefig(p, bbox_inches="tight")
        print(f"  Saved → {p}")

    # Plot 3: reference-gain diagnostics — data and coupled-model fit
    fig3, axes3 = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1.5, 1.0]})
    ax = axes3[0]
    ax.plot(res_freq_sweep, g_centers[best_gi], "o", ms=5, label=r"$|0\rangle$ extracted centers")
    ax.plot(res_freq_sweep, e_centers[best_gi], "s", ms=5, label=r"$|1\rangle$ extracted centers")
    ax.plot(res_freq_sweep, fit["model_g"][best_gi], "-", lw=2, label=r"coupled model fit: $|0\rangle$")
    ax.plot(res_freq_sweep, fit["model_e"][best_gi], "-", lw=2, label=r"coupled model fit: $|1\rangle$")
    ax.axhline(omega_q0, ls=":", lw=1, alpha=0.6, label=rf"fitted $\omega_q$ = {omega_q0:.3f} MHz")
    ax.set_xlabel("Resonator drive frequency [MHz]")
    ax.set_ylabel("Qubit resonance frequency [MHz]")
    ax.set_title(f"Coupled CKP fit at reference gain = {gain_sweep[best_gi]:.4f}")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=9, loc="best")

    ax = axes3[1]
    ax.axis("off")
    lines = [
        r"$\bf{Global\ CKP\ Fit}$",
        "",
        rf"$\omega_{{q,0}}$ = {omega_q0:.4f} MHz",
        rf"$\omega_{{r,m}}$ = {omega_rm:.4f} MHz",
        rf"$\chi$ = {chi:.4f} MHz ({chi*1e3:.1f} kHz)",
        rf"$\kappa$ = {kappa:.4f} MHz ({kappa*1e3:.1f} kHz)",
        rf"$\omega_{{r,|0\rangle}}$ = {center_g:.4f} MHz",
        rf"$\omega_{{r,|1\rangle}}$ = {center_e:.4f} MHz",
        rf"Global RMSE = {fit['rmse']:.4f} MHz",
        "",
        rf"Reference gain = {gain_sweep[best_gi]:.4f}",
        rf"$|A|^2$ at ref gain = {A2_by_gain[best_gi]:.6g}",
        rf"$\bar{{n}}_{{peak,|0\rangle}}$ = {np.max(nbar_g[best_gi]):.2f}",
        rf"$\bar{{n}}_{{peak,|1\rangle}}$ = {np.max(nbar_e[best_gi]):.2f}",
        "",
        rf"Config $|\chi|$ (ref.) = {abs(CHI_CONFIG):.4f} MHz",
    ]
    ax.text(0.05, 0.95, "\n".join(lines), transform=ax.transAxes,
            fontsize=11, va="top", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.5", fc="lightyellow", ec="gray", alpha=0.9))
    fig3.tight_layout()
    if SAVE_FIGS:
        p = os.path.join(OUTPUT_DIR, "ckp_fullmodel_coupled_fit_diagnostic.png")
        fig3.savefig(p, bbox_inches="tight")
        print(f"  Saved → {p}")

    # Plot 4: compare Eq. (6) nbar to the Stark-shift back-out nbar
    fig4, ax4 = plt.subplots(figsize=(7.2, 5.0))
    ax4.plot(gain_sweep, nbar_stark_g[:, on_res_idx], "o", ms=5, label=r"from $\Delta f_q$, $|0\rangle$")
    ax4.plot(gain_sweep, nbar_on_g, "-", lw=2, label=r"from Eq. (6), $|0\rangle$")
    ax4.plot(gain_sweep, nbar_stark_e[:, on_res_idx], "s", ms=5, label=r"from $\Delta f_q$, $|1\rangle$")
    ax4.plot(gain_sweep, nbar_on_e, "-", lw=2, label=r"from Eq. (6), $|1\rangle$")
    ax4.set_xlabel("Resonator drive gain [DAC units]")
    ax4.set_ylabel(r"$\bar{n}$ at on-resonance")
    ax4.set_title("Cross-check: Eq. (6) photon number vs Stark-shift back-out")
    ax4.grid(True, alpha=0.25)
    ax4.legend(fontsize=9)
    fig4.tight_layout()
    if SAVE_FIGS:
        p = os.path.join(OUTPUT_DIR, "ckp_fullmodel_nbar_crosscheck.png")
        fig4.savefig(p, bbox_inches="tight")
        print(f"  Saved → {p}")

    if SHOW_FIGS:
        plt.show()

    # 7. Save results
    npz_path = os.path.join(OUTPUT_DIR, "ckp_fullmodel_results.npz")
    np.savez(
        npz_path,
        chi=chi,
        kappa=kappa,
        omega_q0=omega_q0,
        omega_rm=omega_rm,
        omega_r_g=center_g,
        omega_r_e=center_e,
        gain_sweep=gain_sweep,
        res_freq_sweep=res_freq_sweep,
        qu_freq_sweep=qu_freq_sweep,
        A2_by_gain=A2_by_gain,
        nbar_g=nbar_g,
        nbar_e=nbar_e,
        nbar_stark_g=nbar_stark_g,
        nbar_stark_e=nbar_stark_e,
        g_centers=g_centers,
        e_centers=e_centers,
        model_g=fit["model_g"],
        model_e=fit["model_e"],
        rms_by_gain=fit["rms_by_gain"],
        depth_by_gain=fit["depth_by_gain"],
        reference_gain_index=best_gi,
    )
    print(f"\n  Results saved → {npz_path}")

    print("\n✓ Analysis complete.")
    return {
        "chi": chi,
        "kappa": kappa,
        "omega_q0": omega_q0,
        "omega_rm": omega_rm,
        "A2_by_gain": A2_by_gain,
        "nbar_g": nbar_g,
        "nbar_e": nbar_e,
        "gain_sweep": gain_sweep,
        "res_freq_sweep": res_freq_sweep,
    }


if __name__ == "__main__":
    results = main()
