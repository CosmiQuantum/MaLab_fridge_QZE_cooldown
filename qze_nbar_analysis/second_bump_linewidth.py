import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares

import config as C
from loading import parse_rounds, discover_h5_files, load_sweep_points
from nbar import (average_by_axis, average_over_rounds, calculate_nbar_from_qspec,
                  gain_to_mean_nbar, nbar_lookup)

# the plot with the pink line (R_0 ... 13-41-28) is the high gain zeno qspec
ROUND = 22
SUBDIR = "QSpec_zeno_high_gain"

# the tls shows up as a narrow dip near 3038.6 MHz sitting on the shoulder
# that rises toward the main qubit line
WIN = (3036.5, 3040.5)       # local window fit around the dip
CEN = (3038.0, 3039.2)       # allowed dip center

# call the dip "present" only when the fit is real
DEPTH_MIN = 0.02
R2_MIN = 0.40

# above this the stark shifted main line reaches the window and swamps the dip
GAIN_MAX = 0.035


def lorentzian(x, f0, half_width, amp):
    return amp * half_width ** 2 / ((x - f0) ** 2 + half_width ** 2)


def fit_tls_dip(freq, pop, return_fit=False):
    x = np.asarray(freq, float)
    y = np.asarray(pop, float)
    sel = (x > WIN[0]) & (x < WIN[1])
    x, y = x[sel], y[sel]
    nan = (np.nan, np.nan, np.nan, np.nan)
    if x.size < 15:
        return (nan + (None, None)) if return_fit else nan

    # local linear background minus a lorentzian dip
    def curve(p):
        c0, c1, f0, half_width, depth = p
        return c0 + c1 * (x - 3038.5) - lorentzian(x, f0, half_width, depth)

    c0 = float(np.median(y))
    c1 = float((y[-5:].mean() - y[:5].mean()) / (x[-1] - x[0]))
    try:
        r = least_squares(
            lambda p: curve(p) - y, [c0, c1, 3038.5, 0.4, 0.03],
            bounds=([0, -1, CEN[0], 0.05, 0], [1, 1, CEN[1], 2.0, 0.5]),
            loss="soft_l1", f_scale=0.02, max_nfev=8000,
        )
    except Exception:
        return (nan + (None, None)) if return_fit else nan

    c0, c1, f0, half_width, depth = r.x
    ss = float(np.sum((curve(r.x) - y) ** 2))
    tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss / tot if tot > 0 else 0.0
    if return_fit:
        return f0, 2.0 * half_width, depth, r2, x, curve(r.x)
    return f0, 2.0 * half_width, depth, r2


def fit_all_gains(gain_to_trace, gain_nbar):
    gains = np.array([g for g in sorted(gain_to_trace) if g <= GAIN_MAX])
    nbars = np.array([nbar_lookup(g, gain_nbar) for g in gains])
    centers, fwhms, depths, r2s = [], [], [], []
    for g in gains:
        freq, pop = gain_to_trace[g][0], gain_to_trace[g][1]
        f0, fwhm, depth, r2 = fit_tls_dip(freq, pop)
        centers.append(f0); fwhms.append(fwhm); depths.append(depth); r2s.append(r2)
    return (gains, nbars, np.array(centers), np.array(fwhms), np.array(depths), np.array(r2s))


def plot_linewidth(nbars, fwhms, depths, r2s, title, filename):
    present = (depths >= DEPTH_MIN) & (r2s >= R2_MIN) & np.isfinite(fwhms)
    faint = np.isfinite(fwhms) & ~present
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.plot(nbars[present], fwhms[present], "o-", color="crimson", label="dip present")
    ax.plot(nbars[faint], fwhms[faint], "o", mfc="none", color="gray", alpha=0.6, label="fit not significant")
    ax.set_xlabel(r"$\bar{n}$")
    ax.set_ylabel("TLS dip FWHM (MHz)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(C.OUT_DIR / filename, dpi=300)
    plt.close(fig)
    return present


def plot_overlay(gain_to_trace, gains, centers, fwhms, present, title, filename):
    all_gains = np.array(sorted(gain_to_trace))
    freq_axis = np.asarray(gain_to_trace[all_gains[0]][0], float)
    Z = np.array([gain_to_trace[g][1] for g in all_gains])
    fig, ax = plt.subplots(figsize=(10, 6))
    mesh = ax.pcolormesh(all_gains, freq_axis, Z.T, shading="auto", cmap="viridis")
    fig.colorbar(mesh, ax=ax, label="qubit population (avg)")
    ax.errorbar(gains[present], centers[present], yerr=fwhms[present] / 2.0,
                fmt="o", color="crimson", ms=4, lw=1, capsize=2,
                label=r"TLS dip fit (center $\pm$ HWHM)")
    ax.set_xlabel("gain")
    ax.set_ylabel("qubit frequency (MHz)")
    ax.set_ylim(3033, 3047)
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(C.OUT_DIR / filename, dpi=300)
    plt.close(fig)


C.OUT_DIR.mkdir(exist_ok=True)
rounds = parse_rounds(C.ROUNDS, C.DATA_ROOT, C.QUBIT_INDEX)

# single round (the one with the pink line)
files = discover_h5_files(C.DATA_ROOT, C.QUBIT_INDEX, [ROUND], SUBDIR)
points = load_sweep_points(files, qubit_index=C.QUBIT_INDEX, x_dataset="Frequencies", x_column="frequency_mhz")
single = {g: t for (r, g), t in average_by_axis(points, "frequency_mhz").items()}
gain_nbar = gain_to_mean_nbar(calculate_nbar_from_qspec(
    points, qubit_index=C.QUBIT_INDEX, chi_mhz=C.CHI_MHZ, min_points=C.MIN_QSPEC_POINTS))
gains, nbars, centers, fwhms, depths, r2s = fit_all_gains(single, gain_nbar)
print("=== round %d ===" % ROUND)
print("gain     nbar    center   FWHM(MHz)  depth  r2")
for g, nb, f0, fw, d, r2 in zip(gains, nbars, centers, fwhms, depths, r2s):
    print("%.4f   %5.2f   %7.2f   %6.2f   %.3f  %.2f" % (g, nb, f0, fw, d, r2))
present = plot_linewidth(nbars, fwhms, depths, r2s,
                         r"Q5 TLS dip linewidth vs $\bar{n}$  (round %d, %s)" % (ROUND, SUBDIR),
                         "tls_dip_linewidth_vs_nbar.png")
plot_overlay(single, gains, centers, fwhms, present,
             "Q5 qspec vs gain with fitted TLS dip  (round %d, %s)" % (ROUND, SUBDIR),
             "tls_dip_on_qspec_2d.png")

# averaged across all rounds -> cleaner, especially at low nbar
all_files = discover_h5_files(C.DATA_ROOT, C.QUBIT_INDEX, rounds, SUBDIR)
all_points = load_sweep_points(all_files, qubit_index=C.QUBIT_INDEX, x_dataset="Frequencies", x_column="frequency_mhz")
avg = average_over_rounds(all_points, "frequency_mhz")
avg_trace = {g: (v[0], v[1]) for g, v in avg.items()}
n_rounds = int(np.median([v[2] for v in avg.values()]))
gain_nbar_avg = gain_to_mean_nbar(calculate_nbar_from_qspec(
    all_points, qubit_index=C.QUBIT_INDEX, chi_mhz=C.CHI_MHZ, min_points=C.MIN_QSPEC_POINTS))
gains_a, nbars_a, centers_a, fwhms_a, depths_a, r2s_a = fit_all_gains(avg_trace, gain_nbar_avg)
print("\n=== averaged over %d rounds ===" % n_rounds)
print("gain     nbar    center   FWHM(MHz)  depth  r2")
for g, nb, f0, fw, d, r2 in zip(gains_a, nbars_a, centers_a, fwhms_a, depths_a, r2s_a):
    print("%.4f   %5.2f   %7.2f   %6.2f   %.3f  %.2f" % (g, nb, f0, fw, d, r2))
present_a = plot_linewidth(nbars_a, fwhms_a, depths_a, r2s_a,
                           r"Q5 TLS dip linewidth vs $\bar{n}$  (avg over %d rounds, %s)" % (n_rounds, SUBDIR),
                           "tls_dip_linewidth_vs_nbar_avgrounds.png")
plot_overlay(avg_trace, gains_a, centers_a, fwhms_a, present_a,
             "Q5 qspec vs gain with fitted TLS dip  (avg over %d rounds, %s)" % (n_rounds, SUBDIR),
             "tls_dip_on_qspec_2d_avgrounds.png")
print("\nsaved plots to", C.OUT_DIR)
