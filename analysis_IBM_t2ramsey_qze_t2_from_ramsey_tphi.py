from analysis_006_T1_vs_time_plots import T1VsTime
from analysis_007_T2R_vs_time_plots import T2rVsTime
from analysis_003_q_freqs_vs_time_plots import QubitFreqsVsTime

import os
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict


# =========================
# Configuration
# =========================
save_figs = True
save_individual_qspec = False
save_individual_t2r = False
nbar_from_ramsey = False

figure_quality = 100
final_figure_quality = 200

FRIDGE = "QUIET"
qubits = [5]
round_id = 0

path = '2d_test_high_res'
ramsey_nbar_path = '/bob_run_started_Aug_3_2026/squill/ramsey_n_bar_calibration_more_fringes/run/'

save_dir = f"M:/_Data/20250822 - Olivia/bob_run_started_Aug_3_2026/squill/{path}/all_qubits/analysis_extracted_t2/"
os.makedirs(save_dir, exist_ok=True)

# qspec -> nbar conversion uses Delta f = 2 chi nbar
chi_MHz = -0.137


# =========================
# Helpers
# =========================
def sort_xy(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    x = x[m]
    y = y[m]
    o = np.argsort(x)
    return x[o], y[o]


def match_by_gain(g_ref, y_ref, g_other, y_other, rdigits=12):
    g_ref = np.asarray(g_ref, float)
    y_ref = np.asarray(y_ref, float)
    g_other = np.asarray(g_other, float)
    y_other = np.asarray(y_other, float)

    mref = np.isfinite(g_ref) & np.isfinite(y_ref)
    moth = np.isfinite(g_other) & np.isfinite(y_other)

    g_ref, y_ref = g_ref[mref], y_ref[mref]
    g_other, y_other = g_other[moth], y_other[moth]

    ref_map = {round(float(g), rdigits): float(y) for g, y in zip(g_ref, y_ref)}

    gains_common = []
    y_ref_common = []
    y_other_common = []

    for g, y in zip(g_other, y_other):
        key = round(float(g), rdigits)
        if key in ref_map:
            gains_common.append(float(g))
            y_other_common.append(float(y))
            y_ref_common.append(float(ref_map[key]))

    if len(gains_common) < max(3, int(0.5 * min(len(g_ref), len(g_other)))):
        gains_common = []
        y_ref_common = []
        y_other_common = []

        span = float(np.nanmax(np.r_[g_ref, g_other]) - np.nanmin(np.r_[g_ref, g_other])) if (len(g_ref) and len(g_other)) else 1.0
        tol = max(1e-9, 1e-6 * max(1.0, span))

        for g, y in zip(g_other, y_other):
            j = int(np.argmin(np.abs(g_ref - g)))
            if abs(g_ref[j] - g) <= tol:
                gains_common.append(float(g))
                y_other_common.append(float(y))
                y_ref_common.append(float(y_ref[j]))

    return np.asarray(gains_common, float), np.asarray(y_ref_common, float), np.asarray(y_other_common, float)


def align_nbar_to_gains(n_bars, g_target, round_id_str):
    g_target = np.asarray(g_target, float)

    if n_bars is None:
        return np.full_like(g_target, np.nan, dtype=float)

    entry = None
    if isinstance(n_bars, dict):
        if round_id_str in n_bars:
            entry = n_bars[round_id_str]
        elif str(round_id_str) in n_bars:
            entry = n_bars[str(round_id_str)]
        else:
            entry = n_bars
    else:
        entry = n_bars

    if isinstance(entry, dict):
        for key in ["lorentzian", "gaussian", "nbar", "values"]:
            if key in entry:
                entry = entry[key]
                break

    nb = np.asarray(entry, float).ravel()
    if nb.size == g_target.size:
        return nb

    return np.full_like(g_target, np.nan, dtype=float)


def compute_tphi_and_gamma_phi(t1_us, t2_us):
    T1 = np.asarray(t1_us, float) * 1e-6
    T2 = np.asarray(t2_us, float) * 1e-6

    inv_Tphi = (1.0 / T2) - (1.0 / (2.0 * T1))
    tphi_s = np.where(inv_Tphi > 0, 1.0 / inv_Tphi, np.nan)
    tphi_us = tphi_s * 1e6
    gamma_phi_inv_us = np.where(np.isfinite(tphi_us) & (tphi_us > 0), 1.0 / tphi_us, np.nan)

    return tphi_us, gamma_phi_inv_us


def gamma_from_t1_us(t1_us):
    t1_us = np.asarray(t1_us, float)
    return np.where(np.isfinite(t1_us) & (t1_us > 0), 1.0 / t1_us, np.nan)


def save_plot(x, y, out_png, title, xlabel, ylabel, marker="o", label=None):
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    if label is None:
        ax.plot(x, y, marker=marker, linewidth=1.5)
    else:
        ax.plot(x, y, marker=marker, linewidth=1.5, label=label)
        ax.legend()
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, transparent=False)
    plt.close(fig)
    print(f"Saved: {out_png}")


def interpolate_to_common_x(x_src, y_src, x_target):
    x_src = np.asarray(x_src, float)
    y_src = np.asarray(y_src, float)
    x_target = np.asarray(x_target, float)

    m = np.isfinite(x_src) & np.isfinite(y_src)
    if np.sum(m) < 2:
        return np.full_like(x_target, np.nan, dtype=float)

    xs, ys = sort_xy(x_src[m], y_src[m])
    return np.interp(x_target, xs, ys, left=np.nan, right=np.nan)


# =========================
# Standalone T2R fitter
# =========================
def fit_t2r_and_return(
    t2r_vs_time_obj,
    amps,
    gains,
    rounds,
    delay_times,
    save_path=None,
    save_individual_plots=False,
):
    q = t2r_vs_time_obj.qubit
    gains_q = gains.get(q, [])
    amps_q = amps.get(q, [])
    rounds_q = rounds.get(q, [])
    delay_q = delay_times.get(q, [])

    n = min(len(amps_q), len(gains_q), len(rounds_q), len(delay_q))
    if n == 0 or not (len(amps_q) == len(gains_q) == len(rounds_q) == len(delay_q)):
        print(f"No usable T2R data for qubit {q}.")
        return {}

    all_points = []
    for i in range(n):
        r_id = str(rounds_q[i])

        a_samples = np.asarray(amps_q[i], dtype=float).ravel()
        if a_samples.size == 0:
            continue

        g_i = gains_q[i]
        g_arr = np.asarray(g_i, dtype=float).ravel() if isinstance(g_i, (list, tuple, np.ndarray)) else None
        if g_arr is None or g_arr.size == 1:
            try:
                g_scalar = float(g_i)
            except Exception:
                continue
            g_arr = np.full(a_samples.shape, g_scalar, dtype=float)
        elif g_arr.size != a_samples.size:
            continue

        d_i = delay_q[i]
        d_arr = np.asarray(d_i, dtype=float).ravel() if isinstance(d_i, (list, tuple, np.ndarray)) else None
        if d_arr is None or d_arr.size == 1:
            try:
                d_scalar = float(d_i)
            except Exception:
                continue
            d_arr = np.full(a_samples.shape, d_scalar, dtype=float)
        elif d_arr.size != a_samples.size:
            continue

        mask = np.isfinite(a_samples) & np.isfinite(g_arr) & np.isfinite(d_arr)
        if not np.any(mask):
            continue

        for g, d, a in zip(g_arr[mask], d_arr[mask], a_samples[mask]):
            all_points.append((r_id, float(g), float(d), float(a)))

    if not all_points:
        print(f"No numeric T2R points for qubit {q}.")
        return {}

    def build_grid(points_for_round):
        gains_r = sorted({g for (g, _, __) in points_for_round})
        delays_r = sorted({d for (_, d, __) in points_for_round})

        gi_map = {g: i for i, g in enumerate(gains_r)}
        di_map = {d: i for i, d in enumerate(delays_r)}

        bucket = defaultdict(list)
        for g, d, a in points_for_round:
            bucket[(di_map[d], gi_map[g])].append(a)

        C = np.full((len(delays_r), len(gains_r)), np.nan, dtype=float)
        for (iy, ix), vals in bucket.items():
            C[iy, ix] = float(np.nanmean(vals))
        return gains_r, delays_r, C

    out = {}

    out_root = None
    if save_individual_plots and save_path is not None:
        out_root = os.path.join(save_path, "t2r_fits")
        os.makedirs(out_root, exist_ok=True)

    unique_rounds = sorted({r for (r, _, __, ___) in all_points})
    for r_id in unique_rounds:
        pts = [(g, d, a) for (r, g, d, a) in all_points if r == r_id]
        if not pts:
            continue

        gains_r, delays_r, C = build_grid(pts)

        fit_gains = []
        fit_t2 = []
        fit_t2_err = []

        for ix, g_val in enumerate(gains_r):
            x = np.asarray(delays_r, float)
            y = C[:, ix]

            finite = np.isfinite(x) & np.isfinite(y)
            x_fit = x[finite]
            y_fitdata = y[finite]

            if x_fit.size < 3:
                fit_gains.append(float(g_val))
                fit_t2.append(np.nan)
                fit_t2_err.append(np.nan)
                continue

            try:
                I0 = np.zeros_like(y_fitdata)
                Q0 = np.zeros_like(y_fitdata)

                y_fit, t2_est, t2_err, _ = t2r_vs_time_obj.t2_fit(
                    x_data=x_fit,
                    I=I0,
                    Q=Q0,
                    verbose=False,
                    guess=None,
                    plot=False,
                    amp=y_fitdata,
                )

                fit_gains.append(float(g_val))
                fit_t2.append(float(t2_est))
                fit_t2_err.append(float(t2_err))

                if save_individual_plots and out_root is not None:
                    fig, ax = plt.subplots(figsize=(6.0, 4.0))
                    ax.plot(x_fit, y_fitdata, ".", label="data")
                    ax.plot(x_fit, y_fit, "-", label=f"fit: T2 = {t2_est:.2f} ± {t2_err:.2f} us")
                    ax.set_title(f"Qubit {t2r_vs_time_obj.qubit + 1} — Round {r_id} — Gain {g_val:g}")
                    ax.set_xlabel("Delay time")
                    ax.set_ylabel("Qubit Population")
                    ax.legend(loc="best")
                    fig.tight_layout()

                    gain_str = str(g_val).replace(".", "p").replace("-", "m")
                    out_file = os.path.join(out_root, f"t2r_fit_q{t2r_vs_time_obj.qubit}_round{r_id}_gain{gain_str}.png")
                    fig.savefig(out_file, transparent=False, dpi=t2r_vs_time_obj.final_figure_quality)
                    plt.close(fig)

            except Exception as e:
                print(f"[q{q}] Round {r_id}, gain {g_val}: T2R fit failed ({e})")
                fit_gains.append(float(g_val))
                fit_t2.append(np.nan)
                fit_t2_err.append(np.nan)

        out[str(r_id)] = {
            "gains": fit_gains,
            "T2_us": fit_t2,
            "T2_err_us": fit_t2_err,
        }

    return out


# =========================
# Kofman-Kurizki theory
# =========================
def lorentzian_weight(omega, omega0, gamma_phi):
    return (1.0 / np.pi) * gamma_phi / (gamma_phi**2 + (omega - omega0)**2)


def kk_predict_gamma_from_landscape(nbar_vals, gamma_phi_vals, omega_grid, gammaq_grid,
                                    chi_MHz=-0.137, normalize_window=True):
    """
    Approximate paper-style theory using:
      omega_tilde(nbar) = 2*pi*(2*chi*nbar)   [rad/us]
    where chi is in MHz and omega is rad/us.

    omega_grid should be in rad/us
    gammaq_grid should be in 1/us
    """
    nbar_vals = np.asarray(nbar_vals, float)
    gamma_phi_vals = np.asarray(gamma_phi_vals, float)
    omega_grid = np.asarray(omega_grid, float)
    gammaq_grid = np.asarray(gammaq_grid, float)

    gamma_pred = np.full_like(nbar_vals, np.nan, dtype=float)

    for i, (nb, gphi) in enumerate(zip(nbar_vals, gamma_phi_vals)):
        if not np.isfinite(nb) or not np.isfinite(gphi) or gphi <= 0:
            continue

        delta_f_MHz = 2.0 * chi_MHz * nb
        omega_tilde = 2.0 * np.pi * delta_f_MHz  # rad/us because MHz = 1/us

        L = lorentzian_weight(omega_grid, omega_tilde, gphi)
        raw = np.trapz(gammaq_grid * L, omega_grid)

        if normalize_window:
            norm = np.trapz(L, omega_grid)
            gamma_pred[i] = raw / norm if norm > 0 else np.nan
        else:
            gamma_pred[i] = raw

    return gamma_pred


def build_approx_gammaq_from_measured_gamma(nbar_vals, gamma_vals, chi_MHz=-0.137):
    """
    Construct an approximate gamma_q(omega) using measured Gamma(nbar)=1/T1(nbar),
    mapped onto Stark-shifted frequency:
        omega = 2*pi*(2*chi*nbar)
    This is approximate / self-consistent, not an independent zero-readout gamma_q(omega).
    """
    nbar_vals = np.asarray(nbar_vals, float)
    gamma_vals = np.asarray(gamma_vals, float)

    m = np.isfinite(nbar_vals) & np.isfinite(gamma_vals) & (nbar_vals >= 0) & (gamma_vals > 0)
    if np.sum(m) < 3:
        return np.asarray([]), np.asarray([])

    nbar_vals = nbar_vals[m]
    gamma_vals = gamma_vals[m]

    delta_f_MHz = 2.0 * chi_MHz * nbar_vals
    omega = 2.0 * np.pi * delta_f_MHz  # rad/us

    omega, gamma_vals = sort_xy(omega, gamma_vals)

    # merge duplicates by averaging
    uniq = []
    gavg = []
    cur_w = omega[0]
    cur_vals = [gamma_vals[0]]
    for w, g in zip(omega[1:], gamma_vals[1:]):
        if np.isclose(w, cur_w, rtol=0, atol=1e-15):
            cur_vals.append(g)
        else:
            uniq.append(cur_w)
            gavg.append(np.mean(cur_vals))
            cur_w = w
            cur_vals = [g]
    uniq.append(cur_w)
    gavg.append(np.mean(cur_vals))

    return np.asarray(uniq, float), np.asarray(gavg, float)


# =========================
# Main
# =========================
for qubit in qubits:
    run_name = f'bob_run_started_Aug_3_2026/squill/{path}/all_qubits/'
    top_folder_dates = [f'qubit_{qubit}round{round_id}']

    # -------------------------
    # QSpec -> nbar
    # -------------------------
    q_vs_time = QubitFreqsVsTime(
        figure_quality, final_figure_quality, 6, top_folder_dates, save_figs,
        False, 'None', run_name, fridge=FRIDGE, exp_name='ge', qubit=qubit
    )

    _, _, amps_qspec, gains_qspec, rounds_qspec, freqs_qspec = q_vs_time.run_q_sweep_new(
        exp_extension='_ge', scaling=True
    )

    if nbar_from_ramsey:
        top_folder_dates_ramsey_nbar = ['2025-12-10_13-20-47']
        t2_vs_time = T2rVsTime(
            figure_quality, final_figure_quality, 6, top_folder_dates_ramsey_nbar, save_figs, False,
            'None', ramsey_nbar_path, fridge=FRIDGE, exp_name='ge', qubit=qubit
        )
        n_bars = t2_vs_time.run_ramsey_nbar(
            exp_name='StarkRamsey',
            save_path=save_dir,
        )
    else:
        # This is the code path that already does nbar from qspec.
        n_bars = q_vs_time.calculate_nbar(
            amps_qspec, gains_qspec, rounds_qspec, freqs_qspec, chi_MHz=chi_MHz
        )

        q_vs_time.plot_all_q_heatmaps_nbar(
            amps_qspec, gains_qspec, rounds_qspec, freqs_qspec,
            save_dir,
            individual_subfolder="individual_specs"
        )

    # -------------------------
    # T2R fits
    # -------------------------
    t2r_vs_time = T2rVsTime(
        figure_quality, final_figure_quality, 6,
        top_folder_dates, save_figs, False,
        'None', run_name, fridge=FRIDGE, exp_name='ge', qubit=qubit
    )

    _, _, amps_t2, gains_t2, rounds_t2, delay_times_t2 = t2r_vs_time.run_t2_sweep_new(
        exp_extension='_ge', scaling=True
    )

    t2r_out = fit_t2r_and_return(
        t2r_vs_time,
        amps_t2,
        gains_t2,
        rounds_t2,
        delay_times_t2,
        save_path=save_dir,
        save_individual_plots=save_individual_t2r,
    )

    g_t2 = np.asarray(t2r_out[str(round_id)]["gains"], float)
    t2_us = np.asarray(t2r_out[str(round_id)]["T2_us"], float)

    # -------------------------
    # T1 fits
    # -------------------------
    t1_vs_time = T1VsTime(
        figure_quality, final_figure_quality, 6, top_folder_dates, save_figs, False,
        'None', run_name, FRIDGE, exp_name='ge', qubit=qubit, t1_slice='10us'
    )

    _, _, amps_t1, gains_t1, rounds_t1, delay_times_t1 = t1_vs_time.run_t1_sweep_new(
        exp_extension='_ge', scaling=True, weighted_mean=True
    )

    t1_vs_time.fit_and_save_t1_slices_new_format(
        amps_t1, gains_t1, rounds_t1, delay_times_t1,
        save_dir, n_bar=n_bars
    )

    t1_out = t1_vs_time.plot_all_t1_heatmaps_new_format(
        amps_t1, gains_t1, rounds_t1, delay_times_t1,
        save_dir, n_bar=n_bars, use_linear_x=True
    )

    g_t1 = np.asarray(t1_out[str(round_id)]["gains"], float)
    t1_us = np.asarray(t1_out[str(round_id)]["T1_us"], float)

    # -------------------------
    # Align by gain
    # -------------------------
    gains_common, t1_common, t2_common = match_by_gain(g_t1, t1_us, g_t2, t2_us)
    if len(gains_common) == 0:
        print(f"[q{qubit}] No common gains between T1 and T2R.")
        continue

    # nbar aligned to T1/T2 common gains
    nbar_t1_grid = align_nbar_to_gains(n_bars, g_t1, str(round_id))
    gains_nbar_common, nbar_common, _ = match_by_gain(g_t1, nbar_t1_grid, gains_common, np.ones_like(gains_common))
    gains_final, t1_final, _ = match_by_gain(gains_common, t1_common, gains_nbar_common, np.ones_like(gains_nbar_common))
    _, t2_final, _ = match_by_gain(gains_common, t2_common, gains_nbar_common, np.ones_like(gains_nbar_common))
    _, nbar_final, _ = match_by_gain(gains_nbar_common, nbar_common, gains_final, np.ones_like(gains_final))

    # -------------------------
    # Derived quantities
    # -------------------------
    tphi_us, gamma_phi_inv_us = compute_tphi_and_gamma_phi(t1_final, t2_final)
    gamma_data_inv_us = gamma_from_t1_us(t1_final)

    nbar_plot = np.asarray(nbar_final, float)
    t1_plot = np.asarray(t1_final, float)
    t2_plot = np.asarray(t2_final, float)
    tphi_plot = np.asarray(tphi_us, float)
    gamma_phi_plot = np.asarray(gamma_phi_inv_us, float)
    gamma_data_plot = np.asarray(gamma_data_inv_us, float)

    # -------------------------
    # Build approximate gamma_q(omega)
    # from measured T1(nbar) + Stark shift
    # -------------------------
    omega_grid, gammaq_grid = build_approx_gammaq_from_measured_gamma(
        nbar_plot, gamma_data_plot, chi_MHz=chi_MHz
    )

    gamma_theory_plot = np.full_like(nbar_plot, np.nan, dtype=float)
    if len(omega_grid) >= 3:
        gamma_theory_plot = kk_predict_gamma_from_landscape(
            nbar_vals=nbar_plot,
            gamma_phi_vals=gamma_phi_plot,
            omega_grid=omega_grid,
            gammaq_grid=gammaq_grid,
            chi_MHz=chi_MHz,
            normalize_window=True,
        )

    # shifted frequency axis from nbar calibration
    delta_f_MHz_plot = 2.0 * chi_MHz * nbar_plot

    # -------------------------
    # Save simple plots
    # -------------------------
    x_nbar, y_t2 = sort_xy(nbar_plot, t2_plot)
    _, y_tphi = sort_xy(nbar_plot, tphi_plot)
    _, y_gphi = sort_xy(nbar_plot, gamma_phi_plot)
    _, y_gdata = sort_xy(nbar_plot, gamma_data_plot)
    _, y_gtheory = sort_xy(nbar_plot, gamma_theory_plot)
    _, y_shift = sort_xy(nbar_plot, delta_f_MHz_plot)

    save_plot(
        x_nbar, y_t2,
        os.path.join(save_dir, f"t2_vs_nbar_round{round_id}.png"),
        rf"$T_2^R$ vs $\bar n$ — Round {round_id}",
        r"$\bar n$",
        r"$T_2^R$ ($\mu$s)"
    )

    save_plot(
        x_nbar, y_tphi,
        os.path.join(save_dir, f"tphi_vs_nbar_round{round_id}.png"),
        rf"$T_\phi$ vs $\bar n$ — Round {round_id}",
        r"$\bar n$",
        r"$T_\phi$ ($\mu$s)"
    )

    save_plot(
        x_nbar, y_gphi,
        os.path.join(save_dir, f"gamma_phi_vs_nbar_round{round_id}.png"),
        rf"$\gamma_\phi$ vs $\bar n$ — Round {round_id}",
        r"$\bar n$",
        r"$\gamma_\phi$ (1/$\mu$s)"
    )

    save_plot(
        x_nbar, y_shift,
        os.path.join(save_dir, f"stark_shift_vs_nbar_round{round_id}.png"),
        rf"$\Delta f = 2\chi \bar n$ vs $\bar n$ — Round {round_id}",
        r"$\bar n$",
        r"$\Delta f$ (MHz)"
    )

    save_plot(
        x_nbar, y_gdata,
        os.path.join(save_dir, f"gamma_data_vs_nbar_round{round_id}.png"),
        rf"Data: $\Gamma = 1/T_1$ vs $\bar n$ — Round {round_id}",
        r"$\bar n$",
        r"$\Gamma$ (1/$\mu$s)"
    )

    if np.any(np.isfinite(y_gtheory)):
        fig, ax = plt.subplots(figsize=(6.7, 4.2))
        ax.plot(x_nbar, y_gdata, "o", label="data: $\Gamma = 1/T_1$")
        ax.plot(x_nbar, y_gtheory, "-", linewidth=2.0, label="theory")
        ax.set_title(rf"$\Gamma(\bar n)$ data + approximate theory — Round {round_id}")
        ax.set_xlabel(r"$\bar n$")
        ax.set_ylabel(r"$\Gamma$ (1/$\mu$s)")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        out_png = os.path.join(save_dir, f"gamma_data_theory_vs_nbar_round{round_id}.png")
        fig.savefig(out_png, dpi=300, transparent=False)
        plt.close(fig)
        print(f"Saved: {out_png}")

    # -------------------------
    # Also save gamma_q(omega) and T1 vs shifted frequency
    # -------------------------
    if len(omega_grid) >= 3:
        f_grid_MHz = omega_grid / (2.0 * np.pi)
        t1_freq_us = np.where(gammaq_grid > 0, 1.0 / gammaq_grid, np.nan)

        save_plot(
            f_grid_MHz, t1_freq_us,
            os.path.join(save_dir, f"t1_vs_shifted_freq_round{round_id}.png"),
            rf"Approx. $T_1$ vs shifted qubit frequency — Round {round_id}",
            r"Shifted qubit frequency offset $\Delta f$ (MHz)",
            r"$T_1$ ($\mu$s)"
        )

        save_plot(
            f_grid_MHz, gammaq_grid,
            os.path.join(save_dir, f"gammaq_vs_shifted_freq_round{round_id}.png"),
            rf"Approx. $\gamma_q(\omega)$ vs shifted qubit frequency — Round {round_id}",
            r"Shifted qubit frequency offset $\Delta f$ (MHz)",
            r"$\gamma_q$ (1/$\mu$s)"
        )

    # -------------------------
    # Summary panel
    # -------------------------
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))

    axes[0, 0].plot(x_nbar, y_t2, "o-", linewidth=1.5)
    axes[0, 0].set_title(r"$T_2^R$ vs $\bar n$")
    axes[0, 0].set_xlabel(r"$\bar n$")
    axes[0, 0].set_ylabel(r"$\mu$s")
    axes[0, 0].grid(True, alpha=0.3)

    axes[0, 1].plot(x_nbar, y_tphi, "o-", linewidth=1.5)
    axes[0, 1].set_title(r"$T_\phi$ vs $\bar n$")
    axes[0, 1].set_xlabel(r"$\bar n$")
    axes[0, 1].set_ylabel(r"$\mu$s")
    axes[0, 1].grid(True, alpha=0.3)

    axes[0, 2].plot(x_nbar, y_gphi, "o-", linewidth=1.5)
    axes[0, 2].set_title(r"$\gamma_\phi$ vs $\bar n$")
    axes[0, 2].set_xlabel(r"$\bar n$")
    axes[0, 2].set_ylabel(r"1/$\mu$s")
    axes[0, 2].grid(True, alpha=0.3)

    axes[1, 0].plot(x_nbar, y_shift, "o-", linewidth=1.5)
    axes[1, 0].set_title(r"$\Delta f = 2\chi \bar n$")
    axes[1, 0].set_xlabel(r"$\bar n$")
    axes[1, 0].set_ylabel("MHz")
    axes[1, 0].grid(True, alpha=0.3)

    axes[1, 1].plot(x_nbar, y_gdata, "o-", linewidth=1.5, label="data")
    if np.any(np.isfinite(y_gtheory)):
        axes[1, 1].plot(x_nbar, y_gtheory, "-", linewidth=2.0, label="theory")
        axes[1, 1].legend()
    axes[1, 1].set_title(r"$\Gamma(\bar n)$")
    axes[1, 1].set_xlabel(r"$\bar n$")
    axes[1, 1].set_ylabel(r"1/$\mu$s")
    axes[1, 1].grid(True, alpha=0.3)

    if len(omega_grid) >= 3:
        f_grid_MHz = omega_grid / (2.0 * np.pi)
        axes[1, 2].plot(f_grid_MHz, gammaq_grid, "o-", linewidth=1.5)
        axes[1, 2].set_title(r"Approx. $\gamma_q(\omega)$")
        axes[1, 2].set_xlabel(r"Shifted frequency offset (MHz)")
        axes[1, 2].set_ylabel(r"1/$\mu$s")
        axes[1, 2].grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, "Not enough points for\napprox. $\gamma_q(\omega)$",
                        ha="center", va="center", transform=axes[1, 2].transAxes)
        axes[1, 2].set_axis_off()

    fig.suptitle(f"Qubit {qubit} — Round {round_id}")
    fig.tight_layout()
    out_png = os.path.join(save_dir, f"summary_data_theory_round{round_id}.png")
    fig.savefig(out_png, dpi=300, transparent=False)
    plt.close(fig)
    print(f"Saved: {out_png}")

print("Done.")