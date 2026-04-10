#!/usr/bin/env python3
# gmm_dprime_beta_selection.py
#
# Pooled (multi-NPZ) evaluation of:
#  - Patch-level 2-GMM on WM patches -> signal component stats (mu_S, var_S)
#  - Image-level trimmed means T_n for WM and Random across a grid of n_keep
#  - Standardized separation d'(n) = (mu_R - mu_W) / sqrt(var_W + var_R)
#  - "Crossing" detection: first n where d' >= 3, plus linear-interpolated crossing n≈...
#  - Extra insights: peak, plateau (>=95% of peak), conservative recommended n
#  - Null-calibrated threshold tau = alpha-quantile of Random T_n, with empirical TPR/FPR
#  - Bayes error proxy from d': Pe ≈ Phi(-d'/2)  (Gaussian + equal priors assumption)
#
# Outputs:
#  - dprime_table.csv
#  - plots: means_vs_nkeep.png, vars_vs_nkeep.png, dprime_vs_nkeep_highres.png,
#           bayes_error_vs_nkeep.png, tpr_fpr_points.png
#  - report: report_dprime.md
#
# Example (high-resolution x-axis):
#   python gmm_dprime_beta_selection.py --inputs "all_min_l2_1024_*.npz" --out_dir dprime_outputs --n_min 1 --n_max 80 --n_step 1 --alpha 0.01

import argparse
import glob
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm


# ----------------------------
# NPZ loading (robust)
# ----------------------------

def _pick_2d_matrix(npz: Dict[str, np.ndarray], prefer_substr: Optional[str] = None) -> Optional[Tuple[str, np.ndarray]]:
    """
    Pick a candidate 2D matrix (num_images, K) from NPZ.
    - Prefer keys containing prefer_substr if provided.
    - Otherwise choose the largest 2D array (by number of rows).
    """
    candidates = []
    for k, v in npz.items():
        if isinstance(v, np.ndarray) and v.ndim == 2 and v.shape[1] >= 16:
            candidates.append((k, v))
    if not candidates:
        return None

    if prefer_substr is not None:
        pref = [(k, v) for (k, v) in candidates if prefer_substr.lower() in k.lower()]
        if pref:
            pref.sort(key=lambda kv: kv[1].shape[0], reverse=True)
            return pref[0]

    candidates.sort(key=lambda kv: kv[1].shape[0], reverse=True)
    return candidates[0]


def load_pooled(inputs_glob: str) -> Tuple[np.ndarray, np.ndarray, List[str], Tuple[str, str]]:
    """
    Load multiple NPZ files and pool them into:
      D_wm: (N_wm_images, K)
      D_rand: (N_rand_images, K)
    Returns also: list of file paths and the chosen keys (wm_key, rand_key) for the first file.
    """
    paths = sorted(glob.glob(inputs_glob))
    if not paths:
        raise FileNotFoundError(f"No files match inputs glob: {inputs_glob}")

    wm_list, rand_list = [], []
    first_keys = (None, None)

    for p in paths:
        data = np.load(p, allow_pickle=True)

        wm = (_pick_2d_matrix(data, "wm")
              or _pick_2d_matrix(data, "water")
              or _pick_2d_matrix(data, "mark"))
        rd = (_pick_2d_matrix(data, "rand")
              or _pick_2d_matrix(data, "random")
              or _pick_2d_matrix(data, "null"))

        if wm is None or rd is None:
            raise ValueError(
                f"Could not identify WM/Random 2D arrays in {p}.\n"
                f"Available keys: {list(data.keys())}\n"
                f"Tip: ensure the NPZ contains two 2D arrays (images x patches)."
            )

        wm_key, wm_mat = wm
        rd_key, rd_mat = rd

        if first_keys == (None, None):
            first_keys = (wm_key, rd_key)

        if wm_mat.shape[1] != rd_mat.shape[1]:
            raise ValueError(f"K mismatch in {p}: {wm_key} {wm_mat.shape} vs {rd_key} {rd_mat.shape}")

        wm_list.append(wm_mat.astype(np.float64))
        rand_list.append(rd_mat.astype(np.float64))

    D_wm = np.vstack(wm_list)
    D_rand = np.vstack(rand_list)
    return D_wm, D_rand, paths, (first_keys[0], first_keys[1])


# ----------------------------
# Core computations
# ----------------------------

def trimmed_means(D: np.ndarray, n_keep: int) -> np.ndarray:
    """
    Compute T_n(i) for each image i:
      sort each row (partial), take first n_keep, average.
    """
    if n_keep < 1 or n_keep > D.shape[1]:
        raise ValueError(f"n_keep must be in [1, K], got {n_keep} with K={D.shape[1]}")
    part = np.partition(D, kth=n_keep - 1, axis=1)[:, :n_keep]
    return part.mean(axis=1)


@dataclass
class GMMSignal:
    mu_signal: float
    var_signal: float
    mu_bg: float
    var_bg: float
    weights: Tuple[float, float]


def fit_gmm_signal_from_wm_patches(D_wm: np.ndarray, seed: int = 0) -> GMMSignal:
    """
    Fit a 2-component 1D GMM to pooled WM patches.
    Define "signal" component as the one with smaller mean.
    """
    x = D_wm.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=seed)
    gmm.fit(x)

    means = gmm.means_.flatten()
    covs = gmm.covariances_.flatten()  # 1D
    weights = gmm.weights_.flatten()

    idx_signal = int(np.argmin(means))
    idx_bg = 1 - idx_signal

    return GMMSignal(
        mu_signal=float(means[idx_signal]),
        var_signal=float(covs[idx_signal]),
        mu_bg=float(means[idx_bg]),
        var_bg=float(covs[idx_bg]),
        weights=(float(weights[0]), float(weights[1])),
    )


def compute_grid_metrics(
    D_wm: np.ndarray,
    D_rand: np.ndarray,
    n_grid: List[int],
    alpha: float
) -> pd.DataFrame:
    """
    For each n_keep:
      - compute T_wm, T_rand
      - mu/var on both
      - d'
      - tau = alpha-quantile of random T
      - empirical TPR/FPR
      - Bayes error proxy Pe ≈ Phi(-d'/2)
      - overlap area approx ≈ 2*Pe
    """
    rows = []
    for n in n_grid:
        T_w = trimmed_means(D_wm, n)
        T_r = trimmed_means(D_rand, n)

        mu_w = float(np.mean(T_w))
        mu_r = float(np.mean(T_r))
        var_w = float(np.var(T_w, ddof=1))
        var_r = float(np.var(T_r, ddof=1))

        denom = np.sqrt(var_w + var_r) if (var_w + var_r) > 0 else np.nan
        dprime = (mu_r - mu_w) / denom if np.isfinite(denom) and denom > 0 else np.nan

        tau = float(np.quantile(T_r, alpha))
        tpr = float(np.mean(T_w < tau))
        fpr = float(np.mean(T_r < tau))

        pe = float(norm.cdf(-0.5 * dprime)) if np.isfinite(dprime) else np.nan
        overlap_area = float(2.0 * pe) if np.isfinite(pe) else np.nan

        rows.append({
            "n_keep": int(n),
            "beta": float(n / D_wm.shape[1]),
            "mu_wm": mu_w,
            "mu_rand": mu_r,
            "var_wm": var_w,
            "var_rand": var_r,
            "dprime": dprime,
            "tau_alpha": tau,
            "TPR_at_tau": tpr,
            "FPR_at_tau": fpr,
            "bayes_error_Pe": pe,
            "overlap_area_approx": overlap_area,
        })

    return pd.DataFrame(rows)


# ----------------------------
# Crossing + insights
# ----------------------------

def find_crossing(n_vals: np.ndarray, y_vals: np.ndarray, level: float = 3.0):
    """
    Returns:
      - first_n_ge: first integer n where y >= level (or None)
      - n_cross_interp: linear-interpolated crossing (float) if bracket exists, else None
      - bracket: (n1, y1, n2, y2) used for interpolation, else None
    """
    n_vals = np.asarray(n_vals)
    y_vals = np.asarray(y_vals)

    idx = np.where(y_vals >= level)[0]
    first_n_ge = int(n_vals[idx[0]]) if len(idx) else None

    n_cross_interp = None
    bracket = None
    for k in range(1, len(y_vals)):
        y1, y2 = y_vals[k - 1], y_vals[k]
        if np.isfinite(y1) and np.isfinite(y2) and (y1 < level) and (y2 >= level):
            n1, n2 = float(n_vals[k - 1]), float(n_vals[k])
            if y2 != y1:
                t = (level - y1) / (y2 - y1)
                n_cross_interp = n1 + t * (n2 - n1)
            else:
                n_cross_interp = n2
            bracket = (n1, y1, n2, y2)
            break

    return first_n_ge, n_cross_interp, bracket


def summarize_insights(df: pd.DataFrame, ref: float = 3.0, plateau_frac: float = 0.95):
    """
    Insights:
      - peak d' and where
      - first n where d' >= ref and interpolated crossing
      - plateau range where d' >= plateau_frac * peak
      - conservative recommendation = smallest n in plateau (if exists), else peak
    """
    n = df["n_keep"].to_numpy()
    d = df["dprime"].to_numpy()

    d_finite = d[np.isfinite(d)]
    if len(d_finite) == 0:
        return {
            "n_peak": None, "d_peak": None,
            "first_n_ge_ref": None, "n_cross_interp": None,
            "plateau_range": None, "n_recommended": None
        }

    peak_idx = int(np.nanargmax(d))
    n_peak = int(n[peak_idx])
    d_peak = float(d[peak_idx])

    first_ge, n_cross_interp, _ = find_crossing(n, d, level=ref)

    plateau_mask = d >= plateau_frac * d_peak
    plateau_ns = n[plateau_mask]
    plateau_range = None
    n_reco = n_peak

    if len(plateau_ns):
        plateau_range = (int(np.min(plateau_ns)), int(np.max(plateau_ns)))
        n_reco = int(np.min(plateau_ns))

    return {
        "n_peak": n_peak,
        "d_peak": d_peak,
        "first_n_ge_ref": first_ge,
        "n_cross_interp": n_cross_interp,
        "plateau_range": plateau_range,
        "n_recommended": n_reco
    }


# ----------------------------
# Plotting
# ----------------------------

def save_plot_means(df: pd.DataFrame, mu_signal: float, out_path: str):
    plt.figure(figsize=(9, 4.8))
    plt.plot(df["n_keep"], df["mu_wm"], marker="o", markersize=3.5, linewidth=2, label="mean(T_n | WM)")
    plt.plot(df["n_keep"], df["mu_rand"], marker="o", markersize=3.5, linewidth=2, label="mean(T_n | Random)")
    plt.axhline(mu_signal, linestyle="--", linewidth=1.7, label="GMM signal mean μ_S (patch-level)")
    plt.xlabel("n_keep")
    plt.ylabel("Mean")
    plt.title("Means vs n_keep (with GMM signal mean)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_plot_vars(df: pd.DataFrame, var_signal: float, out_path: str):
    plt.figure(figsize=(9, 4.8))
    plt.plot(df["n_keep"], df["var_wm"], marker="o", markersize=3.5, linewidth=2, label="Var(T_n | WM)")
    plt.plot(df["n_keep"], df["var_rand"], marker="o", markersize=3.5, linewidth=2, label="Var(T_n | Random)")
    plt.axhline(var_signal, linestyle="--", linewidth=1.7, label="GMM signal var σ²_S (patch-level)")
    plt.xlabel("n_keep")
    plt.ylabel("Variance")
    plt.title("Variances vs n_keep (context)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_plot_bayes_error(df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(9, 4.8))
    plt.plot(df["n_keep"], df["bayes_error_Pe"], marker="o", markersize=3.5, linewidth=2,
             label="Pe ≈ Φ(-d'/2)")
    plt.xlabel("n_keep")
    plt.ylabel("Bayes error proxy Pe")
    plt.title("Approx. Bayes error from d' (Gaussian proxy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_plot_operating(df: pd.DataFrame, out_path: str):
    plt.figure(figsize=(7, 6))
    plt.plot(df["FPR_at_tau"], df["TPR_at_tau"], marker="o", markersize=4, linewidth=1.5)
    for _, r in df.iterrows():
        plt.text(r["FPR_at_tau"], r["TPR_at_tau"], str(int(r["n_keep"])), fontsize=8)
    plt.xlabel("FPR (empirical)")
    plt.ylabel("TPR (empirical)")
    plt.title("Operating points at τ = alpha-quantile(Random)")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def save_plot_dprime_highres(df: pd.DataFrame, out_path: str, ref: float = 3.0, plateau_frac: float = 0.95):
    n = df["n_keep"].to_numpy()
    d = df["dprime"].to_numpy()

    first_ge, n_cross_interp, _ = find_crossing(n, d, level=ref)
    insights = summarize_insights(df, ref=ref, plateau_frac=plateau_frac)

    d_finite = d[np.isfinite(d)]
    y_min = float(np.min(d_finite)) if len(d_finite) else 0.0
    y_max = float(np.max(d_finite)) if len(d_finite) else 1.0
    pad = 0.10 * (y_max - y_min + 1e-9)
    y0, y1 = y_min - pad, y_max + pad

    plt.figure(figsize=(10, 5.0))
    plt.plot(n, d, linewidth=2.2, label="d'(n_keep)")
    plt.scatter(n, d, s=18)

    plt.axhline(ref, linestyle="--", linewidth=1.6, label=f"d' = {ref}")

    # Mark peak
    if insights["n_peak"] is not None:
        plt.axvline(insights["n_peak"], linestyle=":", linewidth=1.6)
        plt.text(insights["n_peak"], y1, f"peak @ n={insights['n_peak']}",
                 rotation=90, va="top", ha="right", fontsize=9)

    # Mark first integer crossing
    if first_ge is not None:
        plt.axvline(first_ge, linestyle="--", linewidth=1.2)
        plt.text(first_ge, y0, f"first ≥{ref}: n={first_ge}",
                 rotation=90, va="bottom", ha="left", fontsize=9)

    # Mark interpolated crossing
    if n_cross_interp is not None:
        plt.axvline(n_cross_interp, linestyle="--", linewidth=1.0)
        plt.text(n_cross_interp, (y0 + y1) / 2, f"interp ≈ {n_cross_interp:.2f}",
                 rotation=90, va="center", ha="left", fontsize=9)

    # Plateau shading
    pr = insights["plateau_range"]
    if pr is not None:
        plt.axvspan(pr[0], pr[1], alpha=0.12)
        plt.text((pr[0] + pr[1]) / 2, y1, f"plateau ≥{int(plateau_frac*100)}% max: [{pr[0]},{pr[1]}]",
                 va="top", ha="center", fontsize=9)

    # X ticks: high resolution computed, but label sparsely
    if len(n) > 80:
        step = 5
    elif len(n) > 40:
        step = 2
    else:
        step = 1
    plt.xticks(n[::step])

    # Y ticks: choose step by range
    yr = y1 - y0
    ystep = 0.25 if yr <= 6 else 0.5
    yticks = np.arange(np.floor(y0 / ystep) * ystep, np.ceil(y1 / ystep) * ystep + ystep / 2, ystep)
    plt.yticks(yticks)

    plt.ylim(y0, y1)
    plt.xlabel("n_keep")
    plt.ylabel("d' (standardized separation)")
    plt.title("High-resolution d' vs n_keep (crossing + plateau)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=350)
    plt.close()


# ----------------------------
# Report writing
# ----------------------------

def write_report_with_insights(
    out_dir: str,
    paths_glob: str,
    pooled_paths: List[str],
    wm_key: str,
    rand_key: str,
    D_wm: np.ndarray,
    D_rand: np.ndarray,
    gmm: GMMSignal,
    df: pd.DataFrame,
    alpha: float,
    ref: float = 3.0,
    plateau_frac: float = 0.95
):
    insights = summarize_insights(df, ref=ref, plateau_frac=plateau_frac)

    # Also best-by-d' row
    df_sorted = df.sort_values("dprime", ascending=False)
    best = df_sorted.iloc[0]
    best_n = int(best["n_keep"])
    best_beta = float(best["beta"])
    best_dprime = float(best["dprime"])
    best_pe = float(best["bayes_error_Pe"])

    # first n achieving d' >= ref
    first_ge = insights["first_n_ge_ref"]
    n_cross_interp = insights["n_cross_interp"]

    lines = []
    lines.append("# GMM-signal + d' selection report (high-resolution)\n\n")
    lines.append(f"**Inputs glob:** `{paths_glob}`\n\n")
    lines.append("## Pooled dataset\n")
    lines.append(f"- Files pooled: {len(pooled_paths)}\n")
    lines.append(f"- Example keys used (first file): WM=`{wm_key}`, Random=`{rand_key}`\n")
    lines.append(f"- WM images: {D_wm.shape[0]}\n")
    lines.append(f"- Random images: {D_rand.shape[0]}\n")
    lines.append(f"- K patches/image: {D_wm.shape[1]}\n\n")

    lines.append("## Patch-level GMM (WM patches)\n")
    lines.append(f"- μ_signal = {gmm.mu_signal:.6f}, σ²_signal = {gmm.var_signal:.6f}\n")
    lines.append(f"- μ_bg = {gmm.mu_bg:.6f}, σ²_bg = {gmm.var_bg:.6f}\n")
    lines.append(f"- mixture weights = {gmm.weights}\n\n")

    lines.append("## d' results (standardized separation)\n")
    lines.append(f"- Peak/Best d' in grid: **{best_dprime:.3f}** at n_keep={best_n} (β={best_beta:.6f})\n")
    lines.append(f"- Bayes error proxy at best point: Pe≈Φ(-d'/2) = **{best_pe:.4f}**\n")

    if first_ge is None:
        lines.append(f"- d' does **not** reach {ref} in the evaluated grid.\n")
    else:
        lines.append(f"- First n_keep with d' ≥ {ref}: **{first_ge}**\n")
        if n_cross_interp is not None:
            lines.append(f"- Interpolated crossing of d'={ref}: **n ≈ {n_cross_interp:.2f}**\n")

    pr = insights["plateau_range"]
    if pr is not None:
        pr0, pr1 = pr
        lines.append(f"- Plateau (d' ≥ {int(plateau_frac*100)}% of max): **[{pr0}, {pr1}]**\n")
        lines.append(f"- Conservative recommendation (smallest n in plateau): **{insights['n_recommended']}**\n")
    else:
        lines.append(f"- No plateau detected at ≥{int(plateau_frac*100)}% of max.\n")

    lines.append("\n## Null-controlled thresholding\n")
    lines.append(f"For each n_keep we set τ = Q_alpha(T_n | Random) with alpha={alpha}, "
                 f"then report empirical TPR/FPR at that τ.\n\n")

    lines.append("## Output files\n")
    lines.append("- `dprime_table.csv`\n")
    lines.append("- `dprime_vs_nkeep_highres.png`\n")
    lines.append("- `means_vs_nkeep.png`\n")
    lines.append("- `vars_vs_nkeep.png`\n")
    lines.append("- `bayes_error_vs_nkeep.png`\n")
    lines.append("- `tpr_fpr_points.png`\n")

    report_path = os.path.join(out_dir, "report_dprime.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("".join(lines))


# ----------------------------
# Main
# ----------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", required=True, help='Glob for NPZ files, e.g. "all_min_l2_1024_*.npz"')
    ap.add_argument("--out_dir", default="dprime_outputs", help="Output directory")
    ap.add_argument("--n_min", type=int, default=1)
    ap.add_argument("--n_max", type=int, default=80)
    ap.add_argument("--n_step", type=int, default=1)
    ap.add_argument("--alpha", type=float, default=0.01, help="Quantile level for tau from Random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ref", type=float, default=3.0, help="Reference d' level to mark (default 3.0)")
    ap.add_argument("--plateau_frac", type=float, default=0.95, help="Plateau threshold fraction of peak d'")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    D_wm, D_rand, pooled_paths, (wm_key, rand_key) = load_pooled(args.inputs)
    K = D_wm.shape[1]
    if D_rand.shape[1] != K:
        raise ValueError("K mismatch after pooling.")

    n_max = min(args.n_max, K)
    if args.n_min < 1:
        raise ValueError("--n_min must be >= 1")
    if n_max < args.n_min:
        raise ValueError(f"--n_max ({n_max}) must be >= --n_min ({args.n_min})")

    n_grid = list(range(args.n_min, n_max + 1, args.n_step))

    gmm = fit_gmm_signal_from_wm_patches(D_wm, seed=args.seed)
    df = compute_grid_metrics(D_wm, D_rand, n_grid=n_grid, alpha=args.alpha)

    # Save table
    csv_path = os.path.join(args.out_dir, "dprime_table.csv")
    df.to_csv(csv_path, index=False)

    # Plots
    save_plot_means(df, gmm.mu_signal, os.path.join(args.out_dir, "means_vs_nkeep.png"))
    save_plot_vars(df, gmm.var_signal, os.path.join(args.out_dir, "vars_vs_nkeep.png"))
    save_plot_dprime_highres(
        df,
        os.path.join(args.out_dir, "dprime_vs_nkeep_highres.png"),
        ref=args.ref,
        plateau_frac=args.plateau_frac
    )
    save_plot_bayes_error(df, os.path.join(args.out_dir, "bayes_error_vs_nkeep.png"))
    save_plot_operating(df, os.path.join(args.out_dir, "tpr_fpr_points.png"))

    # Report
    write_report_with_insights(
        args.out_dir,
        args.inputs,
        pooled_paths,
        wm_key,
        rand_key,
        D_wm,
        D_rand,
        gmm,
        df,
        alpha=args.alpha,
        ref=args.ref,
        plateau_frac=args.plateau_frac
    )

    # Console summary with crossing + insights
    insights = summarize_insights(df, ref=args.ref, plateau_frac=args.plateau_frac)
    best = df.sort_values("dprime", ascending=False).iloc[0]

    print("\n=== GMM signal + d' selection summary (high-resolution) ===")
    print(f"Files pooled: {len(pooled_paths)}")
    print(f"Images: WM={D_wm.shape[0]}, Random={D_rand.shape[0]}, K={K}")
    print(f"GMM means (signal,bg by min/other): μ_S={gmm.mu_signal:.6f}, μ_BG={gmm.mu_bg:.6f}")
    print(f"GMM vars: σ²_S={gmm.var_signal:.6f}, σ²_BG={gmm.var_bg:.6f}, weights={gmm.weights}")

    print(f"\nBest d' in grid: d'={best['dprime']:.3f} at n_keep={int(best['n_keep'])} (beta={best['beta']:.6f})")
    print(f"Bayes error proxy at best point: Pe≈Φ(-d'/2)={best['bayes_error_Pe']:.4f}")

    if insights["first_n_ge_ref"] is None:
        print(f"\nCrossing: d' never reaches {args.ref} in evaluated grid.")
    else:
        print(f"\nCrossing: first n_keep with d'≥{args.ref} is n_keep={insights['first_n_ge_ref']}")
        if insights["n_cross_interp"] is not None:
            print(f"Interpolated crossing: n≈{insights['n_cross_interp']:.2f}")

    if insights["plateau_range"] is not None:
        pr0, pr1 = insights["plateau_range"]
        print(f"Plateau (≥{int(args.plateau_frac*100)}% of peak): [{pr0}, {pr1}]")
        print(f"Recommended n_keep (smallest in plateau): {insights['n_recommended']}")

    print(f"\nSaved table: {csv_path}")
    print(f"Saved report: {os.path.join(args.out_dir, 'report_dprime.md')}")
    print(f"Saved plots in: {args.out_dir}")


if __name__ == "__main__":
    main()
