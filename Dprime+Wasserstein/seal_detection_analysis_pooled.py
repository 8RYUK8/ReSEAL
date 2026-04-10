#!/usr/bin/env python3
import argparse
import os
from typing import List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute ROC AUC without external libraries.
    y_true: 1 for watermarked, 0 for random
    y_score: larger = more likely watermarked
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    order = np.argsort(y_score)
    y_true_sorted = y_true[order]

    n_pos = np.sum(y_true_sorted == 1)
    n_neg = np.sum(y_true_sorted == 0)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both positive and negative samples are required for AUC.")

    ranks = np.arange(1, len(y_true_sorted) + 1, dtype=float)
    R_pos = np.sum(ranks[y_true_sorted == 1])

    auc = (R_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def ecdf(x: np.ndarray):
    """Simple empirical CDF helper."""
    x = np.sort(x)
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def _extract_wm_random_from_npz(npz_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Supports two NPZ layouts:
      A) keys: 'watermarked' and 'random' directly, each shape (N,K)
      B) single key like '1024_7' where value is a dict with those arrays
    """
    data = np.load(npz_path, allow_pickle=True)
    files = list(data.files)

    if "watermarked" in files and "random" in files:
        wm = np.asarray(data["watermarked"])
        rnd = np.asarray(data["random"])
        return wm, rnd

    # fallback: single object entry that is a dict
    if len(files) == 1:
        obj = data[files[0]].item()
        if isinstance(obj, dict) and "watermarked" in obj and "random" in obj:
            wm = np.asarray(obj["watermarked"])
            rnd = np.asarray(obj["random"])
            return wm, rnd

    raise KeyError(
        f"{npz_path}: Expected ('watermarked','random') arrays or a single dict entry "
        f"containing them. Found keys: {files}"
    )


def load_and_pool_npz(npz_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load multiple NPZ files and concatenate them along the image dimension:
      wm_all shape: (sum_i Nw_i, K)
      rnd_all shape: (sum_i Nr_i, K)
    """
    wm_list, rnd_list = [], []
    K_ref = None

    for p in npz_paths:
        wm, rnd = _extract_wm_random_from_npz(p)

        if wm.ndim != 2 or rnd.ndim != 2:
            raise ValueError(
                f"{p}: 'watermarked' and 'random' must be 2D (N_images, N_patches). "
                f"Got watermarked={wm.shape}, random={rnd.shape}"
            )
        if wm.shape[1] != rnd.shape[1]:
            raise ValueError(
                f"{p}: Number of patches must match between watermarked and random. "
                f"Got {wm.shape[1]} vs {rnd.shape[1]}"
            )

        if K_ref is None:
            K_ref = wm.shape[1]
        elif wm.shape[1] != K_ref:
            raise ValueError(
                f"{p}: Patch count K mismatch vs previous files. Got K={wm.shape[1]} vs K_ref={K_ref}"
            )

        wm_list.append(wm)
        rnd_list.append(rnd)

    wm_all = np.concatenate(wm_list, axis=0)
    rnd_all = np.concatenate(rnd_list, axis=0)
    return wm_all, rnd_all


def analyze_trimmed_mean(
    npz_paths: Union[str, List[str]],
    beta: float,
    tau: float,
    out_dir: str,
    out_prefix: str,
) -> None:
    """
    Image-level analysis using mean of the lowest β-fraction of L2 distances per image.

    - Load and pool NPZ(s) with 'watermarked' and 'random', each shape (N_images, N_patches).
    - For each image:
        * sort its patch L2 values ascending
        * take the lowest β * N_patches values
        * compute their mean -> this is the image's score.
    - Smaller score => more watermarked-like.
    - τ is fixed manually.
    - Images with score < τ are classified as watermarked.
    - Plots:
        * histogram of scores (wm vs random) with τ line,
        * ECDF of scores with τ and empirical P(random < τ),
        * ROC curve for the score, marking operating point at τ.
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")

    if isinstance(npz_paths, str):
        npz_paths = [npz_paths]

    os.makedirs(out_dir, exist_ok=True)

    wm, rnd = load_and_pool_npz(list(npz_paths))

    n_wm, k = wm.shape
    n_rnd, _ = rnd.shape

    print("Loaded & pooled files:")
    for p in npz_paths:
        print(f"  - {p}")
    print(f"\nPooled shapes:")
    print(f"  watermarked: {wm.shape} (N_wm_images={n_wm}, N_patches={k})")
    print(f"  random     : {rnd.shape} (N_rand_images={n_rnd}, N_patches={k})")
    print(f"  beta (fraction of lowest patches used) = {beta:.3f}")
    print(f"  tau (fixed image-level threshold)      = {tau:.6f}")
    print("-" * 60)

    # ---------- 1. Compute trimmed mean (lowest β-fraction) per image ----------
    n_keep = int(beta * k)
    if n_keep < 1:
        n_keep = 1

    wm_sorted = np.sort(wm, axis=1)
    rnd_sorted = np.sort(rnd, axis=1)

    wm_trimmed_mean = wm_sorted[:, :n_keep].mean(axis=1)
    rnd_trimmed_mean = rnd_sorted[:, :n_keep].mean(axis=1)

    print(f"Using n_keep = {n_keep} patches per image (lowest {beta*100:.1f}%)")
    print("Image-level 'trimmed mean' L2 stats:")
    print(
        f"  watermarked: mean={wm_trimmed_mean.mean():.6f}, "
        f"std={wm_trimmed_mean.std():.6f}, "
        f"min={wm_trimmed_mean.min():.6f}, max={wm_trimmed_mean.max():.6f}"
    )
    print(
        f"  random     : mean={rnd_trimmed_mean.mean():.6f}, "
        f"std={rnd_trimmed_mean.std():.6f}, "
        f"min={rnd_trimmed_mean.min():.6f}, max={rnd_trimmed_mean.max():.6f}"
    )

    # ---------- 2. ROC AUC using trimmed mean as score ----------
    # smaller score => more likely watermarked, so use negative for ROC
    y_true_img = np.concatenate(
        [
            np.ones_like(wm_trimmed_mean, dtype=int),
            np.zeros_like(rnd_trimmed_mean, dtype=int),
        ]
    )
    scores_img = np.concatenate([-wm_trimmed_mean, -rnd_trimmed_mean])
    auc_img = roc_auc_score_manual(y_true_img, scores_img)
    print(f"\nImage-level ROC AUC (using trimmed mean L2): {auc_img:.6f}")

    # ---------- 3. Use fixed τ for classification ----------
    frac_rnd_below = (rnd_trimmed_mean < tau).mean()
    frac_wm_below = (wm_trimmed_mean < tau).mean()

    print("\nImage-level threshold τ (fixed, on trimmed mean L2):")
    print(f"  τ (manual)           = {tau:.6f}")
    print(f"  P(random score < τ)  = {frac_rnd_below:.6f}  (empirical FPR at τ)")
    print(f"  P(wm score    < τ)   = {frac_wm_below:.6f}  (empirical TPR at τ)")

    # ---------- 4. Confusion numbers ----------
    all_scores = np.concatenate([wm_trimmed_mean, rnd_trimmed_mean])
    pred_img = (all_scores < tau).astype(int)  # 1 = watermarked

    tp = int(np.sum((pred_img == 1) & (y_true_img == 1)))
    tn = int(np.sum((pred_img == 0) & (y_true_img == 0)))
    fp = int(np.sum((pred_img == 1) & (y_true_img == 0)))
    fn = int(np.sum((pred_img == 0) & (y_true_img == 1)))

    tpr = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    acc = (tp + tn) / len(y_true_img)

    print("\nImage-level detection with fixed τ on trimmed mean L2:")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  accuracy={acc:.6f}, TPR={tpr:.6f}, FPR={fpr:.6f}")

    # ---------- 5. Plot: histogram ----------
    plt.figure(figsize=(8, 6))
    bins = 80
    plt.hist(wm_trimmed_mean, bins=bins, density=True, alpha=0.6, label="watermarked")
    plt.hist(rnd_trimmed_mean, bins=bins, density=True, alpha=0.6, label="random")
    plt.axvline(tau, linestyle="--", linewidth=2, label=f"τ = {tau:.3f}")

    plt.xlabel("image-level score (trimmed mean L2)")
    plt.ylabel("density")
    plt.title(f"Pooled score distribution (beta={beta:.3f}, tau={tau:.3f})")
    plt.legend()
    plt.tight_layout()

    score_hist_path = os.path.join(
        out_dir, f"{out_prefix}_trimmed_mean_hist_beta{beta:.3f}_tau{tau:.3f}.png"
    )
    plt.savefig(score_hist_path, dpi=180)
    plt.close()
    print(f"Saved histogram to: {score_hist_path}")

    # ---------- 6. Plot: ECDF ----------
    plt.figure(figsize=(8, 6))
    x_rnd, y_rnd = ecdf(rnd_trimmed_mean)
    x_wm, y_wm = ecdf(wm_trimmed_mean)

    plt.plot(x_rnd, y_rnd, label="random (ECDF)")
    plt.plot(x_wm, y_wm, label="watermarked (ECDF)")
    plt.axvline(tau, linestyle="--", label=f"τ = {tau:.3f}")
    plt.axhline(frac_rnd_below, linestyle=":", label=f"FPR≈{frac_rnd_below:.4f}")

    plt.xlabel("image-level score (trimmed mean L2)")
    plt.ylabel("CDF")
    plt.title(f"Pooled ECDF (beta={beta:.3f}, tau={tau:.3f})")
    plt.legend()
    plt.tight_layout()

    cdf_path = os.path.join(
        out_dir, f"{out_prefix}_trimmed_mean_cdf_beta{beta:.3f}_tau{tau:.3f}.png"
    )
    plt.savefig(cdf_path, dpi=180)
    plt.close()
    print(f"Saved ECDF to: {cdf_path}")

    # ---------- 7. Plot: ROC curve ----------
    scores_raw = np.concatenate([wm_trimmed_mean, rnd_trimmed_mean])
    thresholds = np.unique(scores_raw)

    tpr_list = []
    fpr_list = []
    for thr in thresholds:
        pred = (scores_raw <= thr).astype(int)
        tp_ = np.sum((pred == 1) & (y_true_img == 1))
        fn_ = np.sum((pred == 0) & (y_true_img == 1))
        fp_ = np.sum((pred == 1) & (y_true_img == 0))
        tn_ = np.sum((pred == 0) & (y_true_img == 0))
        tpr_list.append(tp_ / (tp_ + fn_ + 1e-12))
        fpr_list.append(fp_ / (fp_ + tn_ + 1e-12))

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_list, tpr_list, label=f"trimmed-mean ROC (beta={beta:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="random guess")

    idx_tau = np.argmin(np.abs(thresholds - tau))
    plt.scatter(
        fpr_list[idx_tau],
        tpr_list[idx_tau],
        marker="o",
        color="red",
        label=f"operating point at τ={tau:.3f}",
    )

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"Pooled ROC (AUC={auc_img:.4f})")
    plt.legend()
    plt.tight_layout()

    roc_path = os.path.join(
        out_dir, f"{out_prefix}_trimmed_mean_roc_beta{beta:.3f}_tau{tau:.3f}.png"
    )
    plt.savefig(roc_path, dpi=180)
    plt.close()
    print(f"Saved ROC to: {roc_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Pooled trimmed-mean verification over multiple NPZ files.\n"
            "For each image: sort patch L2s, average lowest beta*K patches -> score.\n"
            "Classify watermarked if score < tau. Produces histogram, ECDF, ROC."
        )
    )
    parser.add_argument(
        "npz_files",
        nargs="+",
        help="One or more NPZ files. They will be concatenated and treated as one big dataset.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Fraction of lowest-L2 patches per image used in the mean (0 < beta <= 1).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=2.8,
        help="Fixed image-level threshold on trimmed-mean L2.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="pooled_trimmed_mean_out",
        help="Directory to save plots.",
    )
    parser.add_argument(
        "--out_prefix",
        type=str,
        default="pooled",
        help="Prefix for output plot filenames.",
    )
    args = parser.parse_args()

    analyze_trimmed_mean(
        args.npz_files,
        beta=args.beta,
        tau=args.tau,
        out_dir=args.out_dir,
        out_prefix=args.out_prefix,
    )


if __name__ == "__main__":
    main()