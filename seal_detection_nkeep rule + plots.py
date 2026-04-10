#!/usr/bin/env python3
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

# This script performs image-level watermark detection using the trimmed mean of
# patch-wise L2 distances. For each image, it averages the lowest beta fraction
# of patch distances, uses this value as a score, applies a fixed threshold tau
# for classification, and visualizes the score distributions and ROC behavior.

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


def analyze_trimmed_mean(path: str, beta: float, tau: float) -> None:
    """
    Image-level analysis using mean of the lowest β-fraction of L2 distances per image.

    Steps:
        - Load NPZ with 'watermarked' and 'random', each shape (N_images, N_patches).
        - For each image:
            * sort its patch L2 values ascending
            * take the lowest β * N_patches values
            * compute their mean -> this is the image's score.
        - Use these scores as detection scores (smaller = more watermarked).
        - τ is provided manually (e.g. 2.8), not derived from quantiles.
        - Images with score < τ are classified as watermarked.
        - Plots:
            * histogram of scores (wm vs random) with τ line,
            * ECDF of scores with τ and empirical P(random < τ),
            * ROC curve for the score, marking operating point at τ.
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")

    data = np.load(path)

    if "watermarked" not in data.files or "random" not in data.files:
        raise KeyError(
            f"Expected 'watermarked' and 'random' arrays in {path}, "
            f"found: {data.files}"
        )

    wm = np.asarray(data["watermarked"])
    rnd = np.asarray(data["random"])

    if wm.ndim != 2 or rnd.ndim != 2:
        raise ValueError(
            f"'watermarked' and 'random' must be 2D (N_images, N_patches). "
            f"Got shapes: watermarked={wm.shape}, random={rnd.shape}"
        )

    if wm.shape[1] != rnd.shape[1]:
        raise ValueError(
            f"Number of patches must match between watermarked and random. "
            f"Got {wm.shape[1]} vs {rnd.shape[1]}"
        )

    n_wm, k = wm.shape
    n_rnd, _ = rnd.shape

    base = os.path.splitext(os.path.basename(path))[0]

    print(f"Loaded file: {path}")
    print(f"  watermarked shape: {wm.shape} (N_wm_images={n_wm}, N_patches={k})")
    print(f"  random      shape: {rnd.shape} (N_rand_images={n_rnd}, N_patches={k})")
    print(f"  beta (fraction of lowest patches used) = {beta:.3f}")
    print(f"  tau (fixed image-level threshold)      = {tau:.4f}")
    print("-" * 60)

    # ---------- 1. Compute trimmed mean (lowest β-fraction) per image ----------
    n_keep = int(beta * k)
    if n_keep < 1:
        n_keep = 1  # ensure at least one patch is used

    wm_sorted = np.sort(wm, axis=1)
    rnd_sorted = np.sort(rnd, axis=1)

    wm_trimmed_mean = wm_sorted[:, :n_keep].mean(axis=1)
    rnd_trimmed_mean = rnd_sorted[:, :n_keep].mean(axis=1)

    print(f"Using n_keep = {n_keep} patches per image (lowest {beta*100:.1f}%)")
    print("Image-level 'trimmed mean' L2 stats:")
    print(
        f"  watermarked: mean={wm_trimmed_mean.mean():.4f}, "
        f"std={wm_trimmed_mean.std():.4f}, "
        f"min={wm_trimmed_mean.min():.4f}, max={wm_trimmed_mean.max():.4f}"
    )
    print(
        f"  random     : mean={rnd_trimmed_mean.mean():.4f}, "
        f"std={rnd_trimmed_mean.std():.4f}, "
        f"min={rnd_trimmed_mean.min():.4f}, max={rnd_trimmed_mean.max():.4f}"
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
    print(f"\nImage-level ROC AUC (using trimmed mean L2): {auc_img:.4f}")

    # ---------- 3. Use fixed τ for classification ----------
    frac_rnd_below = (rnd_trimmed_mean < tau).mean()
    frac_wm_below = (wm_trimmed_mean < tau).mean()

    print("\nImage-level threshold τ (fixed, on trimmed mean L2):")
    print(f"  τ (manual)       = {tau:.6f}")
    print(f"  P(random score < τ) ≈ {frac_rnd_below:.4f}  (empirical FPR at τ)")
    print(f"  P(wm score    < τ) ≈ {frac_wm_below:.4f}  (empirical TPR at τ)")

    # ---------- 4. Classify images using τ ----------
    all_scores = np.concatenate([wm_trimmed_mean, rnd_trimmed_mean])
    pred_img = (all_scores < tau).astype(int)  # 1 = watermarked, 0 = non-watermarked

    tp = int(np.sum((pred_img == 1) & (y_true_img == 1)))
    tn = int(np.sum((pred_img == 0) & (y_true_img == 0)))
    fp = int(np.sum((pred_img == 1) & (y_true_img == 0)))
    fn = int(np.sum((pred_img == 0) & (y_true_img == 1)))

    tpr = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    acc = (tp + tn) / len(y_true_img)

    print("\nImage-level detection with fixed τ on trimmed mean L2:")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(
        f"  accuracy={acc:.4f}, "
        f"TPR={tpr:.4f}, "
        f"FPR={fpr:.4f}"
    )

    # ---------- 5. Plot: histogram of scores with τ ----------
    plt.figure(figsize=(8, 6))
    bins = 80

    plt.hist(
        wm_trimmed_mean,
        bins=bins,
        density=True,
        alpha=0.6,
        label="watermarked",
    )
    plt.hist(
        rnd_trimmed_mean,
        bins=bins,
        density=True,
        alpha=0.6,
        label="random",
    )

    plt.axvline(tau, linestyle="--", linewidth=2, label=f"τ = {tau:.3f}")

    plt.xlabel("image-level score (trimmed mean L2)")
    plt.ylabel("density")
    plt.title(f"Image-level score distribution (beta={beta:.2f}, tau={tau:.2f})")
    plt.legend()
    plt.tight_layout()

    score_hist_path = f"{base}_trimmed_mean_hist_beta{beta:.2f}_tau{tau:.2f}.png"
    plt.savefig(score_hist_path, dpi=150)
    print(f"Saved image-level score histogram to: {score_hist_path}")

    # ---------- 6. Plot: ECDF of scores ----------
    plt.figure(figsize=(8, 6))

    x_rnd, y_rnd = ecdf(rnd_trimmed_mean)
    x_wm, y_wm = ecdf(wm_trimmed_mean)

    plt.plot(x_rnd, y_rnd, label="random (ECDF)")
    plt.plot(x_wm, y_wm, label="watermarked (ECDF)")

    plt.axvline(tau, linestyle="--", label=f"τ = {tau:.3f}")
    plt.axhline(
        frac_rnd_below,
        linestyle=":",
        label=f"frac_random_below_τ ≈ {frac_rnd_below:.3f}",
    )

    plt.xlabel("image-level score (trimmed mean L2)")
    plt.ylabel("CDF")
    plt.title(f"ECDF of scores (beta={beta:.2f}, tau={tau:.2f})")
    plt.legend()
    plt.tight_layout()

    cdf_path = f"{base}_trimmed_mean_cdf_beta{beta:.2f}_tau{tau:.2f}.png"
    plt.savefig(cdf_path, dpi=150)
    print(f"Saved ECDF plot to: {cdf_path}")

    # ---------- 7. Plot: ROC curve for the score ----------
    # Use raw scores: smaller = more wm-like, thresholding as score <= thr
    scores_raw = np.concatenate([wm_trimmed_mean, rnd_trimmed_mean])
    thresholds = np.unique(scores_raw)

    tpr_list = []
    fpr_list = []

    for thr in thresholds:
        pred = (scores_raw <= thr).astype(int)  # <= thr -> predict watermarked
        tp_ = np.sum((pred == 1) & (y_true_img == 1))
        fn_ = np.sum((pred == 0) & (y_true_img == 1))
        fp_ = np.sum((pred == 1) & (y_true_img == 0))
        tn_ = np.sum((pred == 0) & (y_true_img == 0))

        tpr_list.append(tp_ / (tp_ + fn_ + 1e-12))
        fpr_list.append(fp_ / (fp_ + tn_ + 1e-12))

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_list, tpr_list, label=f"trimmed-mean ROC (beta={beta:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="random guess")

    # mark the operating point corresponding to τ
    idx_tau = np.argmin(np.abs(thresholds - tau))
    plt.scatter(
        fpr_list[idx_tau],
        tpr_list[idx_tau],
        marker="o",
        color="red",
        label=f"operating point at τ={tau:.2f}",
    )

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title("Image-level ROC (trimmed-mean score)")
    plt.legend()
    plt.tight_layout()

    roc_path = f"{base}_trimmed_mean_roc_beta{beta:.2f}_tau{tau:.2f}.png"
    plt.savefig(roc_path, dpi=150)
    print(f"Saved ROC curve to: {roc_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SEAL NPZ using image-level mean of the lowest β-fraction of L2s:\n"
            "  For each image, sort patch L2s, average the lowest β * N_patches,\n"
            "  and use that as the score. τ is fixed manually (e.g. 2.8), and\n"
            "  images with score < τ are classified as watermarked.\n"
            "  Also produces histogram, ECDF, and ROC plots."
        )
    )
    parser.add_argument(
        "npz_file",
        type=str,
        help="Path to NPZ file with 'watermarked' and 'random' arrays "
             "(e.g. all_patch_l2_1024_7.npz).",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help=(
            "Fraction of lowest-L2 patches per image used in the mean "
            "(0 < beta <= 1, e.g. 1.0=all, 0.5=lowest 50%%, 0.3=lowest 30%%)."
        ),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=2.8,
        help="Fixed image-level threshold on trimmed-mean L2 (default: 2.8).",
    )
    args = parser.parse_args()
    analyze_trimmed_mean(args.npz_file, beta=args.beta, tau=args.tau)


if __name__ == "__main__":
    main()
