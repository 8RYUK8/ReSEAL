#!/usr/bin/env python3
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

# This script analyzes image-level watermark detection by averaging the lowest-beta
# fraction of patch-wise L2 distances per image. It evaluates separability between
# watermarked and random images using ROC/AUC and a fixed threshold tau, and also
# visualizes how this trimmed-mean score changes as beta varies.

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


def compute_trimmed_means(wm: np.ndarray, rnd: np.ndarray, beta: float):
    """
    Given watermarked/random arrays (N_images, N_patches) and beta,
    return per-image trimmed-mean L2 for each.
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")

    n_wm, k = wm.shape
    n_rnd, k2 = rnd.shape
    if k != k2:
        raise ValueError("Number of patches must match.")

    n_keep = int(beta * k)
    if n_keep < 1:
        n_keep = 1

    wm_sorted = np.sort(wm, axis=1)
    rnd_sorted = np.sort(rnd, axis=1)

    wm_trimmed_mean = wm_sorted[:, :n_keep].mean(axis=1)
    rnd_trimmed_mean = rnd_sorted[:, :n_keep].mean(axis=1)

    return wm_trimmed_mean, rnd_trimmed_mean, n_keep


def analyze_trimmed_mean(path: str, beta: float, tau: float) -> None:
    """
    Image-level analysis using mean of the lowest β-fraction of L2 distances per image.
    """
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

    n_wm, k = wm.shape
    n_rnd, _ = rnd.shape
    base = os.path.splitext(os.path.basename(path))[0]

    print(f"Loaded file: {path}")
    print(f"  watermarked shape: {wm.shape} (N_wm_images={n_wm}, N_patches={k})")
    print(f"  random      shape: {rnd.shape} (N_rand_images={n_rnd}, N_patches={k})")
    print(f"  beta (fraction of lowest patches used) = {beta:.3f}")
    print(f"  tau  (fixed image-level threshold)     = {tau:.4f}")
    print("-" * 60)

    # ---------- 1. Compute trimmed mean (lowest β-fraction) per image ----------
    wm_trimmed_mean, rnd_trimmed_mean, n_keep = compute_trimmed_means(wm, rnd, beta)

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
    print(f"  τ (manual)          = {tau:.6f}")
    print(f"  P(random score < τ) ≈ {frac_rnd_below:.4f}  (empirical FPR at τ)")
    print(f"  P(wm score    < τ) ≈ {frac_wm_below:.4f}  (empirical TPR at τ)")

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
    print(f"  accuracy={acc:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}")

    # ---------- 4. Plot: histogram of scores with τ ----------
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

    # ---------- 5. Plot: ECDF of scores ----------
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

    # ---------- 6. Plot: ROC curve for the score ----------
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
    plt.plot(fpr_list, tpr_list, label=f"trimmed-mean ROC (beta={beta:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="random guess")

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


def plot_beta_curve(path: str) -> None:
    """
    Plot how the *mean* trimmed-mean L2 changes as we vary beta,
    averaged across all images (watermarked vs random).
    """
    data = np.load(path)
    if "watermarked" not in data.files or "random" not in data.files:
        raise KeyError(
            f"Expected 'watermarked' and 'random' arrays in {path}, "
            f"found: {data.files}"
        )

    wm = np.asarray(data["watermarked"])
    rnd = np.asarray(data["random"])
    base = os.path.splitext(os.path.basename(path))[0]

    betas = np.arange(0.01, 1.001, 0.05)

    wm_means = []
    rnd_means = []

    for beta in betas:
        wm_tm, rnd_tm, n_keep = compute_trimmed_means(wm, rnd, beta)
        wm_means.append(wm_tm.mean())
        rnd_means.append(rnd_tm.mean())

    wm_means = np.array(wm_means)
    rnd_means = np.array(rnd_means)

    plt.figure(figsize=(8, 6))
    plt.plot(betas, wm_means, marker="o", label="watermarked (mean)")
    plt.plot(betas, rnd_means, marker="s", label="random (mean)")

    plt.xlabel("beta (fraction of lowest-L2 patches used)")
    plt.ylabel("mean image-level trimmed-mean L2")
    plt.title("Mean trimmed-mean L2 vs beta (averaged over images)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    out_path = f"{base}_beta_vs_mean_trimmedL2.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved beta-curve plot to: {out_path}")


def plot_beta_envelope(path: str) -> None:
    """
    Plot beta vs trimmed-mean L2 for:
      - the worst watermarked image at each beta (max trimmed mean)
      - the best random image at each beta (min trimmed mean)

    Note: the actual image index can change with beta; these curves are
    envelopes, not single fixed images.
    """
    data = np.load(path)
    if "watermarked" not in data.files or "random" not in data.files:
        raise KeyError(
            f"Expected 'watermarked' and 'random' arrays in {path}, "
            f"found: {data.files}"
        )

    wm = np.asarray(data["watermarked"])
    rnd = np.asarray(data["random"])
    base = os.path.splitext(os.path.basename(path))[0]

    betas = np.arange(0.01, 1.001, 0.05)

    worst_wm_means = []
    best_rnd_means = []

    worst_wm_idx_per_beta = []
    best_rnd_idx_per_beta = []

    for beta in betas:
        wm_tm, rnd_tm, n_keep = compute_trimmed_means(wm, rnd, beta)

        worst_wm_idx = int(np.argmax(wm_tm))
        best_rnd_idx = int(np.argmin(rnd_tm))

        worst_wm_means.append(wm_tm[worst_wm_idx])
        best_rnd_means.append(rnd_tm[best_rnd_idx])

        worst_wm_idx_per_beta.append(worst_wm_idx)
        best_rnd_idx_per_beta.append(best_rnd_idx)

    worst_wm_means = np.array(worst_wm_means)
    best_rnd_means = np.array(best_rnd_means)

    plt.figure(figsize=(8, 6))
    plt.plot(betas, worst_wm_means, marker="o", label="worst watermarked (max mean)")
    plt.plot(betas, best_rnd_means, marker="s", label="best random (min mean)")

    plt.xlabel("beta (fraction of lowest-L2 patches used)")
    plt.ylabel("trimmed-mean L2")
    plt.title("Envelope: worst watermarked vs best random across beta")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = f"{base}_beta_envelope_worstwm_bestrand.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved worst-wm / best-random beta-envelope plot to: {out_path}")

    print("\n[Info] Example extreme indices at a few betas:")
    for beta, w_idx, r_idx in zip(
        betas[::5], worst_wm_idx_per_beta[::5], best_rnd_idx_per_beta[::5]
    ):
        print(
            f"  beta={beta:.2f}: worst watermarked idx={w_idx}, best random idx={r_idx}"
        )


def plot_beta_curve_single(path: str, img_idx: int, image_type: str = "watermarked") -> None:
    """
    Plot beta vs trimmed-mean L2 *for a single image*.

    image_type: "watermarked" or "random"
    img_idx: index within that array.
    """
    data = np.load(path)
    if "watermarked" not in data.files or "random" not in data.files:
        raise KeyError(
            f"Expected 'watermarked' and 'random' arrays in {path}, "
            f"found: {data.files}"
        )

    wm = np.asarray(data["watermarked"])
    rnd = np.asarray(data["random"])

    if image_type == "watermarked":
        arr = wm
    else:
        arr = rnd

    n_images, k = arr.shape
    if not (0 <= img_idx < n_images):
        raise IndexError(
            f"example_index {img_idx} is out of range for {image_type} images "
            f"(0..{n_images-1})"
        )

    base = os.path.splitext(os.path.basename(path))[0]
    npz_dir = os.path.dirname(path) or "."
    out_dir = os.path.join(npz_dir, f"{base}_image_beta_mean_curves")
    os.makedirs(out_dir, exist_ok=True)

    betas = np.arange(0.01, 1.001, 0.05)
    scores = []

    row = np.sort(arr[img_idx])  # sorted patch L2 for that one image

    for beta in betas:
        n_keep = int(beta * k)
        if n_keep < 1:
            n_keep = 1
        score = row[:n_keep].mean()
        scores.append(score)

    scores = np.array(scores)

    plt.figure(figsize=(8, 6))
    plt.plot(betas, scores, marker="o")

    plt.xlabel("beta (fraction of lowest-L2 patches used)")
    plt.ylabel("trimmed-mean L2 (single image)")
    plt.title(
        f"Trimmed-mean L2 vs beta\n{image_type} image index {img_idx}"
    )
    plt.grid(True)
    plt.tight_layout()

    fname = f"{image_type}_idx{img_idx}_beta_vs_trimmedL2.png"
    out_path = os.path.join(out_dir, fname)
    plt.savefig(out_path, dpi=150)
    print(f"Saved single-image beta-curve plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SEAL NPZ using image-level mean of the lowest β-fraction of L2s.\n"
            "Also plots how the mean trimmed-mean L2 changes as beta varies,\n"
            "an envelope of worst watermarked vs best random, and optionally "
            "a per-image beta curve."
        )
    )
    parser.add_argument(
        "npz_file",
        type=str,
        help=(
            "Path to NPZ file with 'watermarked' and 'random' arrays "
            "(e.g. all_patch_l2_1024_7.npz)."
        ),
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help=(
            "Fraction of lowest-L2 patches per image used in the main analysis "
            "(0 < beta <= 1)."
        ),
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=2.8,
        help="Fixed image-level threshold on trimmed-mean L2 (default: 2.8).",
    )
    parser.add_argument(
        "--example_index",
        type=int,
        default=None,
        help=(
            "If set, also plot beta-vs-trimmed-mean curve for a single image "
            "(index within chosen image_type array)."
        ),
    )
    parser.add_argument(
        "--example_type",
        type=str,
        default="watermarked",
        choices=["watermarked", "random"],
        help="Which set the example_index refers to (default: watermarked).",
    )

    args = parser.parse_args()

    analyze_trimmed_mean(args.npz_file, beta=args.beta, tau=args.tau)
    plot_beta_curve(args.npz_file)
    plot_beta_envelope(args.npz_file)

    if args.example_index is not None:
        plot_beta_curve_single(
            args.npz_file, args.example_index, image_type=args.example_type
        )


if __name__ == "__main__":
    main()
