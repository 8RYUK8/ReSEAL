#!/usr/bin/env python3
import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt

# This script extends trimmed-mean watermark detection to support multiple image
# distortions. It computes image-level scores by averaging the lowest beta fraction
# of patch L2 distances, evaluates detection with a fixed threshold tau, and allows
# analyzing specific distortions (e.g., JPEG, blur, noise) from structured NPZ files.
# It also provides beta-sweep plots and optional single-image analysis.

def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute ROC AUC without external libraries.
    y_true: 1 for watermarked, 0 for original/random
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
    x = np.sort(x)
    y = np.arange(1, len(x) + 1, dtype=float) / len(x)
    return x, y


def compute_trimmed_means(wm: np.ndarray, orig: np.ndarray, beta: float):
    """
    Given watermarked/original arrays (N_images, N_patches) and beta,
    return per-image trimmed-mean L2 for each.
    We take the mean of the lowest (beta * K) patches in each image.
    """
    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0, 1], got {beta}")

    n_wm, k = wm.shape
    n_orig, k2 = orig.shape
    if k != k2:
        raise ValueError("Number of patches must match.")

    n_keep = int(beta * k)
    if n_keep < 1:
        n_keep = 1

    wm_sorted = np.sort(wm, axis=1)
    orig_sorted = np.sort(orig, axis=1)

    wm_trimmed_mean = wm_sorted[:, :n_keep].mean(axis=1)
    orig_trimmed_mean = orig_sorted[:, :n_keep].mean(axis=1)

    return wm_trimmed_mean, orig_trimmed_mean, n_keep


def sanitize_key(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s)


def load_npz_pair(path: str, distortion: str | None):
    """
    Supports two NPZ formats:

    (A) Old format:
        - keys: 'watermarked' and 'random' (where random may actually be original)
        - distortion argument is ignored

    (B) New distortions format:
        - keys: 'wm_<DistName>' and 'orig_<DistName>'
          e.g. wm_Clean, orig_Clean, wm_JPEG_80, orig_JPEG_80, ...

        distortion selects which pair to analyze.
    """
    data = np.load(path)
    files = set(data.files)

    # Old format fallback
    if "watermarked" in files and "random" in files:
        wm = np.asarray(data["watermarked"])
        orig = np.asarray(data["random"])
        used_dist = "legacy_watermarked_vs_random"
        return wm, orig, used_dist, sorted(list(files))

    # New format requires distortion selection
    wm_keys = sorted([k for k in files if k.startswith("wm_")])
    orig_keys = sorted([k for k in files if k.startswith("orig_")])

    if not wm_keys or not orig_keys:
        raise KeyError(
            f"NPZ {path} doesn't look like expected formats.\n"
            f"Found keys: {sorted(list(files))}"
        )

    # Build available distortion names from intersections
    # distortion name is suffix after 'wm_' / 'orig_'
    wm_suffix = {k[3:]: k for k in wm_keys}
    orig_suffix = {k[5:]: k for k in orig_keys}
    common = sorted(set(wm_suffix.keys()) & set(orig_suffix.keys()))

    if not common:
        raise KeyError(
            f"No matching wm_/orig_ key pairs found in {path}.\n"
            f"wm keys: {wm_keys}\norig keys: {orig_keys}"
        )

    if distortion is None:
        # default to Clean if present; else first available
        distortion = "Clean" if "Clean" in common else common[0]

    dist_key = sanitize_key(distortion)
    if dist_key not in common:
        raise KeyError(
            f"Requested distortion '{distortion}' (sanitized '{dist_key}') not found.\n"
            f"Available distortions: {common}"
        )

    wm = np.asarray(data[wm_suffix[dist_key]])
    orig = np.asarray(data[orig_suffix[dist_key]])
    return wm, orig, dist_key, sorted(list(files))


def list_available_distortions(path: str):
    data = np.load(path)
    files = set(data.files)

    if "watermarked" in files and "random" in files:
        print("This NPZ is legacy format (keys: watermarked, random). No distortion selection.")
        return

    wm_keys = sorted([k for k in files if k.startswith("wm_")])
    orig_keys = sorted([k for k in files if k.startswith("orig_")])

    wm_suffix = {k[3:]: k for k in wm_keys}
    orig_suffix = {k[5:]: k for k in orig_keys}
    common = sorted(set(wm_suffix.keys()) & set(orig_suffix.keys()))

    print("Available distortions in this NPZ:")
    for d in common:
        print(f"  - {d}")


def analyze_trimmed_mean(path: str, beta: float, tau: float, distortion: str | None) -> None:
    """
    Image-level analysis using mean of the lowest β-fraction of L2 distances per image.
    """
    wm, orig, used_dist, keys = load_npz_pair(path, distortion)

    if wm.ndim != 2 or orig.ndim != 2:
        raise ValueError(
            f"Arrays must be 2D (N_images, N_patches). "
            f"Got shapes: wm={wm.shape}, orig={orig.shape}"
        )

    n_wm, k = wm.shape
    n_orig, _ = orig.shape
    base = os.path.splitext(os.path.basename(path))[0]

    # include distortion in filenames when applicable
    dist_tag = sanitize_key(used_dist)
    out_prefix = f"{base}_{dist_tag}"

    print(f"Loaded file: {path}")
    print(f"Using distortion: {used_dist}")
    print(f"  wm   shape: {wm.shape} (N_wm_images={n_wm}, N_patches={k})")
    print(f"  orig shape: {orig.shape} (N_orig_images={n_orig}, N_patches={k})")
    print(f"  beta (fraction of lowest patches used) = {beta:.3f}")
    print(f"  tau  (fixed image-level threshold)     = {tau:.4f}")
    print("-" * 60)

    # 1) trimmed mean (lowest β fraction) per image
    wm_tm, orig_tm, n_keep = compute_trimmed_means(wm, orig, beta)

    print(f"Using n_keep = {n_keep} patches per image (lowest {beta*100:.1f}%)")
    print("Image-level trimmed-mean L2 stats:")
    print(
        f"  watermarked: mean={wm_tm.mean():.4f}, std={wm_tm.std():.4f}, "
        f"min={wm_tm.min():.4f}, max={wm_tm.max():.4f}"
    )
    print(
        f"  original   : mean={orig_tm.mean():.4f}, std={orig_tm.std():.4f}, "
        f"min={orig_tm.min():.4f}, max={orig_tm.max():.4f}"
    )

    # 2) ROC AUC (use -L2 so "bigger means more watermarked")
    y_true = np.concatenate([np.ones_like(wm_tm, dtype=int), np.zeros_like(orig_tm, dtype=int)])
    scores = np.concatenate([-wm_tm, -orig_tm])
    auc = roc_auc_score_manual(y_true, scores)
    print(f"\nImage-level ROC AUC (trimmed-mean L2 score): {auc:.4f}")

    # 3) Fixed τ classification on trimmed-mean L2 (smaller => more watermarked)
    frac_orig_below = (orig_tm < tau).mean()
    frac_wm_below = (wm_tm < tau).mean()

    print("\nImage-level threshold τ on trimmed-mean L2:")
    print(f"  τ (manual)            = {tau:.6f}")
    print(f"  P(original < τ) ≈ {frac_orig_below:.4f}  (empirical FPR at τ)")
    print(f"  P(wm       < τ) ≈ {frac_wm_below:.4f}  (empirical TPR at τ)")

    all_scores = np.concatenate([wm_tm, orig_tm])
    pred = (all_scores < tau).astype(int)  # 1 = predicted watermarked

    tp = int(np.sum((pred == 1) & (y_true == 1)))
    tn = int(np.sum((pred == 0) & (y_true == 0)))
    fp = int(np.sum((pred == 1) & (y_true == 0)))
    fn = int(np.sum((pred == 0) & (y_true == 1)))

    tpr = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    acc = (tp + tn) / len(y_true)

    print("\nDetection with fixed τ on trimmed-mean L2:")
    print(f"  TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  accuracy={acc:.4f}, TPR={tpr:.4f}, FPR={fpr:.4f}")

    # 4) Histogram
    plt.figure(figsize=(8, 6))
    bins = 80
    plt.hist(wm_tm, bins=bins, density=True, alpha=0.6, label="watermarked")
    plt.hist(orig_tm, bins=bins, density=True, alpha=0.6, label="original")
    plt.axvline(tau, linestyle="--", linewidth=2, label=f"τ = {tau:.3f}")
    plt.xlabel("image-level score (trimmed mean L2)")
    plt.ylabel("density")
    plt.title(f"Score distribution ({used_dist}) (beta={beta:.2f}, tau={tau:.2f})")
    plt.legend()
    plt.tight_layout()

    score_hist_path = f"{out_prefix}_trimmed_mean_hist_beta{beta:.2f}_tau{tau:.2f}.png"
    plt.savefig(score_hist_path, dpi=150)
    print(f"Saved histogram to: {score_hist_path}")

    # 5) ECDF
    plt.figure(figsize=(8, 6))
    x_orig, y_orig = ecdf(orig_tm)
    x_wm, y_wm = ecdf(wm_tm)
    plt.plot(x_orig, y_orig, label="original (ECDF)")
    plt.plot(x_wm, y_wm, label="watermarked (ECDF)")
    plt.axvline(tau, linestyle="--", label=f"τ = {tau:.3f}")
    plt.axhline(frac_orig_below, linestyle=":", label=f"frac_original_below_τ ≈ {frac_orig_below:.3f}")
    plt.xlabel("image-level score (trimmed mean L2)")
    plt.ylabel("CDF")
    plt.title(f"ECDF of scores ({used_dist}) (beta={beta:.2f}, tau={tau:.2f})")
    plt.legend()
    plt.tight_layout()

    cdf_path = f"{out_prefix}_trimmed_mean_cdf_beta{beta:.2f}_tau{tau:.2f}.png"
    plt.savefig(cdf_path, dpi=150)
    print(f"Saved ECDF plot to: {cdf_path}")

    # 6) ROC curve
    scores_raw = np.concatenate([wm_tm, orig_tm])
    thresholds = np.unique(scores_raw)

    tpr_list = []
    fpr_list = []

    for thr in thresholds:
        pred_thr = (scores_raw <= thr).astype(int)
        tp_ = np.sum((pred_thr == 1) & (y_true == 1))
        fn_ = np.sum((pred_thr == 0) & (y_true == 1))
        fp_ = np.sum((pred_thr == 1) & (y_true == 0))
        tn_ = np.sum((pred_thr == 0) & (y_true == 0))
        tpr_list.append(tp_ / (tp_ + fn_ + 1e-12))
        fpr_list.append(fp_ / (fp_ + tn_ + 1e-12))

    plt.figure(figsize=(6, 6))
    plt.plot(fpr_list, tpr_list, label=f"ROC (beta={beta:.2f}), AUC={auc:.3f}")
    plt.plot([0, 1], [0, 1], "k--", label="random guess")

    idx_tau = int(np.argmin(np.abs(thresholds - tau)))
    plt.scatter(
        fpr_list[idx_tau],
        tpr_list[idx_tau],
        marker="o",
        color="red",
        label=f"operating point at τ={tau:.2f}",
    )

    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"Image-level ROC ({used_dist})")
    plt.legend()
    plt.tight_layout()

    roc_path = f"{out_prefix}_trimmed_mean_roc_beta{beta:.2f}_tau{tau:.2f}.png"
    plt.savefig(roc_path, dpi=150)
    print(f"Saved ROC curve to: {roc_path}")


def plot_beta_curve(path: str, distortion: str | None) -> None:
    """
    Plot how the *mean* trimmed-mean L2 changes as we vary beta,
    averaged across all images (watermarked vs original), for one distortion.
    """
    wm, orig, used_dist, _ = load_npz_pair(path, distortion)
    base = os.path.splitext(os.path.basename(path))[0]

    dist_tag = sanitize_key(used_dist)
    out_prefix = f"{base}_{dist_tag}"

    betas = np.arange(0.05, 1.0001, 0.05)

    wm_means = []
    orig_means = []

    for beta in betas:
        wm_tm, orig_tm, _ = compute_trimmed_means(wm, orig, beta)
        wm_means.append(wm_tm.mean())
        orig_means.append(orig_tm.mean())

    wm_means = np.array(wm_means)
    orig_means = np.array(orig_means)

    plt.figure(figsize=(8, 6))
    plt.plot(betas, wm_means, marker="o", label="watermarked (mean)")
    plt.plot(betas, orig_means, marker="s", label="original (mean)")
    plt.xlabel("beta (fraction of lowest-L2 patches used)")
    plt.ylabel("mean image-level trimmed-mean L2")
    plt.title(f"Mean trimmed-mean L2 vs beta ({used_dist})")
    plt.legend()
    plt.grid(True)
    plt.xticks(betas)
    plt.tight_layout()

    out_path = f"{out_prefix}_beta_vs_mean_trimmedL2.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved beta-curve plot to: {out_path}")


def plot_beta_curve_single(path: str, img_idx: int, image_type: str, distortion: str | None) -> None:
    """
    Plot beta vs trimmed-mean L2 for a single image.
    image_type: "watermarked" or "original"
    """
    wm, orig, used_dist, _ = load_npz_pair(path, distortion)

    arr = wm if image_type == "watermarked" else orig
    n_images, k = arr.shape

    if not (0 <= img_idx < n_images):
        raise IndexError(f"img_idx {img_idx} out of range for {image_type}: 0..{n_images-1}")

    base = os.path.splitext(os.path.basename(path))[0]
    dist_tag = sanitize_key(used_dist)

    betas = np.arange(0.05, 1.0001, 0.05)
    row = np.sort(arr[img_idx])

    scores = []
    for beta in betas:
        n_keep = int(beta * k)
        if n_keep < 1:
            n_keep = 1
        scores.append(row[:n_keep].mean())

    scores = np.array(scores)

    plt.figure(figsize=(8, 6))
    plt.plot(betas, scores, marker="o")
    plt.xlabel("beta (fraction of lowest-L2 patches used)")
    plt.ylabel("trimmed-mean L2 (single image)")
    plt.title(f"Single-image curve ({used_dist})\n{image_type} index {img_idx}")
    plt.grid(True)
    plt.xticks(betas)
    plt.tight_layout()

    out_path = f"{base}_{dist_tag}_{image_type}_idx{img_idx}_beta_vs_trimmedL2.png"
    plt.savefig(out_path, dpi=150)
    print(f"Saved single-image beta-curve plot to: {out_path}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SEAL NPZ using image-level trimmed-mean of lowest β-fraction of patch L2s.\n"
            "Supports legacy NPZ (watermarked/random) and new distortion NPZ (wm_*/orig_*).\n"
            "This version REMOVES worst/best envelope plots."
        )
    )
    parser.add_argument(
        "npz_file",
        type=str,
        help="Path to NPZ file.",
    )
    parser.add_argument(
        "--beta",
        type=float,
        default=1.0,
        help="Fraction of lowest-L2 patches per image for main analysis (0 < beta <= 1).",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=2.8,
        help="Fixed image-level threshold on trimmed-mean L2.",
    )
    parser.add_argument(
        "--distortion",
        type=str,
        default=None,
        help=(
            "For distortion NPZ files: which distortion to analyze (e.g. Clean, JPEG_80, Blur_4, Noise_0_05, Bright_2). "
            "If not provided: uses Clean if available else first."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available distortions in the NPZ and exit (only applies to distortion NPZ).",
    )
    parser.add_argument(
        "--example_index",
        type=int,
        default=None,
        help="If set, also plot beta curve for a single image index.",
    )
    parser.add_argument(
        "--example_type",
        type=str,
        default="watermarked",
        choices=["watermarked", "original"],
        help="Which set example_index refers to (default: watermarked).",
    )

    args = parser.parse_args()

    if args.list:
        list_available_distortions(args.npz_file)
        return

    analyze_trimmed_mean(args.npz_file, beta=args.beta, tau=args.tau, distortion=args.distortion)
    plot_beta_curve(args.npz_file, distortion=args.distortion)

    if args.example_index is not None:
        plot_beta_curve_single(
            args.npz_file,
            img_idx=args.example_index,
            image_type=args.example_type,
            distortion=args.distortion,
        )


if __name__ == "__main__":
    main()
