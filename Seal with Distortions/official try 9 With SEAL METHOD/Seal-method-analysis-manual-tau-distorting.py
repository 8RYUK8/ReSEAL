#!/usr/bin/env python3
import argparse
import os
import re

import numpy as np
import matplotlib.pyplot as plt


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


def sanitize_key(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s)


def load_npz_pair(path: str, distortion: str | None):
    """
    Supports two NPZ formats:

    (A) Legacy format:
        - keys: 'watermarked' and 'random' (random may actually be ORIGINAL)
        - distortion ignored

    (B) Distortion format:
        - keys: 'wm_<DistName>' and 'orig_<DistName>'
          e.g. wm_Clean, orig_Clean, wm_JPEG_80, orig_JPEG_80, ...

        distortion selects which pair to analyze
    """
    data = np.load(path)
    files = set(data.files)

    # Legacy
    if "watermarked" in files and "random" in files:
        wm = np.asarray(data["watermarked"])
        orig = np.asarray(data["random"])
        used_dist = "legacy_watermarked_vs_random"
        return wm, orig, used_dist, sorted(list(files))

    # Distortion format
    wm_keys = sorted([k for k in files if k.startswith("wm_")])
    orig_keys = sorted([k for k in files if k.startswith("orig_")])

    if not wm_keys or not orig_keys:
        raise KeyError(
            f"NPZ {path} doesn't match expected formats.\n"
            f"Found keys: {sorted(list(files))}"
        )

    wm_suffix = {k[3:]: k for k in wm_keys}
    orig_suffix = {k[5:]: k for k in orig_keys}
    common = sorted(set(wm_suffix.keys()) & set(orig_suffix.keys()))

    if not common:
        raise KeyError(
            f"No matching wm_/orig_ key pairs found in {path}.\n"
            f"wm keys: {wm_keys}\norig keys: {orig_keys}"
        )

    if distortion is None:
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
        print("Legacy NPZ format detected (keys: watermarked, random). No distortion selection.")
        return

    wm_keys = sorted([k for k in files if k.startswith("wm_")])
    orig_keys = sorted([k for k in files if k.startswith("orig_")])

    wm_suffix = {k[3:]: k for k in wm_keys}
    orig_suffix = {k[5:]: k for k in orig_keys}
    common = sorted(set(wm_suffix.keys()) & set(orig_suffix.keys()))

    print("Available distortions in this NPZ:")
    for d in common:
        print(f"  - {d}")


def analyze_seal_verification(
    path: str,
    alpha_patch: float,
    tau_manual: float | None,
    distortion: str | None,
) -> None:
    """
    SEAL paper-style verification (Algorithm 3-ish):
      1) pick patch threshold τ (from random/original patches quantile or manual)
      2) compute matches per image m = #{patches with L2 < τ}
      3) sweep m_match to classify images
    """
    wm, orig, used_dist, _ = load_npz_pair(path, distortion)

    if wm.ndim != 2 or orig.ndim != 2:
        raise ValueError(
            f"wm and orig must be 2D (N_images, N_patches). "
            f"Got shapes: wm={wm.shape}, orig={orig.shape}"
        )
    if wm.shape[1] != orig.shape[1]:
        raise ValueError(
            f"Number of patches must match between wm and orig. "
            f"Got {wm.shape[1]} vs {orig.shape[1]}"
        )

    n_wm, k = wm.shape
    n_orig, _ = orig.shape

    base = os.path.splitext(os.path.basename(path))[0]
    out_prefix = f"{base}_{sanitize_key(used_dist)}"

    print(f"Loaded file: {path}")
    print(f"Using distortion: {used_dist}")
    print(f"  watermarked shape: {wm.shape} (N_wm_images={n_wm}, N_patches={k})")
    print(f"  original    shape: {orig.shape} (N_orig_images={n_orig}, N_patches={k})")
    print("-" * 60)

    # ----------------------------------------------------------------------
    # 1) Patch-level analysis
    # ----------------------------------------------------------------------
    wm_flat = wm.ravel()
    orig_flat = orig.ravel()

    print("Patch-level L2 stats:")
    print(f"  watermarked patches: {wm_flat.size}")
    print(f"  original    patches: {orig_flat.size}")
    print(f"  mean L2 (watermarked) = {wm_flat.mean():.4f}, std = {wm_flat.std():.4f}")
    print(f"  mean L2 (original)    = {orig_flat.mean():.4f}, std = {orig_flat.std():.4f}")

    # Patch-level ROC AUC (smaller L2 => more likely watermarked)
    y_true_patch = np.concatenate(
        [np.ones_like(wm_flat, dtype=int), np.zeros_like(orig_flat, dtype=int)]
    )
    scores_patch = np.concatenate([-wm_flat, -orig_flat])
    auc_patch = roc_auc_score_manual(y_true_patch, scores_patch)
    print(f"\nPatch-level ROC AUC (L2 as match score): {auc_patch:.4f}")

    # Patch-level histogram
    plt.figure(figsize=(8, 6))
    bins = 80
    plt.hist(wm_flat, bins=bins, density=True, alpha=0.6, label="watermarked")
    plt.hist(orig_flat, bins=bins, density=True, alpha=0.6, label="original")
    plt.xlabel("L2 per patch")
    plt.ylabel("density")
    plt.title(f"Patch-level L2 distribution ({used_dist})")
    plt.legend()
    plt.tight_layout()
    patch_hist_path = f"{out_prefix}_patch_level_hist.png"
    plt.savefig(patch_hist_path, dpi=150)
    print(f"\nSaved patch-level L2 histogram to: {patch_hist_path}")

    # ----------------------------------------------------------------------
    # 2) Choose τ (manual or from alpha_patch quantile of ORIGINAL patches)
    # ----------------------------------------------------------------------
    if tau_manual is not None:
        tau = float(tau_manual)
        source_str = "USER-SPECIFIED"
        print("\nPatch-level threshold τ (manual):")
        print(f"  τ (manual) = {tau:.6f}")
    else:
        tau = np.quantile(orig_flat, alpha_patch)
        source_str = f"quantile_{alpha_patch:.4f} of ORIGINAL-patch L2"
        print("\nPatch-level threshold τ (SEAL-style from alpha_patch):")
        print(f"  alpha_patch (target patch FPR) = {alpha_patch:.4f}")
        print(f"  τ chosen as {source_str} = {tau:.6f}")

    frac_orig_below_tau = (orig_flat < tau).mean()
    frac_wm_below_tau = (wm_flat < tau).mean()
    print(f"  P(original patch L2 < τ) ≈ {frac_orig_below_tau:.4f} (actual patch-level FPR)")
    print(f"  P(wm patch       L2 < τ) ≈ {frac_wm_below_tau:.4f} (patch-level TPR / match rate)")
    print(f"  τ source: {source_str}")
    print("-" * 60)

    # ----------------------------------------------------------------------
    # 3) Image-level match counts
    # ----------------------------------------------------------------------
    wm_matches = (wm < tau).sum(axis=1)      # matches per watermarked image
    orig_matches = (orig < tau).sum(axis=1)  # matches per original image

    print("Image-level match counts (m = #patches with L2 < τ):")
    print(
        f"  watermarked: mean m = {wm_matches.mean():.2f}, median = {np.median(wm_matches):.2f}, "
        f"min = {wm_matches.min()}, max = {wm_matches.max()}"
    )
    print(
        f"  original   : mean m = {orig_matches.mean():.2f}, median = {np.median(orig_matches):.2f}, "
        f"min = {orig_matches.min()}, max = {orig_matches.max()}"
    )

    # Image-level histogram of match counts
    plt.figure(figsize=(8, 6))
    max_m = int(max(wm_matches.max(), orig_matches.max()))
    bins_m = np.arange(0, max_m + 2) - 0.5
    plt.hist(wm_matches, bins=bins_m, density=True, alpha=0.6, label="watermarked")
    plt.hist(orig_matches, bins=bins_m, density=True, alpha=0.6, label="original")
    plt.xlabel("matches per image (m = #patches with L2 < τ)")
    plt.ylabel("density")
    plt.title(f"Image-level distribution of match counts m ({used_dist})")
    plt.legend()
    plt.tight_layout()
    img_hist_path = f"{out_prefix}_image_level_match_hist.png"
    plt.savefig(img_hist_path, dpi=150)
    print(f"Saved image-level match-count histogram to: {img_hist_path}")
    print("-" * 60)

    # Image-level ROC AUC using m as score (higher m => more likely watermarked)
    y_true_img = np.concatenate(
        [np.ones_like(wm_matches, dtype=int), np.zeros_like(orig_matches, dtype=int)]
    )
    scores_img = np.concatenate([wm_matches, orig_matches]).astype(float)
    auc_img = roc_auc_score_manual(y_true_img, scores_img)
    print(f"\nImage-level ROC AUC (using match count m as score): {auc_img:.4f}")

    # ----------------------------------------------------------------------
    # 4) Sweep m_match thresholds
    # ----------------------------------------------------------------------
    max_m = int(max(wm_matches.max(), orig_matches.max()))
    unique_m = np.arange(5, 13)

    print("\nSweeping image-level m_match thresholds...")
    print("m_match | TP  TN  FP  FN | TPR     FPR     accuracy")
    print("-" * 56)

    best_acc = -1.0
    best_config = None
    sweep_results = []

    for m_match in unique_m:
        # predict watermarked if m >= m_match
        pred_wm = (wm_matches >= m_match).astype(int)
        pred_orig = (orig_matches >= m_match).astype(int)

        tp = int(pred_wm.sum())
        fn = int(n_wm - tp)
        fp = int(pred_orig.sum())
        tn = int(n_orig - fp)

        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        acc = (tp + tn) / (n_wm + n_orig)

        sweep_results.append((m_match, tp, tn, fp, fn, tpr, fpr, acc))

        if acc > best_acc:
            best_acc = acc
            best_config = (m_match, tp, tn, fp, fn, tpr, fpr, acc)

        print(
            f"{m_match:6d} | "
            f"{tp:3d} {tn:3d} {fp:3d} {fn:3d} | "
            f"{tpr:7.4f} {fpr:7.4f} {acc:9.4f}"
        )

    print("-" * 56)
    if best_config is not None:
        m_best, tp, tn, fp, fn, tpr, fpr, acc = best_config
        print("Best accuracy configuration (over m_match):")
        print(f"  m_match = {m_best}, accuracy = {acc:.4f}, TPR = {tpr:.4f}, FPR = {fpr:.4f}")

    # ----------------------------------------------------------------------
    # 5) Suggest m_match for target image-level FPRs
    # ----------------------------------------------------------------------
    target_fprs = [0.01, 0.05, 0.10]
    print("\nSuggested m_match for target image-level FPRs:")

    for target in target_fprs:
        candidates = [
            (m_match, tpr, fpr, acc)
            for (m_match, tp, tn, fp, fn, tpr, fpr, acc) in sweep_results
            if fpr <= target
        ]
        if not candidates:
            print(f"  target FPR={target:.2f}: no threshold with FPR <= target.")
            continue

        # choose max TPR among those meeting FPR constraint
        m_sel, tpr_sel, fpr_sel, acc_sel = max(candidates, key=lambda x: x[1])
        print(
            f"  target FPR={target:.2f}: "
            f"m_match={m_sel}, TPR={tpr_sel:.4f}, FPR={fpr_sel:.4f}, accuracy={acc_sel:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "SEAL paper-style analysis (Algorithm-3-like):\n"
            "  pick patch threshold τ (from alpha_patch quantile of ORIGINAL patches or user τ),\n"
            "  compute image matches m (#patches with L2 < τ), sweep m_match.\n"
            "Supports legacy NPZ (watermarked/random) and distortion NPZ (wm_*/orig_*)."
        )
    )
    parser.add_argument("npz_file", type=str, help="Path to NPZ file.")
    parser.add_argument(
        "--alpha_patch",
        type=float,
        default=0.01,
        help="Target patch-level FPR for τ (used if --tau not set). Default=0.01.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=None,
        help="Manual patch-level τ. If set, overrides alpha_patch quantile.",
    )
    parser.add_argument(
        "--distortion",
        type=str,
        default=None,
        help=(
            "For distortion NPZ: which distortion to analyze (e.g. Clean, JPEG_80, Blur_4, Noise_0_05, Bright_2). "
            "If omitted: Clean if present else first available."
        ),
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available distortions in the NPZ and exit (only for distortion NPZ).",
    )

    args = parser.parse_args()

    if args.list:
        list_available_distortions(args.npz_file)
        return

    analyze_seal_verification(
        args.npz_file,
        alpha_patch=args.alpha_patch,
        tau_manual=args.tau,
        distortion=args.distortion,
    )


if __name__ == "__main__":
    main()
