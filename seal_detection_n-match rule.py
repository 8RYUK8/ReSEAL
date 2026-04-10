#!/usr/bin/env python3
import argparse
import numpy as np

# This script evaluates SEAL-style verification from patch-wise L2 distances.
# It chooses a patch threshold tau from random patches, counts how many patches
# match in each image, and then analyzes image-level detection performance by
# sweeping the required number of matches.

def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    Compute ROC AUC without external libraries.
    y_true: 1 for watermarked, 0 for random
    y_score: larger = more likely watermarked
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    # Sort by score (ascending)
    order = np.argsort(y_score)
    y_true_sorted = y_true[order]

    n_pos = np.sum(y_true_sorted == 1)
    n_neg = np.sum(y_true_sorted == 0)

    if n_pos == 0 or n_neg == 0:
        raise ValueError("Both positive and negative samples are required for AUC.")

    # Rank sum of positive samples (Mann–Whitney U interpretation)
    ranks = np.arange(1, len(y_true_sorted) + 1, dtype=float)
    R_pos = np.sum(ranks[y_true_sorted == 1])

    # Mann–Whitney U -> AUC
    auc = (R_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def analyze_seal_verification(path: str, alpha_patch: float) -> None:
    """
    Analyze SEAL-style verification based on Algorithm 3 in the SEAL paper.

    Expected NPZ structure:
        - 'watermarked': shape (N_wm_images, N_patches)   -> L2 distances per patch
        - 'random':      shape (N_rand_images, N_patches) -> L2 distances per patch

    Detection logic (Algorithm 3):
        1) Patch-level: a patch is a "match" if L2 < τ   (patch distance threshold).
        2) Image-level: count matches m; if m >= m_match -> image is 'watermarked'.

    This script:
        - chooses τ from the distribution of RANDOM patches so that
          P(random patch has L2 < τ) ≈ alpha_patch (target patch-level FPR),
        - counts matches per image,
        - evaluates detection for a sweep of m_match values,
        - prints summary statistics and image-level ROC AUC.
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

    if wm.shape[1] != rnd.shape[1]:
        raise ValueError(
            f"Number of patches must match between watermarked and random. "
            f"Got {wm.shape[1]} vs {rnd.shape[1]}"
        )

    n_wm, k = wm.shape
    n_rnd, _ = rnd.shape

    print(f"Loaded file: {path}")
    print(f"  watermarked shape: {wm.shape} (N_wm_images={n_wm}, N_patches={k})")
    print(f"  random      shape: {rnd.shape} (N_rand_images={n_rnd}, N_patches={k})")
    print("-" * 60)

    # ----------------------------------------------------------------------
    # 1. Patch-level analysis (this corresponds to the per-patch test in Alg. 3)
    # ----------------------------------------------------------------------
    wm_flat = wm.ravel()
    rnd_flat = rnd.ravel()

    print("Patch-level L2 stats:")
    print(f"  watermarked patches: {wm_flat.size}")
    print(f"  random      patches: {rnd_flat.size}")
    print(
        f"  mean L2 (watermarked) = {wm_flat.mean():.4f}, "
        f"std = {wm_flat.std():.4f}"
    )
    print(
        f"  mean L2 (random)      = {rnd_flat.mean():.4f}, "
        f"std = {rnd_flat.std():.4f}"
    )

    # Patch-level ROC AUC (smaller L2 => more likely watermarked)
    y_true_patch = np.concatenate(
        [
            np.ones_like(wm_flat, dtype=int),
            np.zeros_like(rnd_flat, dtype=int),
        ]
    )
    scores_patch = np.concatenate([-wm_flat, -rnd_flat])
    auc_patch = roc_auc_score_manual(y_true_patch, scores_patch)
    print(f"\nPatch-level ROC AUC (L2 as match score): {auc_patch:.4f}")

    # Choose τ based on RANDOM patches: P(L2_random < τ) = alpha_patch
    tau = np.quantile(rnd_flat, alpha_patch)
    frac_rnd_below_tau = (rnd_flat < tau).mean()
    frac_wm_below_tau = (wm_flat < tau).mean()

    print("\nPatch-level threshold τ (SEAL-style):")
    print(f"  alpha_patch (target patch FPR) = {alpha_patch:.4f}")
    print(f"  τ chosen as quantile_{alpha_patch:.4f} of random-patch L2 = {tau:.6f}")
    print(
        f"  P(random patch L2 < τ) ≈ {frac_rnd_below_tau:.4f} "
        f"(actual patch-level FPR)"
    )
    print(
        f"  P(wm patch   L2 < τ) ≈ {frac_wm_below_tau:.4f} "
        f"(patch-level TPR / match rate)"
    )
    print("-" * 60)

    # ----------------------------------------------------------------------
    # 2. Image-level SEAL-style detection (Algorithm 3)
    #    For each image: m = number of patches with L2 < τ
    # ----------------------------------------------------------------------
    wm_matches = (wm < tau).sum(axis=1)   # matches per watermarked image
    rnd_matches = (rnd < tau).sum(axis=1) # matches per random image

    print("Image-level match counts (m = #patches with L2 < τ):")
    print(
        f"  watermarked: mean m = {wm_matches.mean():.2f}, "
        f"median = {np.median(wm_matches):.2f}, "
        f"min = {wm_matches.min()}, max = {wm_matches.max()}"
    )
    print(
        f"  random     : mean m = {rnd_matches.mean():.2f}, "
        f"median = {np.median(rnd_matches):.2f}, "
        f"min = {rnd_matches.min()}, max = {rnd_matches.max()}"
    )

    # Image-level ROC AUC using m as a score (more matches => more likely watermarked)
    y_true_img = np.concatenate(
        [
            np.ones_like(wm_matches, dtype=int),
            np.zeros_like(rnd_matches, dtype=int),
        ]
    )
    scores_img = np.concatenate([wm_matches, rnd_matches]).astype(float)
    auc_img = roc_auc_score_manual(y_true_img, scores_img)
    print(f"\nImage-level ROC AUC (using match count m as score): {auc_img:.4f}")

    # ----------------------------------------------------------------------
    # 3. Sweep over m_match thresholds, compute metrics
    # ----------------------------------------------------------------------
    all_matches = np.concatenate([wm_matches, rnd_matches])

    unique_m = np.unique(all_matches)
    unique_m = np.sort(unique_m)

    print("\nSweeping image-level m_match thresholds...")
    print("m_match | TP  TN  FP  FN | TPR     FPR     accuracy")
    print("-" * 56)

    best_acc = -1.0
    best_config = None

    # Store results to help select "nice" points later
    sweep_results = []

    for m_match in unique_m:
        # predict watermarked if m >= m_match
        pred_wm = (wm_matches >= m_match).astype(int)
        pred_rnd = (rnd_matches >= m_match).astype(int)

        tp = int(pred_wm.sum())
        fn = int(n_wm - tp)
        fp = int(pred_rnd.sum())
        tn = int(n_rnd - fp)

        tpr = tp / (tp + fn + 1e-12)
        fpr = fp / (fp + tn + 1e-12)
        acc = (tp + tn) / (n_wm + n_rnd)

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
        print(
            f"  m_match = {m_best}, "
            f"accuracy = {acc:.4f}, TPR = {tpr:.4f}, FPR = {fpr:.4f}"
        )

    # ----------------------------------------------------------------------
    # 4. Suggest m_match for a few target image-level FPRs (optional, handy)
    # ----------------------------------------------------------------------
    target_fprs = [0.01, 0.05, 0.10]
    print("\nSuggested m_match for target image-level FPRs:")

    for target in target_fprs:
        # among thresholds with FPR <= target, choose the one with highest TPR
        candidates = [
            (m_match, tpr, fpr, acc)
            for (m_match, tp, tn, fp, fn, tpr, fpr, acc) in sweep_results
            if fpr <= target
        ]
        if not candidates:
            print(f"  target FPR={target:.2f}: no threshold with FPR <= target.")
            continue

        m_sel, tpr_sel, fpr_sel, acc_sel = max(candidates, key=lambda x: x[1])
        print(
            f"  target FPR={target:.2f}: "
            f"m_match={m_sel}, TPR={tpr_sel:.4f}, FPR={fpr_sel:.4f}, "
            f"accuracy={acc_sel:.4f}"
        )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Analyze SEAL results using Algorithm-3-style verification:\n"
            "  patch-level threshold τ from random patches, and\n"
            "  image-level decision based on number of patch matches m."
        )
    )
    parser.add_argument(
        "npz_file",
        type=str,
        help="Path to NPZ file with 'watermarked' and 'random' arrays "
             "(e.g. all_patch_l2_1024_7.npz).",
    )
    parser.add_argument(
        "--alpha_patch",
        type=float,
        default=0.01,
        help=(
            "Target patch-level false positive rate for τ (default: 0.01, "
            "i.e., ~1%% of random patches fall below τ)."
        ),
    )
    args = parser.parse_args()
    analyze_seal_verification(args.npz_file, alpha_patch=args.alpha_patch)


if __name__ == "__main__":
    main()
