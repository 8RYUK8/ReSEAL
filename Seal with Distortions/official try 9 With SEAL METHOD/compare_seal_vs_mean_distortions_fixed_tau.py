#!/usr/bin/env python3
"""
compare_seal_vs_mean_distortions_fixed_tau.py

Compares:
  (A) SEAL paper-style "n-matches" method with FIXED patch threshold tau_patch (no alpha)
      - patch is a match if L2 < tau_patch
      - image is watermarked if m >= m_match
      - evaluate m_match sweep (default 5..15)

  (B) Mean/trimmed-mean method
      - per image score = mean of lowest (beta*K) patch L2 values
      - image is watermarked if score < tau_mean

Works on NEW distortion NPZ outputs:
  keys: wm_<DistName>, orig_<DistName>
Also supports legacy NPZ:
  keys: watermarked, random   (random is treated as original)

Outputs:
  - prints per-distortion comparison
  - writes a CSV with all results
"""

import argparse
import csv
import os
import re
from typing import List, Tuple, Dict, Any

import numpy as np


# -----------------------------
# Utilities
# -----------------------------
def sanitize_key(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", s)


def roc_auc_score_manual(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """
    ROC AUC without sklearn.
    y_true: 1 for watermarked, 0 for original
    y_score: larger => more likely watermarked
    """
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    order = np.argsort(y_score)
    y_true_sorted = y_true[order]

    n_pos = int(np.sum(y_true_sorted == 1))
    n_neg = int(np.sum(y_true_sorted == 0))
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    ranks = np.arange(1, len(y_true_sorted) + 1, dtype=float)
    R_pos = float(np.sum(ranks[y_true_sorted == 1]))
    auc = (R_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)
    return float(auc)


def compute_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[int, int, int, int, float, float, float]:
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    tpr = tp / (tp + fn + 1e-12)
    fpr = fp / (fp + tn + 1e-12)
    acc = (tp + tn) / len(y_true)
    return tp, tn, fp, fn, tpr, fpr, acc


# -----------------------------
# NPZ loading (new + legacy)
# -----------------------------
def get_available_distortions(npz_path: str) -> List[str]:
    data = np.load(npz_path)
    files = set(data.files)

    # Legacy format
    if "watermarked" in files and "random" in files:
        return ["legacy"]

    wm_keys = [k for k in files if k.startswith("wm_")]
    orig_keys = [k for k in files if k.startswith("orig_")]

    wm_suffix = {k[3:]: k for k in wm_keys}
    orig_suffix = {k[5:]: k for k in orig_keys}
    common = sorted(set(wm_suffix) & set(orig_suffix))
    return common


def load_pair(npz_path: str, distortion: str) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(npz_path)
    files = set(data.files)

    if distortion == "legacy":
        if "watermarked" not in files or "random" not in files:
            raise KeyError("Legacy keys not found (expected watermarked & random).")
        wm = np.asarray(data["watermarked"])
        orig = np.asarray(data["random"])  # treated as ORIGINAL
        return wm, orig

    d = sanitize_key(distortion)
    wm_key = f"wm_{d}"
    orig_key = f"orig_{d}"

    if wm_key not in files or orig_key not in files:
        raise KeyError(
            f"Missing keys for distortion '{distortion}'. "
            f"Expected '{wm_key}' and '{orig_key}'."
        )

    wm = np.asarray(data[wm_key])
    orig = np.asarray(data[orig_key])
    return wm, orig


# -----------------------------
# Method A: SEAL n-matches, fixed tau
# -----------------------------
def seal_method_metrics_fixed_tau(
    wm: np.ndarray,
    orig: np.ndarray,
    tau_patch: float,
    m_min: int,
    m_max: int,
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    SEAL-style method with FIXED patch threshold tau_patch:
      - patch match if L2 < tau_patch
      - per image: m = #matches
      - classify watermarked if m >= m_match
      - sweep m_match from m_min..m_max
    """
    if wm.shape != orig.shape:
        raise ValueError(f"wm and orig shapes must match. Got {wm.shape} vs {orig.shape}")
    if wm.ndim != 2:
        raise ValueError(f"Expected 2D arrays (N_images, N_patches). Got wm ndim={wm.ndim}")

    wm_m = (wm < tau_patch).sum(axis=1)
    orig_m = (orig < tau_patch).sum(axis=1)

    y_true = np.concatenate([
        np.ones_like(wm_m, dtype=int),
        np.zeros_like(orig_m, dtype=int),
    ])
    m_scores = np.concatenate([wm_m, orig_m]).astype(float)
    auc_img = roc_auc_score_manual(y_true, m_scores)

    rows = []
    for m_match in range(m_min, m_max + 1):
        y_pred = np.concatenate([
            (wm_m >= m_match).astype(int),
            (orig_m >= m_match).astype(int),
        ])
        tp, tn, fp, fn, tpr, fpr, acc = compute_confusion(y_true, y_pred)

        rows.append({
            "seal_tau_patch": float(tau_patch),
            "seal_auc_img": float(auc_img),
            "m_match": int(m_match),
            "TP": tp, "TN": tn, "FP": fp, "FN": fn,
            "TPR": float(tpr), "FPR": float(fpr), "ACC": float(acc),
            "wm_m_mean": float(wm_m.mean()),
            "orig_m_mean": float(orig_m.mean()),
            "wm_m_median": float(np.median(wm_m)),
            "orig_m_median": float(np.median(orig_m)),
        })

    return float(auc_img), rows


# -----------------------------
# Method B: trimmed-mean (lowest beta fraction), fixed tau_mean
# -----------------------------
def mean_method_metrics(
    wm: np.ndarray,
    orig: np.ndarray,
    beta: float,
    tau_mean: float,
) -> Dict[str, Any]:
    """
    Mean/trimmed-mean method:
      - score(image) = mean of lowest n_keep = max(1, int(beta*K)) patch L2 values
      - classify watermarked if score < tau_mean
    """
    if wm.shape != orig.shape:
        raise ValueError(f"wm and orig shape mismatch: {wm.shape} vs {orig.shape}")
    if wm.ndim != 2:
        raise ValueError(f"Expected 2D arrays (N_images, N_patches). Got ndim={wm.ndim}")

    if not (0.0 < beta <= 1.0):
        raise ValueError(f"beta must be in (0,1], got {beta}")

    n_wm, k = wm.shape
    n_orig, k2 = orig.shape
    if k != k2:
        raise ValueError("Patch counts mismatch.")

    n_keep = int(beta * k)
    if n_keep < 1:
        n_keep = 1

    wm_sorted = np.sort(wm, axis=1)
    orig_sorted = np.sort(orig, axis=1)

    wm_score = wm_sorted[:, :n_keep].mean(axis=1)
    orig_score = orig_sorted[:, :n_keep].mean(axis=1)

    y_true = np.concatenate([
        np.ones_like(wm_score, dtype=int),
        np.zeros_like(orig_score, dtype=int),
    ])

    # AUC: larger score means more likely watermarked, so use -L2 score
    scores_for_auc = np.concatenate([-wm_score, -orig_score])
    auc = roc_auc_score_manual(y_true, scores_for_auc)

    # Decision rule
    y_pred = np.concatenate([
        (wm_score < tau_mean).astype(int),
        (orig_score < tau_mean).astype(int),
    ])
    tp, tn, fp, fn, tpr, fpr, acc = compute_confusion(y_true, y_pred)

    return {
        "beta": float(beta),
        "tau_mean": float(tau_mean),
        "n_keep": int(n_keep),
        "mean_auc": float(auc),
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
        "TPR": float(tpr), "FPR": float(fpr), "ACC": float(acc),
        "wm_score_mean": float(wm_score.mean()),
        "orig_score_mean": float(orig_score.mean()),
        "wm_score_median": float(np.median(wm_score)),
        "orig_score_median": float(np.median(orig_score)),
    }


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser(
        description="Compare SEAL n-matches (fixed patch tau) vs trimmed-mean (fixed beta & tau) on distortion NPZ."
    )
    ap.add_argument("npz_file", type=str, help="Path to NPZ file (distortion format or legacy).")

    # SEAL fixed tau (NO alpha)
    ap.add_argument("--tau_patch", type=float, default=2.3,
                    help="Fixed patch-level L2 threshold τ for SEAL n-matches method (no alpha).")
    ap.add_argument("--m_min", type=int, default=5, help="Min m_match to evaluate.")
    ap.add_argument("--m_max", type=int, default=15, help="Max m_match to evaluate.")

    # Mean method params (your requested defaults)
    ap.add_argument("--beta", type=float, default=0.01171875,
                    help="Mean-method beta (fraction of lowest patches averaged).")
    ap.add_argument("--tau_mean", type=float, default=2.8,
                    help="Mean-method image threshold (watermarked if score < tau_mean).")

    # Distortions to run
    ap.add_argument("--distortions", nargs="*", default=None,
                    help="Optional list of distortions to run (e.g. Clean Noise_0_5 JPEG_25). If omitted: run all.")
    ap.add_argument("--list", action="store_true",
                    help="List available distortions in the NPZ and exit.")

    # Output
    ap.add_argument("--csv_out", type=str, default=None,
                    help="Optional CSV output path. Default: <npz_basename>_seal_vs_mean_fixedtau.csv")

    args = ap.parse_args()

    avail = get_available_distortions(args.npz_file)

    if args.list:
        print("Available distortions:")
        for d in avail:
            print(f"  - {d}")
        return

    if args.distortions is None or len(args.distortions) == 0:
        dists = avail
    else:
        requested = [sanitize_key(d) for d in args.distortions]
        dists = []
        for r in requested:
            if r == "legacy" and "legacy" in avail:
                dists.append("legacy")
            elif r in avail:
                dists.append(r)
            else:
                raise KeyError(f"Requested distortion '{r}' not found. Available: {avail}")

    base = os.path.splitext(os.path.basename(args.npz_file))[0]
    csv_out = args.csv_out or f"{base}_seal_vs_mean_fixedtau.csv"

    print(f"NPZ: {args.npz_file}")
    print(f"Distortions to run: {dists}")
    print(f"SEAL: tau_patch={args.tau_patch}, m_match={args.m_min}..{args.m_max}")
    print(f"MEAN: beta={args.beta}, tau_mean={args.tau_mean}")
    print("-" * 90)

    all_rows: List[Dict[str, Any]] = []

    for dist in dists:
        wm, orig = load_pair(args.npz_file, dist)

        if wm.ndim != 2 or orig.ndim != 2:
            raise ValueError(f"Expected 2D arrays. Got wm={wm.shape}, orig={orig.shape}")

        # SEAL fixed tau sweep
        seal_auc, seal_rows = seal_method_metrics_fixed_tau(
            wm, orig, tau_patch=args.tau_patch, m_min=args.m_min, m_max=args.m_max
        )

        # Mean method single setting
        mean_row = mean_method_metrics(wm, orig, beta=args.beta, tau_mean=args.tau_mean)

        # Print compact comparison per distortion
        print(f"[{dist}]  K={wm.shape[1]}  wm_images={wm.shape[0]}  orig_images={orig.shape[0]}")
        print(f"  MEAN: n_keep={mean_row['n_keep']}  AUC={mean_row['mean_auc']:.4f}  "
              f"ACC={mean_row['ACC']:.4f}  TPR={mean_row['TPR']:.4f}  FPR={mean_row['FPR']:.4f}")
        print(f"        wm_score_mean={mean_row['wm_score_mean']:.4f}  orig_score_mean={mean_row['orig_score_mean']:.4f}")
        print(f"  SEAL: tau_patch={args.tau_patch:.3f}  AUC(m)={seal_auc:.4f}  "
              f"wm_m_mean={seal_rows[0]['wm_m_mean']:.2f}  orig_m_mean={seal_rows[0]['orig_m_mean']:.2f}")
        print("        m_match sweep (ACC / TPR / FPR):")
        for r in seal_rows:
            print(f"          m={r['m_match']:2d}  ACC={r['ACC']:.4f}  TPR={r['TPR']:.4f}  FPR={r['FPR']:.4f}")
        print("-" * 90)

        # Add MEAN row to CSV
        all_rows.append({
            "distortion": dist,
            "method": "MEAN",
            "tau_patch": "",
            "m_match": "",
            "beta": mean_row["beta"],
            "tau_mean": mean_row["tau_mean"],
            "n_keep": mean_row["n_keep"],
            "AUC": mean_row["mean_auc"],
            "ACC": mean_row["ACC"],
            "TPR": mean_row["TPR"],
            "FPR": mean_row["FPR"],
            "TP": mean_row["TP"],
            "TN": mean_row["TN"],
            "FP": mean_row["FP"],
            "FN": mean_row["FN"],
            "wm_stat": mean_row["wm_score_mean"],      # mean image score (wm)
            "orig_stat": mean_row["orig_score_mean"],  # mean image score (orig)
            "wm_median": mean_row["wm_score_median"],
            "orig_median": mean_row["orig_score_median"],
        })

        # Add SEAL rows (one per m_match)
        for r in seal_rows:
            all_rows.append({
                "distortion": dist,
                "method": "SEAL",
                "tau_patch": r["seal_tau_patch"],
                "m_match": r["m_match"],
                "beta": "",
                "tau_mean": "",
                "n_keep": "",
                "AUC": r["seal_auc_img"],
                "ACC": r["ACC"],
                "TPR": r["TPR"],
                "FPR": r["FPR"],
                "TP": r["TP"],
                "TN": r["TN"],
                "FP": r["FP"],
                "FN": r["FN"],
                "wm_stat": r["wm_m_mean"],      # mean match-count (wm)
                "orig_stat": r["orig_m_mean"],  # mean match-count (orig)
                "wm_median": r["wm_m_median"],
                "orig_median": r["orig_m_median"],
            })

    # Write CSV
    fieldnames = [
        "distortion", "method",
        "tau_patch", "m_match",
        "beta", "tau_mean", "n_keep",
        "AUC", "ACC", "TPR", "FPR",
        "TP", "TN", "FP", "FN",
        "wm_stat", "orig_stat",
        "wm_median", "orig_median"
    ]
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in all_rows:
            w.writerow(row)

    print(f"Saved CSV comparison to: {csv_out}")


if __name__ == "__main__":
    main()
