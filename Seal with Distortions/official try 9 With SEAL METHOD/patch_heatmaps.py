import argparse
import os
import math
import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------
# Utilities
# ---------------------------------------------------

def reshape_to_grid(v, k):
    """(k,) -> (sqrt(k), sqrt(k))"""
    s = int(math.isqrt(k))
    if s * s != k:
        raise ValueError(f"k={k} is not a perfect square")
    return v.reshape(s, s)


# ---------------------------------------------------
# Heatmap generators
# ---------------------------------------------------

def mean_l2_heatmap(arr, k):
    """
    arr: (N, k)
    returns: (sqrt(k), sqrt(k)) mean L2 per patch
    """
    return reshape_to_grid(arr.mean(axis=0), k)


def matchrate_heatmap(arr, k, tau):
    """
    arr: (N, k)
    returns: fraction of patches with L2 < tau
    """
    matches = (arr < tau).astype(np.float32)
    return reshape_to_grid(matches.mean(axis=0), k)


# ---------------------------------------------------
# Plotting
# ---------------------------------------------------

def save_mean_l2_heatmap(H, path, title):
    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        H,
        cmap="viridis_r"  # INVERTED: small L2 = bright
    )
    plt.title(title)
    plt.xlabel("patch-x")
    plt.ylabel("patch-y")
    plt.colorbar(im, label="Mean L2 distance (lower = better)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


def save_matchrate_heatmap(H, path, title):
    plt.figure(figsize=(7, 6))
    im = plt.imshow(
        H,
        cmap="gray",
        vmin=0.0,
        vmax=1.0
    )
    plt.title(title)
    plt.xlabel("patch-x")
    plt.ylabel("patch-y")
    plt.colorbar(im, label="Match rate (white = match)")
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    plt.close()


# ---------------------------------------------------
# Main
# ---------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="SEAL patch-level heatmaps (mean L2 + match-rate)"
    )

    parser.add_argument("npz", type=str, help="NPZ file with L2 distances")
    parser.add_argument("--k", type=int, default=1024, help="Number of patches")
    parser.add_argument("--tau", type=float, default=2.8, help="Match threshold τ")
    parser.add_argument("--outdir", type=str, default="heatmaps")
    parser.add_argument("--single_idx", type=int, default=None,
                        help="Optional single-image heatmap index")

    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    data = np.load(args.npz)

    wm_key = "wm_Clean"
    orig_key = "orig_Clean"

    if wm_key not in data or orig_key not in data:
        print("Available keys:", data.files)
        raise KeyError("Expected keys wm_Clean and orig_Clean")

    wm = data[wm_key]      # (N, k)
    orig = data[orig_key]  # (N, k)

    N, K = wm.shape
    assert K == args.k, f"k mismatch: file has {K}, argument has {args.k}"

    # ---------------------------------------------------
    # Average heatmaps
    # ---------------------------------------------------

    wm_mean = mean_l2_heatmap(wm, args.k)
    orig_mean = mean_l2_heatmap(orig, args.k)

    wm_match = matchrate_heatmap(wm, args.k, args.tau)
    orig_match = matchrate_heatmap(orig, args.k, args.tau)

    save_mean_l2_heatmap(
        wm_mean,
        os.path.join(args.outdir, "meanL2_watermarked_clean.png"),
        f"Mean L2 heatmap (watermarked) – Clean (N={N}, k={args.k})"
    )

    save_mean_l2_heatmap(
        orig_mean,
        os.path.join(args.outdir, "meanL2_orig_clean.png"),
        f"Mean L2 heatmap (orig/random) – Clean (N={N}, k={args.k})"
    )

    save_matchrate_heatmap(
        wm_match,
        os.path.join(args.outdir, f"matchrate_watermarked_tau{args.tau}.png"),
        f"Match-rate heatmap (watermarked) – Clean (τ={args.tau})"
    )

    save_matchrate_heatmap(
        orig_match,
        os.path.join(args.outdir, f"matchrate_orig_tau{args.tau}.png"),
        f"Match-rate heatmap (orig/random) – Clean (τ={args.tau})"
    )

    # ---------------------------------------------------
    # Optional single image
    # ---------------------------------------------------

    if args.single_idx is not None:
        i = args.single_idx
        if i < 0 or i >= N:
            raise IndexError(f"single_idx {i} out of range (N={N})")

        wm_single = reshape_to_grid(wm[i], args.k)
        orig_single = reshape_to_grid(orig[i], args.k)

        save_mean_l2_heatmap(
            wm_single,
            os.path.join(args.outdir, f"single_meanL2_watermarked_{i}.png"),
            f"Single-image Mean L2 (watermarked) – idx {i}"
        )

        save_mean_l2_heatmap(
            orig_single,
            os.path.join(args.outdir, f"single_meanL2_orig_{i}.png"),
            f"Single-image Mean L2 (orig/random) – idx {i}"
        )

        wm_single_match = reshape_to_grid((wm[i] < args.tau).astype(float), args.k)
        orig_single_match = reshape_to_grid((orig[i] < args.tau).astype(float), args.k)

        save_matchrate_heatmap(
            wm_single_match,
            os.path.join(args.outdir, f"single_match_watermarked_{i}.png"),
            f"Single-image Match map (watermarked) – idx {i}"
        )

        save_matchrate_heatmap(
            orig_single_match,
            os.path.join(args.outdir, f"single_match_orig_{i}.png"),
            f"Single-image Match map (orig/random) – idx {i}"
        )

    print("✔ Heatmaps saved to:", args.outdir)


if __name__ == "__main__":
    main()
