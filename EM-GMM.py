#!/usr/bin/env python3
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture
from scipy.optimize import brentq

def load_distances(npz_path: str, split: str) -> np.ndarray:
    """
    Loads patch distances from your NPZ.
    Expected keys: 'watermarked' and 'random'
    Each is shape [num_images, num_patches] (e.g. [50, 1024]).
    """
    data = np.load(npz_path, allow_pickle=True)
    if split not in data.files:
        raise KeyError(f"Split '{split}' not found. Available keys: {data.files}")
    X = data[split]
    if X.ndim != 2:
        raise ValueError(f"Expected 2D array [images, patches], got shape {X.shape}")
    return X.reshape(-1).astype(np.float64)  # flatten to 1D

def fit_gmm_1d(distances: np.ndarray, K: int, seed: int, n_init: int, max_iter: int) -> GaussianMixture:
    """
    Fits a 1D Gaussian Mixture Model via EM using sklearn.
    """
    X = distances.reshape(-1, 1)
    gmm = GaussianMixture(
        n_components=K,
        covariance_type="full",
        random_state=seed,
        n_init=n_init,
        max_iter=max_iter,
        reg_covar=1e-6,
    )
    gmm.fit(X)  # EM happens inside here
    return gmm

def posterior_signal_probability(gmm: GaussianMixture, d: np.ndarray, sig_idx: int) -> np.ndarray:
    """
    Returns p_sig(d) = P(component=sig | d).
    """
    resp = gmm.predict_proba(d.reshape(-1, 1))  # responsibilities
    return resp[:, sig_idx]

def find_boundary_dstar(gmm: GaussianMixture, sig_idx: int, d_min: float, d_max: float) -> float:
    """
    Finds d* such that p_sig(d*) = 0.5.
    We solve f(d) = p_sig(d) - 0.5 = 0 using brentq on [d_min, d_max].
    """
    def f(x: float) -> float:
        p = posterior_signal_probability(gmm, np.array([x], dtype=np.float64), sig_idx)[0]
        return p - 0.5

    # We need an interval where f changes sign.
    # Try to find it by scanning if necessary.
    lo, hi = d_min, d_max
    flo, fhi = f(lo), f(hi)

    if flo == 0:
        return lo
    if fhi == 0:
        return hi

    if flo * fhi > 0:
        # No sign change. Scan to find a bracket.
        grid = np.linspace(d_min, d_max, 500)
        vals = np.array([f(x) for x in grid])
        sign = np.sign(vals)
        idx = np.where(sign[:-1] * sign[1:] < 0)[0]
        if len(idx) == 0:
            raise RuntimeError(
                "Could not bracket a root for p_sig(d)=0.5 in the data range. "
                "This can happen if the signal posterior is always above or always below 0.5."
            )
        lo, hi = grid[idx[0]], grid[idx[0] + 1]

    return float(brentq(f, lo, hi))

def main():
    parser = argparse.ArgumentParser(description="Fit 1D GMM via EM on patch L2 distances.")
    parser.add_argument("--npz", type=str, default="/mnt/data/all_min_l2_1024_7.npz",
                        help="Path to NPZ containing 'watermarked' and 'random' arrays.")
    parser.add_argument("--split", type=str, default="watermarked", choices=["watermarked", "random"],
                        help="Which split to fit the GMM on.")
    parser.add_argument("--K", type=int, default=3, help="Number of mixture components.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
    parser.add_argument("--n_init", type=int, default=10, help="Number of random initializations (best log-likelihood kept).")
    parser.add_argument("--max_iter", type=int, default=1000, help="Max EM iterations.")
    parser.add_argument("--beta", type=float, default=12/1024, help="Tail fraction for trimmed-mean interpretation.")
    args = parser.parse_args()

    # 1) Load distances
    dists = load_distances(args.npz, args.split)
    print(f"Loaded {dists.size:,} distances from '{args.split}'. Range: [{dists.min():.4f}, {dists.max():.4f}]")

    # 2) Fit GMM (EM)
    gmm = fit_gmm_1d(dists, K=args.K, seed=args.seed, n_init=args.n_init, max_iter=args.max_iter)

    # 3) Extract parameters (sort components by mean for readability)
    weights = gmm.weights_.copy()
    means = gmm.means_.flatten().copy()
    vars_ = gmm.covariances_.flatten().copy()  # 1D so each component has one variance
    stds = np.sqrt(vars_)

    order = np.argsort(means)
    weights_s = weights[order]
    means_s = means[order]
    stds_s = stds[order]

    # signal = smallest mean component
    sig_sorted_idx = 0
    sig_idx = order[sig_sorted_idx]  # index in original model ordering

    print("\nFitted GMM parameters (sorted by mean):")
    for i in range(args.K):
        print(f"  comp {i}:  pi={weights_s[i]:.4f},  mu={means_s[i]:.4f},  sigma={stds_s[i]:.4f}")
    print(f"\nSignal component = smallest mean => sorted comp 0 (original index {sig_idx})")

    # 4) Compute d* where p_sig(d*) = 0.5
    d_min, d_max = float(dists.min()), float(dists.max())
    d_star = find_boundary_dstar(gmm, sig_idx=sig_idx, d_min=d_min, d_max=d_max)
    print(f"\nDecision boundary d* where p_sig(d*)=0.5:  d* = {d_star:.4f}")

    # 5) Tail fraction beta: what distance quantile is that?
    beta = float(args.beta)
    if not (0 < beta < 1):
        raise ValueError("--beta must be in (0,1)")
    q_beta = float(np.quantile(dists, beta))
    p_sig_at_qbeta = float(posterior_signal_probability(gmm, np.array([q_beta]), sig_idx)[0])

    print(f"\nTail fraction beta = {beta:.6f}")
    print(f"  Quantile distance d_beta = Q_beta(d) = {q_beta:.4f}")
    print(f"  Posterior p_sig(d_beta) = {p_sig_at_qbeta:.4f}")

    # 6) Optional: quick sanity check values
    test_points = np.array([means_s[0], means_s[1], means_s[2], d_star, q_beta], dtype=np.float64)
    pvals = posterior_signal_probability(gmm, test_points, sig_idx)
    print("\nPosterior p_sig(d) at a few reference points:")
    for x, p in zip(test_points, pvals):
        print(f"  d={x:.4f} -> p_sig={p:.4f}")

if __name__ == "__main__":
    main()
