#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from scipy.stats import norm


def trimmed_mean_per_image(dist_mat: np.ndarray, nkeep: int) -> np.ndarray:
    N, K = dist_mat.shape
    nkeep = max(1, min(nkeep, K))
    part = np.partition(dist_mat, kth=nkeep - 1, axis=1)[:, :nkeep]
    return part.mean(axis=1)


def load_concat_wm_T(npz_files, nkeep):
    Tw_all = []
    K_ref = None
    for path in npz_files:
        d = np.load(path)
        wm = np.asarray(d["watermarked"], dtype=float)
        if K_ref is None:
            K_ref = wm.shape[1]
        elif wm.shape[1] != K_ref:
            raise ValueError(f"Mismatch K across files: {path} has K={wm.shape[1]}, expected {K_ref}")
        Tw_all.append(trimmed_mean_per_image(wm, nkeep))
    return np.concatenate(Tw_all), K_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_files", nargs="+")
    ap.add_argument("--out_dir", default="combined_wm_gmm")
    ap.add_argument("--nkeeps", type=int, nargs="+",
                    default=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66])
    ap.add_argument("--components", type=int, default=2, help="GMM components (default 2).")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    summary_path = os.path.join(args.out_dir, "gmm_params_summary.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("n_keep,beta,K,N,components,weight,mean,std,AIC,BIC\n")

        for n in args.nkeeps:
            Tw, K = load_concat_wm_T(args.npz_files, n)
            beta = n / K

            X = Tw.reshape(-1, 1)
            gmm = GaussianMixture(
                n_components=args.components,
                covariance_type="full",
                random_state=args.seed,
                reg_covar=1e-6,
                max_iter=500
            )
            gmm.fit(X)

            weights = gmm.weights_.flatten()
            means = gmm.means_.flatten()
            stds = np.sqrt(np.array([c[0, 0] for c in gmm.covariances_]))

            # sort by mean
            order = np.argsort(means)
            weights, means, stds = weights[order], means[order], stds[order]

            aic = float(gmm.aic(X))
            bic = float(gmm.bic(X))

            # save per-component params
            for w, m, s in zip(weights, means, stds):
                f.write(f"{n},{beta:.6f},{K},{len(Tw)},{args.components},{w:.10f},{m:.10f},{s:.10f},{aic:.4f},{bic:.4f}\n")

            # plot: histogram + mixture pdf
            xs = np.linspace(float(Tw.min()), float(Tw.max()), 1200)
            pdf_mix = np.exp(gmm.score_samples(xs.reshape(-1, 1)))

            plt.figure(figsize=(8, 5))
            plt.hist(Tw, bins=30, density=True, alpha=0.55, label="WM combined $T_n$")

            plt.plot(xs, pdf_mix, linewidth=2, label=f"GMM({args.components}) pdf")

            # component curves
            for idx, (w, m, s) in enumerate(zip(weights, means, stds), start=1):
                plt.plot(xs, w * norm.pdf(xs, loc=m, scale=s), linestyle="--", linewidth=2,
                         label=f"comp{idx}: w={w:.2f}, μ={m:.2f}, σ={s:.2f}")

            plt.xlabel(rf"$T_{{{n}}}$ (WM combined)")
            plt.ylabel("Density")
            plt.title(f"GMM fit on combined WM means (n_keep={n}, beta={beta:.4f})\nAIC={aic:.1f}  BIC={bic:.1f}")
            plt.legend(fontsize=8)
            plt.tight_layout()

            out_png = os.path.join(args.out_dir, f"gmm_fit_wm_means_nkeep_{n:03d}.png")
            plt.savefig(out_png, dpi=180)
            plt.close()
            print("[saved]", out_png)

    print("[saved]", summary_path)


if __name__ == "__main__":
    main()
