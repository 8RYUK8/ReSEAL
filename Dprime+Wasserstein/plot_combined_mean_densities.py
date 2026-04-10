#!/usr/bin/env python3
import argparse
import os
import numpy as np
import matplotlib.pyplot as plt


def trimmed_mean_per_image(dist_mat: np.ndarray, nkeep: int) -> np.ndarray:
    N, K = dist_mat.shape
    nkeep = max(1, min(nkeep, K))
    part = np.partition(dist_mat, kth=nkeep - 1, axis=1)[:, :nkeep]
    return part.mean(axis=1)


def stats_dict(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    return {
        "N": int(x.size),
        "mean": float(x.mean()),
        "var": float(x.var(ddof=1)) if x.size > 1 else 0.0,
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "min": float(x.min()),
        "q10": float(np.quantile(x, 0.10)),
        "q50": float(np.quantile(x, 0.50)),
        "q90": float(np.quantile(x, 0.90)),
        "max": float(x.max()),
    }


def load_and_concat_T(npz_files, nkeep):
    Tw_all = []
    Tr_all = []
    K_ref = None

    for path in npz_files:
        d = np.load(path)
        wm = np.asarray(d["watermarked"], dtype=float)
        rnd = np.asarray(d["random"], dtype=float)

        if wm.ndim != 2 or rnd.ndim != 2 or wm.shape[1] != rnd.shape[1]:
            raise ValueError(f"Bad shapes in {path}: wm={wm.shape}, random={rnd.shape}")

        if K_ref is None:
            K_ref = wm.shape[1]
        elif wm.shape[1] != K_ref:
            raise ValueError(f"Mismatch K across files: {path} has K={wm.shape[1]}, expected {K_ref}")

        Tw_all.append(trimmed_mean_per_image(wm, nkeep))
        Tr_all.append(trimmed_mean_per_image(rnd, nkeep))

    return np.concatenate(Tw_all), np.concatenate(Tr_all), K_ref


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz_files", nargs="+", help="8 NPZ files (or more). Each must contain 'watermarked' and 'random'.")
    ap.add_argument("--out_dir", default="combined_mean_plots")
    ap.add_argument("--nkeeps", type=int, nargs="+",
                    default=[12, 18, 24, 30, 36, 42, 48, 54, 60, 66])
    ap.add_argument("--bins", type=int, default=30)
    ap.add_argument("--xlim", type=float, nargs=2, default=None)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Summary CSV across all nkeeps
    csv_path = os.path.join(args.out_dir, "combined_mean_density_stats.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "n_keep,beta,K,"
            "wm_N,wm_mean,wm_var,wm_std,wm_min,wm_q10,wm_q50,wm_q90,wm_max,"
            "rnd_N,rnd_mean,rnd_var,rnd_std,rnd_min,rnd_q10,rnd_q50,rnd_q90,rnd_max\n"
        )

        for n in args.nkeeps:
            Tw, Tr, K = load_and_concat_T(args.npz_files, n)
            beta = n / K

            sw = stats_dict(Tw)
            sr = stats_dict(Tr)

            # common bins for comparable density
            all_vals = np.concatenate([Tw, Tr])
            lo, hi = float(all_vals.min()), float(all_vals.max())
            edges = np.linspace(lo, hi, args.bins + 1)

            plt.figure(figsize=(8, 5))
            plt.hist(Tw, bins=edges, density=True, alpha=0.55, label="WM (combined)")
            plt.hist(Tr, bins=edges, density=True, alpha=0.55, label="Random (combined)")

            plt.xlabel(rf"Per-image trimmed mean $T_{{{n}}}$")
            plt.ylabel("Density")
            plt.title(f"Combined per-image mean distribution (n_keep={n}, beta={beta:.4f})")

            if args.xlim is not None:
                plt.xlim(args.xlim[0], args.xlim[1])

            # annotate stats (mean/var/std + quantiles)
            txt = (
                f"WM:   mean={sw['mean']:.3f}, var={sw['var']:.4f}, std={sw['std']:.3f}\n"
                f"      q10={sw['q10']:.3f}, q50={sw['q50']:.3f}, q90={sw['q90']:.3f}\n"
                f"Rnd:  mean={sr['mean']:.3f}, var={sr['var']:.4f}, std={sr['std']:.3f}\n"
                f"      q10={sr['q10']:.3f}, q50={sr['q50']:.3f}, q90={sr['q90']:.3f}"
            )
            plt.text(0.02, 0.98, txt, transform=plt.gca().transAxes, va="top", fontsize=9)

            plt.legend()
            plt.tight_layout()
            out_png = os.path.join(args.out_dir, f"combined_mean_density_nkeep_{n:03d}.png")
            plt.savefig(out_png, dpi=180)
            plt.close()
            print("[saved]", out_png)

            # write CSV row
            f.write(
                f"{n},{beta:.6f},{K},"
                f"{sw['N']},{sw['mean']:.10f},{sw['var']:.10f},{sw['std']:.10f},{sw['min']:.10f},{sw['q10']:.10f},{sw['q50']:.10f},{sw['q90']:.10f},{sw['max']:.10f},"
                f"{sr['N']},{sr['mean']:.10f},{sr['var']:.10f},{sr['std']:.10f},{sr['min']:.10f},{sr['q10']:.10f},{sr['q50']:.10f},{sr['q90']:.10f},{sr['max']:.10f}\n"
            )

    print("[saved]", csv_path)


if __name__ == "__main__":
    main()