import numpy as np
from scipy import stats
from scipy.stats import binom
from math import erf, sqrt

NPZ_PATH = "all_min_l2_1024_7_distortions.npz"

# --- settings ---
TAU = 2.3
PATCHES_PER_IMAGE = 1024
K_MIN, K_MAX = 5, 15

# --- helper: Normal CDF (vectorized) ---
def normal_cdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    z = (x - mu) / (sigma * sqrt(2.0))
    return 0.5 * (1.0 + np.vectorize(erf)(z))

# --- helper: Anderson–Darling statistic for Uniform(0,1) ---
# If data x follows baseline CDF F0, then u=F0(x) should be Uniform(0,1).
def anderson_darling_uniform(u: np.ndarray, eps: float = 1e-12) -> float:
    u = np.asarray(u, dtype=np.float64)
    u = np.clip(u, eps, 1.0 - eps)
    u = np.sort(u)
    n = u.size
    i = np.arange(1, n + 1, dtype=np.float64)
    a2 = -n - np.mean((2*i - 1) * (np.log(u) + np.log(1.0 - u[::-1])))
    return float(a2)

# --- helper: AUC from scores (Mann–Whitney / rank method) ---
def auc_rank(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores).astype(float)
    ranks = stats.rankdata(scores)
    n1 = int((labels == 1).sum())
    n0 = int((labels == 0).sum())
    sum_ranks_pos = float(ranks[labels == 1].sum())
    auc = (sum_ranks_pos - n1 * (n1 + 1) / 2.0) / (n0 * n1)
    return float(auc)

def main():
    data = np.load(NPZ_PATH, allow_pickle=True)
    print("Available keys:", data.files)

    # ---- clean arrays only ----
    orig = data["orig_Clean"]  # shape (num_images, 1024)
    wm   = data["wm_Clean"]

    assert orig.ndim == 2 and wm.ndim == 2, "Expected (num_images, num_patches) arrays"
    assert orig.shape[1] == PATCHES_PER_IMAGE, "Patch count mismatch vs PATCHES_PER_IMAGE"
    assert wm.shape[1]   == PATCHES_PER_IMAGE, "Patch count mismatch vs PATCHES_PER_IMAGE"

    n_orig, n_wm = orig.shape[0], wm.shape[0]
    print(f"\nClean set: orig images={n_orig}, wm images={n_wm}, patches/image={PATCHES_PER_IMAGE}")

    # ============================================================
    # 1) Tau justification: estimate p0(tau) from clean/orig patches
    # ============================================================
    orig_all = orig.reshape(-1)
    p0 = float((orig_all < TAU).mean())
    count0 = int((orig_all < TAU).sum())
    total0 = int(orig_all.size)

    print("\n=== Tau justification (from clean/orig) ===")
    print(f"tau = {TAU}")
    print(f"p0 = P(clean patch L2 < tau) ≈ {p0:.10f}  ({count0}/{total0})")
    print(f"Expected matches per clean image: N*p0 = {PATCHES_PER_IMAGE*p0:.4f}")

    # ============================================================
    # 2) Binomial tail probabilities for n-matches k=5..15
    #    This is the theoretical per-image false-positive rate alpha(tau,k)
    # ============================================================
    print("\n=== Binomial tail probabilities under H0 (theoretical FPR) ===")
    print("k   alpha = P(K >= k | clean, tau)")
    for k in range(K_MIN, K_MAX + 1):
        alpha = float(1.0 - binom.cdf(k - 1, PATCHES_PER_IMAGE, p0))
        print(f"{k:2d}  {alpha:.6e}")

    # ============================================================
    # 3) Empirical n-matches detector performance on your 50 clean images
    # ============================================================
    K_orig = (orig < TAU).sum(axis=1)
    K_wm   = (wm   < TAU).sum(axis=1)

    print("\n=== Empirical n-matches performance on this NPZ (clean only) ===")
    print("k   FPR_emp   TPR_emp   (rule: predict wm if K >= k)")
    for k in range(K_MIN, K_MAX + 1):
        fpr_emp = float((K_orig >= k).mean())
        tpr_emp = float((K_wm   >= k).mean())
        print(f"{k:2d}  {fpr_emp:7.3f}   {tpr_emp:7.3f}")

    # Also give AUC when using K as a continuous score
    labels = np.array([0]*n_orig + [1]*n_wm, dtype=int)
    scores_nm = np.concatenate([K_orig, K_wm])
    auc_nm = auc_rank(labels, scores_nm)
    print(f"\nAUC using n-matches score K (higher => more wm): {auc_nm:.6f}")

    # ============================================================
    # 4) AD score detector (per image), compared to n-matches
    #    Baseline: fit Normal(mu0,s0) to pooled clean/orig patches
    # ============================================================
    mu0 = float(orig_all.mean())
    s0  = float(orig_all.std(ddof=1))
    print("\n=== AD detector vs baseline Normal fitted on clean/orig ===")
    print(f"Baseline Normal: mu0={mu0:.6f}, sigma0={s0:.6f}")

    ad_orig = []
    ad_wm = []
    for img_patches in orig:
        u = normal_cdf(img_patches, mu0, s0)
        ad_orig.append(anderson_darling_uniform(u))
    for img_patches in wm:
        u = normal_cdf(img_patches, mu0, s0)
        ad_wm.append(anderson_darling_uniform(u))

    ad_orig = np.array(ad_orig, dtype=float)
    ad_wm   = np.array(ad_wm, dtype=float)

    print(f"AD score (orig): mean={ad_orig.mean():.3f}, min={ad_orig.min():.3f}, max={ad_orig.max():.3f}")
    print(f"AD score (wm):   mean={ad_wm.mean():.3f}, min={ad_wm.min():.3f}, max={ad_wm.max():.3f}")

    scores_ad = np.concatenate([ad_orig, ad_wm])
    auc_ad = auc_rank(labels, scores_ad)
    print(f"\nAUC using AD score (higher => less like clean): {auc_ad:.6f}")

    # Show a few simple AD thresholds (not exhaustive ROC; just illustrative)
    print("\nExample AD thresholds (rule: predict wm if AD >= T)")
    for T in [1, 2, 5, 10, 50, 100]:
        fpr = float((ad_orig >= T).mean())
        tpr = float((ad_wm   >= T).mean())
        print(f"T={T:>4}  FPR={fpr:6.3f}  TPR={tpr:6.3f}")

if __name__ == "__main__":
    main()
