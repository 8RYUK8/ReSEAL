# GMM-signal + d' selection report (high-resolution)

**Inputs glob:** `all_min_l2_1024_*.npz`

## Pooled dataset
- Files pooled: 8
- Example keys used (first file): WM=`watermarked`, Random=`random`
- WM images: 525
- Random images: 525
- K patches/image: 1024

## Patch-level GMM (WM patches)
- μ_signal = 1.547012, σ²_signal = 0.228426
- μ_bg = 5.462492, σ²_bg = 1.052270
- mixture weights = (0.9179400299061136, 0.08205997009388645)

## d' results (standardized separation)
- Peak/Best d' in grid: **5.086** at n_keep=2 (β=0.001953)
- Bayes error proxy at best point: Pe≈Φ(-d'/2) = **0.0055**
- First n_keep with d' ≥ 3.0: **1**
- Plateau (d' ≥ 95% of max): **[2, 3]**
- Conservative recommendation (smallest n in plateau): **2**

## Null-controlled thresholding
For each n_keep we set τ = Q_alpha(T_n | Random) with alpha=0.0, then report empirical TPR/FPR at that τ.

## Output files
- `dprime_table.csv`
- `dprime_vs_nkeep_highres.png`
- `means_vs_nkeep.png`
- `vars_vs_nkeep.png`
- `bayes_error_vs_nkeep.png`
- `tpr_fpr_points.png`
