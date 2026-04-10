import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA

# =====================
# CONFIG
# =====================
NPZ_PATH = "all_min_l2_1024_7.npz"
N_COMPONENTS = 3        # set to 2 or 3
EMBED_METHOD = "tsne"  # "tsne" or "pca"
SEED = 0

# =====================
# LOAD DATA
# =====================
data = np.load(NPZ_PATH, allow_pickle=True)
wm = data["watermarked"]          # shape (N_images, 1024)

N, K = wm.shape
side = int(np.sqrt(K))

# Flatten all patches across all images
d_all = wm.reshape(-1, 1)         # shape (N*K, 1)

# =====================
# FIT GMM (GLOBAL)
# =====================
gmm = GaussianMixture(
    n_components=N_COMPONENTS,
    covariance_type="full",
    random_state=SEED
)
gmm.fit(d_all)

# Sort components by mean (0 = lowest mean = signal)
means = gmm.means_.flatten()
order = np.argsort(means)

labels_raw = gmm.predict(d_all)
labels = np.zeros_like(labels_raw)
for new_k, old_k in enumerate(order):
    labels[labels_raw == old_k] = new_k

# =====================
# BUILD FEATURES FOR 2D VISUALIZATION
# =====================
# Patch spatial coordinates (repeated for all images)
xs, ys = np.meshgrid(
    np.linspace(0, 1, side),
    np.linspace(0, 1, side)
)
xs = np.tile(xs.flatten(), N)
ys = np.tile(ys.flatten(), N)

# Normalize distances
d_norm = (d_all.flatten() - d_all.mean()) / (d_all.std() + 1e-12)

# Feature vector per patch
# (distance + spatial info → separable clusters)
X = np.stack([d_norm, xs, ys], axis=1)

# =====================
# 2D EMBEDDING
# =====================
if EMBED_METHOD == "tsne":
    embed = TSNE(
        n_components=2,
        perplexity=30,
        init="pca",
        learning_rate="auto",
        random_state=SEED
    ).fit_transform(X)
elif EMBED_METHOD == "pca":
    embed = PCA(n_components=2, random_state=SEED).fit_transform(X)
else:
    raise ValueError("EMBED_METHOD must be 'tsne' or 'pca'")

# =====================
# PLOT GLOBAL MOSAIC
# =====================
plt.figure(figsize=(6, 6))
plt.scatter(
    embed[:, 0],
    embed[:, 1],
    c=labels,
    s=6,
    cmap="tab10",
    alpha=0.9,
    linewidths=0
)
plt.xticks([])
plt.yticks([])
plt.title(
    f"Global patch clusters ({N_COMPONENTS}-GMM, {EMBED_METHOD.upper()})\n"
    f"All patches from all images"
)
plt.tight_layout()
plt.show()
