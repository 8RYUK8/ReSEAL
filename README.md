# ReSEAL
Evaluations of seal watermark



# Analysis Scripts Guide

This repository contains several analysis scripts used to evaluate SEAL and related watermark verification behavior from saved `.npz` result files.  
Most scripts operate on arrays of patch-wise L2 distances stored in the following format:

- `watermarked`: patch-wise L2 distances for watermarked images
- `random`: patch-wise L2 distances for non-watermarked / random images

These scripts are mainly used to study patch-level and image-level separability, threshold behavior, and alternative decision rules.

---

## Folder Notes

### `Wasserstein+dprime`
This folder contains the scripts, plots, analysis outputs, and `.npz` files from all runs and tgheir Wasserstein-distance and d-prime based analysis.  
It serves as the main location for these statistical evaluation experiments and their generated figures.

### `attacked images`
This folder contains the adversarially perturbed images generated using the adversary epsilon attack.  
These images are used to evaluate how watermark detection behaves under adversarial perturbations.

### `SEAL with Distortions`
This folder contains the analysis scripts and images that underwent distortions, these include: Blur, Brightness, JPEG Compression ... .


---

## Script Overview

### 1. Global patch clustering / visualization script
This script loads patch-wise L2 distances from watermarked images, fits a global Gaussian Mixture Model (GMM) to all patches, and visualizes patch clusters in 2D using either t-SNE or PCA.

**Purpose**
- Identify whether patch distances naturally separate into groups such as likely signal patches and likely noise patches
- Visualize patch structure across all images

**Main idea**
- Flatten all patches from all images
- Fit a GMM on patch L2 values
- Build a feature vector using patch distance and spatial coordinates
- Reduce to 2D for visualization

---

### 2. Trimmed-mean analysis with beta sweep
This script performs image-level detection by averaging only the lowest `beta` fraction of patch-wise L2 distances for each image.

**Purpose**
- Test whether focusing on the best-matching patches improves separation between watermarked and random images
- Evaluate score distributions and threshold behavior

**Main outputs**
- Histogram of image-level scores
- ECDF plot
- ROC curve
- Optional beta sweep and single-image beta curves

**Key parameters**
- `beta`: fraction of lowest-L2 patches kept per image
- `tau`: fixed image-level threshold

**Interpretation**
- Smaller trimmed-mean L2 suggests the image is more likely to be watermarked

---

### 3. SEAL-style verification analysis
This script follows a verification procedure similar to Algorithm 3 in the SEAL paper.

**Purpose**
- Evaluate watermark detection using patch-level thresholding followed by image-level match counting

**Main idea**
1. Choose a patch threshold `tau` from random patches using a target patch false positive rate
2. Mark a patch as matched if `L2 < tau`
3. Count the number of matched patches per image
4. Classify an image as watermarked if the number of matches exceeds a threshold `m_match`

**Main outputs**
- Patch-level statistics
- Patch-level ROC AUC
- Image-level match-count statistics
- Sweep over `m_match` thresholds
- Suggested operating points for selected FPR targets

---

### 4. Fixed-threshold trimmed-mean analysis
This script is a simpler image-level detector based on the trimmed mean of patch-wise L2 values, using a manually chosen threshold `tau`.

**Purpose**
- Provide a direct and interpretable image-level decision rule
- Compare watermarked and random score distributions under a fixed threshold

**Main outputs**
- Image-level statistics
- ROC AUC
- Histogram
- ECDF
- ROC curve with operating point

**Key idea**
- For each image, sort patch L2 values
- Average the lowest `beta` fraction
- Predict watermarked if the resulting score is below `tau`

---

### Distortion-Aware Trimmed-Mean Analysis

This script extends the trimmed-mean detection approach to support `.npz` files containing multiple distortions (e.g., JPEG, blur, noise).

- Supports two formats:
  - Legacy: `watermarked` vs `random`
  - Distortion-based: `wm_<distortion>` and `orig_<distortion>`

- Allows selecting a specific distortion via:
  ```bash
  --distortion JPEG_80
  
## General Usage

Most scripts are run from the command line with an `.npz` file as input.

Example:
```bash
python script_name.py all_patch_l2_1024_7.npz
