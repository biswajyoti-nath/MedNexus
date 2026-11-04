"""
Brain MRI Segmentation — Data Exploration & Visualization
=========================================================
Generates publication-quality figures that analyze the Kaggle LGG MRI
Segmentation dataset. All outputs are saved to results/exploration/.

Run:
    python explore_data.py
"""

import os
import glob
import random
import numpy as np
from PIL import Image
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")                       # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap

# ─── Config ───────────────────────────────────────────────────────────
DATA_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kaggle_3m")
OUT_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", "exploration")
SEED       = 42
IMG_SIZE   = 256            # native size of the dataset images

os.makedirs(OUT_DIR, exist_ok=True)
random.seed(SEED)
np.random.seed(SEED)

# ─── Styling ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor":   "#0d1117",
    "axes.facecolor":     "#161b22",
    "axes.edgecolor":     "#30363d",
    "axes.labelcolor":    "#c9d1d9",
    "text.color":         "#c9d1d9",
    "xtick.color":        "#8b949e",
    "ytick.color":        "#8b949e",
    "grid.color":         "#21262d",
    "font.family":        "sans-serif",
    "font.size":          11,
    "axes.titlesize":     13,
    "axes.labelsize":     11,
})

PALETTE = {
    "teal":      "#58a6ff",
    "green":     "#3fb950",
    "red":       "#f85149",
    "purple":    "#bc8cff",
    "orange":    "#d29922",
    "pink":      "#f778ba",
    "cyan":      "#39d2c0",
    "gray":      "#8b949e",
}


# =====================================================================
#  1. Gather dataset statistics
# =====================================================================
def gather_stats():
    """Walk dataset and collect per-patient and per-slice statistics."""
    patients = sorted([d for d in os.listdir(DATA_DIR)
                       if os.path.isdir(os.path.join(DATA_DIR, d))])

    patient_stats = {}      # patient -> { total, positive, negative }
    all_mask_areas  = []    # non-zero mask pixel counts (for positive slices)
    all_intensities = []    # mean intensity of each MRI slice
    sample_images   = []    # (img_path, mask_path, has_tumor) for later vis

    for patient in patients:
        pdir = os.path.join(DATA_DIR, patient)
        masks = sorted(glob.glob(os.path.join(pdir, "*_mask.tif")))
        pos = 0
        neg = 0
        for m in masks:
            img_path = m.replace("_mask.tif", ".tif")
            mask_arr = np.array(Image.open(m).convert("L"))
            has_tumor = mask_arr.max() > 0
            if has_tumor:
                pos += 1
                area = (mask_arr > 0).sum()
                all_mask_areas.append(area)
            else:
                neg += 1

            # Mean intensity of MRI image
            img_arr = np.array(Image.open(img_path).convert("L"))
            all_intensities.append(img_arr.mean())

            sample_images.append((img_path, m, has_tumor))

        patient_stats[patient] = {"total": pos + neg, "positive": pos, "negative": neg}

    return patients, patient_stats, all_mask_areas, all_intensities, sample_images


print("Scanning dataset ...")
patients, patient_stats, all_mask_areas, all_intensities, sample_images = gather_stats()

total_slices   = sum(s["total"]    for s in patient_stats.values())
total_positive = sum(s["positive"] for s in patient_stats.values())
total_negative = sum(s["negative"] for s in patient_stats.values())

print(f"  Patients       : {len(patients)}")
print(f"  Total slices   : {total_slices}")
print(f"  With tumor     : {total_positive}  ({total_positive/total_slices:.1%})")
print(f"  Without tumor  : {total_negative}  ({total_negative/total_slices:.1%})")


# =====================================================================
#  2. Figure 1 — Dataset Overview (Class Distribution)
# =====================================================================
def plot_class_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Dataset Class Distribution", fontsize=16, fontweight="bold", y=1.02)

    # --- Bar chart ---
    ax = axes[0]
    categories = ["No Tumor\n(Negative)", "Tumor\n(Positive)"]
    counts     = [total_negative, total_positive]
    colors     = [PALETTE["teal"], PALETTE["red"]]
    bars = ax.bar(categories, counts, color=colors, width=0.55, edgecolor="#30363d", linewidth=0.8)
    for bar, cnt in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 30,
                f"{cnt}", ha="center", va="bottom", fontweight="bold", fontsize=13, color="#c9d1d9")
    ax.set_ylabel("Number of Slices")
    ax.set_title("Slice Count by Class")
    ax.set_ylim(0, max(counts) * 1.15)
    ax.grid(axis="y", alpha=0.3)

    # --- Pie chart ---
    ax = axes[1]
    explode = (0.03, 0.03)
    wedges, texts, autotexts = ax.pie(
        counts, labels=categories, autopct="%1.1f%%", colors=colors,
        explode=explode, startangle=140,
        textprops={"color": "#c9d1d9", "fontsize": 12},
        wedgeprops={"edgecolor": "#0d1117", "linewidth": 1.5},
    )
    for at in autotexts:
        at.set_fontweight("bold")
    ax.set_title("Class Proportions")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "01_class_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# =====================================================================
#  3. Figure 2 — Per-Patient Slice Distribution
# =====================================================================
def plot_patient_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("Per-Patient Analysis", fontsize=16, fontweight="bold", y=1.02)

    # Sort patients by total slices (descending)
    sorted_patients = sorted(patient_stats.items(), key=lambda x: x[1]["total"], reverse=True)
    names      = [p[0].replace("TCGA_", "") for p in sorted_patients]
    positives  = [p[1]["positive"] for p in sorted_patients]
    negatives  = [p[1]["negative"] for p in sorted_patients]
    totals     = [p[1]["total"]    for p in sorted_patients]

    # --- Stacked horizontal bar ---
    ax = axes[0]
    y = np.arange(len(names))
    ax.barh(y, negatives, color=PALETTE["teal"],  label="No Tumor", edgecolor="#30363d", linewidth=0.3)
    ax.barh(y, positives, left=negatives, color=PALETTE["red"], label="Tumor", edgecolor="#30363d", linewidth=0.3)
    ax.set_yticks(y[::5])
    ax.set_yticklabels([names[i] for i in range(0, len(names), 5)], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Number of Slices")
    ax.set_title(f"Slices per Patient (n={len(patients)})")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)

    # --- Histogram of slices per patient ---
    ax = axes[1]
    ax.hist(totals, bins=20, color=PALETTE["purple"], edgecolor="#30363d", linewidth=0.8, alpha=0.85)
    ax.axvline(np.mean(totals), color=PALETTE["orange"], linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(totals):.1f}")
    ax.axvline(np.median(totals), color=PALETTE["cyan"], linestyle="--", linewidth=1.5,
               label=f"Median = {np.median(totals):.0f}")
    ax.set_xlabel("Slices per Patient")
    ax.set_ylabel("Number of Patients")
    ax.set_title("Distribution of Slices per Patient")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "02_patient_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# =====================================================================
#  4. Figure 3 — Sample MRI Images + Masks (positive cases)
# =====================================================================
def plot_sample_positive():
    positives = [(ip, mp) for ip, mp, ht in sample_images if ht]
    random.shuffle(positives)
    chosen = positives[:10]

    fig, axes = plt.subplots(2, 10, figsize=(22, 5))
    fig.suptitle("Sample Brain MRI Slices with Tumor Masks", fontsize=16, fontweight="bold", y=1.04)

    # Custom red-transparent colormap for mask overlay
    cmap_mask = LinearSegmentedColormap.from_list("mask_red",
        [(0, 0, 0, 0), (1, 0.2, 0.2, 0.7)])

    for i, (img_path, mask_path) in enumerate(chosen):
        img  = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Row 0: MRI image
        axes[0, i].imshow(img)
        axes[0, i].axis("off")
        if i == 0:
            axes[0, i].set_ylabel("MRI", fontsize=11, fontweight="bold", rotation=0, labelpad=35)

        # Row 1: MRI + mask overlay
        axes[1, i].imshow(img)
        axes[1, i].imshow(mask, cmap=cmap_mask, alpha=0.6)
        axes[1, i].axis("off")
        if i == 0:
            axes[1, i].set_ylabel("+ Mask", fontsize=11, fontweight="bold", rotation=0, labelpad=35)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "03_sample_positive.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# =====================================================================
#  5. Figure 4 — Sample MRI Images (negative cases)
# =====================================================================
def plot_sample_negative():
    negatives = [(ip, mp) for ip, mp, ht in sample_images if not ht]
    random.shuffle(negatives)
    chosen = negatives[:10]

    fig, axes = plt.subplots(1, 10, figsize=(22, 2.5))
    fig.suptitle("Sample Brain MRI Slices Without Tumor (Healthy)", fontsize=16, fontweight="bold", y=1.1)

    for i, (img_path, _) in enumerate(chosen):
        img = np.array(Image.open(img_path).convert("RGB"))
        axes[i].imshow(img)
        axes[i].axis("off")

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "04_sample_negative.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# =====================================================================
#  6. Figure 5 — Tumor Mask Area Distribution
# =====================================================================
def plot_mask_area_distribution():
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Tumor Mask Area Analysis", fontsize=16, fontweight="bold", y=1.02)

    areas = np.array(all_mask_areas)
    total_pixels = IMG_SIZE * IMG_SIZE
    area_pcts = areas / total_pixels * 100

    # --- Histogram (pixel count) ---
    ax = axes[0]
    ax.hist(areas, bins=50, color=PALETTE["green"], edgecolor="#30363d", linewidth=0.6, alpha=0.85)
    ax.axvline(np.mean(areas), color=PALETTE["orange"], linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(areas):.0f} px")
    ax.axvline(np.median(areas), color=PALETTE["cyan"], linestyle="--", linewidth=1.5,
               label=f"Median = {np.median(areas):.0f} px")
    ax.set_xlabel("Tumor Mask Area (pixels)")
    ax.set_ylabel("Number of Slices")
    ax.set_title("Absolute Mask Area")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    # --- Histogram (% of image) ---
    ax = axes[1]
    ax.hist(area_pcts, bins=50, color=PALETTE["pink"], edgecolor="#30363d", linewidth=0.6, alpha=0.85)
    ax.axvline(np.mean(area_pcts), color=PALETTE["orange"], linestyle="--", linewidth=1.5,
               label=f"Mean = {np.mean(area_pcts):.2f}%")
    ax.axvline(np.median(area_pcts), color=PALETTE["cyan"], linestyle="--", linewidth=1.5,
               label=f"Median = {np.median(area_pcts):.2f}%")
    ax.set_xlabel("Tumor Mask Area (% of image)")
    ax.set_ylabel("Number of Slices")
    ax.set_title("Relative Mask Area")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "05_mask_area_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# =====================================================================
#  7. Figure 6 — Pixel Intensity Distribution
# =====================================================================
def plot_intensity_analysis():
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("MRI Pixel Intensity Analysis", fontsize=16, fontweight="bold", y=1.02)

    # Collect intensity histograms for a sample of positive & negative images
    pos_samples = [(ip, mp) for ip, mp, ht in sample_images if ht]
    neg_samples = [(ip, mp) for ip, mp, ht in sample_images if not ht]

    random.shuffle(pos_samples)
    random.shuffle(neg_samples)

    n_sample = min(200, len(pos_samples), len(neg_samples))

    pos_intensities = []
    neg_intensities = []
    for ip, _ in pos_samples[:n_sample]:
        arr = np.array(Image.open(ip).convert("L")).ravel()
        pos_intensities.extend(arr.tolist())
    for ip, _ in neg_samples[:n_sample]:
        arr = np.array(Image.open(ip).convert("L")).ravel()
        neg_intensities.extend(arr.tolist())

    ax.hist(neg_intensities, bins=128, density=True, alpha=0.55, color=PALETTE["teal"],
            label=f"No Tumor (n={n_sample})", edgecolor="none")
    ax.hist(pos_intensities, bins=128, density=True, alpha=0.55, color=PALETTE["red"],
            label=f"Tumor (n={n_sample})", edgecolor="none")
    ax.set_xlabel("Pixel Intensity (0–255)")
    ax.set_ylabel("Density")
    ax.set_title("Intensity Distribution: Tumor vs No Tumor")
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    path = os.path.join(OUT_DIR, "06_intensity_distribution.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# =====================================================================
#  8. Figure 7 — Summary Dashboard
# =====================================================================
def plot_summary_dashboard():
    fig = plt.figure(figsize=(16, 9))
    fig.suptitle("Brain MRI Segmentation — Dataset Dashboard",
                 fontsize=20, fontweight="bold", color="#58a6ff", y=0.98)

    gs = gridspec.GridSpec(2, 3, hspace=0.4, wspace=0.35)

    # ---- (0,0) Key Stats ----
    ax = fig.add_subplot(gs[0, 0])
    ax.axis("off")
    stats_text = (
        f"{'Patients':<20}{len(patients):>6}\n"
        f"{'Total Slices':<20}{total_slices:>6}\n"
        f"{'With Tumor':<20}{total_positive:>6}  ({total_positive/total_slices:.1%})\n"
        f"{'Without Tumor':<20}{total_negative:>6}  ({total_negative/total_slices:.1%})\n"
        f"{'Image Size':<20}{IMG_SIZE}×{IMG_SIZE:>1}\n"
        f"{'Mean Slices/Patient':<20}{total_slices/len(patients):>6.1f}\n"
    )
    ax.text(0.05, 0.95, stats_text, transform=ax.transAxes,
            fontfamily="monospace", fontsize=12, verticalalignment="top", color="#c9d1d9")
    ax.set_title("Key Statistics", fontsize=14, fontweight="bold")

    # ---- (0,1) Class bar ----
    ax = fig.add_subplot(gs[0, 1])
    bars = ax.bar(["Negative", "Positive"], [total_negative, total_positive],
                  color=[PALETTE["teal"], PALETTE["red"]], width=0.5, edgecolor="#30363d")
    for bar, cnt in zip(bars, [total_negative, total_positive]):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20,
                str(cnt), ha="center", va="bottom", fontweight="bold", fontsize=11, color="#c9d1d9")
    ax.set_ylabel("Count")
    ax.set_title("Class Balance", fontsize=14, fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    # ---- (0,2) Mask area hist ----
    ax = fig.add_subplot(gs[0, 2])
    area_pcts = np.array(all_mask_areas) / (IMG_SIZE*IMG_SIZE) * 100
    ax.hist(area_pcts, bins=40, color=PALETTE["green"], edgecolor="#30363d", linewidth=0.5, alpha=0.85)
    ax.axvline(np.median(area_pcts), color=PALETTE["orange"], ls="--", lw=1.5,
               label=f"Median {np.median(area_pcts):.1f}%")
    ax.set_xlabel("Mask Area (%)")
    ax.set_ylabel("Count")
    ax.set_title("Tumor Size Distribution", fontsize=14, fontweight="bold")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # ---- (1,0:3) Sample images ----
    inner_gs = gridspec.GridSpecFromSubplotSpec(2, 6, subplot_spec=gs[1, :], hspace=0.05, wspace=0.05)
    cmap_mask = LinearSegmentedColormap.from_list("r", [(0,0,0,0), (1,0.2,0.2,0.7)])

    positives = [(ip, mp) for ip, mp, ht in sample_images if ht]
    random.shuffle(positives)
    for i in range(6):
        ip, mp = positives[i]
        img  = np.array(Image.open(ip).convert("RGB"))
        mask = np.array(Image.open(mp).convert("L"))

        ax1 = fig.add_subplot(inner_gs[0, i])
        ax1.imshow(img)
        ax1.axis("off")
        if i == 0:
            ax1.set_title("MRI", fontsize=10, color=PALETTE["teal"])

        ax2 = fig.add_subplot(inner_gs[1, i])
        ax2.imshow(img)
        ax2.imshow(mask, cmap=cmap_mask, alpha=0.6)
        ax2.axis("off")
        if i == 0:
            ax2.set_title("+ Mask", fontsize=10, color=PALETTE["red"])

    path = os.path.join(OUT_DIR, "00_dashboard.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved -> {path}")


# =====================================================================
#  Run all
# =====================================================================
if __name__ == "__main__":
    print("\nGenerating visualizations ...\n")
    plot_summary_dashboard()
    plot_class_distribution()
    plot_patient_distribution()
    plot_sample_positive()
    plot_sample_negative()
    plot_mask_area_distribution()
    plot_intensity_analysis()
    print(f"\nDone! All figures saved to: {OUT_DIR}\n")
