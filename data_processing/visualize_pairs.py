import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Set global matplotlib font sizes
plt.rcParams.update({
    'font.size': 14,
    'axes.titlesize': 16,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
})

def print_geometry(name, pts):
    pts = np.asarray(pts)

    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)

    extents = maxs - mins
    center = (mins + maxs) / 2

    print(f"\n{name}")
    print(f"  center  : {center}")
    print(f"  extents : {extents}")
    print(f"  max dim : {extents.max():.4f}")
    print(f"  min dim : {extents.min():.4f}")
    print(f"  ratio   : {extents / extents.max()}")

def print_pca(name, pts):
    pca = PCA(n_components=3)
    pca.fit(pts)

    print(f"\n{name} PCA")
    print("explained variance:", pca.explained_variance_ratio_)
    print("axes:")
    print(np.round(pca.components_, 3))

def visualize_saved_pair_truthful(pt_file_path):
    print(f"\n🔍 Verifying alignment for: {pt_file_path}")
    data = torch.load(pt_file_path, map_location='cpu')

    print("\nMetadata")
    dataset = data.get("dataset", "Unknown")

    print(f"\nDataset: {dataset}")

    if dataset == "ScanNet++":
        print("scene   :", data["scene_id"])

    print("object  :", data["object_id"])
    print("category:", data["category"])

    # Extract
    partial_pts = data["partial_pts"].numpy()
    gt_pts = data["gt_pts"].numpy()

    print_geometry("Partial", partial_pts)
    print_geometry("GT", gt_pts)

    print_pca("Partial", partial_pts)
    print_pca("GT", gt_pts)

    # Clean
    partial_pts = partial_pts[np.any(partial_pts != 0, axis=1)]

    print("\nPartial mean:")
    print(partial_pts.mean(axis=0))

    print("\nGT mean:")
    print(gt_pts.mean(axis=0))

    print("\nPartial covariance:")
    print(np.round(np.cov(partial_pts.T), 3))

    print("\nGT covariance:")
    print(np.round(np.cov(gt_pts.T), 3))

    print("\nFeature shapes:")
    print(f"  Partial: {partial_pts.shape[0]} points")
    print(f"  GT: {gt_pts.shape[0]} points")

    # PCA Coloring for the scan
    try:
        pca = PCA(n_components=3)
        pca_colors = pca.fit_transform(partial_pts)
        pca_colors = (pca_colors - pca_colors.min(axis=0)) / (pca_colors.max(axis=0) - pca_colors.min(axis=0) + 1e-8)
    except:
        pca_colors = 'red'

    fig = plt.figure(figsize=(20, 7))
    
    # Subplot 1: The Raw Scan
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(partial_pts[:, 0], partial_pts[:, 1], partial_pts[:, 2], c=pca_colors, s=2, alpha=0.8)
    ax1.set_title("Partial ScanNet++ scan\n(normalized space)", fontsize=20, pad=12)

    # Subplot 2: The GT CAD
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], c='lightgreen', s=2, alpha=0.5)
    ax2.set_title("GT Scannotate++ CAD\n(normalized space)", fontsize=20, pad=12)

    # Subplot 3: The Overlay (The "Truth")
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], c='gray', s=2, alpha=0.1)
    ax3.scatter(partial_pts[:, 0], partial_pts[:, 1], partial_pts[:, 2], c=pca_colors, s=2, alpha=0.8)
    ax3.set_title("Alignment overlay", fontsize=20, pad=12)

    # Format axis limits to [-0.5, 0.5] and update font sizes
    for ax in [ax1, ax2, ax3]:
        ax.set_box_aspect([1, 1, 1])
        ax.set_xlim([-0.5, 0.5])
        ax.set_ylim([-0.5, 0.5])
        ax.set_zlim([-0.5, 0.5])
        
        ax.set_xlabel("X", fontsize=14, labelpad=8)
        ax.set_ylabel("Y", fontsize=14, labelpad=8)
        ax.set_zlabel("Z", fontsize=14, labelpad=8)
        
        ax.tick_params(axis='both', which='major', labelsize=11)

    plt.tight_layout()
    plt.show()

# Target object pair
TARGET_PAIR = "pair_00020_7b6477cb95_display_29.pt"

# Usage
visualize_saved_pair_truthful(
    f"./data/pairs_scannet/train/{TARGET_PAIR}"
)