import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

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

    fig = plt.figure(figsize=(18, 6))
    
    # Subplot 1: The Raw Scan
    ax1 = fig.add_subplot(131, projection='3d')
    ax1.scatter(partial_pts[:, 0], partial_pts[:, 1], partial_pts[:, 2], c=pca_colors, s=1, alpha=0.8)
    ax1.set_title("Partial ScanNet++ Scan (Normalized Space)")

    # Subplot 2: The GT CAD
    ax2 = fig.add_subplot(132, projection='3d')
    ax2.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], c='lightgreen', s=1, alpha=0.5)
    ax2.set_title("GT Scannotate++ CAD (Normalized Space)")

    # Subplot 3: The Overlay (The "Truth")
    # If this looks messy, it means the rotation matrix (R_pure) 
    # in your data_factory2.py does not match the CAD's front-facing axis.
    ax3 = fig.add_subplot(133, projection='3d')
    ax3.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], c='gray', s=1, alpha=0.05)
    ax3.scatter(partial_pts[:, 0], partial_pts[:, 1], partial_pts[:, 2], c=pca_colors, s=1, alpha=0.8)
    ax3.set_title("Alignment Overlay")

    for ax in [ax1, ax2, ax3]:
        ax.set_box_aspect([1,1,1])
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])

    plt.tight_layout()
    plt.show()

# Usage
visualize_saved_pair_truthful("./data/pairs_scannet/train/pair_01815_acd95847c5_display_75.pt")