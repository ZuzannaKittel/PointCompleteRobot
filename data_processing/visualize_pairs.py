import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def visualize_saved_pair_truthful(pt_file_path):
    print(f"\n🔍 Verifying alignment for: {pt_file_path}")
    data = torch.load(pt_file_path, map_location='cpu')

    # Extract
    partial_pts = data["partial_pts"].numpy()
    gt_pts = data["gt_pts"].numpy()
    # The anchor_world is the object's translation vector used during generation
    # We use this to ensure we aren't "re-centering" incorrectly
    anchor = data["anchor_world"].numpy() 

    # Clean
    partial_pts = partial_pts[np.any(partial_pts != 0, axis=1)]

    # We do NOT center independently anymore.
    # We treat both as being in a local canonical space, but we keep 
    # their relative orientation intact.
    
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
visualize_saved_pair_truthful("./data/geometric_pairs_dataset2/train/pair_00049_355e5e32db_sofa_28.pt")