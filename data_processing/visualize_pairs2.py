import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

def apply_rotation(pts, axis='z', degrees=0):
    theta = np.radians(degrees)
    cos_t, sin_t = np.cos(theta), np.sin(theta)
    
    if axis == 'z':
        R = np.array([
            [cos_t, -sin_t, 0],
            [sin_t,  cos_t, 0],
            [0,      0,     1]
        ])
    elif axis == 'x':
        R = np.array([
            [1, 0,      0],
            [0, cos_t, -sin_t],
            [0, sin_t,  cos_t]
        ])
    elif axis == 'y':
        R = np.array([
            [cos_t,  0, sin_t],
            [0,      1, 0],
            [-sin_t, 0, cos_t]
        ])
    else:
        return pts
    return pts @ R.T

def test_pure_axis_sweeps(pt_file_path):
    print(f"\n🔄 Running pure sweeps on: {pt_file_path}")
    data = torch.load(pt_file_path, map_location='cpu')

    partial_pts = data["partial_pts"].numpy()
    gt_pts = data["gt_pts"].numpy()
    partial_pts = partial_pts[np.any(partial_pts != 0, axis=1)]

    # Test pure Y-axis rotations directly on the original points

    gt_y_90  = apply_rotation(gt_pts, axis='y', degrees=90)
    gt_y_180 = apply_rotation(gt_pts, axis='y', degrees=180)
    gt_y_270 = apply_rotation(gt_pts, axis='y', degrees=270)

    try:
        pca = PCA(n_components=3)
        pca_colors = pca.fit_transform(partial_pts)
        pca_colors = (pca_colors - pca_colors.min(axis=0)) / (pca_colors.max(axis=0) - pca_colors.min(axis=0) + 1e-8)
    except:
        pca_colors = 'red'

    fig = plt.figure(figsize=(20, 5))
    
    def setup_ax(ax, title):
        ax.set_title(title)
        ax.set_box_aspect([1,1,1])
        ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1]); ax.set_zlim([-1, 1])
        ax.set_xlabel('X'); ax.set_ylabel('Y'); ax.set_zlabel('Z')

    # Plot 1: Original Uncorrected (from your very first image)
    ax1 = fig.add_subplot(141, projection='3d')
    ax1.scatter(gt_pts[:, 0], gt_pts[:, 1], gt_pts[:, 2], c='blue', s=1, alpha=0.1)
    ax1.scatter(partial_pts[:, 0], partial_pts[:, 1], partial_pts[:, 2], c=pca_colors, s=2, alpha=0.3)
    setup_ax(ax1, "1. Original (Uncorrected)")

    # Plot 2: Pure Y: 90
    ax2 = fig.add_subplot(142, projection='3d')
    ax2.scatter(gt_y_90[:, 0], gt_y_90[:, 1], gt_y_90[:, 2], c='orange', s=1, alpha=0.1)
    ax2.scatter(partial_pts[:, 0], partial_pts[:, 1], partial_pts[:, 2], c=pca_colors, s=2, alpha=0.3)
    setup_ax(ax2, "2. Pure Rotation Y: 90")

    # Plot 3: Pure Y: 180
    ax3 = fig.add_subplot(143, projection='3d')
    ax3.scatter(gt_y_180[:, 0], gt_y_180[:, 1], gt_y_180[:, 2], c='purple', s=1, alpha=0.1)
    ax3.scatter(partial_pts[:, 0], partial_pts[:, 1], partial_pts[:, 2], c=pca_colors, s=2, alpha=0.3)
    setup_ax(ax3, "3. Pure Rotation Y: 180")

    # Plot 4: Pure Y: 270
    ax4 = fig.add_subplot(144, projection='3d')
    ax4.scatter(gt_y_270[:, 0], gt_y_270[:, 1], gt_y_270[:, 2], c='green', s=1, alpha=0.1)
    ax4.scatter(partial_pts[:, 0], partial_pts[:, 1], partial_pts[:, 2], c=pca_colors, s=2, alpha=0.3)
    setup_ax(ax4, "4. Pure Rotation Y: 270")

    plt.tight_layout()
    plt.show()

# Run it fresh on the file
test_pure_axis_sweeps("./data/geometric_pairs_dataset/train/pair_00006_7b6477cb95_chair_27.pt")