import os
import sys
import torch
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import KDTree

# Use the modern import for spconv
import spconv.pytorch as spconv 
from core.utonia_backbone import UtoniaBackbone

def visualize_features_complete(sparse_coords, sparse_feats, original_pc, save_path="chair_features.png"):
    s_coords = sparse_coords.detach().cpu().numpy()
    if s_coords.shape[1] == 4: s_coords = s_coords[:, 1:]
    
    s_feats = sparse_feats.detach().cpu().numpy().squeeze()
    full_pc = original_pc.detach().cpu().numpy().squeeze()

    # PCA to get colors
    pca = PCA(n_components=3)
    rgb_raw = pca.fit_transform(s_feats)
    rgb = (rgb_raw - rgb_raw.min()) / (rgb_raw.max() - rgb_raw.min() + 1e-8)

    # CRITICAL: Scale full_pc to match the voxel coordinate range of s_coords
    # This assumes the backbone voxels the input internally. 
    # If Utonia uses a 64x64x64 grid:
    pc_min, pc_max = full_pc.min(0), full_pc.max(0)
    s_min, s_max = s_coords.min(0), s_coords.max(0)
    
    # Normalize full_pc to [0, 1] then scale to [s_min, s_max]
    full_pc_scaled = (full_pc - pc_min) / (pc_max - pc_min + 1e-8)
    full_pc_scaled = full_pc_scaled * (s_max - s_min) + s_min

    # Now the KDTree query will work logically
    tree = KDTree(s_coords)
    _, indices = tree.query(full_pc_scaled)
    full_rgb = rgb[indices]

    # ... rest of your plotting code ...

    # 4. Save Plot
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(full_pc[:, 0], full_pc[:, 1], full_pc[:, 2], c=full_rgb, s=2)
    
    # Keep it centered for consistent view
    ax.set_xlim(-0.5, 0.5); ax.set_ylim(-0.5, 0.5); ax.set_zlim(-0.5, 0.5)
    
    plt.title("Utonia Feature Map: Segmented Chair")
    plt.savefig(save_path)
    print(f"✨ SUCCESS: saved to {save_path}")

def get_gpu_safe_chair(path):
    # 1. Load and immediately downsample
    pcd = o3d.io.read_point_cloud(path)
    
    # This physically collapses all points within a 5mm box into one.
    # It is the only 100% way to prevent coordinate collisions in spconv.
    pcd = pcd.voxel_down_sample(voxel_size=0.005)
    
    # 2. Remove any "Ghost" points (NaNs or non-finite numbers)
    pcd.remove_non_finite_points()
    
    points = np.asarray(pcd.points).astype(np.float32)
    
    # 3. Strict ShapeNet Normalization
    # Centering
    points -= np.mean(points, axis=0)
    # Scaling to fit strictly inside [-0.5, 0.5]
    max_dist = np.max(np.sqrt(np.sum(points**2, axis=1)))
    points /= (max_dist * 2.1 + 1e-8) # 2.1 adds a tiny safety "buffer" at the edges
    
    # 4. Force exact point count
    if len(points) > 8192:
        points = points[np.random.choice(len(points), 8192, replace=False)]
    else:
        points = points[np.random.choice(len(points), 8192, replace=True)]

    print(f"✅ Cleaned Chair: {len(points)} points, Range: {points.min():.3f} to {points.max():.3f}")
    return torch.from_numpy(points).float().unsqueeze(0)

if __name__ == "__main__":
    device = torch.device('cuda')
    backbone = UtoniaBackbone(use_mock=False).to(device)
    
    checkpoint = torch.load("checkpoints/utoniadreamer/latest.pth", map_location=device)
    full_sd = checkpoint['model'] if 'model' in checkpoint else checkpoint
    backbone_sd = {k.replace('backbone.', ''): v for k, v in full_sd.items() if k.startswith('backbone.')}
    backbone.load_state_dict(backbone_sd, strict=False)
    backbone.eval()

    print("🧹 Cleaning chair data...")
    complete_pc = get_gpu_safe_chair("assets/segmented_chair.pcd").to(device)

    print(f"🧠 Inference on {torch.cuda.get_device_name(0)}...")
    with torch.no_grad():
        # Utonia returns: global_feat, (sparse_coords, point_feats)
        _, (sparse_coords, point_feats) = backbone(complete_pc)
    
    # --- THE MISSING CALL ---
    visualize_features_complete(sparse_coords, point_feats, complete_pc)