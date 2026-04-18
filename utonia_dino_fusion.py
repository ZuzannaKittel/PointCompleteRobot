import os
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import cv2
from segment_anything import sam_model_registry, SamPredictor


"""
Utonia-DINO Fusion: A Multi-modal Feature Extraction Pipeline
Description:
This script demonstrates how to fuse geometric features from Utonia with semantic features from DINOv2, 
using SAM for object segmentation. The final output is a triple PCA visualization comparing the three feature spaces.
Key Steps:
1. Load RGB-D data from Replica and segment the object using SAM.
2. Extract DINOv2 features from the RGB image and Utonia features from the point cloud.
3. Fuse the features via L2-normalization and concatenation.
4. Visualize the results in a triple PCA plot, showcasing geometry, semantics, and their fusion.
Requirements:
- PyTorch, Open3D, Matplotlib, Scikit-learn, SciPy, 
- SAM and DINOv2 models (weights should be downloaded separately).
"""

# --- CLUSTER FIXES (L40S / CUDA) ---
os.environ["SPCONV_ALGO_FILTER"] = "MaskImplicitGemm"
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
from core.utonia_backbone import UtoniaBackbone

# --- 1. DATA & SAM TOOLS ---

def load_replica_data(rgb_p, dep_p):
    """Load and scale RGB-D from Replica."""
    rgb = plt.imread(rgb_p)
    depth = cv2.imread(dep_p, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    return rgb, depth

def get_sam_mask(rgb, ckpt, point=(800, 500)):
    """Segment object with SAM."""
    sam = sam_model_registry["vit_h"](checkpoint=ckpt).to(DEVICE)
    predictor = SamPredictor(sam)
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(
        point_coords=np.array([point]), 
        point_labels=np.array([1]), 
        multimask_output=True
    )
    return masks[np.argmax(scores)]

# --- 2. FEATURE EXTRACTION ---

def get_dino_features(rgb_image):
    print("🦖 Extracting DINOv2 Features...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE).eval()
    
    img_t = torch.from_numpy(rgb_image).permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 255.0
    h, w = img_t.shape[2:]
    nh, nw = (h // 14) * 14, (w // 14) * 14
    img_t = F.interpolate(img_t, (nh, nw), mode='bilinear', align_corners=False)

    with torch.no_grad():
        feat = model.forward_features(img_t)["x_norm_patchtokens"]
    
    feat = feat.reshape(1, nh//14, nw//14, -1).permute(0, 3, 1, 2)
    return F.interpolate(feat, (h, w), mode='bilinear').squeeze(0).permute(1,2,0).cpu().numpy()

def get_utonia_features(pc_tensor, ckpt_path):
    print("🧠 Running Utonia Inference...")
    backbone = UtoniaBackbone(use_mock=False).to(DEVICE).eval()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    backbone.load_state_dict({k.replace('backbone.', ''): v for k, v in sd.items() if k.startswith('backbone.')}, strict=False)

    with torch.no_grad():
        _, (sparse_coords, point_feats) = backbone(pc_tensor)
    
    # Map back to query points
    s_coords = sparse_coords.detach().cpu().numpy()[:, -3:]
    s_feats = point_feats.detach().cpu().numpy().squeeze()
    tree = KDTree(s_coords)
    _, indices = tree.query(pc_tensor.squeeze().cpu().numpy())
    return s_feats[indices]

# --- 3. FUSION & VISUALIZATION ---

def fuse_features(f1, f2):
    """L2-Normalization and Concatenation."""
    f1_n = f1 / (np.linalg.norm(f1, axis=1, keepdims=True) + 1e-8)
    f2_n = f2 / (np.linalg.norm(f2, axis=1, keepdims=True) + 1e-8)
    return np.hstack([f1_n, f2_n])

def plot_multimodal_results(points, utonia, dino, fused, save_path="multimodal_fusion_pca.png"):
    print("🎨 Generating Triple PCA Plot (Standardized View)...")
    
    def to_rgb(feat):
        pca = PCA(n_components=3).fit_transform(feat)
        # Force a consistent scale for colors
        return (pca - pca.min(0)) / (pca.max(0) - pca.min(0) + 1e-8)

    fig = plt.figure(figsize=(24, 9))
    titles = ["1. Geometry (Utonia)", "2. 2D Semantics (DINOv2)", "3. Features Fusion (multi-modal)"]
    data = [utonia, dino, fused]

    for i, feat in enumerate(data):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Rotate the points slightly for better "human-readable" visualization
        # In Replica, the Y-axis is often 'up', but in many 3D plots, Z is 'up'
        # Adjusting the scatter for clarity:
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=to_rgb(feat), s=5, alpha=0.8)
        
        ax.set_title(titles[i], fontsize=20, pad=40, fontweight='bold')
        
        # Fix the "Cutoff" headers by adding extra space
        ax.set_box_aspect([1,1,1]) # Equal aspect ratio
        
        # Coordinate Labels
        ax.set_xlabel('X (m)', labelpad=10)
        ax.set_ylabel('Y (m)', labelpad=10)
        ax.set_zlabel('Z (m)', labelpad=10)
        
        # --- THE FIX FOR ROTATION ---
        # elevation=15 (looking slightly down), azimuth=-60 (slight side angle)
        ax.view_init(elev=15, azim=-60)
        
        # Tighten limits to the chair's bounding box to stop it from looking "lost"
        ax.set_xlim(points[:,0].min()-0.1, points[:,0].max()+0.1)
        ax.set_ylim(points[:,1].min()-0.1, points[:,1].max()+0.1)
        ax.set_zlim(points[:,2].min()-0.1, points[:,2].max()+0.1)

    plt.tight_layout(rect=[0, 0, 1, 0.93]) 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✨ SUCCESS: Standardized plot saved to {save_path}")

# --- 4. MAIN ---

if __name__ == "__main__":
    # Paths
    PATHS = {
        "rgb": "data/Replica/room_0/imap/00/rgb/rgb_0.png",
        "dep": "data/Replica/room_0/imap/00/depth/depth_0.png",
        "sam": "checkpoints/weights/sam_vit_h_4b8939.pth",
        "uto": "checkpoints/utoniadreamer/latest.pth"
    }

    # Pipeline Flow
    rgb, depth = load_replica_data(PATHS["rgb"], PATHS["dep"])
    mask = get_sam_mask(rgb, PATHS["sam"])
    dino_map = get_dino_features(rgb)
    
    # Lifting & Voxelization
    h, w = depth.shape
    v, u = np.indices((h, w))
    z = depth[mask]
    fx, fy, cx, cy = 600.0, 600.0, 599.5, 339.5
    dense_xyz = np.stack([(u[mask]-cx)*z/fx, (v[mask]-cy)*z/fy, z], axis=1)

    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dense_xyz)).voxel_down_sample(0.01)
    pts = np.asarray(pcd.points).astype(np.float32)
    pts -= pts.mean(0)
    pts /= (pts.max() + 1e-8)
    
    # Sync & Sample 8192
    idx = np.random.choice(len(pts), 8192, replace=(len(pts) < 8192))
    final_pts = pts[idx]
    _, d_idx = KDTree(dense_xyz).query(final_pts)
    
    # Inference & Fusion
    u_feats = get_utonia_features(torch.from_numpy(final_pts).unsqueeze(0).to(DEVICE), PATHS["uto"])
    d_feats = dino_map[mask][d_idx]
    f_feats = fuse_features(u_feats, d_feats)

    # Result
    plot_multimodal_results(final_pts, u_feats, d_feats, f_feats)