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

# --- 0. INTRINSICS of DATASETS ---

def get_intrinsics(path):
    """Returns intrinsics based on the dataset detected in the path."""
    if "Replica" in path:
        return {"fx": 600.0, "fy": 600.0, "cx": 599.5, "cy": 339.5}
    elif "ScanNetPP" in path:
        # Standard ScanNet++ iPhone intrinsics (roughly)
        # Note: Actual Scannet++ often provides these in a json file per frame!
        return {"fx": 1050.0, "fy": 1050.0, "cx": 960.0, "cy": 720.0}
    else:
        print("⚠️ Unknown dataset, using Replica defaults.")
        return {"fx": 600.0, "fy": 600.0, "cx": 599.5, "cy": 339.5}

# --- 1. DATA & SAM TOOLS ---

def load_data(rgb_p, dep_p):
    """Loads RGB and Depth, forcing them into the same pixel grid."""
    # Use cv2 for both to ensure consistent coordinate handling
    rgb = cv2.imread(rgb_p)
    if rgb is None: raise FileNotFoundError(f"Could not load RGB at {rgb_p}")
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    
    depth_raw = cv2.imread(dep_p, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if depth_raw is None: raise FileNotFoundError(f"Could not load Depth at {dep_p}")
    
    # Scale depth (standard for ScanNet++ is /1000.0 for meters)
    depth = depth_raw / 1000.0
    
    # --- THE CRITICAL FIX ---
    rgb_h, rgb_w = rgb.shape[:2]
    dep_h, dep_w = depth.shape[:2]
    
    if (rgb_h, rgb_w) != (dep_h, dep_w):
        print(f"🔄 Resizing Depth ({dep_h}, {dep_w}) -> RGB ({rgb_h}, {rgb_w})")
        # OpenCV resize expects (Width, Height)
        depth = cv2.resize(depth, (rgb_w, rgb_h), interpolation=cv2.INTER_NEAREST)
        
    return rgb, depth
    

def get_sam_mask(rgb, ckpt, point=(500, 400)):
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
    """This function performs L2-Normalization and Concatenation
    to fuse Utonia and DINO features into a single multi-modal representation."""
    f1_n = f1 / (np.linalg.norm(f1, axis=1, keepdims=True) + 1e-8)
    f2_n = f2 / (np.linalg.norm(f2, axis=1, keepdims=True) + 1e-8)
    return np.hstack([f1_n, f2_n])

def plot_multimodal_results(points, utonia, dino, fused, save_path="multimodal_fusion_pca_scannet.png"):
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
        
        ax.set_title(titles[i], fontsize=20, pad=40)
        
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
    # --- TOGGLE DATASET PATHS HERE (Replica vs ScanNet) ---
    # Paths for Replica (adjust if using ScanNet or other datasets)
    """PATHS = {
    "rgb": "data/Replica/room_0/imap/00/rgb/rgb_0.png",
    "dep": "data/Replica/room_0/imap/00/depth/depth_0.png",
    "sam": "checkpoints/weights/sam_vit_h_4b8939.pth",
    "uto": "checkpoints/utoniadreamer/latest.pth"
    }"""

    # Paths for ScanNet (uncomment if testing on ScanNet)
    PATHS = {
    "rgb": "data/ScanNetPP/30966f4c6e/iphone/rgb/frame_000000.jpg",
    "dep": "data/ScanNetPP/30966f4c6e/iphone/depth/frame_000000.png",
    "sam": "checkpoints/weights/sam_vit_h_4b8939.pth",
    "uto": "checkpoints/utoniadreamer/latest.pth"
    }

    # 2. Pipeline Flow
    rgb, depth = load_data(PATHS["rgb"], PATHS["dep"])
    
    # 3. Get Intrinsics dynamically
    K = get_intrinsics(PATHS["rgb"])
    
    # 4. SAM Masking
    mask = get_sam_mask(rgb, PATHS["sam"])
    
    # 5. DINO Features
    dino_map = get_dino_features(rgb)
    
    # 6. Lifting & Voxelization
    # Double check shapes here to prevent the IndexError
    print(f"DEBUG: Depth shape {depth.shape}, Mask shape {mask.shape}")
    
    # Use the mask to extract valid pixels
    z = depth[mask]
    
    h, w = depth.shape
    v, u = np.indices((h, w))
    u_m, v_m = u[mask], v[mask]
    
    # Pinhole Projection
    x = (u_m - K['cx']) * z / K['fx']
    y = (v_m - K['cy']) * z / K['fy']
    dense_xyz = np.stack([x, y, z], axis=1)

    # 7. Geometry Cleanup
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dense_xyz))
    pcd = pcd.voxel_down_sample(voxel_size=0.01)
    pts = np.asarray(pcd.points).astype(np.float32)
    
    # Normalize for Utonia
    pts -= pts.mean(0)
    pts /= (np.max(np.linalg.norm(pts, axis=1)) + 1e-8)
    
    # 8. Sampling & Feature Sync
    num_pts = 8192
    if len(pts) >= num_pts:
        idx = np.random.choice(len(pts), num_pts, replace=False)
    else:
        idx = np.random.choice(len(pts), num_pts, replace=True)
    
    final_pts = pts[idx]
    
    # Sync DINO features (using KDTree on the masked dense points)
    # dense_dino contains features for every pixel where the mask was True
    dense_dino = dino_map[mask] 
    tree = KDTree(dense_xyz)
    _, d_idx = tree.query(final_pts)
    d_feats = dense_dino[d_idx]

    # 9. Utonia Inference
    u_input = torch.from_numpy(final_pts).unsqueeze(0).to(DEVICE)
    u_feats = get_utonia_features(u_input, PATHS["uto"])
    
    # 10. Fusion & Plot
    f_feats = fuse_features(u_feats, d_feats)
    plot_multimodal_results(final_pts, u_feats, d_feats, f_feats)