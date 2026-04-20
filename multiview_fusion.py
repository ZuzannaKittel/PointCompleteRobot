import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree
from segment_anything import sam_model_registry, SamPredictor
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# These are your custom modules - ensure they are in your working directory
try:
    from core.utonia_backbone import UtoniaBackbone
    from utonia_dino_fusion import fuse_features
except ImportError:
    print("⚠️ Warning: Custom modules 'core' or 'utonia_dino_fusion' not found.")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

PATHS = {
    "sam": "checkpoints/weights/sam_vit_h_4b8939.pth",
    "uto": "checkpoints/utoniadreamer/latest.pth",
    "json": "data/ScanNetPP/30966f4c6e/iphone/pose_intrinsic_imu.json"
}

# --- Initialization Functions ---

def init_sam(ckpt):
    print("🎯 Initializing SAM...")
    sam = sam_model_registry["vit_h"](checkpoint=ckpt).to(DEVICE)
    return SamPredictor(sam)

def init_dino():
    print("🦖 Initializing DINOv2...")
    model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
    return model.to(DEVICE).eval()

def init_utonia(ckpt_path):
    print("🧠 Initializing Utonia...")
    backbone = UtoniaBackbone(use_mock=False).to(DEVICE).eval()
    ckpt = torch.load(ckpt_path, map_location=DEVICE, weights_only=True)
    sd = ckpt['model'] if 'model' in ckpt else ckpt
    # Clean up state dict keys if they have the 'backbone.' prefix
    backbone.load_state_dict(
        {k.replace('backbone.', ''): v for k, v in sd.items() if k.startswith('backbone.')},
        strict=False
    )
    return backbone

# --- Data Loading Helpers ---

def load_data(rgb_path, dep_path):
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
    # Load depth (assuming mm scale, converting to meters)
    depth = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    return rgb, depth

def load_metadata(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def get_frame_info(meta, frame_key):
    entry = meta[frame_key]
    pose = np.array(entry["aligned_pose"], dtype=np.float32)
    K_mat = np.array(entry["intrinsic"], dtype=np.float32)
    K = {
        "fx": K_mat[0, 0], "fy": K_mat[1, 1],
        "cx": K_mat[0, 2], "cy": K_mat[1, 2],
    }
    return pose, K

# --- Feature Extraction Functions ---

def get_sam_mask(predictor, rgb, points):
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(
        point_coords=np.array(points),
        point_labels=np.ones(len(points)),
        multimask_output=True
    )
    return masks[np.argmax(scores)]

def get_dino_features(model, rgb_image):
    h, w = rgb_image.shape[:2]
    img_t = torch.from_numpy(rgb_image).permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 255.0
    
    # DINOv2 needs dimensions divisible by 14
    nh, nw = (h // 14) * 14, (w // 14) * 14
    img_t = F.interpolate(img_t, (nh, nw), mode='bilinear', align_corners=False)

    with torch.no_grad():
        feat = model.forward_features(img_t)["x_norm_patchtokens"]
    
    # Reshape and upscale back to original resolution
    feat = feat.reshape(1, nh//14, nw//14, -1).permute(0, 3, 1, 2)
    feat_upsampled = F.interpolate(feat, (h, w), mode='bilinear').squeeze(0)
    return feat_upsampled.permute(1,2,0).cpu().numpy()

def get_utonia_features(backbone, pc_tensor):
    with torch.no_grad():
        _, (sparse_coords, point_feats) = backbone(pc_tensor)
    
    s_coords = sparse_coords.detach().cpu().numpy()[:, -3:]
    s_feats = point_feats.detach().cpu().numpy().squeeze()
    
    # Map sparse features to the query points using KDTree
    tree = KDTree(s_coords)
    _, indices = tree.query(pc_tensor.squeeze().cpu().numpy())
    return s_feats[indices]

# --- 3D Processing ---

def lift_to_world(depth, mask, K, pose):
    h, w = depth.shape
    v, u = np.indices((h, w))

    z = depth[mask]
    u_m, v_m = u[mask], v[mask]

    x = (u_m - K['cx']) * z / K['fx']
    y = (v_m - K['cy']) * z / K['fy']

    pts_cam = np.stack([x, y, z], axis=1)
    pts_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
    pts_world = (pose @ pts_h.T).T[:, :3]

    return pts_world

def voxel_fusion(points, features, voxel_size=0.02):
    """Optimized NumPy-based voxelization."""
    print(f"🧊 Voxelizing {len(points)} points...")
    coords = np.floor(points / voxel_size).astype(np.int32)
    
    # Find unique voxels and their mapping
    _, inverse_indices, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
    
    # Compute mean position per voxel
    fused_pts = np.zeros((len(counts), 3))
    np.add.at(fused_pts, inverse_indices, points)
    fused_pts /= counts[:, None]
    
    # Compute mean features per voxel
    fused_feats = np.zeros((len(counts), features.shape[1]))
    np.add.at(fused_feats, inverse_indices, features)
    fused_feats /= counts[:, None]
    
    print(f"  🔹 Reduced to {len(fused_pts)} voxels")
    return fused_pts, fused_feats

# --- Main Pipeline Logic ---

def build_global_object(frame_indices, paths, sam_predictor, dino_model, prompt_points):
    print("🌍 Phase 1: Multi-frame extraction and lifting...")
    meta = load_metadata(paths["json"])

    all_points = []
    all_dino = []

    for idx in frame_indices:
        frame_key = f"frame_{idx:06d}"
        rgb_path = f"data/ScanNetPP/30966f4c6e/iphone/rgb/{frame_key}.jpg"
        dep_path = f"data/ScanNetPP/30966f4c6e/iphone/depth/{frame_key}.png"

        try:
            rgb, depth = load_data(rgb_path, dep_path)
        except Exception as e:
            print(f"  ❌ Skipping {frame_key}: {e}")
            continue

        # 1. Segment object at high-res (RGB resolution)
        full_res_mask = get_sam_mask(sam_predictor, rgb, prompt_points)
        
        # DEBUG: Save the mask to see if SAM is actually picking up the desk
        cv2.imwrite(f"debug_mask_{frame_key}.png", (full_res_mask * 255).astype(np.uint8))
        
        # 2. RESIZE MASK to match depth resolution
        # We use INTER_NEAREST because it's a boolean/binary mask
        mask_depth_res = cv2.resize(
            full_res_mask.astype(np.uint8), 
            (depth.shape[1], depth.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        # 3. Lift to 3D using the resized mask
        pose, K = get_frame_info(meta, frame_key)
        
        # IMPORTANT: If your K matrix is for the high-res RGB, 
        # you must scale it to the depth resolution!
        K_scaled = K.copy()
        scale_x = depth.shape[1] / rgb.shape[1]
        scale_y = depth.shape[0] / rgb.shape[0]
        K_scaled['fx'] *= scale_x
        K_scaled['fy'] *= scale_y
        K_scaled['cx'] *= scale_x
        K_scaled['cy'] *= scale_y

        pts_world = lift_to_world(depth, mask_depth_res, K_scaled, pose)

        # 4. Extract DINO features 
        # (DINO features should be sampled at the same resolution as the lifting)
        # CRITICAL: Resize the DINO map to depth resolution BEFORE indexing
        dino_map = get_dino_features(dino_model, rgb)
        dino_map_res = cv2.resize(dino_map, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_LINEAR)
        dino_feats = dino_map_res[mask_depth_res]

        # DEBUG: Check if features are all zeros
        print(f"Frame {idx}: DINO Mean={dino_feats.mean():.4f}, Max={dino_feats.max():.4f}")

        all_points.append(pts_world)
        all_dino.append(dino_feats)

        print(f"  ✅ {frame_key} processed ({len(pts_world)} pts)")

    return np.vstack(all_points), np.vstack(all_dino)

def run_utonia_on_fused(backbone, points):
    print("🧠 Running Utonia on fused geometry...")
    pts = points.copy()

    # Normalization (Crucial for most point-based backbones)
    pts -= pts.mean(0)
    pts /= (np.max(np.linalg.norm(pts, axis=1)) + 1e-8)

    # Sample standard point count for Utonia
    num_pts = 8192
    if len(pts) > num_pts:
        idx = np.random.choice(len(pts), num_pts, replace=False)
    else:
        idx = np.random.choice(len(pts), num_pts, replace=True)
        
    pts_sampled = pts[idx]
    u_input = torch.from_numpy(pts_sampled).float().unsqueeze(0).to(DEVICE)
    u_feats = get_utonia_features(backbone, u_input)

    return pts_sampled, u_feats, idx

def plot_multimodal_results(points, utonia, dino, fused, save_path="multimodal_fusion_pca_scannet.png"):
    print("🎨 Generating Triple PCA Plot (Standardized View)...")
    
    def to_rgb(feat):
        # 1. Standardize (Zero mean, Unit variance)
        feat_norm = StandardScaler().fit_transform(feat)
        
        # 2. PCA
        pca_features = PCA(n_components=3).fit_transform(feat_norm)
        
        # 3. CONTRAST BOOST: Instead of min/max, use tight percentiles
        # This ignores the "floor" features if they are hogging the variance
        p_low, p_high = np.percentile(pca_features, [5, 95], axis=0)
        
        # Scale and Clip
        rgb = (pca_features - p_low) / (p_high - p_low + 1e-8)
        return np.clip(rgb, 0, 1)

    fig = plt.figure(figsize=(24, 9))
    titles = ["1. Geometry (Utonia)", "2. 2D Semantics (DINOv2)", "3. Features Fusion (multi-modal)"]
    data = [utonia, dino, fused]

    for i, feat in enumerate(data):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        
        # Get the colors using robust function
        colors = to_rgb(feat)
        
        # Scatter plot
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=5, alpha=0.8)
        
        ax.set_title(titles[i], fontsize=20, pad=40)
        ax.set_box_aspect([1,1,1]) 
        
        # Labels
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        
        # View angle
        ax.view_init(elev=20, azim=-45)
        
        # Tighten limits
        ax.set_xlim(points[:,0].min(), points[:,0].max())
        ax.set_ylim(points[:,1].min(), points[:,1].max())
        ax.set_zlim(points[:,2].min(), points[:,2].max())

    plt.tight_layout(rect=[0, 0, 1, 0.95]) 
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close() # Good practice to close figure to save memory
    print(f"✨ SUCCESS: Standardized plot saved to {save_path}")

if __name__ == "__main__":
    # 1. Init all models
    sam_predictor = init_sam(PATHS["sam"])
    dino_model = init_dino()
    utonia_model = init_utonia(PATHS["uto"])

    # 2. Parameters
    frame_indices = [0, 10, 20, 30, 40]
    # Center of image for SAM prompt, or provide specific object coordinates
    prompt_points = [[512, 512]] 

    # 3. Build global 3D cloud with 2D semantic features
    points, dino = build_global_object(frame_indices, PATHS, sam_predictor, dino_model, prompt_points)

    # 4. Fuse points from multiple frames into unique voxels
    f_points, f_dino = voxel_fusion(points, dino, voxel_size=0.02)

    # 5. Get geometric features from Utonia
    s_pts, u_feats, sampled_indices = run_utonia_on_fused(utonia_model, f_points)

    # 6. Align DINO features to the exact same sampled points (NO KDTree needed!)
    d_feats = f_dino[sampled_indices]

    # Extract the unnormalized world points for accurate plotting
    real_world_pts = f_points[sampled_indices]

    # 7. Final Fusion and Visualization
    fused = fuse_features(u_feats, d_feats)
    
    # Pass real_world_pts instead of s_pts
    plot_multimodal_results(real_world_pts, u_feats, d_feats, fused, save_path="multiview_fusion_pca_scannet.png")