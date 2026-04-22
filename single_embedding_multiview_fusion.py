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
    backbone.load_state_dict(
        {k.replace('backbone.', ''): v for k, v in sd.items() if k.startswith('backbone.')},
        strict=False
    )
    return backbone

# --- Data Loading Helpers ---

def load_data(rgb_path, dep_path):
    rgb = cv2.imread(rgb_path)
    rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
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

# --- NEW: Bilinear Sub-pixel Sampling for DINO ---
def get_dino_features_bilinear(model, rgb_image, mask):
    """
    Replaces the blocky resize/indexing with smooth bilinear grid sampling.
    """
    h, w = rgb_image.shape[:2]
    img_t = torch.from_numpy(rgb_image).permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 255.0
    
    # DINO needs dimensions divisible by 14
    nh, nw = (h // 14) * 14, (w // 14) * 14
    img_t = F.interpolate(img_t, (nh, nw), mode='bilinear', align_corners=False)

    with torch.no_grad():
        feat = model.forward_features(img_t)["x_norm_patchtokens"]
        c = feat.shape[-1]
        # Reshape to spatial grid: (1, Channels, Height, Width)
        feat = feat.reshape(1, nh//14, nw//14, c).permute(0, 3, 1, 2)
    
    # Find the specific pixel coordinates from our mask
    v, u = np.where(mask)
    if len(v) == 0:
        return np.zeros((0, c))

    # Normalize coordinates to [-1, 1] range required by grid_sample
    u_norm = (2.0 * u / (mask.shape[1] - 1)) - 1.0
    v_norm = (2.0 * v / (mask.shape[0] - 1)) - 1.0
    
    grid = torch.stack([torch.tensor(u_norm), torch.tensor(v_norm)], dim=-1)
    grid = grid.view(1, 1, -1, 2).float().to(DEVICE) # Shape: (1, 1, N, 2)
    
    # Sample the feature map exactly at these coordinates
    sampled_feats = F.grid_sample(feat, grid, mode='bilinear', align_corners=True)
    return sampled_feats.squeeze().T.cpu().numpy() # Returns (N, 384)

def get_utonia_features(backbone, pc_tensor):
    with torch.no_grad():
        _, (sparse_coords, point_feats) = backbone(pc_tensor)
    
    s_coords = sparse_coords.detach().cpu().numpy()[:, -3:]
    s_feats = point_feats.detach().cpu().numpy().squeeze()
    
    tree = KDTree(s_coords)
    _, indices = tree.query(pc_tensor.squeeze().cpu().numpy())
    return s_feats[indices]

# --- NEW: Global Aggregator ---
# [MODIFIED] Changed to accept fused features directly to ensure proper 1024 dimension
def create_single_embedding(fused_feats):
    """
    Compresses the point-wise multimodal features into one global vector.
    """
    print("📦 Compressing into a Single Global Embedding...")
    features_t = torch.from_numpy(fused_feats).float().to(DEVICE)
    
    # [MODIFIED] Added keepdim=True so it outputs shape (1, D) instead of (D,)
    global_embedding = torch.max(features_t, dim=0, keepdim=True)[0]
    return global_embedding.cpu().numpy()

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

def voxel_fusion(points, features, voxel_size=0.015):
    print(f"🧊 Voxelizing {len(points)} points...")
    coords = np.floor(points / voxel_size).astype(np.int32)
    
    _, inverse_indices, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
    
    fused_pts = np.zeros((len(counts), 3))
    np.add.at(fused_pts, inverse_indices, points)
    fused_pts /= counts[:, None]
    
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

        full_res_mask = get_sam_mask(sam_predictor, rgb, prompt_points)

        # --- NEW: VISUAL VERIFICATION ---
        # This saves a "verification" image for every frame in the indices
        verify_path = f"verify_frames/verify_{frame_key}.jpg"
        save_verification_image(rgb, full_res_mask, prompt_points, verify_path)
        print(f"  📸 Saved target verification to: {verify_path}")
        
        mask_depth_res = cv2.resize(
            full_res_mask.astype(np.uint8), 
            (depth.shape[1], depth.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)

        pose, K = get_frame_info(meta, frame_key)
        
        K_scaled = K.copy()
        scale_x = depth.shape[1] / rgb.shape[1]
        scale_y = depth.shape[0] / rgb.shape[0]
        K_scaled['fx'] *= scale_x
        K_scaled['fy'] *= scale_y
        K_scaled['cx'] *= scale_x
        K_scaled['cy'] *= scale_y

        pts_world = lift_to_world(depth, mask_depth_res, K_scaled, pose)

        # --- NEW: Swap to Bilinear DINO Sampling ---
        dino_feats = get_dino_features_bilinear(dino_model, rgb, mask_depth_res)

        print(f"Frame {idx}: DINO Mean={dino_feats.mean():.4f}, Max={dino_feats.max():.4f}")

        all_points.append(pts_world)
        all_dino.append(dino_feats)

        print(f"  ✅ {frame_key} processed ({len(pts_world)} pts)")

    return np.vstack(all_points), np.vstack(all_dino)

# [MODIFIED] Added `num_pts` parameter and defaulted to 16384 (16k)
def run_utonia_on_fused(backbone, points, num_pts=16384):
    print(f"🧠 Running Utonia on {num_pts} sampled geometry points...")
    pts = points.copy()

    pts -= pts.mean(0)
    pts /= (np.max(np.linalg.norm(pts, axis=1)) + 1e-8)

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
        # --- IMPROVED SCATTER PARAMETERS ---
        ax.scatter(
            points[:, 0], 
            points[:, 1], 
            points[:, 2], 
            c=colors, 
            s=15,          # INCREASED: Helps fill gaps in sparse scans
            alpha=0.9,     # OPAQUE: Makes the PCA colors pop
            edgecolor='none', # REMOVE EDGES: Prevents the "black mesh" look
            marker='o'     # SMOOTH: Circular points blend better than squares
        )
        # --- CLEANER BACKGROUND (MODERN AI LOOK) ---
        # Remove the gray "panes" so the colorful points are the focus
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        
        # Make the grid lines very subtle
        ax.grid(True, linestyle='--', alpha=0.2)

        ax.set_title(titles[i], fontsize=20, pad=40)
        
        # Revert to your original 'Auto-Fit' style
        ax.set_box_aspect(None)

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


# --- NEW: Single Embedding Visualization ---
# This function creates a "barcode" style visualization of the single global embedding vector.
# It reshapes the 1D embedding into a 2D grid and applies a colormap to show the distribution of values.
# This can help us visually inspect the global embedding and identify any patterns or salient features it captures.
# [MODIFIED] Updated to handle non-square dimensions like 1408
def save_barcode(embedding, save_path="desk_identity_barcode.png"):
    """
    Visualizes the 1408-d global embedding as a rectangular fingerprint.
    """
    emb_flat = embedding.flatten()
    print(f"📊 Creating barcode for {len(emb_flat)} features...")

    # Normalize for visual contrast
    emb_norm = (emb_flat - emb_flat.mean()) / (emb_flat.std() + 1e-8)
    
    # 1408 features factor perfectly into 32 x 44
    # This ensures no data is truncated.
    try:
        grid = emb_norm.reshape(32, 44)
    except ValueError:
        # Fallback if dimensions change again
        side = int(np.sqrt(len(emb_norm)))
        grid = emb_norm[:side*side].reshape(side, side)

    plt.figure(figsize=(12, 8))
    plt.imshow(grid, cmap='magma', aspect='auto', interpolation='nearest')
    plt.title(f"Global Object Fingerprint (Dim: {len(emb_flat)})", fontsize=16)
    plt.colorbar(label="Activation Strength")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📁 Global Embedding Barcode saved to: {save_path}")


# NEW: This function creates a visualization of the RGB image with the SAM mask overlayed and the prompt point marked.
def save_verification_image(rgb, mask, prompt_points, save_path):
    """
    Saves an RGB image with the SAM mask overlayed and the prompt point marked.
    """
    # 1. Create a copy to avoid modifying the original data
    # Convert RGB to BGR for OpenCV saving
    vis_img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    
    # 2. Create the mask overlay (Neon Green)
    overlay = vis_img.copy()
    overlay[mask] = [0, 255, 0]  # BGR green
    
    # 3. Blend the original and the green overlay
    alpha = 0.35  # Transparency factor
    cv2.addWeighted(overlay, alpha, vis_img, 1 - alpha, 0, vis_img)
    
    # 4. Draw a white outline around the mask for clarity
    contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis_img, contours, -1, (255, 255, 255), 2)
    
    # 5. Draw the prompt point (Red Dot) so you know where SAM started
    for pt in prompt_points:
        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)
        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 12, (255, 255, 255), 2)

    cv2.imwrite(save_path, vis_img)

if __name__ == "__main__":
    # 1. Setup Models
    sam_predictor = init_sam(PATHS["sam"])
    dino_model = init_dino()
    utonia_model = init_utonia(PATHS["uto"])

    # 2. Extract Data from multi-view
    frame_indices = [0, 10, 20, 30, 40]
    prompt_points = [[470, 370]]

    points, dino = build_global_object(frame_indices, PATHS, sam_predictor, dino_model, prompt_points)
    f_points, f_dino = voxel_fusion(points, dino, voxel_size=0.02)
    
    # 3. SAMPLE 16K POINTS (Geometry)
    # [ACTION] Plotting the 16k point cloud results
    s_pts, u_feats, sampled_indices = run_utonia_on_fused(utonia_model, f_points, num_pts=16384)
    
    d_feats = f_dino[sampled_indices]
    real_world_pts = f_points[sampled_indices]
    
    # 4. FUSE & PLOT 16K PCA (The "Visual" part)
    fused_pointwise = fuse_features(u_feats, d_feats)
    
    print("\n--- Plotting Step 1: 16k Point Cloud Analysis ---")
    plot_multimodal_results(
        real_world_pts, 
        u_feats, 
        d_feats, 
        fused_pointwise, 
        save_path="desk_16k_multimodal_pca_2.png"
    )

    # 5. CREATE & PLOT GLOBAL EMBEDDING (The "Identity" part)
    # [ACTION] Condensing 16k points into 1 single vector
    global_embedding = create_single_embedding(fused_pointwise)
    
    # --- NEW: Visualize the Single Global Embedding ---
    print("\n--- Plotting Step 2: Global Identity Barcode ---")
    save_barcode(global_embedding, "desk_global_identity_1408_2.png")
    
    print("=========================================")
    print("✅ Pipeline Complete.")
    print(f"🔹 Final Point Cloud Size: {len(real_world_pts)} points")
    print(f"🔹 Single Object Embedding Shape: {global_embedding.shape}")
    print("=========================================")