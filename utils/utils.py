import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- I/O & METADATA ---
def load_data(rgb_path, dep_path):
    rgb = cv2.cvtColor(cv2.imread(rgb_path), cv2.COLOR_BGR2RGB)
    depth = cv2.imread(dep_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 1000.0
    return rgb, depth

def load_metadata(json_path):
    with open(json_path, "r") as f:
        return json.load(f)

def get_frame_info(meta, frame_key):
    entry = meta[frame_key]
    pose = np.array(entry["aligned_pose"], dtype=np.float32)
    K_mat = np.array(entry["intrinsic"], dtype=np.float32)
    return pose, {"fx": K_mat[0, 0], "fy": K_mat[1, 1], "cx": K_mat[0, 2], "cy": K_mat[1, 2]}

# --- VISION & FEATURES ---
def get_sam_mask(predictor, rgb, points):
    predictor.set_image(rgb)
    masks, scores, _ = predictor.predict(np.array(points), np.ones(len(points)), multimask_output=True)
    return masks[np.argmax(scores)]

def get_dino_features_bilinear(model, rgb_image, mask):
    h, w = rgb_image.shape[:2]
    img_t = torch.from_numpy(rgb_image).permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 255.0
    nh, nw = (h // 14) * 14, (w // 14) * 14
    img_t = F.interpolate(img_t, (nh, nw), mode='bilinear', align_corners=False)

    with torch.no_grad():
        feat = model.forward_features(img_t)["x_norm_patchtokens"]
        c = feat.shape[-1]
        feat = feat.reshape(1, nh//14, nw//14, c).permute(0, 3, 1, 2)
    
    v, u = np.where(mask)
    if len(v) == 0: return np.zeros((0, c))

    u_norm, v_norm = (2.0 * u / (mask.shape[1] - 1)) - 1.0, (2.0 * v / (mask.shape[0] - 1)) - 1.0
    grid = torch.stack([torch.tensor(u_norm), torch.tensor(v_norm)], dim=-1).view(1, 1, -1, 2).float().to(DEVICE)
    return F.grid_sample(feat, grid, mode='bilinear', align_corners=True).squeeze().T.cpu().numpy()

def get_utonia_features(backbone, pc_tensor, return_pointwise=False):
    """
    Extracts geometric features from the Utonia backbone.
    If return_pointwise=True, it returns the (1, N, 1024) features.
    """
    with torch.no_grad():
        global_feat, (sparse_coords, point_feats) = backbone(pc_tensor)
    
    if return_pointwise:
        # We need to map the output features back to the EXACT input point order.
        # Since PointTransformer might subsample or shuffle via voxelization internally, 
        # we use KDTree to map the output features back to the original pc_tensor order.
        
        # Ensure we are working with CPU numpy arrays for the KDTree
        s_coords = sparse_coords.detach().cpu().numpy()[:, -3:]
        s_feats = point_feats.detach().cpu().numpy()
        
        # Original input points
        orig_pts = pc_tensor.squeeze(0).cpu().numpy()
        
        # Map Utonia's output points to our input points
        tree = KDTree(s_coords)
        _, indices = tree.query(orig_pts)
        
        # Get the pointwise features in the correct order
        aligned_point_feats = s_feats[indices]
        
        # Return as a tensor with batch dimension (1, N, 1024) to match expectations
        return torch.from_numpy(aligned_point_feats).unsqueeze(0).to(pc_tensor.device)
        
    # If not pointwise, return the global feature
    return global_feat

def fuse_features(f1, f2):
    """This function performs L2-Normalization and Concatenation
    to fuse Utonia and DINO features into a single multi-modal representation."""
    f1_n = f1 / (np.linalg.norm(f1, axis=1, keepdims=True) + 1e-8)
    f2_n = f2 / (np.linalg.norm(f2, axis=1, keepdims=True) + 1e-8)
    return np.hstack([f1_n, f2_n])

# --- GEOMETRY ---
def lift_to_world(depth, mask, K, pose):
    h, w = depth.shape
    v, u = np.indices((h, w))
    z = depth[mask]
    pts_cam = np.stack([(u[mask] - K['cx']) * z / K['fx'], (v[mask] - K['cy']) * z / K['fy'], z], axis=1)
    pts_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
    return (pose @ pts_h.T).T[:, :3]

def project_world_to_pixel(point_world, pose, K):
    cam_h = np.linalg.inv(pose) @ np.append(point_world, 1.0)
    pts_cam = cam_h[:3]
    u = (pts_cam[0] * K['fx'] / pts_cam[2]) + K['cx']
    v = (pts_cam[1] * K['fy'] / pts_cam[2]) + K['cy']
    return np.array([[u, v]])

def voxel_fusion(points, features, voxel_size=0.015):
    coords = np.floor(points / voxel_size).astype(np.int32)
    _, inv, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
    f_pts = np.zeros((len(counts), 3))
    np.add.at(f_pts, inv, points)
    f_feats = np.zeros((len(counts), features.shape[1]))
    np.add.at(f_feats, inv, features)
    return f_pts / counts[:, None], f_feats / counts[:, None]

# --- PLOTTING ---
# Basic 3D scatter with PCA-based coloring for better visualization of multi-modal features
# This function standardizes the features before PCA to ensure that both Utonia and DINO contributions are visible in the color space.
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

# --- Single Embedding Visualization ---
# This function creates a "barcode" style visualization of the single global embedding vector.
# It reshapes the 1D embedding into a 2D grid and applies a colormap to show the distribution of values.
# This can help us visually inspect the global embedding and identify any patterns or salient features it captures.
# Updated to handle non-square dimensions like 1408
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

def save_verification_image(rgb, mask, prompt_points, save_path):
    vis_img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    overlay = vis_img.copy()
    overlay[mask] = [0, 255, 0]
    cv2.addWeighted(overlay, 0.35, vis_img, 0.65, 0, vis_img)
    for pt in prompt_points:
        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)
    cv2.imwrite(save_path, vis_img)