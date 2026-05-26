import json
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

if not hasattr(np, 'product'):
    np.product = np.prod

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
    """
    Returns the Camera-to-World (c2w) pose and the raw intrinsic components.
    """
    entry = meta[frame_key]
    pose = np.array(entry["aligned_pose"], dtype=np.float32)
    K_mat = np.array(entry["intrinsic"], dtype=np.float32)
    return pose, {"fx": K_mat[0, 0], "fy": K_mat[1, 1], "cx": K_mat[0, 2], "cy": K_mat[1, 2]}

# --- VISION & FEATURES ---
def get_dino_features_bilinear(model, rgb_image, uv_coords):
    """
    Bilinearly samples DINOv2 patch features at continuous sub-pixel coordinates.
    """
    h, w = rgb_image.shape[:2]
    img_t = torch.from_numpy(rgb_image).permute(2,0,1).unsqueeze(0).float().to(DEVICE) / 255.0
    nh, nw = (h // 14) * 14, (w // 14) * 14
    img_t = F.interpolate(img_t, (nh, nw), mode='bilinear', align_corners=False)

    with torch.no_grad():
        feat = model.forward_features(img_t)["x_norm_patchtokens"]
        c = feat.shape[-1]
        feat = feat.reshape(1, nh//14, nw//14, c).permute(0, 3, 1, 2)
    
    if len(uv_coords) == 0: 
        return np.zeros((0, c))

    u = uv_coords[:, 0]
    v = uv_coords[:, 1]
    u_norm = (2.0 * u / (w - 1)) - 1.0
    v_norm = (2.0 * v / (h - 1)) - 1.0
    
    grid = torch.stack([torch.tensor(u_norm), torch.tensor(v_norm)], dim=-1).view(1, 1, -1, 2).float().to(DEVICE)
    sampled_feats = F.grid_sample(feat, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
    
    return sampled_feats.squeeze(0).squeeze(1).T.cpu().numpy()

def get_utonia_features(backbone, pc_tensor, return_pointwise=False):
    """
    Extracts geometric features from the Utonia backbone.
    """
    with torch.no_grad():
        outputs = backbone(pc_tensor)
    
    if isinstance(outputs, tuple):
        global_feat, internal_data = outputs
        if isinstance(internal_data, tuple):
            sparse_coords, point_feats = internal_data
        else:
            point_feats = internal_data
            sparse_coords = pc_tensor.squeeze(0).cpu().numpy()
    else:
        global_feat = outputs
        point_feats = outputs
        sparse_coords = pc_tensor.squeeze(0).cpu().numpy()

    if return_pointwise:
        if hasattr(sparse_coords, 'detach'):
            s_coords = sparse_coords.detach().cpu().numpy()[:, -3:]
        else:
            s_coords = sparse_coords[:, -3:] if sparse_coords.shape[-1] == 4 else sparse_coords
            
        s_feats = point_feats.detach().cpu().numpy() if hasattr(point_feats, 'detach') else point_feats
        orig_pts = pc_tensor.squeeze(0).cpu().numpy()
        
        tree = KDTree(s_coords)
        distances, indices = tree.query(orig_pts, k=4)
        
        weights = 1.0 / (distances + 1e-8)
        weights /= np.sum(weights, axis=1, keepdims=True)
        aligned_point_feats = np.sum(s_feats[indices] * weights[:, :, None], axis=1)
        
        return torch.from_numpy(aligned_point_feats).unsqueeze(0).to(pc_tensor.device)
        
    return global_feat

def fuse_features(f1, f2):
    """
    Inline L2-Normalization and Concatenation array builder.
    """
    f1_n = f1 / (np.linalg.norm(f1, axis=1, keepdims=True) + 1e-8)
    f2_n = f2 / (np.linalg.norm(f2, axis=1, keepdims=True) + 1e-8)
    return np.hstack([f1_n, f2_n])

# --- GEOMETRY ---
def lift_to_world(depth, mask, K, c2w):
    """
    Lifts depth maps directly to 3D world coordinates via Camera-to-World (c2w).
    """
    h, w = depth.shape
    v, u = np.indices((h, w))
    z = depth[mask]
    pts_cam = np.stack([(u[mask] - K['cx']) * z / K['fx'], (v[mask] - K['cy']) * z / K['fy'], z], axis=1)
    pts_h = np.hstack([pts_cam, np.ones((pts_cam.shape[0], 1))])
    return (c2w @ pts_h.T).T[:, :3]

def project_world_to_pixel(points_world, w2c, K):
    """
    Projects 3D world coordinates into 2D continuous screen space.
    Accepts explicit World-to-Camera (w2c) matrices.
    """
    if points_world.ndim == 1:
        points_world = points_world[None, :]
        
    pts_h = np.hstack([points_world, np.ones((len(points_world), 1))])
    cam_h = (w2c @ pts_h.T).T
    pts_cam = cam_h[:, :3]
    
    u = (pts_cam[:, 0] * K['fx'] / (pts_cam[:, 2] + 1e-8)) + K['cx']
    v = (pts_cam[:, 1] * K['fy'] / (pts_cam[:, 2] + 1e-8)) + K['cy']
    
    return np.stack([u, v], axis=1)

def voxel_fusion(points, features, voxel_size=0.015):
    coords = np.floor(points / voxel_size).astype(np.int32)
    _, inv, counts = np.unique(coords, axis=0, return_inverse=True, return_counts=True)
    f_pts = np.zeros((len(counts), 3))
    np.add.at(f_pts, inv, points)
    f_feats = np.zeros((len(counts), features.shape[1]))
    np.add.at(f_feats, inv, features)
    return f_pts / counts[:, None], f_feats / counts[:, None]

# --- PLOTTING ---
def plot_multimodal_results(points, utonia, dino, fused, save_path="multimodal_fusion_pca_scannet.png"):
    def to_rgb(feat):
        feat_norm = StandardScaler().fit_transform(feat)
        pca_features = PCA(n_components=3).fit_transform(feat_norm)
        p_low, p_high = np.percentile(pca_features, [5, 95], axis=0)
        rgb = (pca_features - p_low) / (p_high - p_low + 1e-8)
        return np.clip(rgb, 0, 1)

    fig = plt.figure(figsize=(24, 9))
    titles = ["1. Geometry (Utonia)", "2. 2D Semantics (DINOv2)", "3. Features Fusion (multi-modal)"]
    data = [utonia, dino, fused]

    for i, feat in enumerate(data):
        ax = fig.add_subplot(1, 3, i+1, projection='3d')
        colors = to_rgb(feat)
        ax.scatter(points[:, 0], points[:, 1], points[:, 2], c=colors, s=15, alpha=0.9, edgecolor='none', marker='o')
        
        ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
        ax.grid(True, linestyle='--', alpha=0.2)
        ax.set_title(titles[i], fontsize=20, pad=40)
        ax.set_box_aspect(None)
        ax.set_xlabel('X (m)')
        ax.set_ylabel('Y (m)')
        ax.set_zlabel('Z (m)')
        ax.view_init(elev=20, azim=-45)
        
        ax.set_xlim(points[:,0].min(), points[:,0].max())
        ax.set_ylim(points[:,1].min(), points[:,1].max())
        ax.set_zlim(points[:,2].min(), points[:,2].max())

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_barcode(embedding, save_path="desk_identity_barcode.png"):
    emb_flat = embedding.flatten()
    emb_norm = (emb_flat - emb_flat.mean()) / (emb_flat.std() + 1e-8)
    
    try:
        grid = emb_norm.reshape(32, 44)
    except ValueError:
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

def save_voxel_density_map(tensor, filename, title="Voxel Density Projection"):
    if hasattr(tensor, 'detach'):
        tensor = tensor.detach().cpu()
    intensity = tensor[0].abs().sum(dim=0).numpy()
    top_down = np.max(intensity, axis=0) 
    
    plt.figure(figsize=(8, 6))
    im = plt.imshow(top_down, cmap='magma', origin='lower')
    plt.colorbar(im, label='Aggregate Feature Intensity')
    plt.title(title)
    plt.axis('off') 
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()