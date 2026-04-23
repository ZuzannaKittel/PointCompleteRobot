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
def save_barcode(embedding, save_path):
    emb_flat = embedding.flatten()
    emb_norm = (emb_flat - emb_flat.mean()) / (emb_flat.std() + 1e-8)
    grid = emb_norm.reshape(32, 44) if len(emb_flat) == 1408 else emb_norm[:32*32].reshape(32,32)
    plt.imshow(grid, cmap='magma', aspect='auto')
    plt.axis('off')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def save_verification_image(rgb, mask, prompt_points, save_path):
    vis_img = cv2.cvtColor(rgb.copy(), cv2.COLOR_RGB2BGR)
    overlay = vis_img.copy()
    overlay[mask] = [0, 255, 0]
    cv2.addWeighted(overlay, 0.35, vis_img, 0.65, 0, vis_img)
    for pt in prompt_points:
        cv2.circle(vis_img, (int(pt[0]), int(pt[1])), 10, (0, 0, 255), -1)
    cv2.imwrite(save_path, vis_img)