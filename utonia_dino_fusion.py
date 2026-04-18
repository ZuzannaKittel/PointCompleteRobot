import os
import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import cv2

import open3d as o3d
from segment_anything import sam_model_registry, SamPredictor

# Modern spconv import
# Force spconv to use the more stable 'SpatiallySparseConvolution' algorithm
os.environ["SPCONV_ALGO_FILTER"] = "MaskImplicitGemm"
os.environ["SPCONV_OFFLINE_CONF_PATH"] = "spconv_conf.json"
import spconv.pytorch as spconv 
from core.utonia_backbone import UtoniaBackbone

# --- 1. CONFIGURATION ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
INTRINSICS = {"fx": 600.0, "fy": 600.0, "cx": 599.5, "cy": 339.5}

# --- 2. DINOv2 FEATURE EXTRACTION ---
def get_dino_features(rgb_image):
    print("🦖 Extracting DINOv2 Features...")
    # Using ViT-Small for speed/memory, but ViT-Large is better if you have VRAM
    dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14').to(DEVICE)
    dinov2.eval()

    # Preprocess image (DINO requires shape [B, C, H, W] and floats 0-1)
    img_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE) / 255.0
    h, w = img_tensor.shape[2:]
    
    # DINO needs dimensions to be multiples of 14 (patch size)
    new_h, new_w = (h // 14) * 14, (w // 14) * 14
    img_resized = F.interpolate(img_tensor, (new_h, new_w), mode='bilinear', align_corners=False)

    with torch.no_grad():
        features = dinov2.forward_features(img_resized)["x_norm_patchtokens"]
        
    emb_dim = features.shape[-1]
    features = features.reshape(1, new_h // 14, new_w // 14, emb_dim).permute(0, 3, 1, 2)
    
    # Upsample back to exact original image resolution so it matches the SAM mask
    full_res_features = F.interpolate(features, (h, w), mode='bilinear', align_corners=False)
    
    # Return as [H, W, Channels] on CPU for easy numpy masking
    return full_res_features.squeeze(0).permute(1, 2, 0).cpu().numpy()

# --- 3. 3D PROJECTION & LIFTING ---
def lift_to_3d_with_features(depth_meters, mask, dino_features):
    print("🏗️ Lifting pixels to 3D with Semantic Features...")
    h, w = depth_meters.shape
    v, u = np.indices((h, w))
    
    # Filter by SAM mask
    z = depth_meters[mask]
    u = u[mask]
    v = v[mask]
    
    # Pinhole Math
    x = (u - INTRINSICS['cx']) * z / INTRINSICS['fx']
    y = (v - INTRINSICS['cy']) * z / INTRINSICS['fy']
    
    dense_points = np.stack([x, y, z], axis=1)
    dense_dino_feats = dino_features[mask] # Extract features for these exact pixels
    
    return dense_points, dense_dino_feats

# --- 4. SAFE DOWNSAMPLING (The Trick!) ---
def prepare_for_utonia(dense_points, dense_dino_feats):
    # 1. Clean up input
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(dense_points)
    pcd_down = pcd.voxel_down_sample(voxel_size=0.01) 
    sparse_points = np.asarray(pcd_down.points).astype(np.float32)
    
    # 2. Re-center and Normalize to [-1, 1]
    sparse_points -= np.mean(sparse_points, axis=0)
    max_dist = np.max(np.linalg.norm(sparse_points, axis=1))
    sparse_points /= (max_dist + 1e-8) 

    # 3. CRITICAL: Voxel-Grid Safety
    # Using 128 ensures the sparse grid isn't too large for the L40S kernels
    precision = 128 
    sparse_points = np.unique(np.floor(sparse_points * precision) / precision, axis=0)

    # 4. Force exact point count (8192)
    if len(sparse_points) > 8192:
        idx = np.random.choice(len(sparse_points), 8192, replace=False)
    else:
        # If we have too few points, we duplicate some to keep the tensor shape consistent
        idx = np.random.choice(len(sparse_points), 8192, replace=True)
        
    final_points = sparse_points[idx]
    
    # 5. Sync DINO features back to these unique points
    tree = KDTree(dense_points)
    _, indices = tree.query(final_points)
    final_dino = dense_dino_feats[indices]
    
    # Return as Torch tensor [Batch, Points, XYZ]
    return torch.from_numpy(final_points).float().unsqueeze(0), final_dino

# --- 5. VISUALIZATION ---
def plot_pca_comparison(points, utonia_feats, dino_feats, save_path="multimodal_comparison.png"):
    print("🎨 Running PCA and Plotting...")
    
    # Normalize features for PCA
    utonia_pca = PCA(n_components=3).fit_transform(utonia_feats)
    dino_pca = PCA(n_components=3).fit_transform(dino_feats)
    
    # Normalize colors to [0, 1]
    u_rgb = (utonia_pca - utonia_pca.min(0)) / (utonia_pca.max(0) - utonia_pca.min(0) + 1e-8)
    d_rgb = (dino_pca - dino_pca.min(0)) / (dino_pca.max(0) - dino_pca.min(0) + 1e-8)
    
    fig = plt.figure(figsize=(16, 7))
    
    # Plot 1: Utonia Only (Geometry)
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.scatter(points[:, 0], points[:, 1], points[:, 2], c=u_rgb, s=2)
    ax1.set_title("Utonia Only (Geometry Features)")
    ax1.set_xlim(-0.5, 0.5); ax1.set_ylim(-0.5, 0.5); ax1.set_zlim(-0.5, 0.5)
    
    # Plot 2: DINO Only (Semantic)
    ax2 = fig.add_subplot(122, projection='3d')
    ax2.scatter(points[:, 0], points[:, 1], points[:, 2], c=d_rgb, s=2)
    ax2.set_title("DINOv2 Only (Semantic Features)")
    ax2.set_xlim(-0.5, 0.5); ax2.set_ylim(-0.5, 0.5); ax2.set_zlim(-0.5, 0.5)

    plt.savefig(save_path)
    print(f"✨ SUCCESS: Saved comparison to {save_path}")

# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # --- DATA LOADING (Replica RGB/Depth/Mask) ---
    # example paths (replace with your actual paths or loading code)
    # rgb_img = cv2.imread("frame_0.jpg")[..., ::-1] 
    # depth_map = np.load("depth_0.npy") 
    # sam_mask = np.load("best_mask.npy") # The output from your SAM script
    # ⚠️ Make sure to load your actual RGB, Depth, and SAM mask here!

    # --- SET ABSOLUTE PATHS ---
    # Replace these with the output from the 'find' command above
    RGB_PATH = "data/Replica/room_0/imap/00/rgb/rgb_0.png"
    DEPTH_PATH = "data/Replica/room_0/imap/00/depth/depth_0.png"

    # --- LOAD DATA ---
    if not os.path.exists(RGB_PATH):
        raise FileNotFoundError(f"❌ Could not find RGB at {RGB_PATH}")

    rgb = plt.imread(RGB_PATH)
    depth_raw = cv2.imread(DEPTH_PATH, cv2.IMREAD_UNCHANGED)

    if depth_raw is None:
        print(f"❌ Error: Could not load depth map at {DEPTH_PATH}")
    else:
        # Divide by 1000.0 because raw 5007 / 1000 = 5.007 meters (Correct for iMAP)
        depth_meters = depth_raw.astype(np.float32) / 1000.0

    print(f"✅ Loaded Image: {rgb.shape}, Max Depth: {depth_meters.max():.2f}m")

    CHECKPOINT = "checkpoints/weights/sam_vit_h_4b8939.pth"
    MODEL_TYPE = "vit_h"

    # Initialize SAM on Mac GPU
    sam = sam_model_registry[MODEL_TYPE](checkpoint=CHECKPOINT).to(DEVICE)
    predictor = SamPredictor(sam)

    # --- EXECUTION ---
    # Let's target the chair. In frame_0, a point on the chair is roughly (x=800, y=500)
    input_point = np.array([[800, 500]]) 
    input_label = np.array([1]) # 1 = foreground

    predictor.set_image(rgb)
    masks, scores, logits = predictor.predict(
        point_coords=input_point,
        point_labels=input_label,
        multimask_output=True,
    )

    # Use the mask with the highest score
    sam_mask = masks[np.argmax(scores)]
    
    # DINO extraction
    dino_feat_map = get_dino_features(rgb)
    
    # Lift to 3D
    dense_xyz, dense_dino = lift_to_3d_with_features(depth_meters, sam_mask, dino_feat_map)
    
    # Clean and Downsample (Maintaining feature alignment!)
    utonia_input_pc, dino_input_feats = prepare_for_utonia(dense_xyz, dense_dino)
    utonia_input_pc = utonia_input_pc.to(DEVICE)
    
    # --- UTONIA INFERENCE ---
    backbone = UtoniaBackbone(use_mock=False).to(DEVICE)
    # Fixed the security warning by adding weights_only=True
    checkpoint = torch.load("checkpoints/utoniadreamer/latest.pth", map_location=DEVICE, weights_only=True)
    full_sd = checkpoint['model'] if 'model' in checkpoint else checkpoint
    backbone_sd = {k.replace('backbone.', ''): v for k, v in full_sd.items() if k.startswith('backbone.')}
    backbone.load_state_dict(backbone_sd, strict=False)
    backbone.eval()

    print(f"🧠 Running Utonia Inference on {torch.cuda.get_device_name(0)}...")
    with torch.no_grad():
        _, (sparse_coords, point_feats) = backbone(utonia_input_pc)
        
    # Map Utonia sparse features back to the 8192 dense points
    s_coords = sparse_coords.detach().cpu().numpy()
    if s_coords.shape[1] == 4: s_coords = s_coords[:, 1:]
    s_feats = point_feats.detach().cpu().numpy().squeeze()
    
    pc_np = utonia_input_pc.squeeze().cpu().numpy()
    
    # Scale pc_np to match voxel coordinates for KDTree
    pc_min, pc_max = pc_np.min(0), pc_np.max(0)
    s_min, s_max = s_coords.min(0), s_coords.max(0)
    pc_scaled = (pc_np - pc_min) / (pc_max - pc_min + 1e-8) * (s_max - s_min) + s_min
    
    tree = KDTree(s_coords)
    _, indices = tree.query(pc_scaled)
    final_utonia_feats = s_feats[indices]

    # Plot the results side by side!
    plot_pca_comparison(pc_np, final_utonia_feats, dino_input_feats)