import json
import numpy as np
import torch
import cv2
import os
from models.models import init_sam, init_dino, init_utonia
from utils.utils import (load_data, load_metadata,get_frame_info, get_sam_mask, get_dino_features_bilinear,
                   lift_to_world, project_world_to_pixel, voxel_fusion, save_barcode, 
                   save_verification_image, get_utonia_features, fuse_features, plot_multimodal_results)

from utils.gt_extractor import extract_gt_object

PATHS = {
    "sam": "checkpoints/weights/sam_vit_h_4b8939.pth",
    "uto": "checkpoints/utoniadreamer/latest.pth",
    "json": "data/ScanNetPP/30966f4c6e/iphone/pose_intrinsic_imu.json",
    "mesh": "data/ScanNetPP/30966f4c6e/scans/mesh_aligned_0.05_semantic.ply"
}
os.makedirs("verify_frames", exist_ok=True)

def build_global_object(frame_indices, paths, sam_predictor, dino_model, initial_prompt):
    print("🌍 Phase 1: Multi-frame extraction and lifting...")
    meta = load_metadata(paths["json"])

    all_points = []
    all_dino = []
    
    # We will store the "Master 3D Point" once we calculate it in the first frame
    target_world_point = None 

    for i, idx in enumerate(frame_indices):
        frame_key = f"frame_{idx:06d}"
        rgb_path = f"data/ScanNetPP/30966f4c6e/iphone/rgb/{frame_key}.jpg"
        dep_path = f"data/ScanNetPP/30966f4c6e/iphone/depth/{frame_key}.png"

        try:
            rgb, depth = load_data(rgb_path, dep_path)
        except Exception as e:
            print(f"  ❌ Skipping {frame_key}: {e}")
            continue

        pose, K = get_frame_info(meta, frame_key)

        # --- Dynamic Prompt Logic ---
        if i == 0:
            # 1. Get original RGB click coordinates
            u_rgb, v_rgb = initial_prompt[0]
            
            # 2. Calculate scaling factors (RGB -> Depth)
            scale_u = depth.shape[1] / rgb.shape[1]
            scale_v = depth.shape[0] / rgb.shape[0]
            
            # 3. Map coordinates to depth resolution
            u_depth = int(u_rgb * scale_u)
            v_depth = int(v_rgb * scale_v)
            
            # 4. Safely get depth
            z = depth[v_depth, u_depth] 
            
            # Lift to world using the SCALED Intrinsics (K_scaled is handled below, 
            # but for the first frame's math, we use the depth-res version)
            # We'll use your existing K_scaled logic but apply it here once
            scale_x = depth.shape[1] / rgb.shape[1]
            scale_y = depth.shape[0] / rgb.shape[0]
            
            fx_s, fy_s = K['fx'] * scale_x, K['fy'] * scale_y
            cx_s, cy_s = K['cx'] * scale_x, K['cy'] * scale_y
            
            x = (u_depth - cx_s) * z / fx_s
            y = (v_depth - cy_s) * z / fy_s
            pts_cam = np.array([x, y, z, 1.0])
            target_world_point = (pose @ pts_cam)[:3]
            
            current_prompt = initial_prompt
            print(f"  📌 Anchored 3D point at depth px ({u_depth}, {v_depth}) -> World: {target_world_point}")
        else:
            # Later frames: CALCULATE where that 3D point is on the screen now
            current_prompt = project_world_to_pixel(target_world_point, pose, K)
            print(f"  🎯 Projected prompt to 2D: {current_prompt}")

        # Run SAM with the dynamic prompt
        full_res_mask = get_sam_mask(sam_predictor, rgb, current_prompt)

        # Visual Verification
        verify_path = f"verify_frames/verify_{frame_key}.jpg"
        save_verification_image(rgb, full_res_mask, current_prompt, verify_path)
        print(f"  📸 Saved target verification to: {verify_path}")
        
        mask_depth_res = cv2.resize(
            full_res_mask.astype(np.uint8), 
            (depth.shape[1], depth.shape[0]), 
            interpolation=cv2.INTER_NEAREST
        ).astype(bool)
        
        K_scaled = K.copy()
        scale_x = depth.shape[1] / rgb.shape[1]
        scale_y = depth.shape[0] / rgb.shape[0]
        K_scaled['fx'] *= scale_x
        K_scaled['fy'] *= scale_y
        K_scaled['cx'] *= scale_x
        K_scaled['cy'] *= scale_y

        pts_world = lift_to_world(depth, mask_depth_res, K_scaled, pose)
        dino_feats = get_dino_features_bilinear(dino_model, rgb, mask_depth_res)

        print(f"Frame {idx}: DINO Mean={dino_feats.mean():.4f}, Max={dino_feats.max():.4f}")

        all_points.append(pts_world)
        all_dino.append(dino_feats)

        print(f"  ✅ {frame_key} processed ({len(pts_world)} pts)")
    
    # Return also the anchor point
    return np.vstack(all_points), np.vstack(all_dino), target_world_point


def run_utonia_on_fused(backbone, points, num_pts=16384):
    print(f"🧠 Running Utonia Geometry Extraction on {num_pts} points...")
    pts = points.copy()
    pts -= pts.mean(0)
    pts /= (np.max(np.linalg.norm(pts, axis=1)) + 1e-8)
    
    idx = np.random.choice(len(pts), num_pts, replace=(len(pts) < num_pts)).astype(np.int64)
    pts_s = pts[idx]
    
    u_input = torch.from_numpy(pts_s).float().unsqueeze(0).cuda()
    u_feats = get_utonia_features(backbone, u_input)
    
    # Return idx so the main block can use it to slice f_dino and f_pts
    return pts_s, u_feats, idx

if __name__ == "__main__":
    # 0. Ensure GT Output Directory Exists
    os.makedirs("data/gt_output", exist_ok=True)

    # 1. Init
    sam = init_sam(PATHS["sam"])
    dino = init_dino()
    uto = init_utonia(PATHS["uto"])

    # 2. Extract Partial Scan & 3D Anchor
    raw_pts, raw_dino, anchor = build_global_object([0, 10, 20, 30, 40], PATHS, sam, dino, [[470, 370]])
    f_pts, f_dino = voxel_fusion(raw_pts, raw_dino)

    # --- Phase 2.5 - GT Extraction ---
    print("\n⚖️ Phase 2.5: Harvesting Ground Truth Laser Scan...")
    gt_save_path = "data/gt_output/gt_desk_30966.ply"
    gt_points, gt_meta = extract_gt_object(PATHS["mesh"], anchor, gt_save_path)
    # --------------------------------------

    # 3. Create Embedding
    s_pts, u_feats, sampled_indices = run_utonia_on_fused(uto, f_pts)

    # Align DINO and real-world points with the Utonia samples
    d_feats = f_dino[sampled_indices]
    real_world_pts = f_pts[sampled_indices]

    # 4. FUSE & PLOT 16K PCA (The "Visual" part)
    fused_pointwise = fuse_features(u_feats, d_feats)

    print("\n--- Plotting Step 1: 16k Point Cloud Analysis ---")
    plot_multimodal_results(
        real_world_pts, 
        u_feats, 
        d_feats, 
        fused_pointwise, 
        save_path="desk_16k_multimodal_pca_4.png"
    )
    
    # 5. CREATE & PLOT GLOBAL EMBEDDING (The "Identity" part)
    print("📦 Creating Global Object Fingerprint...")
    global_embedding = torch.max(torch.from_numpy(fused_pointwise).cuda(), dim=0, keepdim=True)[0].cpu().numpy()
    save_barcode(global_embedding, "desk_final_identity_4.png")
    
    print(f"\n🚀 SUCCESS: Identity barcode (shape: {global_embedding.shape}) created and GT Desk saved at {gt_save_path}")

    print("\n" + "="*40)
    print("✅ Pipeline Complete.")
    print(f"🔹 Input (Partial): {len(real_world_pts)} points")
    print(f"🔹 Target (GT):     {len(gt_points)} points")
    print(f"🔹 Embedding Shape: {global_embedding.shape}")
    print("="*40)