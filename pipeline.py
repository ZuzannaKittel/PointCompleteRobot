import json
import numpy as np
import torch
import cv2
import os
from models import init_sam, init_dino, init_utonia
from utils import (load_data, get_frame_info, get_sam_mask, get_dino_features_bilinear,
                   lift_to_world, project_world_to_pixel, voxel_fusion, save_barcode, 
                   save_verification_image)
from utonia_dino_fusion import fuse_features # Assuming this is in your folder

# Config
PATHS = {
    "sam": "checkpoints/weights/sam_vit_h_4b8939.pth",
    "uto": "checkpoints/utoniadreamer/latest.pth",
    "json": "data/ScanNetPP/30966f4c6e/iphone/pose_intrinsic_imu.json"
}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
os.makedirs("verify_frames", exist_ok=True)

def build_global_object(frame_indices, sam_predictor, dino_model, initial_prompt):
    with open(PATHS["json"], "r") as f: meta = json.load(f)
    all_points, all_dino, target_world_point = [], [], None

    for i, idx in enumerate(frame_indices):
        frame_key = f"frame_{idx:06d}"
        rgb, depth = load_data(f"data/ScanNetPP/30966f4c6e/iphone/rgb/{frame_key}.jpg", 
                               f"data/ScanNetPP/30966f4c6e/iphone/depth/{frame_key}.png")
        pose, K = get_frame_info(meta, frame_key)

        # Coordinate & Prompt Logic
        if i == 0:
            u_rgb, v_rgb = initial_prompt[0]
            # [FIXED] Scaling for depth resolution lookup
            u_d, v_d = int(u_rgb * (depth.shape[1]/rgb.shape[1])), int(v_rgb * (depth.shape[0]/rgb.shape[0]))
            z = depth[v_d, u_d]
            
            # Lift initial point to world
            s_x, s_y = depth.shape[1]/rgb.shape[1], depth.shape[0]/rgb.shape[0]
            pts_cam = np.array([(u_d - K['cx']*s_x)*z/(K['fx']*s_x), (v_d - K['cy']*s_y)*z/(K['fy']*s_y), z, 1.0])
            target_world_point = (pose @ pts_cam)[:3]
            current_prompt = initial_prompt
        else:
            current_prompt = project_world_to_pixel(target_world_point, pose, K)

        # Vision Processing
        mask = get_sam_mask(sam_predictor, rgb, current_prompt)
        save_verification_image(rgb, mask, current_prompt, f"verify_frames/v_{frame_key}.jpg")
        
        # Feature Lifting
        m_depth = cv2.resize(mask.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        pts_world = lift_to_world(depth, m_depth, K, pose) # Add K scaling if needed here
        dino_f = get_dino_features_bilinear(dino_model, rgb, m_depth)
        
        all_points.append(pts_world); all_dino.append(dino_f)
        print(f"✅ Frame {idx} processed.")

    return np.vstack(all_points), np.vstack(all_dino)

if __name__ == "__main__":
    sam = init_sam(PATHS["sam"])
    dino = init_dino()
    uto = init_utonia(PATHS["uto"])

    points, dino_feats = build_global_object([0, 10, 20, 30, 40], sam, dino, [[470, 370]])
    f_pts, f_dino = voxel_fusion(points, dino_feats)

    # Simplified Utonia + Fusion run...
    # (Insert your run_utonia_on_fused logic here)
    
    print("🚀 Pipeline Cleaned and Ready for Decoder Phase.")