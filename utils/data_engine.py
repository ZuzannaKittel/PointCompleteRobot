import torch
import numpy as np
import cv2
import os
import trimesh
from utils.utils import (load_data, load_metadata, get_frame_info, get_sam_mask, 
                         get_dino_features_bilinear, lift_to_world, 
                         project_world_to_pixel, voxel_fusion, 
                         save_verification_image, get_utonia_features, fuse_features)
from utils.gt_extractor_scannotatepp import extract_scannotate_gt

class SceneDataEngine:
    def __init__(self, paths, sam, dino, utonia):
        """
        paths: Dictionary containing 'json', 'mesh', and dataset roots.
        sam: Initialized SAM predictor.
        dino: Initialized DINOv2 model.
        utonia: Initialized Utonia backbone.
        """
        self.paths = paths
        self.sam = sam
        self.dino = dino
        self.uto = utonia
        self.meta = load_metadata(paths["json"])

    def get_fused_object(self, frame_indices, initial_prompt, num_pts=16384):
        """
        Orchestrates the entire Phase 1-4 pipeline.
        """
        # 1. Lift multi-view frames to world space
        raw_pts, raw_dino, anchor = self._build_global_object(frame_indices, initial_prompt)
        
        # 2. Voxel Fusion (Resolves spatial redundancy from multiple views)
        f_pts, f_dino = voxel_fusion(raw_pts, raw_dino)

        # 3. Extract Utonia Geometry Features
        s_pts, u_feats, sampled_indices = self._run_utonia_on_fused(f_pts, num_pts)

        # 4. Final Feature Fusion
        # Slice the DINO features to match the sampled Utonia indices
        d_feats = f_dino[sampled_indices]
        fused_pointwise = fuse_features(u_feats, d_feats)

        return s_pts, u_feats, d_feats, fused_pointwise, anchor

    def _build_global_object(self, frame_indices, initial_prompt):
        print("🌍 Phase 1: Multi-frame extraction and lifting...")
        all_points = []
        all_dino = []
        target_world_point = None 

        for i, idx in enumerate(frame_indices):
            frame_key = f"frame_{idx:06d}"
            # Paths constructed based on ScanNetpp structure
            # paths["json"] is '.../30966f4c6e/iphone/pose_intrinsic_imu.json'
            # scene_root becomes '.../30966f4c6e/iphone/'
            scene_root = os.path.dirname(self.paths["json"])
            rgb_path = os.path.join(scene_root, "rgb", f"{frame_key}.jpg")
            dep_path = os.path.join(scene_root, "depth", f"{frame_key}.png")

            try:
                rgb, depth = load_data(rgb_path, dep_path)
            except Exception as e:
                print(f"  ❌ Skipping {frame_key}: {e}")
                continue

            pose, K = get_frame_info(self.meta, frame_key)

            # --- Dynamic Prompt Logic ---
            if i == 0:
                # Map RGB click to Depth resolution for anchor calculation
                scale_u = depth.shape[1] / rgb.shape[1]
                scale_v = depth.shape[0] / rgb.shape[0]
                u_depth, v_depth = int(initial_prompt[0][0] * scale_u), int(initial_prompt[0][1] * scale_v)
                
                z = depth[v_depth, u_depth] 
                fx_s, fy_s = K['fx'] * scale_u, K['fy'] * scale_v
                cx_s, cy_s = K['cx'] * scale_u, K['cy'] * scale_v
                
                x = (u_depth - cx_s) * z / fx_s
                y = (v_depth - cy_s) * z / fy_s
                target_world_point = (pose @ np.array([x, y, z, 1.0]))[:3]
                current_prompt = initial_prompt
                print(f"  📌 Anchored 3D point -> World: {target_world_point}")
            else:
                current_prompt = project_world_to_pixel(target_world_point, pose, K)

            # --- Segmentation & Lifting ---
            full_res_mask = get_sam_mask(self.sam, rgb, current_prompt)
            
            # Save verification for debugging
            verify_path = f"verify_frames/verify_{frame_key}.jpg"
            save_verification_image(rgb, full_res_mask, current_prompt, verify_path)
            
            # Prepare scaled intrinsics for current depth resolution
            mask_depth_res = cv2.resize(full_res_mask.astype(np.uint8), 
                                      (depth.shape[1], depth.shape[0]), 
                                      interpolation=cv2.INTER_NEAREST).astype(bool)
            
            K_scaled = K.copy()
            K_scaled.update({'fx': K['fx']*(depth.shape[1]/rgb.shape[1]), 
                             'fy': K['fy']*(depth.shape[0]/rgb.shape[0]),
                             'cx': K['cx']*(depth.shape[1]/rgb.shape[1]), 
                             'cy': K['cy']*(depth.shape[0]/rgb.shape[0])})

            pts_world = lift_to_world(depth, mask_depth_res, K_scaled, pose)
            dino_feats = get_dino_features_bilinear(self.dino, rgb, mask_depth_res)

            all_points.append(pts_world)
            all_dino.append(dino_feats)
            print(f"  ✅ {frame_key} processed ({len(pts_world)} pts)")
        
        return np.vstack(all_points), np.vstack(all_dino), target_world_point

    def _run_utonia_on_fused(self, points, num_pts):
        print(f"🧠 Running Utonia Geometry Extraction on {num_pts} points...")
        pts = points.copy()
        
        # Mandatory Utonia Normalization (Unit Sphere)
        pts -= pts.mean(0)
        pts /= (np.max(np.linalg.norm(pts, axis=1)) + 1e-8)
        
        idx = np.random.choice(len(pts), num_pts, replace=(len(pts) < num_pts)).astype(np.int64)
        pts_s = pts[idx]
        
        u_input = torch.from_numpy(pts_s).float().unsqueeze(0).cuda()
        
        # Pointwise feature extraction
        u_feats_pointwise = get_utonia_features(self.uto, u_input, return_pointwise=True)
        
        if u_feats_pointwise.dim() == 3:
            u_feats_pointwise = u_feats_pointwise.squeeze(0)

        return pts_s, u_feats_pointwise.cpu().numpy(), idx

    def get_ground_truth(self, pkl_path, shapenet_root, anchor):
        print("\n💎 Phase 2.5: Harvesting High-Fidelity CAD Ground Truth...")
        gt_points, gt_metadata = extract_scannotate_gt(pkl_path, shapenet_root, anchor)
        return gt_points