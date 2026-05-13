import torch
import numpy as np
import cv2
import os
import trimesh
import matplotlib.pyplot as plt
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

    def save_thesis_visuals(self, raw_pts, fused_pts, save_dir="figures"):
        os.makedirs(save_dir, exist_ok=True)
        
        # 1. Visualize (c): The "Raw" Lifted Stack (All frames combined)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        # We use a small alpha because the raw stack is very dense/messy
        ax.scatter(raw_pts[:, 0], raw_pts[:, 1], raw_pts[:, 2], s=1, c='crimson', alpha=0.2)
        ax.set_title("Accumulated Multi-View Points (Raw)")
        ax.set_axis_off()
        plt.savefig(f"{save_dir}/lifted_raw_stack.png", dpi=300, bbox_inches='tight')
        plt.close()

        # 2. Visualize (d): After Voxel Fusion (Clean & Uniform)
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(fused_pts[:, 0], fused_pts[:, 1], fused_pts[:, 2], s=1, c='forestgreen', alpha=0.8)
        ax.set_title("Unified Point Cloud (After 1.5cm Fusion)")
        ax.set_axis_off()
        plt.savefig(f"{save_dir}/fused_clean_cloud.png", dpi=300, bbox_inches='tight')
        plt.close()
    
        print(f"✨ Thesis visuals saved to {save_dir}/")

    def get_fused_object(self, frame_indices, initial_prompt, num_pts=16384, save_visuals=False):
        """
        Orchestrates the entire Phase 1-4 pipeline.
        """
        ## 1. Lift multi-view frames to world space
        # We pass the box here
        raw_pts, raw_dino, anchor = self._build_global_object(frame_indices, initial_prompt)
        
        # 2. Voxel Fusion (Resolves spatial redundancy from multiple views)
        f_pts, f_dino = voxel_fusion(raw_pts, raw_dino)

        # Optional: Save visuals before we sub-sample for Utonia
        if save_visuals:
            self.save_thesis_visuals(raw_pts, f_pts)

        # 3. Extract Utonia Geometry Features
        s_pts, u_feats, sampled_indices = self._run_utonia_on_fused(f_pts, num_pts)

        # 4. Final Feature Fusion
        # Slice the DINO features to match the sampled Utonia indices
        d_feats = f_dino[sampled_indices]
        fused_pointwise = fuse_features(u_feats, d_feats)

        return s_pts, u_feats, d_feats, fused_pointwise, anchor

    def _build_global_object(self, frame_indices, initial_prompt):
        print("🌍 Phase 1: Multi-frame extraction (Bounding Box Mode)...")
        all_points = []
        all_dino = []
        target_world_point = None 
        last_pts_world = None # Used to refine the box in subsequent frames

        for i, idx in enumerate(frame_indices):
            frame_key = f"frame_{idx:06d}"
            scene_root = os.path.dirname(self.paths["json"])
            rgb_path = os.path.join(scene_root, "rgb", f"{frame_key}.jpg")
            dep_path = os.path.join(scene_root, "depth", f"{frame_key}.png")

            try:
                rgb, depth = load_data(rgb_path, dep_path)
            except Exception as e:
                print(f"  ❌ Skipping {frame_key}: {e}")
                continue

            pose, K = get_frame_info(self.meta, frame_key)

            # --- Updated Dynamic Box Prompt Logic ---
            if i == 0:
                # 1. Calculate 3D Anchor from Box Center
                x1, y1, x2, y2 = initial_prompt
                center_u, center_v = (x1 + x2) / 2, (y1 + y2) / 2
                
                scale_u = depth.shape[1] / rgb.shape[1]
                scale_v = depth.shape[0] / rgb.shape[0]
                u_depth, v_depth = int(center_u * scale_u), int(center_v * scale_v)
                
                # Ensure we don't index out of bounds
                u_depth = np.clip(u_depth, 0, depth.shape[1] - 1)
                v_depth = np.clip(v_depth, 0, depth.shape[0] - 1)
                
                z = depth[v_depth, u_depth] 
                fx_s, fy_s = K['fx'] * scale_u, K['fy'] * scale_v
                cx_s, cy_s = K['cx'] * scale_u, K['cy'] * scale_v
                
                x = (u_depth - cx_s) * z / fx_s
                y = (v_depth - cy_s) * z / fy_s
                target_world_point = (pose @ np.array([x, y, z, 1.0]))[:3]
                
                current_box = np.array(initial_prompt)
                print(f"  📌 Box Anchor -> World Center: {target_world_point}")
            else:
                # 2. Project the 3D Anchor to get the new box center
                center_proj = project_world_to_pixel(target_world_point, pose, K) # returns [[u, v]]
                cp_u, cp_v = center_proj[0]
                
                # We can estimate box size based on movement, but a robust way is to 
                # take the bounding box of the points we found in the PREVIOUS frame 
                # re-projected into this frame. 
                if last_pts_world is not None:
                    # Project all points from previous frame to current frame
                    pts_h = np.hstack([last_pts_world, np.ones((len(last_pts_world), 1))])
                    pts_cam = (np.linalg.inv(pose) @ pts_h.T).T[:, :3]
                    
                    u_proj = (pts_cam[:, 0] * K['fx'] / pts_cam[:, 2]) + K['cx']
                    v_proj = (pts_cam[:, 1] * K['fy'] / pts_cam[:, 2]) + K['cy']
                    
                    # Create a box with a 10% padding margin for SAM
                    pad = 20
                    current_box = [np.min(u_proj)-pad, np.min(v_proj)-pad, np.max(u_proj)+pad, np.max(v_proj)+pad]
                else:
                    # Fallback: Square box around projected center
                    current_box = [cp_u-100, cp_v-100, cp_u+100, cp_v+100]

            # --- Segmentation with BOX ---
            # Pass box=current_box to your updated get_sam_mask utility
            full_res_mask = get_sam_mask(self.sam, rgb, box=current_box)
            
            # Save verification (Now draws a rectangle)
            verify_path = f"verify_frames/verify_{frame_key}.jpg"
            save_verification_image(rgb, full_res_mask, current_box, verify_path)
            
            # --- Lifting (Standard) ---
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
            last_pts_world = pts_world # Store for the next frame's box projection
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