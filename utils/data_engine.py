import os
import json
import pickle

import cv2
import numpy as np
import torch
import trimesh
from scipy.spatial import KDTree


def _matrix_to_intrinsics(raw_k):
    K_mat = np.asarray(raw_k, dtype=np.float64)
    return {
        "fx": K_mat[0, 0],
        "fy": K_mat[1, 1],
        "cx": K_mat[0, 2],
        "cy": K_mat[1, 2],
    }


class SceneDataEngine:
    def __init__(self, paths, sam, dino, utonia):
        self.paths = paths
        self.sam = sam
        print(f"DEBUG: Engine initialized. SAM type: {type(self.sam)}")
        self.dino = dino
        self.uto = utonia
        with open(paths["json"], "r") as f:
            self.meta = json.load(f)

    def get_obb_mask(self, points, obb):
        centroid = np.asarray(obb['centroid'], dtype=np.float64)
        axes = np.asarray(obb['normalizedAxes'], dtype=np.float64).reshape(3, 3)
        axes = axes / (np.linalg.norm(axes, axis=1, keepdims=True) + 1e-8)
        extents = np.asarray(obb['axesLengths'], dtype=np.float64)
        half_extents = extents * 0.5

        # Transform points from world space into the OBB local frame.
        pts_local = (points - centroid) @ axes.T
        mask = np.all(np.abs(pts_local) <= (half_extents + 0.02), axis=1)

        # Diagnostic counts for alternative local axis orientations.
        pts_local_A = (points - centroid) @ axes
        pts_local_B = (points - centroid) @ axes.T
        mask_A = np.all(np.abs(pts_local_A) <= (half_extents + 0.02), axis=1)
        mask_B = np.all(np.abs(pts_local_B) <= (half_extents + 0.02), axis=1)
        print("mask_A count:", mask_A.sum())
        print("mask_B count:", mask_B.sum())

        return mask

    def _run_utonia_on_points(self, points, num_pts):
        pts_sampled = points.copy()
        idx = np.random.choice(len(pts_sampled), num_pts, replace=(len(pts_sampled) < num_pts)).astype(np.int64)
        pts_sampled = pts_sampled[idx]

        # Sample or pad points to the requested Utonia input size.
        u_input = torch.from_numpy(pts_sampled).float().unsqueeze(0).cuda()
        with torch.no_grad():
            outputs = self.uto(u_input)

        if isinstance(outputs, tuple):
            _, internal_data = outputs
            if isinstance(internal_data, tuple):
                sparse_coords, point_feats = internal_data
            else:
                point_feats = internal_data
                sparse_coords = pts_sampled
        else:
            sparse_coords = pts_sampled
            point_feats = outputs

        # Infer the sparse coordinates and Utonia point features from the model output.

        if hasattr(sparse_coords, 'detach'):
            sparse_coords = sparse_coords.detach().cpu().numpy()
            if sparse_coords.shape[-1] == 4:
                sparse_coords = sparse_coords[:, -3:]
        if hasattr(point_feats, 'detach'):
            point_feats = point_feats.detach().cpu().numpy()

        print("\n================ UTONIA DEBUG ================")
        print("pts_sampled shape:", pts_sampled.shape)
        print("sparse_coords shape:", np.shape(sparse_coords))
        print("point_feats shape:", np.shape(point_feats))

        if len(sparse_coords) < 4:
            print("WARNING: sparse_coords too small, using identity fallback")
            return pts_sampled, np.zeros((len(pts_sampled), 32)), idx

        tree = KDTree(sparse_coords)
        distances, indices = tree.query(pts_sampled, k=4)
        weights = 1.0 / (distances + 1e-8)
        weights /= np.sum(weights, axis=1, keepdims=True)
        aligned_point_feats = np.sum(point_feats[indices] * weights[:, :, None], axis=1)

        return pts_sampled, aligned_point_feats, idx

    def get_ground_truth(self, pkl_path, shapenet_root, target_obj_id, category):
        """Load and normalize the ground truth CAD model for the target object."""
        with open(pkl_path, 'rb') as f:
            scene_obj = pickle.load(f)

        selected_box = next((b for b in scene_obj.obj_annotation_list if str(getattr(b, 'object_id', '')) == str(target_obj_id)), None)
        if selected_box is None:
            return None

        cad_path = os.path.join(shapenet_root, selected_box.catid_cad, selected_box.id_cad, 'models', 'model_normalized.obj')
        if not os.path.exists(cad_path):
            return None

        mesh = trimesh.load(cad_path, force='mesh')
        gt_pts = mesh.sample(16384)

        # Normalize GT
        gt_centered = gt_pts - gt_pts.mean(axis=0)
        gt_max_size = np.max(gt_centered.max(axis=0) - gt_centered.min(axis=0))
        
        return (gt_centered / (gt_max_size + 1e-8)) * 0.9

    def get_multi_view_tsdf_object(self, frame_indices, obj_transform, obb_data, num_pts=8192):
        from utils.utils import load_data, get_frame_info, project_world_to_pixel, get_dino_features_bilinear

        # Extract the native 4x4 transform matrix and anchor translation.
        matrix_np = (obj_transform.detach().cpu().numpy() if hasattr(obj_transform, 'detach') else np.array(obj_transform)).astype(np.float64)
        T = matrix_np[3, :3]

        scene_root = os.path.dirname(self.paths["json"])
        all_fused_points = []
        best_K, best_rgb, best_c2w = None, None, None

        # ============================================================
        # 1. USE EVERYTHING IN ALIGNED SCANNET SPACE
        # ============================================================
        obb_center = np.array(obb_data['centroid'], dtype=np.float64)
        print("OBB center:", obb_center)
        print("matrix translation:", matrix_np[3, :3])

        obb_axes = np.array(obb_data['normalizedAxes'], dtype=np.float64).reshape(3, 3)
        extents = np.array(obb_data['axesLengths'], dtype=np.float64)

        # ============================================================
        # 2. LOAD FRAMES AND PROJECT
        # ============================================================
        for i, f_idx in enumerate(frame_indices):
            frame_key = f"frame_{f_idx:06d}"
            rgb_path = os.path.join(scene_root, "rgb", f"{frame_key}.jpg")
            dep_path = os.path.join(scene_root, "depth", f"{frame_key}.png")
            if not os.path.exists(dep_path):
                continue

            rgb, depth = load_data(rgb_path, dep_path)
            frame_entry = self.meta[frame_key]

            print("depth stats:", depth.min(), depth.max())
            
            c2w = np.array(frame_entry.get('aligned_pose')).astype(np.float64)
            w2c = np.linalg.inv(c2w)

            print("OBB centroid:", obb_center)
            print("Camera position:", c2w[:3, 3])
            print("Distance:", np.linalg.norm(obb_center - c2w[:3, 3]))

            # Extract camera intrinsics from the frame metadata.
            raw_k = frame_entry.get('intrinsics') or frame_entry.get('intrinsic')
            K = _matrix_to_intrinsics(raw_k)

            if i == len(frame_indices) // 2:
                best_K, best_rgb, best_c2w = K, rgb, c2w

            ## SAM MASKING: dynamic 2D bbox from projected OBB corners
            self.sam.set_image(rgb)

            # 1. Calculate all 8 corners of the 3D OBB in world space.
            h = extents * 0.5
            local_corners = np.array([
                [x, y, z] for x in [-h[0], h[0]] for y in [-h[1], h[1]] for z in [-h[2], h[2]]
            ])
            world_corners = obb_center + local_corners @ obb_axes

            # 2. Project all 8 corners into the 2D image plane
            pixels = []
            for pt in world_corners:
                px = project_world_to_pixel(pt, w2c, K)
                # Force the output to a flat 1D array [u, v] to prevent (8, 1, 2) nesting.
                pixels.append(np.array(px).flatten()) 
            
            # Now pixels will safely be shape (8, 2)
            pixels = np.array(pixels)

            # Clamp using the high-res RGB frame boundaries where SAM runs
            img_h, img_w = rgb.shape[:2]
            
            # These slices will now safely grab the X column [:, 0] and Y column [:, 1]
            x_min = np.clip(np.min(pixels[:, 0]), 0, img_w - 1)
            y_min = np.clip(np.min(pixels[:, 1]), 0, img_h - 1)
            x_max = np.clip(np.max(pixels[:, 0]), 0, img_w - 1)
            y_max = np.clip(np.max(pixels[:, 1]), 0, img_h - 1)
            
            bbox_2d = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

            masks, scores, _ = self.sam.predict(box=bbox_2d, multimask_output=True)
            mask_2d = masks[np.argmax(scores)]
            
            # Align SAM mask size to the depth image resolution when needed.
            if mask_2d.shape != depth.shape:
                mask_2d = cv2.resize(mask_2d.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

            semantic_valid = (mask_2d > 0) & (depth > 0.1) & (depth < 4.5)
            z_cam = depth[semantic_valid]
            if len(z_cam) == 0:
                continue

            h_dep, w_dep = depth.shape[:2]
            v_grid, u_grid = np.indices((h_dep, w_dep))
            u_valid = u_grid[semantic_valid]
            v_valid = v_grid[semantic_valid]

            # Dynamically scale camera intrinsics to match depth image resolution.
            scale_x = w_dep / img_w
            scale_y = h_dep / img_h
            
            fx_depth = K["fx"] * scale_x
            fy_depth = K["fy"] * scale_y
            cx_depth = K["cx"] * scale_x
            cy_depth = K["cy"] * scale_y

            x_cam = (u_valid - cx_depth) * z_cam / fx_depth
            y_cam = (v_valid - cy_depth) * z_cam / fy_depth
            
            pts_cam = np.stack([x_cam, y_cam, z_cam, np.ones_like(z_cam)], axis=-1)
            pts_world = (c2w @ pts_cam.T).T[:, :3]

            if np.isnan(pts_world).any():
                continue

            crop_mask = self.get_obb_mask(pts_world, obb_data)
            pts_cropped = pts_world[crop_mask]

            if len(pts_cropped) == 0:
                print("EMPTY OBB CROP")
                continue

            if len(pts_cropped) < 300:
                print("   ⚠️ rejected frame due to low point count:", len(pts_cropped))
                continue

            if len(pts_cropped) > 10:
                all_fused_points.append(pts_cropped)

        # ============================================================
        # 3. CHECK FUSION
        # ============================================================
        if not all_fused_points:
            raise ValueError("Fusion failed: No valid geometry extracted.")

        pts_world_all = np.concatenate(all_fused_points, axis=0)
        balanced = []
        max_per_frame = 3000

        for chunk in all_fused_points:
            if len(chunk) > max_per_frame:
                idx = np.random.choice(len(chunk), max_per_frame, replace=False)
                chunk = chunk[idx]
            balanced.append(chunk)

        pts_world_all = np.concatenate(balanced, axis=0)

        # ============================================================
        # 4. CANONICALIZATION (ALIGN TO ANNOTATED OBB FRAME)
        # ============================================================
        local_pts = (pts_world_all - obb_center) @ obb_axes.T

        # ============================================================
        # 5. VOXEL DOWNSAMPLE & NORMALIZE the canonical point cloud.
        # ============================================================
        voxel_coords = np.round(local_pts / 0.005).astype(np.int32)
        _, unique_indices = np.unique(voxel_coords, axis=0, return_index=True)
        final_pts_local = local_pts[unique_indices]

        shape_center = np.median(final_pts_local, axis=0)
        clean_pts_centered = final_pts_local - shape_center

        extent = clean_pts_centered.max(axis=0) - clean_pts_centered.min(axis=0)
        max_size = np.max(extent)
        clean_pts_canonical = clean_pts_centered / (max_size + 1e-8) * 0.9

        # ============================================================
        # 6. DINO FEATURES (project canonical points back into RGB frame).
        # ============================================================
        # Project canonical points back into aligned world space for pixel lookup.
        clean_pts_world_actual = final_pts_local @ obb_axes + obb_center
        
        w2c_best = np.linalg.inv(best_c2w)
        uv_continuous = project_world_to_pixel(clean_pts_world_actual, w2c_best, best_K)
        d_feats = get_dino_features_bilinear(self.dino, best_rgb, uv_continuous)

        # ============================================================
        # 7. UTONIA FEATURES & FUSION
        # ============================================================
        # Extract Utonia point features on the canonical object shape.
        s_pts, u_feats, sampled_indices = self._run_utonia_on_points(clean_pts_canonical, num_pts)
        d_feats = d_feats[sampled_indices]

        fused_pointwise = np.hstack([
            (u_feats.detach().cpu().numpy() if hasattr(u_feats, 'detach') else u_feats),
            d_feats
        ])

        return s_pts, u_feats, d_feats, fused_pointwise, T