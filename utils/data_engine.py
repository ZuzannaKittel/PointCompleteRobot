import os
import json
import pickle

import cv2
import numpy as np
import torch
import trimesh
import open3d as o3d
from scipy.spatial import KDTree
from typing import Tuple


def _matrix_to_intrinsics(raw_k):
    K_mat = np.asarray(raw_k, dtype=np.float64)
    return {
        "fx": K_mat[0, 0],
        "fy": K_mat[1, 1],
        "cx": K_mat[0, 2],
        "cy": K_mat[1, 2],
    }

def project_world_to_camera(points_world, w2c):
    if points_world.ndim == 1:
        points_world = points_world[None, :]
    pts_h = np.hstack([points_world, np.ones((len(points_world), 1))])
    cam = (w2c @ pts_h.T).T
    return cam[:, :3]


def project_to_pixel(points_cam, K):
    x = points_cam[:, 0] / (points_cam[:, 2] + 1e-8)
    y = points_cam[:, 1] / (points_cam[:, 2] + 1e-8)

    u = K["fx"] * x + K["cx"]
    v = K["fy"] * y + K["cy"]
    return np.stack([u, v], axis=1)


def inside_image(uv, shape):
    h, w = shape[:2]
    return (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)


def sample_depth(depth, uv):
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, depth.shape[1] - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, depth.shape[0] - 1)
    return depth[v, u]


def mask_check(mask, uv):
    u = np.clip(np.round(uv[:, 0]).astype(int), 0, mask.shape[1] - 1)
    v = np.clip(np.round(uv[:, 1]).astype(int), 0, mask.shape[0] - 1)
    return mask[v, u] > 0


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

        #mask = np.all(np.abs(pts_local) <= (half_extents + 0.05), axis=1)

        inside = np.all(np.abs(pts_local) <= half_extents, axis=1)

        soft = np.linalg.norm(
            np.maximum(0, np.abs(pts_local) - half_extents),
            axis=1
        )

        mask = inside | (soft < 0.15 * np.max(half_extents))

        # Diagnostic counts for alternative local axis orientations.
        """pts_local_A = (points - centroid) @ axes
        pts_local_B = (points - centroid) @ axes.T
        mask_A = np.all(np.abs(pts_local_A) <= (half_extents + 0.02), axis=1)
        mask_B = np.all(np.abs(pts_local_B) <= (half_extents + 0.02), axis=1)
        print("mask_A count:", mask_A.sum())
        print("mask_B count:", mask_B.sum())"""

        print("\nOBB DEBUG")
        print("Half extents:", half_extents)
        print("Local min:", pts_local.min(axis=0))
        print("Local max:", pts_local.max(axis=0))

        pts_local_A = (points - centroid) @ axes
        pts_local_B = (points - centroid) @ axes.T

        mask_A = np.all(np.abs(pts_local_A) <= half_extents, axis=1)
        mask_B = np.all(np.abs(pts_local_B) <= half_extents, axis=1)

        print("mask_A:", mask_A.sum())
        print("mask_B:", mask_B.sum())

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

    def get_ground_truth(self, selected_box, shapenet_root, category, obb_data):
        """
        Load and anisotropically scale the ground truth CAD model 
        to match the true aspect ratio of the real-world object.
        """
        if selected_box is None or not hasattr(selected_box, 'catid_cad'):
            return None

        cad_path = os.path.join(shapenet_root, selected_box.catid_cad, selected_box.id_cad, 'models', 'model_normalized.obj')
        if not os.path.exists(cad_path):
            return None

        # Load and sample surface points
        mesh = trimesh.load(cad_path, force='mesh')
        gt_pts = mesh.sample(16384)

        # 1. Calculate the current tight bounds of the raw CAD points
        gt_min = gt_pts.min(axis=0)
        gt_max = gt_pts.max(axis=0)
        gt_center = (gt_max + gt_min) / 2.0
        
        # Center the CAD points at the origin
        gt_centered = gt_pts - gt_center
        
        # 2. Determine the unique extent lengths of this specific CAD asset
        cad_extents = gt_max - gt_min  # Shape: (3,) -> [length, width, height]
        
        # 3. Completely normalize the CAD per-axis into a clean unit cube [-0.5, 0.5]^3
        gt_unit_cube = gt_centered / (cad_extents + 1e-8)
        
        # --- GLOBAL FIX: Universal +180-degree Y-axis rotation ---
        # For theta = +180 degrees: cos(180) = -1, sin(180) = 0
        # R_y = [[-1,  0,  0],
        #        [ 0,  1,  0],
        #        [ 0,  0, -1]]
        R_y_180 = np.array([
            [-1.0,  0.0,  0.0],
            [ 0.0,  1.0,  0.0],
            [ 0.0,  0.0, -1.0]
        ], dtype=np.float64)
    
        gt_unit_cube = gt_unit_cube @ R_y_180.T
        # --------------------------------------------------------
        
        # 4. Apply Anisotropic Scaling using real-world OBB proportions
        real_extents = np.asarray(obb_data['axesLengths'], dtype=np.float64)  # Real object [L, W, H]
        shared_max_size = np.max(real_extents)
        
        # Stretch/compress each axis to match the real-world aspect ratio inside canonical space
        aspect_ratio_scale = real_extents / (shared_max_size + 1e-8)
        gt_canonical = gt_unit_cube * aspect_ratio_scale * 0.9

        return gt_canonical
    
    def _tsdf_hash(self, p, voxel_size):
        return tuple(np.floor(p / voxel_size).astype(np.int32))
    
    def multiview_verify_points(self, points_world, selected_frames, depth_tol=0.03, confidence_thresh=0.5):
        print("\nRunning global multi-view verification...")

        support = np.zeros(len(points_world), dtype=np.float32)
        visibility = np.zeros(len(points_world), dtype=np.float32)

        for frame in selected_frames:
            w2c = frame["w2c"]
            K = frame["K"]
            depth = frame["depth"]
            mask = frame["mask"]

            pts_h = np.hstack([points_world, np.ones((len(points_world), 1))])
            pts_cam = (w2c @ pts_h.T).T[:, :3]

            u = pts_cam[:, 0] * K["fx"] / (pts_cam[:, 2] + 1e-8) + K["cx"]
            v = pts_cam[:, 1] * K["fy"] / (pts_cam[:, 2] + 1e-8) + K["cy"]

            u = np.round(u).astype(np.int32)
            v = np.round(v).astype(np.int32)

            H, W = depth.shape

            inside = ((pts_cam[:, 2] > 0.05) & (u >= 0) & (u < W) & (v >= 0) & (v < H))

            visibility += inside.astype(np.float32)

            idx = np.where(inside)[0]
            if len(idx) == 0:
                continue

            measured_depth = depth[v[idx], u[idx]]
            inside_mask = mask[v[idx], u[idx]]

            depth_ok = np.abs(
                pts_cam[idx, 2] - measured_depth
            ) < depth_tol

            good = depth_ok & inside_mask
            support[idx] += good.astype(np.float32)
        
        confidence = support / np.maximum(visibility, 1)

        print(
            "Visibility:",
            visibility.min(),
            visibility.mean(),
            visibility.max()
        )

        print(
            "Support:",
            support.min(),
            support.mean(),
            support.max()
        )

        print(
            "Confidence:",
            confidence.min(),
            confidence.mean(),
            confidence.max()
        )
        keep = confidence >= confidence_thresh
        filtered = points_world[keep]
        print("Verification kept", len(filtered), "/", len(points_world),"points")

        return filtered

    def get_multi_view_tsdf_object(self, frame_indices, obj_transform, obb_data, num_pts=2048):
        # This is the main function that extracts the multi-view TSDF object point cloud, 
        # applies SAM masking, canonicalizes it to the OBB frame, and fuses DINO + Utonia features.
        
        from utils.utils import load_data, get_frame_info, project_world_to_pixel, get_dino_features_bilinear

        # Extract the native 4x4 transform matrix and anchor translation.
        matrix_np = (obj_transform.detach().cpu().numpy() if hasattr(obj_transform, 'detach') else np.array(obj_transform)).astype(np.float64)
        T = matrix_np[3, :3]

        scene_root = os.path.dirname(self.paths["json"])

        # Debug: only save the first processed object
        debug_saved = False

        frame_clouds = []
        best_K, best_rgb, best_c2w = None, None, None
        # ============================================================
        # 1. USE EVERYTHING IN ALIGNED SCANNET SPACE
        # ============================================================
        obb_center = np.array(obb_data['centroid'], dtype=np.float64)
        # DEBUGGING: Print OBB and camera stats to verify they are in the expected range and location before projection.
        """print("OBB center:", obb_center)
        print("matrix translation:", matrix_np[3, :3])"""

        obb_axes = np.array(obb_data['normalizedAxes'], dtype=np.float64).reshape(3, 3)
        extents = np.array(obb_data['axesLengths'], dtype=np.float64)

        # ============================
        # TSDF ACCUMULATION GRID
        # ============================
        voxel_size = 0.02  # 2cm (tunable)
        tsdf = {}  # sparse hash grid
        tsdf_count = {}

        # ============================================================
        # 2. LOAD FRAMES AND PROJECT
        # ============================================================
        for i, f_idx in enumerate(frame_indices):
            frame_key = f"frame_{f_idx:06d}"
            print(f"\n==============================")
            print(f"Processing {frame_key}")
            rgb_path = os.path.join(scene_root, "rgb", f"{frame_key}.jpg")
            dep_path = os.path.join(scene_root, "depth", f"{frame_key}.png")
            if not os.path.exists(dep_path):
                continue

            rgb, depth = load_data(rgb_path, dep_path)
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float64) / 1000.0  # Convert mm to meters

            frame_entry = self.meta[frame_key]

            # DEBUGGING: Print depth stats to verify the depth map is valid and in the expected range (e.g., 0.1m to 4.5m for indoor scenes).
            # print("depth stats:", depth.min(), depth.max())
            
            c2w = np.array(frame_entry.get('aligned_pose')).astype(np.float64)
            w2c = np.linalg.inv(c2w)

            # DEBUGGING: Print camera and OBB stats to verify they are in the expected range and location.
            """print("OBB centroid:", obb_center)
            print("Camera position:", c2w[:3, 3])
            print("Distance:", np.linalg.norm(obb_center - c2w[:3, 3]))"""

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

            #world_corners = obb_center + local_corners @ obb_axes

            obb_axes = obb_axes / (np.linalg.norm(obb_axes, axis=0, keepdims=True) + 1e-8)

            world_corners = np.zeros_like(local_corners)

            for i in range(3):
                world_corners += np.outer(local_corners[:, i], obb_axes[:, i])

            world_corners = world_corners + obb_center

            # 2. Project all 8 corners into the 2D image plane
            pixels = []
            for pt in world_corners:
                # Transform to camera space first to check Z
                pt_h = np.append(pt, 1.0)
                cam_pt = w2c @ pt_h
                if cam_pt[2] > 0.05:  # Only project points safely in front of the lens
                    px = project_world_to_pixel(pt, w2c, K)
                    pixels.append(np.array(px).flatten())

            # Edge case safety check
            if len(pixels) < 4: 
                continue  # Skip frame if the camera is inside/overwhelmingly behind the box
            
            # Now pixels will safely be shape (8, 2)
            pixels = np.array(pixels)

            # Clamp using the high-res RGB frame boundaries where SAM runs
            img_h, img_w = rgb.shape[:2]
            
            # These slices will now safely grab the X column [:, 0] and Y column [:, 1]
            # padding added to stablize sam bbox
            pad = 0.1  # 10% padding

            x_min = np.min(pixels[:, 0])
            y_min = np.min(pixels[:, 1])
            x_max = np.max(pixels[:, 0])
            y_max = np.max(pixels[:, 1])

            dx = (x_max - x_min) * pad
            dy = (y_max - y_min) * pad

            x_min = np.clip(x_min - dx, 0, img_w - 1)
            y_min = np.clip(y_min - dy, 0, img_h - 1)
            x_max = np.clip(x_max + dx, 0, img_w - 1)
            y_max = np.clip(y_max + dy, 0, img_h - 1)
            
            bbox_2d = np.array([x_min, y_min, x_max, y_max], dtype=np.float32)

            print(">>> Running SAM...")
            masks, scores, _ = self.sam.predict(box=bbox_2d, multimask_output=True)
            print(">>> SAM finished")
            mask_2d = masks[np.argmax(scores)]
            
            # Align SAM mask size to the depth image resolution when needed.
            if mask_2d.shape != depth.shape:
                mask_2d = cv2.resize(mask_2d.astype(np.uint8), (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST) > 0

            depth_valid = (depth > 0.1) & (depth < 4.5)

            semantic_valid = (mask_2d > 0) & (depth > 0.1) & (depth < 4.5)

            if np.count_nonzero(semantic_valid) < 500:
                continue

            print(
                "Mask stats: \n",
                frame_key,
                "mask:", np.count_nonzero(mask_2d),
                "depth valid:", np.count_nonzero(depth_valid),
                "semantic:", np.count_nonzero(semantic_valid),
            )   

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
            print(">>> Backprojecting points...")
            pts_world = (c2w @ pts_cam.T).T[:, :3]
            print(f">>> World points: {len(pts_world)}")

            print(">>> Cropping with OBB...")
            crop_mask = self.get_obb_mask(pts_world, obb_data)
            print(">>> OBB crop finished")
            pts_cropped = pts_world[crop_mask]

            if len(pts_cropped) == 0:
                print("⚠️ EMPTY CROPPED CLOUD — check OBB projection / SAM bbox")
                continue

            pts_verified_frame = pts_cropped
            support = np.ones(len(pts_verified_frame), dtype=np.float32)

            # ============================================================
            # SAFE OUTPUT (NO BROKEN INDEXING)
            # ============================================================

            support_mean = support.mean() if len(support) > 0 else 0.0

            keep_ratio = (
                len(pts_verified_frame) / (len(pts_cropped) + 1e-8)
                if len(pts_cropped) > 0 else 0.0
            )

            print(
                frame_key,
                "cropped:", len(pts_cropped),
                "verified:", len(pts_verified_frame),
                "keep ratio:", keep_ratio,
                "support mean:", support_mean
            )

            print(
                frame_key,
                "mask pixels:", np.count_nonzero(mask_2d),
                "valid depth:", np.count_nonzero(semantic_valid),
                "cropped pts:", len(pts_cropped)
            )

            # ============================================================
            # FRAME FILTER
            # ============================================================

            min_points = 80 # to keep small but valid frames

            if len(pts_verified_frame) < min_points:
                print("   ⚠️ rejected frame:", len(pts_verified_frame))
                continue
                            
            # Score based on the geometry richness and mask quality and consistency
            #score = len(pts_cropped) * (np.count_nonzero(semantic_valid) / (mask_2d.size + 1e-8))
            coverage = len(pts_verified_frame)
            distance = np.median(depth[semantic_valid])

            score = coverage / (distance + 1e-6)

            if len(pts_cropped) > 10:
                print(">>> Appending frame to frame_clouds")
                frame_clouds.append({
                    "raw_points": pts_world.copy(),
                    "cropped_points": pts_cropped.copy(),
                    "points": pts_verified_frame.copy(),

                    "score": score,
                    "frame": frame_key,

                    "mask": mask_2d.copy(),
                    "semantic_valid": semantic_valid.copy(),
                    "depth": depth.copy(),

                    "K": K,
                    "c2w": c2w,
                    "w2c": w2c,
                })
                print(">>> Frame appended")

            print(frame_key, score)

        print("\n==============================")
        print("Finished processing ALL frames")
        print("==============================")

        # ============================================================
        # 3. CHECK FUSION
        # ============================================================
        if len(frame_clouds) < 2:
            print("⚠️ Not enough good views — skipping object")
            return None

        # ------------------------------------------------------------------
        # Rank frames by quality (currently: geometry richness and mask quality + closer, more detailes views)
        # ------------------------------------------------------------------
        frame_clouds.sort(key=lambda x: x["score"], reverse=True)

        max_frames = 12
        selected_frames = frame_clouds[:max_frames]

        if debug_saved is False:

            debug_package = {
                "selected_frames": selected_frames,
                "obb_center": obb_center,
                "obb_axes": obb_axes,
                "extents": extents,
            }

            torch.save(debug_package, "debug_multiview.pt")

            debug_saved = True

        print("\nSelected frames:")
        for f in selected_frames:
            print(f"  {f['frame']}: {f['score']} points")
            print(f"Finished processing {frame_key}")

        # Frame balancing
        max_per_frame = 2000

        balanced = []

        for item in selected_frames:

            chunk = item["points"]

            if len(chunk) >= max_per_frame:
                idx = np.random.choice(len(chunk), max_per_frame, replace=False)
            else:
                idx = np.random.choice(len(chunk), max_per_frame, replace=True)

            balanced.append(chunk[idx])

        pts_world_all = np.concatenate(balanced, axis=0)

        # Frames verification
        pts_world_all = self.multiview_verify_points(pts_world_all, selected_frames)

        # ============================================================
        # 4. CANONICALIZATION (ALIGN TO ANNOTATED OBB FRAME)
        # ============================================================
        # This properly centers and rotates the points using the GT anchor
        local_pts = (pts_world_all - obb_center) @ obb_axes.T

        # ============================================================
        # 5. FILTERING & SHARED NORMALIZATION
        # ============================================================
        # A. Voxel Downsample
        debug_local_pts = local_pts.copy()
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(local_pts)

        pcd = pcd.voxel_down_sample(voxel_size=0.005)

        final_pts_local = np.asarray(pcd.points)

        debug_voxel = final_pts_local.copy()

        # light trimming only (outlier suppression, not shape destruction)
        centroid = final_pts_local.mean(axis=0)
        dist = np.linalg.norm(final_pts_local - centroid, axis=1)

        # only remove extreme outliers (very safe)
        threshold = np.percentile(dist, 99.5)
        final_pts_local = final_pts_local[dist < threshold]

        # B. Statistical Outlier Removal (Cleans the floating noise)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(final_pts_local)
        cl, ind = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        clean_pts_centered = np.asarray(pcd.points)[ind] # Already centered via OBB!
        debug_clean = clean_pts_centered.copy()

        # C. Shared Normalization (Use OBB size, NOT partial scan size)
        # extents is obb_data['axesLengths']
        shared_max_size = np.max(extents) 
        clean_pts_canonical = clean_pts_centered / (shared_max_size + 1e-8) * 0.9
        debug_canonical = clean_pts_canonical.copy()

        # ============================================================
        # 6. DINO FEATURES (project canonical points back into RGB frame).
        # ============================================================
        # Project canonical points back into aligned world space for pixel lookup.
        clean_pts_world_actual = clean_pts_centered @ obb_axes + obb_center
        clean_pts_world_actual = clean_pts_world_actual.astype(np.float64)
        
        w2c_best = np.linalg.inv(best_c2w)
        uv_continuous = project_world_to_pixel(clean_pts_world_actual, w2c_best, best_K)
        d_feats = get_dino_features_bilinear(self.dino, best_rgb, uv_continuous)

        # ============================================================
        # 7. UTONIA FEATURES & FUSION
        # ============================================================
        # Extract Utonia point features on the canonical object shape.
        s_pts, u_feats, sampled_indices = self._run_utonia_on_points(clean_pts_canonical, num_pts)
        debug_sampled = s_pts.copy()
        d_feats = d_feats[sampled_indices]

        torch.save({
            "merged": pts_world_all,
            "local": debug_local_pts,
            "voxel": debug_voxel,
            "clean": debug_clean,
            "canonical": debug_canonical,
            "sampled": debug_sampled,
        }, "debug_pipeline.pt")

        fused_pointwise = np.hstack([
            (u_feats.detach().cpu().numpy() if hasattr(u_feats, 'detach') else u_feats),
            d_feats
        ])

        print("FINAL CLOUD SIZE:", len(clean_pts_centered))
        print("FINAL CANONICAL SIZE:", len(clean_pts_canonical))

        return s_pts, u_feats, d_feats, fused_pointwise, T