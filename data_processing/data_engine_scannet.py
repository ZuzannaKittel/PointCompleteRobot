import os
import json

import cv2
import numpy as np
import torch
import trimesh
import open3d as o3d
from scipy.spatial import KDTree


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

        inside = np.all(np.abs(pts_local) <= half_extents, axis=1)

        soft = np.linalg.norm(
            np.maximum(0, np.abs(pts_local) - half_extents),
            axis=1
        )

        mask = inside | (soft < 0.15 * np.max(half_extents))

        return mask

    def _run_utonia_on_points(self, points, num_pts):
        pts = np.asarray(points, dtype=np.float32)
        if pts.size == 0:
            return pts, np.zeros((0, 32), dtype=np.float32), np.array([], dtype=np.int64)

        num_pts = max(int(num_pts), 1)
        if len(pts) <= num_pts:
            idx = np.arange(len(pts), dtype=np.int64)
        else:
            idx = np.random.choice(len(pts), num_pts, replace=False).astype(np.int64)
        pts_sampled = pts[idx]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        u_input = torch.from_numpy(pts_sampled).float().unsqueeze(0).to(device)
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

        if hasattr(sparse_coords, "detach"):
            sparse_coords = sparse_coords.detach().cpu().numpy()
            if sparse_coords.shape[-1] == 4:
                sparse_coords = sparse_coords[:, -3:]
        if hasattr(point_feats, "detach"):
            point_feats = point_feats.detach().cpu().numpy()

        if len(sparse_coords) < 4:
            return pts_sampled, np.zeros((len(pts_sampled), 32), dtype=np.float32), idx

        tree = KDTree(sparse_coords)
        distances, indices = tree.query(pts_sampled, k=min(4, len(sparse_coords)))
        weights = 1.0 / (distances + 1e-8)
        weights /= np.sum(weights, axis=1, keepdims=True)
        point_feats = np.asarray(point_feats)
        if point_feats.ndim == 1:
            point_feats = point_feats[:, None]
        aligned_point_feats = np.sum(point_feats[indices] * weights[:, :, None], axis=1)

        return pts_sampled, aligned_point_feats, idx

    def get_ground_truth(self, selected_box, shapenet_root, category, obb_data):
    
        """Load and rescale a CAD model into a canonical reference frame."""
        if selected_box is None or not hasattr(selected_box, 'catid_cad'):
            return None

        cad_path = os.path.join(shapenet_root, selected_box.catid_cad, selected_box.id_cad, 'models', 'model_normalized.obj')
        if not os.path.exists(cad_path):
            return None

        mesh = trimesh.load(cad_path, force='mesh')

        vertices = np.asarray(mesh.vertices)

        # Compute the axis-aligned bounding box of the CAD model
        gt_min = vertices.min(axis=0)
        gt_max = vertices.max(axis=0)

        gt_pts = mesh.sample(16384)

        # 1. Center the CAD points at the origin
        gt_center = (gt_max + gt_min) / 2.0
        
        # 2. Center the CAD points at the origin
        gt_centered = gt_pts - gt_center
        
        # 3. Determine the largest extent of the CAD model to normalize it into a unit cube
        extent = np.max(gt_max - gt_min)
    
        # 4. Apply a uniform scaling factor to fit the CAD into a canonical space, leaving a small margin (0.9) to avoid touching the boundaries
        gt_canonical = gt_centered / (extent + 1e-8) * 0.9

        return gt_canonical

    def _tsdf_hash(self, p, voxel_size):
        return tuple(np.floor(p / voxel_size).astype(np.int32))
    
    def multiview_verify_points(self, points_world, selected_frames, depth_tol=0.03, confidence_thresh=0.5):
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

        keep = confidence >= confidence_thresh
        filtered = points_world[keep]
        return filtered

    def get_multi_view_tsdf_object(self, frame_indices, selected_box, obb_data, num_pts=2048):
        # Build a multi-view TSDF object cloud, crop it to the OBB,
        # canonicalize it with the annotated CAD transform, and fuse
        # DINO and Utonia features.
        from utils.utils import load_data, project_world_to_pixel, get_dino_features_bilinear

        scene_root = os.path.dirname(self.paths["json"])

        frame_clouds = []
        best_K, best_rgb, best_c2w = None, None, None
        # ============================================================
        # 1. USE EVERYTHING IN ALIGNED SCANNET SPACE
        # ============================================================
        obb_center = np.array(obb_data['centroid'], dtype=np.float64)
        obb_axes = np.array(obb_data['normalizedAxes'], dtype=np.float64).reshape(3, 3)
        extents = np.array(obb_data['axesLengths'], dtype=np.float64)

        # ============================================================
        # 2. LOAD FRAMES AND PROJECT
        # ============================================================
        for f_idx in frame_indices:
            frame_key = f"frame_{f_idx:06d}"
            print(f"Processing {frame_key}")
            rgb_path = os.path.join(scene_root, "rgb", f"{frame_key}.jpg")
            dep_path = os.path.join(scene_root, "depth", f"{frame_key}.png")
            if not os.path.exists(dep_path):
                continue

            rgb, depth = load_data(rgb_path, dep_path)
            if depth.dtype == np.uint16:
                depth = depth.astype(np.float64) / 1000.0  # Convert mm to meters

            frame_entry = self.meta[frame_key]

            c2w = np.array(frame_entry.get('aligned_pose')).astype(np.float64)
            w2c = np.linalg.inv(c2w)

            # Extract camera intrinsics from the frame metadata.
            raw_k = frame_entry.get('intrinsics') or frame_entry.get('intrinsic')
            K = _matrix_to_intrinsics(raw_k)

            if best_K is None:
                best_K, best_rgb, best_c2w = K, rgb, c2w

            ## SAM MASKING: dynamic 2D bbox from projected OBB corners
            self.sam.set_image(rgb)

            # Compute the 8 corners of the 3D OBB in world space.
            half_extents = extents * 0.5
            local_corners = np.array([
                [x, y, z] for x in [-half_extents[0], half_extents[0]]
                for y in [-half_extents[1], half_extents[1]]
                for z in [-half_extents[2], half_extents[2]]
            ], dtype=np.float64)

            obb_axes = obb_axes / (np.linalg.norm(obb_axes, axis=1, keepdims=True) + 1e-8)
            world_corners = local_corners @ obb_axes.T + obb_center

            # 2. Project all 8 corners into the 2D image plane
            pixels = []
            for pt in world_corners:
                # Transform to camera space first to check Z
                pt_h = np.append(pt, 1.0)
                cam_pt = w2c @ pt_h
                if cam_pt[2] > 0.05:  # Only project points safely in front of the lens
                    px = project_world_to_pixel(pt, w2c, K)
                    pixels.append(np.array(px).flatten())

            if len(pixels) < 4:
                continue

            pixels = np.array(pixels, dtype=np.float32)
            img_h, img_w = rgb.shape[:2]
            pad = 0.1

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

            semantic_valid = (mask_2d > 0) & (depth > 0.1) & (depth < 4.5)

            if np.count_nonzero(semantic_valid) < 500:
                continue

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
            crop_mask = self.get_obb_mask(pts_world, obb_data)
            pts_cropped = pts_world[crop_mask]

            # Skip frames where the OBB crop removes too much geometry.
            frame_pass_rate = len(pts_cropped) / max(len(pts_world), 1)
            if frame_pass_rate < 0.3:
                print(f"  ⚠️ frame {frame_key}: only {frame_pass_rate:.1%} of backprojected points "
                    f"fall inside the OBB -- likely mask leakage, skipping")
                continue

            if len(pts_cropped) == 0:
                print("⚠️ EMPTY CROPPED CLOUD — check OBB projection / SAM bbox")
                continue

            pts_verified_frame = pts_cropped

            # ============================================================
            # FRAME FILTER
            # ============================================================

            min_points = 80  # Keep small but valid frames.

            if len(pts_verified_frame) < min_points:
                continue

            # Rank frames by the amount of geometry they contribute relative
            # to the median observed depth.
            coverage = len(pts_verified_frame)
            distance = np.median(depth[semantic_valid])

            score = coverage / (distance + 1e-6)

            if len(pts_cropped) > 10:
                frame_clouds.append({
                    "raw_points": pts_world.copy(),
                    "cropped_points": pts_cropped.copy(),
                    "points": pts_verified_frame.copy(),

                    "score": score,
                    "frame": frame_key,

                    "mask": mask_2d.copy(),
                    "semantic_valid": semantic_valid.copy(),
                    "depth": depth.copy(),
                    "rgb": rgb.copy(),

                    "K": K,
                    "c2w": c2w,
                    "w2c": w2c,
                })

        print("Finished processing all frames")

        # ============================================================
        # 3. CHECK FUSION
        # ============================================================
        if len(frame_clouds) < 2:
            return None

        # ------------------------------------------------------------------
        # Rank frames by quality (currently: geometry richness and mask quality + closer, more detailes views)
        # ------------------------------------------------------------------
        frame_clouds.sort(key=lambda x: x["score"], reverse=True)

        max_frames = 12
        selected_frames = frame_clouds[:max_frames]

        print("Selected frames:")
        for f in selected_frames:
            print(f"  {f['frame']}: score={f['score']:.3f}")

        ####################################################
        # TSDF MULTI-VIEW FUSION
        ####################################################

        tsdf = o3d.pipelines.integration.ScalableTSDFVolume(voxel_length=0.005, sdf_trunc=0.03, color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)

        for item in selected_frames:

            rgb = item["rgb"]
            depth = item["depth"].copy()
            mask = item["mask"]

            depth[~mask] = 0

            rgb_o3d = o3d.geometry.Image(rgb.astype(np.uint8))
            depth_o3d = o3d.geometry.Image(depth.astype(np.float32))

            rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d, depth_scale=1.0, depth_trunc=5.0, convert_rgb_to_intensity=False)

            K = item["K"]

            intrinsic = o3d.camera.PinholeCameraIntrinsic(width=rgb.shape[1], height=rgb.shape[0], fx=K["fx"], fy=K["fy"], cx=K["cx"], cy=K["cy"])

            tsdf.integrate(
                rgbd,
                intrinsic,
                np.linalg.inv(item["c2w"])
            )

        mesh = tsdf.extract_triangle_mesh()

        # Guard against empty meshes after TSDF fusion
        if len(mesh.vertices) == 0:
            print("⚠️ Empty TSDF mesh")
            return None
        
        pcd = mesh.sample_points_uniformly(number_of_points=12000)

        pts_world_all = np.asarray(pcd.points)

        # Guard against empty or too small point clouds after TSDF fusion
        if len(pts_world_all) < 500:
            return None

        # Crop with OBB again to remove any stray points outside the object
        mask = self.get_obb_mask(pts_world_all, obb_data)
        pts_world_all = pts_world_all[mask]

        # Guard against empty or too small point clouds after OBB cropping
        if len(pts_world_all) < 500:
            return None

        # ============================================================
        # 4. CANONICALIZATION
        # ============================================================
        # Rotation: selected_box's own annotated CAD alignment
        # (rotate_transform), NOT obb_axes -- obb_axes is a box fit to the
        # visible scan and has no reliable relationship to the CAD's
        # canonical pose (confirmed via exhaustive signed-permutation
        # search finding no match at all).
        #
        # Translation & scale: from obb_data, validated consistent with
        # the CAD's own annotated position (within ~2-10cm across
        # different scenes) once COORD_FIX is applied.
        #
        # COORD_FIX: fixed axis-order difference between obb_data's and
        # transform_dict's world-coordinate serialization. Confirmed
        # identical across >=2 different scenes -- a dataset convention,
        # not a per-object fit.
        COORD_FIX = np.array([
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ], dtype=np.float64)

        translation_anchor = obb_center
        rotate_transform = selected_box.transform_dict["rotate_transform"]

        cad_native_centered = (pts_world_all - translation_anchor) @ COORD_FIX
        pts_t = torch.from_numpy(cad_native_centered).float().unsqueeze(0)
        local_pts = rotate_transform.inverse().transform_points(pts_t).squeeze(0).cpu().numpy()

        rotate_transform = selected_box.transform_dict["rotate_transform"]

        # ============================================================
        # 5. FILTERING
        # ============================================================

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(local_pts)

        pcd = pcd.voxel_down_sample(voxel_size=0.005)

        filtered_pts = np.asarray(pcd.points)

        centroid = filtered_pts.mean(axis=0)
        dist = np.linalg.norm(filtered_pts - centroid, axis=1)

        threshold = np.percentile(dist, 99.5)
        filtered_pts = filtered_pts[dist < threshold]

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(filtered_pts)

        _, ind = pcd.remove_statistical_outlier(
            nb_neighbors=20,
            std_ratio=2.0
        )

        filtered_pts = filtered_pts[ind]

        if len(filtered_pts) < 500:
            return None

        # ============================================================
        # 6. NORMALIZATION -- unaffected by rotation, box scale is fine as-is
        # ============================================================
        scale = np.max(np.asarray(obb_data["axesLengths"], dtype=np.float64))
        clean_pts_canonical = filtered_pts / (scale + 1e-8) * 0.9

        # ============================================================
        # 7. DINO FEATURES
        # ============================================================
        pts_t = torch.from_numpy(filtered_pts).float().unsqueeze(0)
        cad_native_pt = rotate_transform.transform_points(pts_t).squeeze(0).cpu().numpy()
        clean_pts_world_actual = cad_native_pt @ COORD_FIX.T + translation_anchor

        w2c_best = np.linalg.inv(best_c2w)
        uv_continuous = project_world_to_pixel(clean_pts_world_actual, w2c_best, best_K)
        d_feats = get_dino_features_bilinear(self.dino, best_rgb, uv_continuous)

        # ============================================================
        # 8. UTONIA FEATURES & FUSION
        # ============================================================
        # Extract Utonia point features on the canonical object shape.
        s_pts, u_feats, sampled_indices = self._run_utonia_on_points(clean_pts_canonical, num_pts)
        d_feats = d_feats[sampled_indices]

        fused_pointwise = np.hstack([
            (u_feats.detach().cpu().numpy() if hasattr(u_feats, 'detach') else u_feats),
            d_feats
        ])

        return s_pts, u_feats, d_feats, fused_pointwise