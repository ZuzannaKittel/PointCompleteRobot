#!/usr/bin/env python3
"""
Single-object ScanNet++ defence visualization
=============================================

Runs the existing ScanNet++ object-centric pipeline for ONE object only and
saves a sequence of visual intermediate stages for a thesis-defence slide.

Output:
    defence_pipeline_one_object/
      01_rgb.png
      02_sam_mask.png
      03_segmented_single_frame.png
      04_tsdf_fusion.png
      05_filtered_canonical.png
      06_final_partial.png
      07_pipeline_overview.png
      metadata.json
      *.npy                 (intermediate point clouds)

The script reuses the project's existing model initializers and
SceneDataEngine. It never writes training-pair .pt files.
"""

import os
# Prevent Intel/GNU OpenMP runtime collision
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPEN3D_NUM_THREADS"] = "1"

import cv2
cv2.setNumThreads(0)

import sys
import json
import pickle
import glob
import traceback

# Import Open3D before PyTorch CUDA hooks
import open3d as o3d
import torch
import numpy as np
import matplotlib.pyplot as plt

from models.models import init_dino, init_sam, init_utonia

import utils
from data_processing.data_engine_scannet import SceneDataEngine

# Dataset annotation compatibility patch
if hasattr(utils, "get_frame_info"):
    _orig_get_frame_info = utils.get_frame_info

    def secure_get_frame_info(*args, **kwargs):
        pose, intrinsic = _orig_get_frame_info(*args, **kwargs)
        return np.array(pose, dtype=np.float64, order="C"), intrinsic

    utils.get_frame_info = secure_get_frame_info
    if hasattr(utils, "data_engine"):
        utils.data_engine.get_frame_info = secure_get_frame_info

import SCANnotatepp.ScanNetAnnotation as scannet_mod
sys.modules["ScanNetAnnotation"] = scannet_mod


# ============================================================
# USER CONFIGURATION
# ============================================================

# ---- Exact object mode ----
# Put the scene ID and object ID here if you know the exact object.
# Target pair: pair_00020_7b6477cb95_display_29.pt
TARGET_SCENE = "7b6477cb95"
TARGET_OBJECT_ID = "29"

# If False, the script will automatically find the first valid target
# category/object it can reconstruct, then STOP immediately after success.
USE_EXACT_TARGET = True

# Used only when USE_EXACT_TARGET == False.
AUTO_TARGET_CATEGORIES = [
    "chair", "table", "sofa", "cabinet", "bookshelf", "shelf",
    "bed", "bench", "lamp", "display", "monitor", "printer",
    "trash", "bin", "basket", "mug", "bowl", "bottle", "pot"
]

# Number of views passed to TSDF. This follows your current pipeline.
NUM_VIEWS_FOR_TSDF = 6

# Number of points used for Utonia in the final representation.
NUM_FINAL_POINTS = 2048

TSDF_VOXEL_SIZE = 0.005
TSDF_SDF_TRUNC = 0.03
FINAL_VOXEL_SIZE = 0.005

# Output folder.
OUTPUT_DIR = "defence_pipeline_one_object"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Paths matching your dataset-generation script.
BASE_DATA_DIR = "data/ScanNetpp"
SHAPENET_ROOT = "data/ShapeNet/ShapeNet_preprocessed"
UTONIA_CKPT = "checkpoints/utoniadreamer/latest.pth"
SAM_CHECKPOINT = "checkpoints/weights/sam_vit_h_4b8939.pth"


# ============================================================
# BASIC GEOMETRY HELPERS
# ============================================================

def matrix_to_intrinsics(raw_k):
    K_mat = np.asarray(raw_k, dtype=np.float64)
    return {
        "fx": K_mat[0, 0],
        "fy": K_mat[1, 1],
        "cx": K_mat[0, 2],
        "cy": K_mat[1, 2],
    }


def project_world_to_camera(points_world, w2c):
    points_world = np.asarray(points_world)
    if points_world.ndim == 1:
        points_world = points_world[None, :]
    pts_h = np.hstack([
        points_world,
        np.ones((len(points_world), 1), dtype=points_world.dtype)
    ])
    cam = (w2c @ pts_h.T).T
    return cam[:, :3]


def project_to_pixel(points_cam, K):
    z = points_cam[:, 2]
    x = points_cam[:, 0] / (z + 1e-8)
    y = points_cam[:, 1] / (z + 1e-8)
    u = K["fx"] * x + K["cx"]
    v = K["fy"] * y + K["cy"]
    return np.stack([u, v], axis=1)


def select_best_sam_mask(
    masks,
    scores,
    bbox,
    depth=None,
    min_area_ratio=0.005,
    max_area_ratio=0.60,
):
    """
    Select the most geometrically consistent SAM mask.

    The selection combines:
      - SAM confidence
      - overlap with the supplied bounding box
      - penalty for mask leakage outside the box
      - valid-depth ratio
      - connected-component cleanup
    """

    H, W = masks.shape[1:3]

    x1, y1, x2, y2 = bbox.astype(int)

    x1 = np.clip(x1, 0, W - 1)
    x2 = np.clip(x2, 0, W - 1)
    y1 = np.clip(y1, 0, H - 1)
    y2 = np.clip(y2, 0, H - 1)

    bbox_mask = np.zeros((H, W), dtype=bool)
    bbox_mask[y1:y2 + 1, x1:x2 + 1] = True

    bbox_area = bbox_mask.sum()

    candidates = []

    for i, mask in enumerate(masks):

        mask = mask.astype(bool)

        area = mask.sum()
        area_ratio = area / float(H * W)

        # --------------------------------------------------
        # 1. Basic area sanity
        # --------------------------------------------------
        if area == 0:
            continue

        if area_ratio < min_area_ratio:
            continue

        if area_ratio > max_area_ratio:
            continue

        # --------------------------------------------------
        # 2. Bounding-box overlap
        # --------------------------------------------------
        intersection = np.logical_and(mask, bbox_mask).sum()

        mask_area = mask.sum()

        bbox_iou = intersection / (
            mask_area + bbox_area - intersection + 1e-8
        )

        mask_inside_bbox = intersection / (
            mask_area + 1e-8
        )

        bbox_coverage = intersection / (
            bbox_area + 1e-8
        )

        # --------------------------------------------------
        # 3. Penalize leakage outside the projected OBB
        # --------------------------------------------------
        leakage = 1.0 - mask_inside_bbox

        # --------------------------------------------------
        # 4. Depth consistency
        # --------------------------------------------------
        if depth is not None:

            valid_depth = (
                np.isfinite(depth) &
                (depth > 0.1) &
                (depth < 6.0)
            )

            depth_ratio = np.logical_and(
                mask,
                valid_depth
            ).sum() / (mask_area + 1e-8)

        else:
            depth_ratio = 1.0

        # --------------------------------------------------
        # 5. Combined quality score
        # --------------------------------------------------
        quality = (
            0.30 * float(scores[i]) +
            0.30 * bbox_iou +
            0.20 * bbox_coverage +
            0.15 * depth_ratio -
            0.25 * leakage
        )

        candidates.append({
            "index": i,
            "quality": quality,
            "sam_score": float(scores[i]),
            "bbox_iou": bbox_iou,
            "bbox_coverage": bbox_coverage,
            "depth_ratio": depth_ratio,
            "leakage": leakage,
            "area_ratio": area_ratio,
        })

    if not candidates:
        return None, None

    # Best candidate
    candidates.sort(
        key=lambda x: x["quality"],
        reverse=True
    )

    best = candidates[0]

    best_mask = masks[best["index"]].astype(bool)

    # ------------------------------------------------------
    # 6. Keep largest connected component
    # ------------------------------------------------------
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        best_mask.astype(np.uint8),
        connectivity=8
    )

    if num_labels > 1:

        largest_label = 1 + np.argmax(
            stats[1:, cv2.CC_STAT_AREA]
        )

        best_mask = labels == largest_label

    return best_mask, best


# ============================================================
# VIEW SELECTION — SAME LOGIC AS YOUR CURRENT SCRIPT
# ============================================================

def find_multi_view_frames_vectorized(
    world_center,
    camera_json_data,
    scene_path,
    num_frames=5,
    fully_extracted_frames=None,
):
    """Select trajectory-spread frames that see the object center."""

    if fully_extracted_frames is None:
        rgb_dir = os.path.join(scene_path, "iphone", "rgb")
        depth_dir = os.path.join(scene_path, "iphone", "depth")
        if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir):
            return None

        try:
            available_rgb = {
                int(f.split("_")[1].split(".")[0])
                for f in os.listdir(rgb_dir)
                if f.startswith("frame_")
                and (f.endswith(".png") or f.endswith(".jpg"))
            }
            available_depth = {
                int(f.split("_")[1].split(".")[0])
                for f in os.listdir(depth_dir)
                if f.startswith("frame_") and f.endswith(".png")
            }
            fully_extracted_frames = available_rgb.intersection(available_depth)
        except Exception as exc:
            print(f"⚠️ Error parsing disk frames: {exc}")
            return None

    frame_keys = []
    for key in camera_json_data.keys():
        if not key.startswith("frame_"):
            continue
        try:
            frame_num = int(key.split("_")[1])
        except (ValueError, IndexError):
            continue
        if frame_num in fully_extracted_frames:
            frame_keys.append(key)

    if not frame_keys:
        return None

    poses = np.array(
        [camera_json_data[k].get("aligned_pose", np.eye(4)) for k in frame_keys],
        dtype=np.float64,
    )

    cam_positions = poses[:, :3, 3]
    distances = np.linalg.norm(cam_positions - world_center, axis=1)

    in_range = (distances > 0.3) & (distances < 6.0)
    candidate_indices = np.where(in_range)[0]
    if len(candidate_indices) == 0:
        return None

    filtered_poses = poses[candidate_indices]
    filtered_keys = [frame_keys[i] for i in candidate_indices]
    filtered_distances = distances[candidate_indices]

    R = filtered_poses[:, :3, :3]
    t = filtered_poses[:, :3, 3]
    R_T = R.transpose(0, 2, 1)

    cam_pts = (
        np.einsum("mij,j->mi", R_T, world_center)
        - np.einsum("mij,mj->mi", R_T, t)
    )

    valid_z = cam_pts[:, 2] > 0.0
    valid_frames = []

    for idx in np.where(valid_z)[0]:
        key = filtered_keys[idx]
        entry = camera_json_data[key]
        cam_pt = cam_pts[idx]

        raw_k = entry.get("intrinsics") or entry.get("intrinsic")
        if raw_k is None:
            K = np.array([
                [525.0, 0.0, 320.0],
                [0.0, 525.0, 240.0],
                [0.0, 0.0, 1.0],
            ])
        else:
            K = np.array(raw_k)

        u = cam_pt[0] * K[0, 0] / cam_pt[2] + K[0, 2]
        v = cam_pt[1] * K[1, 1] / cam_pt[2] + K[1, 2]

        # Your current script assumes 1920x1440 RGB resolution here.
        if 0 <= u < 1920 and 0 <= v < 1440:
            frame_number = int(key.split("_")[1])
            valid_frames.append((frame_number, filtered_distances[idx]))

    if not valid_frames:
        return None

    valid_frames.sort(key=lambda x: x[0])

    if len(valid_frames) <= num_frames:
        return [f[0] for f in valid_frames]

    indices = np.linspace(
        0,
        len(valid_frames) - 1,
        num_frames,
        dtype=int,
    )

    return [valid_frames[i][0] for i in indices]


# ============================================================
# VISUALIZATION HELPERS
# ============================================================

def save_rgb(rgb, filename):
    path = os.path.join(OUTPUT_DIR, filename)
    rgb = np.asarray(rgb)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    return path


def save_sam_overlay(rgb, mask, bbox=None):
    """Save clean RGB + SAM overlay, optionally with the prompting box."""
    rgb = np.asarray(rgb).copy()
    mask = np.asarray(mask).astype(bool)

    overlay = rgb.copy()
    tint = np.zeros_like(overlay)
    tint[..., 0] = 255
    tint[..., 1] = 70
    tint[..., 2] = 70

    alpha = 0.45
    overlay[mask] = (
        (1.0 - alpha) * overlay[mask]
        + alpha * tint[mask]
    ).astype(np.uint8)

    if bbox is not None:
        x1, y1, x2, y2 = np.round(bbox).astype(int)
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            3,
        )

    return save_rgb(overlay, "02_sam_mask.png")


def save_depth(depth):
    depth = np.asarray(depth, dtype=np.float32)
    valid = depth > 0

    vis = np.zeros_like(depth)
    if np.any(valid):
        lo = np.percentile(depth[valid], 2)
        hi = np.percentile(depth[valid], 98)
        vis = np.clip((depth - lo) / max(hi - lo, 1e-8), 0, 1)

    image = (vis * 255).astype(np.uint8)
    path = os.path.join(OUTPUT_DIR, "03_depth.png")
    cv2.imwrite(path, image)
    return path


def equalize_3d_axes(ax, points):
    points = np.asarray(points)
    mins = points.min(axis=0)
    maxs = points.max(axis=0)
    center = (mins + maxs) / 2.0
    extent = np.max(maxs - mins)
    extent = max(extent, 1e-6)

    ax.set_xlim(center[0] - extent / 2, center[0] + extent / 2)
    ax.set_ylim(center[1] - extent / 2, center[1] + extent / 2)
    ax.set_zlim(center[2] - extent / 2, center[2] + extent / 2)


def save_pointcloud(points, filename, title):
    points = np.asarray(points)
    if len(points) == 0:
        raise ValueError(f"Cannot visualize empty point cloud: {filename}")

    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection="3d")

    ax.scatter(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        s=1.0,
        alpha=0.85,
    )

    ax.view_init(elev=15, azim=-90)
    ax.set_axis_off()
    ax.set_title(title, pad=8)
    equalize_3d_axes(ax, points)

    path = os.path.join(OUTPUT_DIR, filename)
    plt.tight_layout()
    plt.savefig(path, dpi=250, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    return path


def save_pipeline_overview(stages):
    """Save all geometric stages as one horizontal overview image."""
    fig = plt.figure(figsize=(16, 4.2))

    for i, stage in enumerate(stages):
        ax = fig.add_subplot(1, len(stages), i + 1, projection="3d")
        points = np.asarray(stage["points"])

        ax.scatter(
            points[:, 0],
            points[:, 1],
            points[:, 2],
            s=0.75,
            alpha=0.85,
        )

        ax.view_init(elev=15, azim=-90)
        ax.set_axis_off()
        ax.set_title(stage["title"], fontsize=11, pad=5)
        equalize_3d_axes(ax, points)

    plt.tight_layout()

    path = os.path.join(
        OUTPUT_DIR,
        "07_pipeline_overview.png"
    )

    plt.savefig(
        path,
        dpi=250,
        bbox_inches="tight",
        pad_inches=0.03,
    )

    plt.close(fig)
    return path


# ============================================================
# ENGINE HELPERS
# ============================================================


def get_obb_mask(points, obb):
    centroid = np.asarray(obb["centroid"], dtype=np.float64)
    axes = np.asarray(
        obb["normalizedAxes"],
        dtype=np.float64,
    ).reshape(3, 3)

    axes /= np.linalg.norm(
        axes,
        axis=1,
        keepdims=True,
    ) + 1e-8

    extents = np.asarray(
        obb["axesLengths"],
        dtype=np.float64,
    )

    half_extents = extents * 0.5
    pts_local = (points - centroid) @ axes.T

    inside = np.all(
        np.abs(pts_local) <= half_extents,
        axis=1,
    )

    soft = np.linalg.norm(
        np.maximum(
            0,
            np.abs(pts_local) - half_extents,
        ),
        axis=1,
    )

    return (
        inside
        | (soft < 0.15 * np.max(half_extents))
    )


# ============================================================
# MAIN ONE-OBJECT PROCESSOR
# ============================================================


def visualize_object(
    engine,
    scene_path,
    selected_box,
    obb_data,
    frame_indices,
    scene_id,
    obj_id,
):
    """Run the same core pipeline, saving visualization intermediates."""

    from utils.utils import (
        load_data,
        get_dino_features_bilinear,
    )

    scene_root = os.path.dirname(engine.paths["json"])
    frame_clouds = []

    # ----------------------------------------------
    # OBB data
    # ----------------------------------------------
    obb_center = np.asarray(
        obb_data["centroid"],
        dtype=np.float64,
    )

    obb_axes = np.asarray(
        obb_data["normalizedAxes"],
        dtype=np.float64,
    ).reshape(3, 3)

    extents = np.asarray(
        obb_data["axesLengths"],
        dtype=np.float64,
    )

    # ----------------------------------------------
    # Process candidate frames
    # ----------------------------------------------
    for f_idx in frame_indices:
        frame_key = f"frame_{f_idx:06d}"
        print(f"    Processing {frame_key}")

        rgb_path = os.path.join(
            scene_root,
            "rgb",
            f"{frame_key}.jpg",
        )

        dep_path = os.path.join(
            scene_root,
            "depth",
            f"{frame_key}.png",
        )

        if not os.path.exists(dep_path):
            print("      depth missing")
            continue

        rgb, depth = load_data(rgb_path, dep_path)

        if depth.dtype == np.uint16:
            depth = depth.astype(np.float64) / 1000.0

        frame_entry = engine.meta[frame_key]
        c2w = np.asarray(
            frame_entry.get("aligned_pose"),
            dtype=np.float64,
        )
        w2c = np.linalg.inv(c2w)

        raw_k = (
            frame_entry.get("intrinsics")
            or frame_entry.get("intrinsic")
        )
        K = matrix_to_intrinsics(raw_k)

        # ------------------------------------------
        # OBB -> 2D bbox -> SAM
        # ------------------------------------------
        engine.sam.set_image(rgb)

        half_extents = extents * 0.5
        local_corners = np.array([
            [x, y, z]
            for x in [-half_extents[0], half_extents[0]]
            for y in [-half_extents[1], half_extents[1]]
            for z in [-half_extents[2], half_extents[2]]
        ], dtype=np.float64)

        normalized_axes = obb_axes / (
            np.linalg.norm(
                obb_axes,
                axis=1,
                keepdims=True,
            ) + 1e-8
        )

        world_corners = (
            local_corners @ normalized_axes.T
            + obb_center
        )

        corners_cam = project_world_to_camera(
            world_corners,
            w2c,
        )

        in_front = corners_cam[:, 2] > 0.05
        if np.count_nonzero(in_front) < 4:
            continue

        pixels = project_to_pixel(
            corners_cam[in_front],
            K,
        ).astype(np.float32)

        img_h, img_w = rgb.shape[:2]
        pad = 0.1

        x_min = np.min(pixels[:, 0])
        y_min = np.min(pixels[:, 1])
        x_max = np.max(pixels[:, 0])
        y_max = np.max(pixels[:, 1])

        dx = (x_max - x_min) * pad
        dy = (y_max - y_min) * pad

        bbox_2d = np.array([
            np.clip(x_min - dx, 0, img_w - 1),
            np.clip(y_min - dy, 0, img_h - 1),
            np.clip(x_max + dx, 0, img_w - 1),
            np.clip(y_max + dy, 0, img_h - 1),
        ], dtype=np.float32)

        print("      SAM...")

        masks, scores, _ = engine.sam.predict(
            box=bbox_2d,
            multimask_output=True
        )

        mask_2d, mask_quality = select_best_sam_mask(
            masks,
            scores,
            bbox_2d,
            depth=depth
        )

        if mask_2d is None:
            print("⚠️ No SAM candidate passed quality checks.")
            continue

        print(
            f"   ✓ SAM mask selected | "
            f"quality={mask_quality['quality']:.3f} | "
            f"IoU={mask_quality['bbox_iou']:.3f} | "
            f"leakage={mask_quality['leakage']:.3f} | "
            f"depth={mask_quality['depth_ratio']:.3f}"
        )

        if mask_2d.shape != depth.shape:
            mask_2d = cv2.resize(
                mask_2d.astype(np.uint8),
                (depth.shape[1], depth.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ) > 0

        semantic_valid = (
            (mask_2d > 0)
            & (depth > 0.1)
            & (depth < 6.0)
        )

        if np.count_nonzero(semantic_valid) < 300:
            continue

        # ------------------------------------------
        # Back-project masked depth into world space
        # ------------------------------------------
        z_cam = depth[semantic_valid]
        h_dep, w_dep = depth.shape[:2]
        v_grid, u_grid = np.indices((h_dep, w_dep))
        u_valid = u_grid[semantic_valid]
        v_valid = v_grid[semantic_valid]

        scale_x = w_dep / img_w
        scale_y = h_dep / img_h

        fx_depth = K["fx"] * scale_x
        fy_depth = K["fy"] * scale_y
        cx_depth = K["cx"] * scale_x
        cy_depth = K["cy"] * scale_y

        x_cam = (
            (u_valid - cx_depth)
            * z_cam
            / fx_depth
        )

        y_cam = (
            (v_valid - cy_depth)
            * z_cam
            / fy_depth
        )

        pts_cam = np.stack([
            x_cam,
            y_cam,
            z_cam,
            np.ones_like(z_cam),
        ], axis=-1)

        pts_world = (
            c2w @ pts_cam.T
        ).T[:, :3]

        crop_mask = get_obb_mask(
            pts_world,
            obb_data,
        )

        pts_cropped = pts_world[crop_mask]

        frame_pass_rate = (
            len(pts_cropped)
            / max(len(pts_world), 1)
        )

        if frame_pass_rate < 0.2:
            continue

        if len(pts_cropped) < 80:
            continue

        coverage = len(pts_cropped)
        distance = np.median(depth[semantic_valid])
        score = coverage / (distance + 1e-6)

        frame_clouds.append({
            "frame": frame_key,
            "rgb": rgb.copy(),
            "depth": depth.copy(),
            "mask": mask_2d.copy(),
            "semantic_valid": semantic_valid.copy(),
            "raw_points": pts_world.copy(),
            "cropped_points": pts_cropped.copy(),
            "score": float(score),
            "bbox": bbox_2d.copy(),
            "K": K,
            "c2w": c2w.copy(),
            "w2c": w2c.copy(),
        })

    if len(frame_clouds) < 2:
        raise RuntimeError(
            f"Only {len(frame_clouds)} usable frames; need at least 2."
        )

    # ----------------------------------------------
    # Rank and select views
    # ----------------------------------------------
    frame_clouds.sort(
        key=lambda item: item["score"],
        reverse=True,
    )

    selected_frames = frame_clouds[:NUM_VIEWS_FOR_TSDF]

    print("\nSelected frames:")
    for item in selected_frames:
        print(
            f"  {item['frame']}"
            f" | score={item['score']:.2f}"
            f" | points={len(item['cropped_points'])}"
        )

    # ----------------------------------------------
    # Pick the highest-quality selected frame for the
    # 2D visualizations.
    # ----------------------------------------------
    representative = selected_frames[0]

    save_rgb(
        representative["rgb"],
        "01_rgb.png",
    )

    save_sam_overlay(
        representative["rgb"],
        representative["mask"],
        representative["bbox"],
    )

    save_depth(
        representative["depth"]
    )

    save_pointcloud(
        representative["cropped_points"],
        "03_segmented_single_frame.png",
        "Single-frame segmented object",
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "segmented_single_frame.npy",
        ),
        representative["cropped_points"],
    )

    # =======================================================
    # TSDF FUSION
    # =======================================================
    print("\n[4] Multi-view TSDF fusion...")

    tsdf = (
        o3d.pipelines.integration
        .ScalableTSDFVolume(
            voxel_length=TSDF_VOXEL_SIZE,
            sdf_trunc=TSDF_SDF_TRUNC,
            color_type=(
                o3d.pipelines.integration
                .TSDFVolumeColorType.RGB8
            ),
        )
    )

    for item in selected_frames:
        rgb = item["rgb"]
        depth = item["depth"].copy()
        mask = item["mask"]
        depth[~mask] = 0

        rgb_o3d = o3d.geometry.Image(
            rgb.astype(np.uint8)
        )
        depth_o3d = o3d.geometry.Image(
            depth.astype(np.float32)
        )

        rgbd = (
            o3d.geometry.RGBDImage
            .create_from_color_and_depth(
                rgb_o3d,
                depth_o3d,
                depth_scale=1.0,
                depth_trunc=5.0,
                convert_rgb_to_intensity=False,
            )
        )

        K = item["K"]
        intrinsic = (
            o3d.camera.PinholeCameraIntrinsic(
                width=rgb.shape[1],
                height=rgb.shape[0],
                fx=K["fx"],
                fy=K["fy"],
                cx=K["cx"],
                cy=K["cy"],
            )
        )

        tsdf.integrate(
            rgbd,
            intrinsic,
            np.linalg.inv(item["c2w"]),
        )

    mesh = tsdf.extract_triangle_mesh()

    if len(mesh.vertices) == 0:
        raise RuntimeError("Empty TSDF mesh.")

    pcd = mesh.sample_points_uniformly(
        number_of_points=6000
    )

    pts_world_all = np.asarray(pcd.points)

    if len(pts_world_all) < 100:
        raise RuntimeError("TSDF point cloud too small.")

    crop_mask = get_obb_mask(
        pts_world_all,
        obb_data,
    )
    pts_world_all = pts_world_all[crop_mask]

    if len(pts_world_all) < 80:
        raise RuntimeError(
            "TSDF point cloud too small after OBB crop."
        )

    save_pointcloud(
        pts_world_all,
        "04_tsdf_fusion.png",
        f"TSDF fusion ({len(selected_frames)} views)",
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "tsdf_points_world.npy",
        ),
        pts_world_all,
    )

    # =======================================================
    # CANONICALIZATION — SAME TRANSFORM LOGIC AS YOUR CODE
    # =======================================================
    print("[5] Canonicalization...")

    COORD_FIX = np.array([
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
    ], dtype=np.float64)

    translation_anchor = obb_center
    rotate_transform = (
        selected_box
        .transform_dict["rotate_transform"]
    )

    cad_native_centered = (
        pts_world_all - translation_anchor
    ) @ COORD_FIX

    pts_t = (
        torch
        .from_numpy(cad_native_centered)
        .float()
        .unsqueeze(0)
    )

    local_pts = (
        rotate_transform
        .inverse()
        .transform_points(pts_t)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    # =======================================================
    # FILTERING
    # =======================================================
    print("[6] Filtering...")

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(local_pts)

    pcd = pcd.voxel_down_sample(
        voxel_size=FINAL_VOXEL_SIZE
    )

    filtered_pts = np.asarray(pcd.points)

    if len(filtered_pts) == 0:
        raise RuntimeError("Empty cloud after voxel filtering.")

    centroid = filtered_pts.mean(axis=0)
    dist = np.linalg.norm(
        filtered_pts - centroid,
        axis=1,
    )

    threshold = np.percentile(
        dist,
        99.5,
    )

    filtered_pts = filtered_pts[
        dist < threshold
    ]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(
        filtered_pts
    )

    _, ind = pcd.remove_statistical_outlier(
        nb_neighbors=20,
        std_ratio=2.0,
    )

    filtered_pts = filtered_pts[ind]

    if len(filtered_pts) < 80:
        raise RuntimeError(
            "Too few points after statistical filtering."
        )

    # Same isotropic scale as your current code.
    scale = np.max(
        np.asarray(
            obb_data["axesLengths"],
            dtype=np.float64,
        )
    )

    clean_pts_canonical = (
        filtered_pts
        / (scale + 1e-8)
        * 0.9
    )

    save_pointcloud(
        filtered_pts,
        "05_filtered_canonical.png",
        "Filtered + canonicalized",
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "filtered_canonical_points.npy",
        ),
        filtered_pts,
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "clean_points_canonical.npy",
        ),
        clean_pts_canonical,
    )

    # =======================================================
    # DINO + UTONIA + FUSION
    # =======================================================
    print("[7] DINOv2 + Utonia features...")

    # Map canonical/filtered points back to world coordinates
    # so DINO can be projected into the representative RGB frame.
    pts_t = (
        torch
        .from_numpy(filtered_pts)
        .float()
        .unsqueeze(0)
    )

    cad_native_pt = (
        rotate_transform
        .transform_points(pts_t)
        .squeeze(0)
        .cpu()
        .numpy()
    )

    clean_pts_world_actual = (
        cad_native_pt @ COORD_FIX.T
        + translation_anchor
    )

    w2c_best = representative["w2c"]

    pts_cam_best = project_world_to_camera(
        clean_pts_world_actual,
        w2c_best,
    )

    uv_continuous = project_to_pixel(
        pts_cam_best,
        representative["K"],
    )

    d_feats = get_dino_features_bilinear(
        engine.dino,
        representative["rgb"],
        uv_continuous,
    )

    # Use the engine's own Utonia helper, preserving your normal feature path.
    sampled_pts, u_feats, sampled_indices = (
        engine._run_utonia_on_points(
            clean_pts_canonical,
            NUM_FINAL_POINTS,
        )
    )

    d_feats = d_feats[sampled_indices]

    if hasattr(u_feats, "detach"):
        u_feats_np = (
            u_feats
            .detach()
            .cpu()
            .numpy()
        )
    else:
        u_feats_np = np.asarray(u_feats)

    d_feats = np.asarray(d_feats)
    fused_pointwise = np.hstack([
        u_feats_np,
        d_feats,
    ])

    print(
        f"    Utonia: {u_feats_np.shape}"
    )
    print(
        f"    DINOv2: {d_feats.shape}"
    )
    print(
        f"    Fused:  {fused_pointwise.shape}"
    )

    # Final sampled canonical point cloud.
    save_pointcloud(
        sampled_pts,
        "06_final_partial.png",
        "Final canonical partial observation",
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "final_partial_points.npy",
        ),
        sampled_pts,
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "utonia_features.npy",
        ),
        u_feats_np,
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "dino_features.npy",
        ),
        d_feats,
    )

    np.save(
        os.path.join(
            OUTPUT_DIR,
            "fused_features.npy",
        ),
        fused_pointwise,
    )

    # =======================================================
    # FINAL OVERVIEW
    # =======================================================
    save_pipeline_overview([
        {
            "points": representative["cropped_points"],
            "title": "Single view",
        },
        {
            "points": pts_world_all,
            "title": "TSDF fusion",
        },
        {
            "points": filtered_pts,
            "title": "Canonical + filtered",
        },
        {
            "points": sampled_pts,
            "title": "Final partial",
        },
    ])

    # =======================================================
    # METADATA
    # =======================================================
    metadata = {
        "scene_id": scene_id,
        "object_id": obj_id,
        "category": str(
            getattr(
                selected_box,
                "category_label",
                "unknown",
            )
        ),
        "representative_frame": representative["frame"],
        "candidate_frames": len(frame_clouds),
        "selected_frames": [
            x["frame"] for x in selected_frames
        ],
        "selected_frame_scores": {
            x["frame"]: x["score"]
            for x in selected_frames
        },
        "tsdf_points_after_obb": int(len(pts_world_all)),
        "filtered_points": int(len(filtered_pts)),
        "final_points": int(len(sampled_pts)),
        "utonia_dim": int(u_feats_np.shape[-1]),
        "dino_dim": int(d_feats.shape[-1]),
        "fused_dim": int(fused_pointwise.shape[-1]),
        "tsdf_voxel_size": TSDF_VOXEL_SIZE,
        "tsdf_sdf_trunc": TSDF_SDF_TRUNC,
        "final_voxel_size": FINAL_VOXEL_SIZE,
    }

    with open(
        os.path.join(OUTPUT_DIR, "metadata.json"),
        "w",
    ) as f:
        json.dump(metadata, f, indent=2)

    print("\n============================================================")
    print("SUCCESS — ONE OBJECT VISUALIZED")
    print("============================================================")
    print(f"Scene:       {scene_id}")
    print(f"Object:      {obj_id}")
    print(f"Frames used: {len(selected_frames)}")
    print(f"Final pts:   {len(sampled_pts)}")
    print(f"Output:      {OUTPUT_DIR}")
    print("============================================================\n")

    return True


# ============================================================
# INITIALIZATION
# ============================================================


def initialize_models():
    print("\nInitializing foundation models...")

    dino = init_dino()
    uto = init_utonia(
        ckpt_path=UTONIA_CKPT
    )
    sam = init_sam(
        SAM_CHECKPOINT
    )

    return dino, uto, sam


# ============================================================
# ONE-OBJECT DRIVER
# ============================================================


def main():

    dino, uto, sam = initialize_models()

    scene_data_dir = os.path.join(
        BASE_DATA_DIR,
        "data/data",
    )

    scene_paths = [
        os.path.join(scene_data_dir, entry.name)
        for entry in os.scandir(scene_data_dir)
        if entry.is_dir()
    ] if os.path.isdir(scene_data_dir) else []

    if USE_EXACT_TARGET:
        if TARGET_SCENE == "YOUR_SCENE_ID" or TARGET_OBJECT_ID == "YOUR_OBJECT_ID":
            raise RuntimeError(
                "Set TARGET_SCENE and TARGET_OBJECT_ID, "
                "or set USE_EXACT_TARGET = False."
            )

        scene_paths = [
            p for p in scene_paths
            if os.path.basename(p) == TARGET_SCENE
        ]

        if not scene_paths:
            raise RuntimeError(
                f"Target scene not found: {TARGET_SCENE}"
            )

    # This is intentionally ONE successful object only.
    for scene_path in scene_paths:

        scene_id = os.path.basename(scene_path)
        print("\n" + "=" * 70)
        print(f"SCENE: {scene_id}")
        print("=" * 70)

        iphone_dir = os.path.join(
            scene_path,
            "iphone"
        )

        if not os.path.isdir(iphone_dir):
            print("No iphone directory; skipping.")
            continue

        json_path = os.path.join(
            scene_path,
            "iphone/pose_intrinsic_imu.json"
        )

        pkl_path = os.path.join(
            BASE_DATA_DIR,
            f"annotations/{scene_id}/{scene_id}.pkl"
        )

        if not os.path.exists(json_path):
            print("Missing camera JSON; skipping.")
            continue

        if not os.path.exists(pkl_path):
            print("Missing annotation PKL; skipping.")
            continue

        # Frame availability
        rgb_dir = os.path.join(
            scene_path,
            "iphone/rgb"
        )
        depth_dir = os.path.join(
            scene_path,
            "iphone/depth"
        )

        try:
            available_rgb = {
                int(f.split("_")[1].split(".")[0])
                for f in os.listdir(rgb_dir)
                if f.startswith("frame_")
                and (f.endswith(".png") or f.endswith(".jpg"))
            }
            available_depth = {
                int(f.split("_")[1].split(".")[0])
                for f in os.listdir(depth_dir)
                if f.startswith("frame_") and f.endswith(".png")
            }
            fully_extracted_frames = (
                available_rgb & available_depth
            )
        except Exception as exc:
            print(f"Frame scan failed: {exc}")
            continue

        with open(json_path, "r") as f:
            camera_json_data = json.load(f)

        try:
            with open(pkl_path, "rb") as f:
                annotation_obj = pickle.load(f)
        except Exception as exc:
            print(f"Annotation load failed: {exc}")
            continue

        paths = {
            "json": json_path,
            "pkl": pkl_path,
            "shapenet": SHAPENET_ROOT,
        }

        engine = SceneDataEngine(
            paths=paths,
            sam=sam,
            dino=dino,
            utonia=uto,
        )

        print(
            f"Annotated objects: "
            f"{len(annotation_obj.obj_annotation_list)}"
        )

        for obj in annotation_obj.obj_annotation_list:

            raw_category = str(
                getattr(
                    obj,
                    "category_label",
                    getattr(
                        obj,
                        "scannet_category_label",
                        ""
                    ),
                )
            ).lower()

            obj_id = str(
                getattr(
                    obj,
                    "object_id",
                    "unknown"
                )
            )

            # --------------------------------------------------
            # Exact-object guard
            # --------------------------------------------------
            if USE_EXACT_TARGET:
                if obj_id != str(TARGET_OBJECT_ID):
                    continue
            else:
                if not any(
                    category in raw_category
                    for category in AUTO_TARGET_CATEGORIES
                ):
                    continue

            annot_dict = getattr(
                obj,
                "scan2cad_annotation_dict",
                {}
            )

            obb_data = annot_dict.get("obb")

            if obb_data is None:
                print(
                    f"Object {obj_id}: missing OBB; skipping."
                )
                continue

            world_center = np.asarray(
                obb_data["centroid"],
                dtype=np.float64,
            )

            print(
                f"\nTARGET CANDIDATE: "
                f"object={obj_id}, category={raw_category}"
            )

            # --------------------------------------------------
            # Select candidate views
            # --------------------------------------------------
            sweep_frames = find_multi_view_frames_vectorized(
                world_center,
                camera_json_data,
                scene_path=scene_path,
                num_frames=5,
                fully_extracted_frames=fully_extracted_frames,
            )

            if not sweep_frames or len(sweep_frames) < 3:
                print(
                    "Insufficient valid frames; skipping object."
                )
                continue

            print(
                f"Frame candidates: {sweep_frames}"
            )

            # --------------------------------------------------
            # RUN VISUALIZATION
            # --------------------------------------------------
            try:
                success = visualize_object(
                    engine=engine,
                    scene_path=scene_path,
                    selected_box=obj,
                    obb_data=obb_data,
                    frame_indices=sweep_frames,
                    scene_id=scene_id,
                    obj_id=obj_id,
                )
            except Exception as exc:
                print(
                    f"\n❌ Visualization failed for "
                    f"scene={scene_id}, object={obj_id}: {exc}"
                )
                traceback.print_exc()
                continue

            if success:
                # HARD STOP: do not process another object.
                return

    raise RuntimeError(
        "No successfully visualized object was found."
    )


if __name__ == "__main__":
    main()
