import os
# 1. Prevent the Intel/GNU OpenMP runtime collision from crashing the process
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPEN3D_NUM_THREADS"] = "1"

import sys
import json
import pickle
import glob
# 2. CRITICAL: Import Open3D BEFORE PyTorch sets up its CUDA hooks
import open3d as o3d
import torch
import numpy as np
from tqdm import tqdm

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from models.models import init_dino, init_sam, init_utonia

import utils
from utils.data_engine import SceneDataEngine

# 3. Dynamic Safety Patch: Force any sub-module matrix extractions to Float64
if hasattr(utils, 'get_frame_info'):
    _orig_get_frame_info = utils.get_frame_info
    def secure_get_frame_info(*args, **kwargs):
        pose, intrinsic = _orig_get_frame_info(*args, **kwargs)
        return np.array(pose, dtype=np.float64, order='C'), intrinsic
    utils.get_frame_info = secure_get_frame_info
    if hasattr(utils, 'data_engine'):
        utils.data_engine.get_frame_info = secure_get_frame_info

import SCANnotatepp.ScanNetAnnotation as scannet_mod
sys.modules['ScanNetAnnotation'] = scannet_mod


def is_visible(world_pt, pose, K, img_shape):
    # 1. Project to camera space
    # Ensure world_pt is [x, y, z, 1]
    world_pt_h = np.append(world_pt, 1.0)
    w2c = np.linalg.inv(pose)
    cam_pt = w2c @ world_pt_h
    
    # Check if point is behind the camera
    if cam_pt[2] <= 0:
        return False
        
    # 2. Project to pixel space
    u = (cam_pt[0] * K["fx"] / cam_pt[2]) + K["cx"]
    v = (cam_pt[1] * K["fy"] / cam_pt[2]) + K["cy"]
    
    # 3. Check bounds
    return (0 <= u < img_shape[1]) and (0 <= v < img_shape[0])

def find_multi_view_frames(world_center, camera_json_data, scene_root, num_frames=5):
    # --- DEBUGGING LINE ---
    print(f"DEBUG: Input camera_json_data has {len(camera_json_data)} frames.")
    print(f"DEBUG: Processing object at {world_center}")
    valid_frames = []

    for frame_key, entry in camera_json_data.items():
        if not frame_key.startswith("frame_"): continue
        
        # 1. Get Camera Pose
        pose = np.array(entry.get('aligned_pose', np.eye(4)))
        # 2. Get Intrinsics (Need to load K)
        # Inside find_multi_view_frames, retrieve the frame's specific intrinsics
        # Retrieve the raw intrinsic matrix from JSON
        # Usually it's either named 'intrinsics' or 'intrinsic'
        raw_k = entry.get('intrinsics') or entry.get('intrinsic')
        
        if raw_k is None:
            # Fallback if not found in frame (some datasets store it in a global header)
            # Or just use your default
            K_dict = {'fx': 525, 'fy': 525, 'cx': 320, 'cy': 240}
        else:
            # 2. Convert to numpy array to handle list-of-lists or flat list formats
            K_mat = np.array(raw_k)
            
            # 3. Extract the components (assuming 3x3 or 4x4 matrix)
            # K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            K_dict = {
                'fx': K_mat[0, 0],
                'fy': K_mat[1, 1],
                'cx': K_mat[0, 2],
                'cy': K_mat[1, 2]
            }
        
        # 3. Check Distance AND Visibility
        cam_position = pose[:3, 3] # Camera position is the right-most column
        dist = np.linalg.norm(cam_position - world_center)
        if 0.3 < dist < 6.0: # Only consider frames where the camera is reasonably close to the object
            # ONLY include if the camera is actually looking at the object center
            if is_visible(world_center, pose, K_dict, (1440, 1920)): # Adjust resolution
                valid_frames.append((int(frame_key.split("_")[1]), dist))
                
    if not valid_frames: return None

    # 2. Sort by frame index
    valid_frames.sort(key=lambda x: x[0])
    
    # 3. DIVERSITY SAMPLING:
    # Instead of focusing on the 'center_idx' (the closest approach),
    # we take the full range of 'valid_frames' and pick points evenly
    # distributed throughout the whole duration of visibility.
    
    if len(valid_frames) <= num_frames:
        return [f[0] for f in valid_frames]
    
    # Use linspace to pick N frames spread evenly across the entire list
    # This guarantees we don't just get a cluster from the 'closest' moment
    indices = np.linspace(0, len(valid_frames) - 1, num_frames, dtype=int)
    sampled_frames = [valid_frames[i][0] for i in indices]
    
    return sampled_frames


def find_multi_view_frames_vectorized(world_center, camera_json_data, scene_path, num_frames=5):
    """ 
    Vectorized version of find_multi_view_frames. 
    This is a critical function that is called for every single object during dataset generation, 
    so optimizing it can have a huge impact on overall processing time. 
    The original version iterates through each frame sequentially, 
    which can be very slow when there are hundreds of frames per scene.
    Ensure the frames are actually extracted on disk before proceeding with visibility checks. 
    """
     
    # 1. Identify which frames are ACTUALLY fully extracted on disk
    rgb_dir = os.path.join(scene_path, "iphone", "rgb")
    depth_dir = os.path.join(scene_path, "iphone", "depth")
    
    if not os.path.exists(rgb_dir) or not os.path.exists(depth_dir):
        return None
        
    # Scan directories once to build a lightning-fast O(1) look-up set of frame IDs
    try:
        # Check for both .jpg and .png in the RGB folder!
        available_rgb = {
            int(f.split('_')[1].split('.')[0]) 
            for f in os.listdir(rgb_dir) 
            if f.startswith("frame_") and (f.endswith(".png") or f.endswith(".jpg"))
        }
        
        # Depth maps are basically always .png
        available_depth = {
            int(f.split('_')[1].split('.')[0]) 
            for f in os.listdir(depth_dir) 
            if f.startswith("frame_") and f.endswith(".png")
        }
        
        # A frame must have BOTH RGB and Depth present to be valid
        fully_extracted_frames = available_rgb.intersection(available_depth)
        
    except Exception as e:
        print(f"   ⚠️ Error parsing disk frames: {e}")
        return None

    # 2. Filter JSON keys: Only keep frames that are valid in metadata AND exist on disk
    frame_keys = []
    for k in camera_json_data.keys():
        if k.startswith("frame_"):
            try:
                f_num = int(k.split("_")[1])
                if f_num in fully_extracted_frames:
                    frame_keys.append(k)
            except (ValueError, IndexError):
                continue
                
    if not frame_keys:
        return None
        
    # 3. Extract all poses for existing frames into array of shape (N, 4, 4)
    poses = np.array([camera_json_data[k].get('aligned_pose', np.eye(4)) for k in frame_keys], dtype=np.float64)
    
    # 4. Vectorized distance calculation
    cam_positions = poses[:, :3, 3] 
    distances = np.linalg.norm(cam_positions - world_center, axis=1)
    
    # Apply distance threshold filter
    in_range_mask = (distances > 0.3) & (distances < 6.0)
    candidate_indices = np.where(in_range_mask)[0]
    
    if len(candidate_indices) == 0:
        return None
        
    filtered_poses = poses[candidate_indices]
    filtered_keys = [frame_keys[idx] for idx in candidate_indices]
    filtered_distances = distances[candidate_indices]
    
    # 5. Apply the Rigid Inverse Trick in Batch
    R = filtered_poses[:, :3, :3]
    t = filtered_poses[:, :3, 3]
    R_T = R.transpose(0, 2, 1)
    
    cam_pts = np.einsum('mij,j->mi', R_T, world_center) - np.einsum('mij,mj->mi', R_T, t)
    
    # 6. Check Visibility (Z > 0)
    valid_z_mask = cam_pts[:, 2] > 0.0
    valid_frames = []
    
    for idx in np.where(valid_z_mask)[0]:
        frame_key = filtered_keys[idx]
        entry = camera_json_data[frame_key]
        cam_pt = cam_pts[idx]
        
        raw_k = entry.get('intrinsics') or entry.get('intrinsic')
        if raw_k is None:
            K_mat = np.array([[525.0, 0.0, 320.0], [0.0, 525.0, 240.0], [0.0, 0.0, 1.0]])
        else:
            K_mat = np.array(raw_k)
            
        u = (cam_pt[0] * K_mat[0, 0] / cam_pt[2]) + K_mat[0, 2]
        v = (cam_pt[1] * K_mat[1, 1] / cam_pt[2]) + K_mat[1, 2]
        
        if (0 <= u < 1920) and (0 <= v < 1440):
            frame_number = int(frame_key.split("_")[1])
            valid_frames.append((frame_number, filtered_distances[idx]))
            
    if not valid_frames:
        return None
        
    # 7. Uniform Trajectory Sampling
    valid_frames.sort(key=lambda x: x[0])
    if len(valid_frames) <= num_frames:
        return [f[0] for f in valid_frames]
        
    indices = np.linspace(0, len(valid_frames) - 1, num_frames, dtype=int)
    return [valid_frames[i][0] for i in indices]


def generate_universal_dataset(base_data_dir, output_dir, target_categories, checkpoint_path):
    PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    absolute_output_dir = os.path.join(PROJECT_ROOT, output_dir, "train")
    os.makedirs(absolute_output_dir, exist_ok=True)

    print("🤖 Initializing Foundation Backbones...")
    dino = init_dino()
    uto = init_utonia(ckpt_path=checkpoint_path)
    sam = init_sam(os.path.join(PROJECT_ROOT, "checkpoints/weights/sam_vit_h_4b8939.pth"))

    scene_paths = glob.glob(os.path.join(PROJECT_ROOT, base_data_dir, "data/data/*"))
    pair_counter = 0

    for scene_path in tqdm(scene_paths, desc="Processing Scenes"):
        if not os.path.isdir(scene_path): 
            continue

        scene_id = os.path.basename(scene_path)

        # This skips the ghost folders (which does not contain "iphone" folder) 
        # instantly before printing any debug statements
        if not os.path.isdir(os.path.join(scene_path, "iphone")):
            continue

        paths = {
            "json": os.path.join(scene_path, "iphone/pose_intrinsic_imu.json"),
            "pkl": os.path.join(PROJECT_ROOT, base_data_dir, f"annotations/{scene_id}/{scene_id}.pkl"),
            "shapenet": os.path.join(PROJECT_ROOT, "data/ShapeNet/ShapeNet_preprocessed")
        }

        # --- FOR DEBUGGING ---
        print(f"\nDEBUG checking scene: {scene_id}")
        print(f" -> Looking for JSON at: {paths['json']} (Exists: {os.path.exists(paths['json'])})")
        print(f" -> Looking for PKL at: {paths['pkl']} (Exists: {os.path.exists(paths['pkl'])})")

        if not os.path.exists(paths["pkl"]) or not os.path.exists(paths["json"]): 
            continue

        with open(paths["json"], 'r') as f: 
            camera_json_data = json.load(f)
            
        # GUARD 1: Prevent crash from truncated/corrupted pickle files
        try:
            with open(paths["pkl"], "rb") as f: 
                annotation_obj = pickle.load(f)
        except Exception as e:
            print(f"\n❌ [SKIPPING SCENE] Annotation file for {scene_id} is corrupted: {e}")
            continue

        engine = SceneDataEngine(
            paths=paths, 
            sam=sam, 
            dino=dino, 
            utonia=uto)

        print(f"\n📋 Scene '{scene_id}' contains {len(annotation_obj.obj_annotation_list)} total annotated objects.")

        for obj in annotation_obj.obj_annotation_list:
            raw_category = str(getattr(obj, 'category_label', getattr(obj, 'scannet_category_label', ''))).lower()
            obj_id = getattr(obj, 'object_id', 'unknown')
            
            if not any(tc in raw_category for tc in target_categories): 
                continue

            # --- 1. PRE-CALCULATE AND INITIALIZE ---
            annot_dict = getattr(obj, 'scan2cad_annotation_dict', {})
            obb_data = annot_dict.get('obb')
            
            T_obj = obj.transform3d.get_matrix()[0].detach().cpu().numpy()
            T_obj = np.array(T_obj, dtype=np.float64, order='C')
            world_center = T_obj[3, :3] 

            if obb_data is None:
                print(f"   ↳ ID {obj_id}: ⚠️ OBB missing from scan2cad dict.")
                continue
            else:
                world_center = np.array(obb_data['centroid'])

            print(f"DEBUG: Processing object ID {obj_id} at {world_center}")

            # --- 2. SAMPLING ---
            # GUARD 2: Pass scene_path to ensure we only select frames that actually exist on disk
            sweep_frames = find_multi_view_frames_vectorized(
                world_center, 
                camera_json_data, 
                scene_path=scene_path, 
                num_frames=5
            )

            # GUARD 3: Ensure we have enough real frames to perform a valid TSDF fusion
            if not sweep_frames or len(sweep_frames) < 3:
                print(f"   ↳ ID {obj_id} ({raw_category}): ⚠️ Skipped - Insufficient valid frames on disk.")
                continue
            else: 
                print(f"   ↳ ID {obj_id} ({raw_category}): Found {len(sweep_frames)} valid frames on disk: {sweep_frames}")

            # --- 3. EXECUTION ---
            try:
                s_pts, u_feats, d_feats, fused_pointwise, _ = engine.get_multi_view_tsdf_object(
                    sweep_frames, 
                    T_obj, 
                    obb_data, 
                    num_pts=2048 
                )
                print(f"   ☑️ Partial point cloud extracted with {s_pts.shape[0]} points.")
            except Exception as e:
                print(f"   ❌ Extraction failed for ID {obj_id} at {world_center}: {e}")
                continue

            # Harvest Ground Truth
            #gt_points = engine.get_ground_truth(paths["pkl"], paths["shapenet"], obj_id, raw_category, obb_data)
            gt_points = engine.get_ground_truth(obj, paths["shapenet"], raw_category, obb_data)
            if gt_points is None or len(gt_points) == 0:
                print(f"      ⚠️ Skipping: ShapeNet CAD model missing.")
                continue

            # Complete package saved as a single .pt file for easy loading in training scripts
            data_pair = {
                "scene_id": scene_id,
                "object_id": obj_id,
                "category": raw_category,
                "partial_pts": torch.from_numpy(s_pts).half(), # Save space by using half precision for the raw points
                "partial_feats": torch.from_numpy(fused_pointwise).half(), # Save space by using half precision for the raw features
                "gt_pts": torch.from_numpy(gt_points).half(), # Save space by using half precision for the GT points
                "anchor_world": torch.from_numpy(world_center).float() # Keep anchor float32 for precision
            }

            save_filename = os.path.join(
                absolute_output_dir, f"pair_{pair_counter:05d}_{scene_id}_{raw_category}_{obj_id}.pt"
            )
            torch.save(data_pair, save_filename)
            pair_counter += 1
            print(f"      ✅ Saved training package: {os.path.basename(save_filename)}")

    print(f"\n🎉 Generation complete! Successfully produced {pair_counter} perfect canonical training pairs.")


if __name__ == "__main__":
    CKPT_PATH = os.path.join(PROJECT_ROOT, "checkpoints/utoniadreamer/latest.pth")
    TARGETS = ["desk", "chair", "table", "sofa", "furniture", "workstation", "bin"]
    
    generate_universal_dataset(
        base_data_dir="data/ScanNetpp", 
        output_dir="data/geometric_pairs_dataset", 
        target_categories=TARGETS,
        checkpoint_path=CKPT_PATH
    )