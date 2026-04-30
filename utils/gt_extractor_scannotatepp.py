import pickle
import trimesh
import numpy as np
import torch
import os
import sys
import SCANnotatepp.ScanNetAnnotation as scannet_mod

sys.modules['ScanNetAnnotation'] = scannet_mod

def extract_scannotate_gt(pkl_path, shapenet_root, anchor_point, device="cuda"):
    """
    Finds the CAD model closest to the user's 3D anchor and returns its points.
    """
    print(f"🔍 Searching SCANnotate annotations in {os.path.basename(pkl_path)}...")
    
    with open(pkl_path, 'rb') as f:
        scene_obj = pickle.load(f)

    print(f"DEBUG: Anchor Point is {anchor_point}")
    print(f"DEBUG: Found {len(scene_obj.obj_annotation_list)} objects in scene.")

    best_match = None
    min_dist = float('inf')

    for box_item in scene_obj.obj_annotation_list:
        # Get center
        center = box_item.transform3d.get_matrix()[0, 3, :3].detach().cpu().numpy()
        dist = np.linalg.norm(center - anchor_point)
        
        # PRINT EVERY DISTANCE TO SEE WHAT'S HAPPENING
        print(f"DEBUG: Checking {box_item.category_label} at {center} (Dist: {dist:.2f}m)")

        if dist < min_dist:
            min_dist = dist
            best_match = box_item

    # Loosen threshold to 5m to account for any annotation inaccuracies or anchor misplacement
    if best_match is None or min_dist > 5.0: 
        print(f"❌ Closest object was {best_match.category_label if best_match else 'None'} at {min_dist:.2f}m")
        return None, None

    print(f"✅ Found Match: {best_match.category_label} (Dist: {min_dist:.2f}m)")

    # 2. Robust Path Construction
    # Make sure base_cad_dir is defined correctly here
    base_cad_dir = os.path.join(shapenet_root, best_match.catid_cad, best_match.id_cad)
    cad_path = os.path.join(base_cad_dir, 'models', 'model_normalized.obj')

    # Graceful handling of missing files
    if not os.path.exists(cad_path):
        print(f"⚠️ CAD file missing at {cad_path}. Skipping GT extraction for {best_match.category_label}.")
        return None, None 

    print(f"✅ Found CAD file at: {cad_path}")
    mesh = trimesh.load(cad_path, force='mesh')
    
    # 3. Transform CAD to World Space
    # ShapeNet is usually centered at origin; apply the ScannetPP transform
    T = best_match.transform3d.get_matrix()[0].detach().cpu().numpy()
    mesh.apply_transform(T)
    
    # 4. Sample points (this is your Decoder Target)
    # We sample 16384 to match your Utonia/Object-X input density
    gt_points = mesh.sample(16384)
    
    return gt_points, best_match