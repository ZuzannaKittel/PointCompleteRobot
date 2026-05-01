import pickle
import trimesh
import numpy as np
import torch
import os
import sys
import SCANnotatepp.ScanNetAnnotation as scannet_mod

sys.modules['ScanNetAnnotation'] = scannet_mod

import numpy as np
import pickle
import os
import trimesh

def extract_scannotate_gt(pkl_path, shapenet_root, anchor_point, device="cuda"):
    """
    Automatic Volume & Structure-Weighted Extractor.
    Fixed Matrix Indexing for PyTorch3D row-major transforms.
    """
    print(f"🔍 Analyzing ScanNet++ Scene Geometry...")
    
    with open(pkl_path, 'rb') as f:
        scene_obj = pickle.load(f)

    candidates = []

    # 1. Gather all objects in the vicinity
    for box_item in scene_obj.obj_annotation_list:
        T = box_item.transform3d.get_matrix()[0].detach().cpu().numpy()
        
        # THE FIX: PyTorch3D translation is in the 4th row (index 3)!
        center = T[3, :3]  
        dist = np.linalg.norm(center - anchor_point)
        
        if dist < 5.0:
            # THE FIX: Scale is the length of the first 3 rows
            scale_x = np.linalg.norm(T[0, :3])
            scale_y = np.linalg.norm(T[1, :3])
            scale_z = np.linalg.norm(T[2, :3])
            volume = scale_x * scale_y * scale_z
            
            candidates.append({
                'item': box_item,
                'label': box_item.category_label.lower(),
                'dist': dist,
                'volume': volume
            })

    if not candidates:
        print("❌ No objects found within 5 meters of anchor.")
        return None, None

    print(f"📊 Found {len(candidates)} candidates nearby. Ranking them...")

    best_candidate = None
    min_effective_score = float('inf')

    structural_classes = ['table', 'desk', 'cabinet', 'sofa', 'bookshelf', 'bed', 'counter']

    # 2. Score them mathematically
    for c in candidates:
        score = c['dist']
        
        # Structure Bonus
        if any(sc in c['label'] for sc in structural_classes):
            score -= 2.0 
            
        # Volume Bonus (Max 1.0m pull)
        vol_bonus = min(c['volume'] * 0.3, 1.0)
        score -= vol_bonus

        print(f"   - {c['label']}: dist={c['dist']:.2f}m, vol={c['volume']:.2f}m³ -> Score: {score:.2f}")

        if score < min_effective_score:
            min_effective_score = score
            best_candidate = c

    best_match = best_candidate['item']
    print(f"✅ WINNER: {best_candidate['label']} (Real Dist: {best_candidate['dist']:.2f}m)")

    # 3. Standard CAD Loading
    base_cad_dir = os.path.join(shapenet_root, best_match.catid_cad, best_match.id_cad)
    cad_path = os.path.join(base_cad_dir, 'models', 'model_normalized.obj')

    if not os.path.exists(cad_path):
        print(f"⚠️ CAD file missing at {cad_path}.")
        return None, None 

    mesh = trimesh.load(cad_path, force='mesh')
    T = best_match.transform3d.get_matrix()[0].detach().cpu().numpy()
    mesh.apply_transform(T)
    
    return mesh.sample(16384), best_match