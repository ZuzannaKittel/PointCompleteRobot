import os
import torch
import numpy as np
import trimesh

# Model and Engine Imports
from models.models import init_sam, init_dino, init_utonia
from utils.data_engine import SceneDataEngine
from models.objectX_complete import ObjectX_System
from models.objectX import create_structured_latent # Utility for voxelization

# Utility for plotting/verification
from utils.utils import save_barcode, plot_multimodal_results

# Configuration
PATHS = {
    "sam": "checkpoints/weights/sam_vit_h_4b8939.pth",
    "uto": "checkpoints/utoniadreamer/latest.pth",
    "json": "data/ScanNetpp/data/30966f4c6e/iphone/pose_intrinsic_imu.json",
    "pkl":  "data/ScanNetpp/annotations/30966f4c6e/30966f4c6e.pkl",
    "shapenet": "data/ShapeNet/ShapeNet_preprocessed"
}

def main():
    # 0. Setup Directories
    os.makedirs("data/gt_output", exist_ok=True)
    os.makedirs("verify_frames", exist_ok=True)

    # 1. Initialize Heavy Foundation Models
    print("🚀 Initializing Foundation Models...")
    sam = init_sam(PATHS["sam"])
    dino = init_dino()
    uto = init_utonia(PATHS["uto"])

    # 2. Initialize our Custom Modules
    engine = SceneDataEngine(PATHS, sam, dino, uto)
    model = ObjectX_System(in_channels=1408).cuda()

    # 3. Phase 1-4: Data Acquisition via Engine
    # Lifting frames -> Voxel Fusion -> Utonia Geometries -> Fusing with DINO
    real_world_pts, u_feats, d_feats, fused_pointwise, anchor = engine.get_fused_object(
        frame_indices=[0, 10, 20, 30, 40], 
        initial_prompt=[[1000, 700]], 
        num_pts=16384
    )

    # 4. Phase 2.5: Ground Truth Harvesting
    gt_points = engine.get_ground_truth(PATHS["pkl"], PATHS["shapenet"], anchor)
    if gt_points is not None:
        trimesh.PointCloud(gt_points).export("data/gt_output/perfect_cad_target.ply")

    # 5. Visual Diagnostics (Optional but good for thesis figures)
    save_barcode(np.max(fused_pointwise, axis=0, keepdims=True), "desk_identity_barcode.png")
    plot_multimodal_results(
        real_world_pts, 
        u_feats, 
        d_feats, 
        fused_pointwise, 
        save_path="multimodal_pca.png"
    )

    # 6. Phase 6: Prepare Input Volume (Voxelization)
    # This prepares the 1408-channel grid for the Object-X funnel
    raw_vol, mask_64 = create_structured_latent(real_world_pts, fused_pointwise, grid_res=64)

    # 7. Phase 7: Model Forward Pass & Extraction
    print("\n🧊 Running Object-X Inference & Reconstruction...")
    with torch.no_grad():
        # The model performs the whole funnel internally
        results = model(raw_vol.cuda(), mask_64.cuda())
        
        # Use the results to extract the points
        completed_points = model.decode_from_results(results)

    if completed_points is not None:
        trimesh.PointCloud(completed_points).export("data/gt_output/predicted_completion.ply")
        print(f"🌟 Saved {len(completed_points)} points to data/gt_output/predicted_completion.ply")
    else:
        print("⚠️ Decoder predicted an empty space.")

    print("\n✅ Pipeline Finished.")

if __name__ == "__main__":
    main()