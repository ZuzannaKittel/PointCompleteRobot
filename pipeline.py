import os
import torch
import numpy as np
# Fix for trimesh compatibility with Numpy 2.0
if not hasattr(np, 'product'):
    np.product = np.prod
import trimesh

# Model and Engine Imports
from models.models import init_sam, init_dino, init_utonia
from utils.data_engine import SceneDataEngine
from models.objectX_complete import ObjectX_System
from models.objectX import create_structured_latent, SLatCompressor, U3DGS_SpatialCompressor  # Utility for voxelization

# Utility for plotting/verification
from utils.utils import save_barcode, plot_multimodal_results, save_voxel_density_map

# Configuration
PATHS = {
    "sam": "checkpoints/weights/sam_vit_h_4b8939.pth",
    "uto": "checkpoints/utoniadreamer/latest.pth",
    "json": "data/ScanNetpp/data/30966f4c6e/iphone/pose_intrinsic_imu.json",
    "pkl":  "data/ScanNetpp/annotations/30966f4c6e/30966f4c6e.pkl",
    "shapenet": "data/ShapeNet/ShapeNet_preprocessed"
}

def main():
    # Set deterministic behavior for reproducibility (important for thesis results)
    torch.cuda.empty_cache()
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

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

    # Bbox format: [x_min, y_min, x_max, y_max]
    bbox_prompt = [300, 450, 1400, 1400]
    real_world_pts, u_feats, d_feats, fused_pointwise, anchor = engine.get_fused_object(
        frame_indices=[0, 10, 20, 30, 40], 
        initial_prompt=bbox_prompt, 
        num_pts=16384,
        save_visuals=True  # This will save visuals of the raw vs fused point clouds for thesis figures
    )

    # 4. Phase 2.5: Ground Truth Harvesting
    gt_points = engine.get_ground_truth(PATHS["pkl"], PATHS["shapenet"], anchor)
    if gt_points is not None:
        trimesh.PointCloud(gt_points).export("data/gt_output/perfect_cad_target.ply")

    # 5. Visual Diagnostics (Optional but good for thesis figures)
    save_barcode(np.max(fused_pointwise, axis=0, keepdims=True), "figures/desk_identity_barcode.png")
    plot_multimodal_results(
        real_world_pts, 
        u_feats, 
        d_feats, 
        fused_pointwise, 
        save_path="figures/multimodal_pca.png"
    )

    # --- 6: The Full Object-X Funnel ---
    print("\n🧊 Phase 6: Generating Object-X Structured Latent & U-3DGS Embedding...")

    # A. Voxelize + Get Mask
    raw_vol, mask_64 = create_structured_latent(real_world_pts, fused_pointwise, grid_res=64)

    # B. Generate 64^3 SLat
    compressor_64 = SLatCompressor(in_channels=1408, out_channels=8).cuda()
    with torch.no_grad():
        slat_64 = compressor_64(raw_vol)
        
        # CRITICAL: Apply Mask to kill hallucinations
        # This turns your "solid block" back into a "desk shape"
        slat_64 = slat_64 * mask_64 

    # C. Generate 16^3 U-3DGS Embedding (The spatial compression)
    spatial_compressor = U3DGS_SpatialCompressor().cuda()
    with torch.no_grad():
        u3dgs_16 = spatial_compressor(slat_64)

    # --- DEBUG STATS ---
    active_64 = (slat_64.abs().sum(1) > 1e-5).sum().item()
    active_16 = (u3dgs_16.abs().sum(1) > 1e-5).sum().item()
    print(f"📊 Sparsity Check:")
    print(f"   - SLat 64^3 Active: {active_64} / {64**3} ({active_64/64**3:.2%})")
    print(f"   - U-3DGS 16^3 Active: {active_16} / {16**3} ({active_16/16**3:.2%})")

    # --- DEBUG & SAVE ---
    # Save the raw data for future processing
    torch.save(slat_64.cpu(), "desk_slat_64_6.pt")
    torch.save(u3dgs_16.cpu(), "desk_u3dgs_16_6.pt")

    # Save the visuals for the thesis
    print("📊 Generating Sparsity Visuals...")
    save_voxel_density_map(
        slat_64, 
        "figures/density_slat_64.png", 
        "Object-X SLat (64^3) High-Res Projection"
    )
    save_voxel_density_map(
        u3dgs_16, 
        "figures/density_u3dgs_16.png", 
        "Object-X U-3DGS (16^3) Low-Res Projection"
    )

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