import os
import glob
import torch
import numpy as np
import trimesh
import matplotlib.pyplot as plt

from models.objectX import create_structured_latent
from models.objectX_complete import ObjectX_System

def plot_density_map(tensor_volume, save_path, title):
    """
    Compresses a 5D feature volume [B, C, D, H, W] into a 2D heatmap
    by summing across channels and taking the max projection along the Z-axis.
    """
    # 1. Sum across feature channels to get raw "activation energy" per voxel
    activation_energy = tensor_volume.abs().sum(dim=1).squeeze(0) # Shape: [D, H, W]
    
    # 2. Max projection along the Depth (Z) axis to create a 2D image
    projection = activation_energy.max(dim=0)[0].cpu().numpy() # Shape: [H, W]
    
    # 3. Plot and save
    plt.figure(figsize=(6, 6))
    plt.imshow(projection, cmap='magma', interpolation='nearest')
    plt.colorbar(label='Feature Activation Magnitude')
    plt.title(title)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"📊 Saved visualization: {save_path}")

def run_diagnostic(data_folder):
    print("🚀 Initializing Diagnostic Run...")
    
    # 1. Find a sample file
    files = glob.glob(os.path.join(data_folder, "*.pt"))
    if not files:
        raise FileNotFoundError(f"No .pt files found in {data_folder}")
    
    test_file = files[2]
    print(f"📂 Loading sample: {os.path.basename(test_file)}")
    data = torch.load(test_file)
    
    pts = data["partial_pts"].numpy()
    feats = data["partial_feats"].numpy()
    print(f"   ↳ Loaded {len(pts)} points with {feats.shape[1]} features.")

    # 2. Initialize the heavy model
    print("\n🧠 Booting Object-X System (Heavy FP32 Mode)...")
    model = ObjectX_System(in_channels=1408).cuda()
    model.eval() # Set to eval to disable batchnorm drift during testing

    # 3. Voxelize
    # Make sure create_structured_latent is returning tensors for this test
    raw_vol, mask_64 = create_structured_latent(pts, feats, grid_res=64)
    raw_vol = raw_vol.cuda()
    mask_64 = mask_64.cuda()

    # 4. Forward Pass
    print("\n⚡ Running Forward Pass...")
    with torch.no_grad():
        results = model(raw_vol, mask_64)

    slat = results["slat"]           # [1, 8, 64, 64, 64]
    bottleneck = results["bottleneck"] # [1, 8, 16, 16, 16]

    # --- DIAGNOSTICS & VISUALIZATIONS ---
    os.makedirs("diagnostic_outputs", exist_ok=True)

    # A. Sparsity Check
    active_64 = (slat.abs().sum(1) > 1e-5).sum().item()
    active_16 = (bottleneck.abs().sum(1) > 1e-5).sum().item()
    print(f"\n📏 Embedding Sparsity Check:")
    print(f"   - SLat (64^3) Active Voxels: {active_64} / {64**3} ({active_64/64**3:.2%})")
    print(f"   - U-3DGS (16^3) Active Voxels: {active_16} / {16**3} ({active_16/16**3:.2%})")

    # B. Generate 2D Heatmaps
    print("\n🎨 Generating Feature Density Heatmaps...")
    plot_density_map(
        slat, 
        "diagnostic_outputs/01_slat_64_heatmap.png", 
        "SLat High-Res Features (64x64)"
    )
    plot_density_map(
        bottleneck, 
        "diagnostic_outputs/02_u3dgs_16_heatmap.png", 
        "U-3DGS Spatial Bottleneck (16x16)"
    )

    """# C. Test Decoder Plumbing - PLACEHOLDER for now, since the decoder is still being built
    print("\n🧊 Testing Decoder Geometry Extraction...")
    predicted_points = model.decode_from_results(results, threshold=0.5)

    if predicted_points is not None and len(predicted_points) > 0:
        output_ply = "diagnostic_outputs/03_reconstructed_plumbing_test.ply"
        trimesh.PointCloud(predicted_points).export(output_ply)
        print(f"   ✅ Decoder successfully extracted {len(predicted_points)} points.")
        print(f"   ✅ Saved geometry to: {output_ply}")
    else:
        print("   ⚠️ Decoder predicted an empty space (Expected behavior for random initialization!).")

    print("\n✅ Diagnostic Complete.")"""

if __name__ == "__main__":
    # Point this to the folder where the .pt files live
    DATA_DIR = "data/geometric_pairs_dataset/train" 
    run_diagnostic(DATA_DIR)