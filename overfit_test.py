import os
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

# Internal Imports
from models.voxelizer import create_structured_latent
from models.architectures import ObjectX_System

def plot_diagnostic(tensor_volume, save_path, title):
    """Visualizes the 3D feature volume as a 2D projection."""
    with torch.no_grad():
        projection = tensor_volume.abs().sum(dim=1).squeeze(0).max(dim=0)[0].cpu().numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(projection, cmap='magma')
    plt.title(title)
    plt.axis('off')
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_3d_point_cloud(points, save_path, title, limits=None):
    """Generates a static 3D scatter plot with fixed axis bounds for direct comparison."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color points by their Z height to visually show depth profile
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                    c=points[:, 2], cmap='viridis', s=6, alpha=0.8)
    
    ax.set_xlabel('X (Meters)')
    ax.set_ylabel('Y (Meters)')
    ax.set_zlabel('Z (Meters)')
    ax.set_title(title, fontsize=14, pad=20)
    
    # Lock the camera angle so both plots match perfectly
    ax.view_init(elev=20, azim=45)
    
    # Force both plots to share the exact same spatial box
    if limits is not None:
        min_b, max_b = limits
        ax.set_xlim(min_b[0], max_b[0])
        ax.set_ylim(min_b[1], max_b[1])
        ax.set_zlim(min_b[2], max_b[2])
        
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📸 Visual saved to: {save_path}")

if __name__ == "__main__":
    print("🚀 Initializing Single-Scene Overfit Engine...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Data Selection
    data_files = glob.glob("data/geometric_pairs_dataset/train/*.pt")
    if not data_files:
        raise FileNotFoundError("Check your data path; no .pt files found.")
    
    sample = torch.load(data_files[2], weights_only=False)
    partial_pts = sample['partial_pts'].float().to(device)
    partial_feats = sample['partial_feats'].float().to(device)
    gt_pts = sample['gt_pts'].float().to(device)

    # 2. Oracle Boundary Setup
    o_min, o_max = gt_pts.min(dim=0)[0], gt_pts.max(dim=0)[0]
    
    # 3. Create Training Targets (Voxelize GT)
    print("🎯 Creating Ground Truth occupancy grid...")
    _, gt_mask = create_structured_latent(gt_pts, torch.zeros_like(gt_pts), o_min, o_max, grid_res=64)
    target_occupancy = gt_mask.float().to(device) # Shape: [1, 1, 64, 64, 64]

    # 4. Model & Optimization
    model = ObjectX_System(in_channels=1408).to(device)
    model.train()
    
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion_occ = nn.BCEWithLogitsLoss() 

    # 5. Pre-process Input
    voxel_in, mask_in = create_structured_latent(partial_pts, partial_feats, o_min, o_max, grid_res=64)

    # 6. Training Loop
    epochs = 1000
    print(f"🔥 Starting burn-in for {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        preds = model(voxel_in, mask_in)
        logits = preds["occ"] 
        
        loss = criterion_occ(logits, target_occupancy)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"   [Epoch {epoch:03d}] Loss: {loss.item():.6f}")

    # 7. Final Verification
    print("\n✅ Training Complete. Running final diagnostic...")
    model.eval()
    with torch.no_grad():
        # 🔴 FIX APPLIED: Changed mask_64 to mask_in
        final_out = model(voxel_in, mask_in)
        points = model.decode_from_results_oracle(final_out, o_min, o_max, threshold=0.0)
        
    if points is not None:
        print(f"✨ Successfully reconstructed {len(points)} points from the latent grid.")
        
        # 1. Prepare bounds for matching aspect ratios
        limits_tuple = (o_min.cpu().numpy(), o_max.cpu().numpy())
        
        # 2. Convert raw GPU partial points to NumPy for plotting
        partial_pts_np = partial_pts.detach().cpu().numpy()
        
        # 3. Save the Partial Input Cloud
        plot_3d_point_cloud(
            partial_pts_np, 
            save_path="diagnostics/partial_input_pc.png", 
            title=f"Partial Input Point Cloud ({len(partial_pts_np)} Points)",
            limits=limits_tuple
        )
        
        # 4. Save the Reconstructed Cloud (using identical bounds)
        plot_3d_point_cloud(
            points, 
            save_path="diagnostics/final_reconstructed_pc.png", 
            title=f"Decoded Point Cloud ({len(points)} Points)",
            limits=limits_tuple
        )

        # 5. Save the Occupancy Heatmap
        plot_diagnostic(
            final_out["occ"], 
            save_path="diagnostics/occupancy_heatmap.png", 
            title="Occupancy Logits Heatmap (64^3)"
        )
    else:
        print("❌ Model failed to reconstruct any points above threshold.")