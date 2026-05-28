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

def plot_3d_point_cloud(points, save_path="diagnostic_outputs/03_reconstructed_pc.png"):
    """Generates a static 3D scatter plot of the decoded points."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color points by their Z height to visually show depth profile
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                    c=points[:, 2], cmap='viridis', s=5, alpha=0.8)
    
    ax.set_xlabel('X (Meters)')
    ax.set_ylabel('Y (Meters)')
    ax.set_zlabel('Z (Meters)')
    ax.set_title(f"Decoded Point Cloud ({len(points)} Points)")
    
    # Adjust viewpoint for a clear perspective
    ax.view_init(elev=20, azim=45)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"📸 Static 3D visual saved to: {save_path}")

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
    # Using Logits version for stability
    criterion_occ = nn.BCEWithLogitsLoss() 

    # 5. Pre-process Input
    voxel_in, mask_in = create_structured_latent(partial_pts, partial_feats, o_min, o_max, grid_res=64)

    # 6. Training Loop
    epochs = 1000
    print(f"🔥 Starting burn-in for {epochs} epochs...")
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        
        preds = model(voxel_in, mask_in)
        logits = preds["occ"] # Raw output from the new Conv3d head
        
        loss = criterion_occ(logits, target_occupancy)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0 or epoch == epochs - 1:
            print(f"   [Epoch {epoch:03d}] Loss: {loss.item():.6f}")

    # 7. Final Verification
    print("\n✅ Training Complete. Running final diagnostic...")
    model.eval()
    os.makedirs("diagnostics", exist_ok=True)
    
    with torch.no_grad():
        final_preds = model(voxel_in, mask_in)
        decoded = model.decode_from_results_oracle(final_preds, o_min, o_max)
        
    if decoded is not None:
        print(f"✨ Successfully reconstructed {len(decoded)} points from the latent grid.")
        print(f"Initial Loss: ~0.69 | Final Loss: {loss.item():.6f}")

        plot_3d_point_cloud(decoded, save_path="diagnostics/final_reconstructed_pc.png")
        print(f"📊 Final point cloud visualization saved to diagnostics/final_reconstructed_pc.png")
    
        plot_diagnostic(final_preds["slat"], "diagnostics/final_slat.png", "Final Latent Features")
        print(f"📊 Final occupancy heatmap saved to diagnostics/final_slat.png")
    else:
        print("⚠️ Warning: No points passed the occupancy threshold post-training.")