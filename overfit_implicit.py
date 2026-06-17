import os
import glob
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import trimesh

# ==========================================
# 1. IMPLICIT DECODER ARCHITECTURE
# ==========================================
class ImplicitDecoder(nn.Module):
    """
    An implicit function decoder. It takes a shape's latent vector 
    and a batch of arbitrary (x, y, z) query coordinates, then predicts 
    whether each coordinate is inside (1) or outside (0) the shape.
    """
    def __init__(self, latent_dim=512, hidden_dim=256):
        super().__init__()
        # Map 3D coordinates to a higher dimensional space
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Combine the structural geometry features with the spatial query coordinates
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Outputs a single occupancy logit per query point
        )

    def forward(self, query_pts, latent_vector):
        # query_pts: [B, N, 3]
        # latent_vector: [B, latent_dim]
        B, N, _ = query_pts.shape
        
        # Encode coordinates
        coord_feats = self.coord_encoder(query_pts) # [B, N, hidden_dim]
        
        # Expand latent vector across all N query points
        latent_expanded = latent_vector.unsqueeze(1).repeat(1, N, 1) # [B, N, latent_dim]
        
        # Concatenate features and query
        combined = torch.cat([coord_feats, latent_expanded], dim=-1) # [B, N, hidden_dim + latent_dim]
        logits = self.decoder(combined) # [B, N, 1]
        
        return logits.squeeze(-1) # [B, N]

class SimplePointEncoder(nn.Module):
    """A placeholder encoder simulating how Utonia compresses the input scan into a shape embedding."""
    def __init__(self, in_channels=3, latent_dim=512):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.ReLU()
        )
    def forward(self, x):
        # x: [B, Num_Points, In_Channels] -> Simple Max Pooling for global vector
        feats = self.mlp(x)
        return torch.max(feats, dim=1)[0]


# ==========================================
# 2. DIAGNOSTIC PLOTTING FUNCTIONS
# ==========================================
def plot_3d_point_cloud(points, save_path, title, bounds=0.5):
    """Generates a static 3D scatter plot within a fixed canonical unit cube."""
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    sc = ax.scatter(points[:, 0], points[:, 1], points[:, 2], 
                    c=points[:, 2], cmap='viridis', s=6, alpha=0.8)
    
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14, pad=20)
    ax.view_init(elev=20, azim=45)
    
    # Force a rigid canonical coordinate window (No Oracle bounding boxes!)
    ax.set_xlim(-bounds, bounds)
    ax.set_ylim(-bounds, bounds)
    ax.set_zlim(-bounds, bounds)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"🎨 Visual saved to: {save_path}")

# =========================================
# * IMPLICIT SHAPE EXTRACTION FUNCTION *
# =========================================
def extract_implicit_shape(encoder, decoder, partial_pts, device, resolution=64):
    """Interrogates the latent space to extract the current reconstructed shape."""
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        final_latent = encoder(partial_pts)
        linear_spaces = torch.linspace(-0.5, 0.5, resolution, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(linear_spaces, linear_spaces, linear_spaces, indexing='ij')
        eval_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(1, -1, 3)
        
        chunk_size = 50000
        pred_slices = []
        for i in range(0, eval_coords.shape[1], chunk_size):
            chunk = eval_coords[:, i:i+chunk_size, :]
            logits = decoder(chunk, final_latent)
            pred_slices.append(logits)
        
        total_logits = torch.cat(pred_slices, dim=1)
        probabilities = torch.sigmoid(total_logits).squeeze(0)
        reconstructed_mask = probabilities > 0.5
        reconstructed_points = eval_coords.squeeze(0)[reconstructed_mask].cpu().numpy()
        
    encoder.train()
    decoder.train()
    return reconstructed_points

# ==========================================
# 3. CORE RUNTIME ENGINE
# ==========================================
if __name__ == "__main__":
    print("🚀 Initializing Implicit Coordinate Overfit Engine...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Data Selection
    data_files = glob.glob("data/geometric_pairs_dataset/train/*.pt")
    if not data_files:
        raise FileNotFoundError("Check your data path; no .pt files found.")
    
    sample = torch.load(data_files[6], weights_only=False)

    print(f"📦 Files number: {len(data_files)} | \nUsing sample: {data_files[6]} \n")
    
    # Ensure raw coordinates are placed cleanly inside a [-0.5, 0.5] bounding zone
    partial_pts = sample['partial_pts'].float().to(device).unsqueeze(0) # [1, N, 3]
    gt_pts = sample['gt_pts'].float().to(device).unsqueeze(0)           # [1, M, 3]

    # Initialize Models
    encoder = SimplePointEncoder(in_channels=3, latent_dim=512).to(device)
    decoder = ImplicitDecoder(latent_dim=512, hidden_dim=256).to(device)
    
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    # Save the input point clouds for reference
    os.makedirs("diagnostics", exist_ok=True)
    os.makedirs("diagnostics/snapshots", exist_ok=True)

    partial_points_np = partial_pts.squeeze(0).cpu().numpy().astype(np.float32)
    gt_points_np = gt_pts.squeeze(0).cpu().numpy().astype(np.float32)

    trimesh.points.PointCloud(
        partial_points_np
    ).export("diagnostics/input_partial.ply")

    trimesh.points.PointCloud(
        gt_points_np
    ).export("diagnostics/input_gt.ply")

    print("📸 Saved input partial point cloud: diagnostics/input_partial.ply")
    print("📸 Saved input ground-truth point cloud: diagnostics/input_gt.ply")

    plot_3d_point_cloud(partial_points_np, "diagnostics/input_partial.png", "Input Partial Scan")
    plot_3d_point_cloud(gt_points_np, "diagnostics/input_gt.png", "Input Ground Truth Scan")

    print("\n🔥 Starting implicit coordinate burn-in...")
    epochs = 1000
    
    # Force centering/normalization to protect against alignment drift
    gt_centroid = gt_pts.mean(dim=1, keepdim=True)
    gt_pts = gt_pts - gt_centroid
    partial_pts = partial_pts - gt_centroid

    # --- Initialize scale factor so it's always accessible ---
    scale_factor = 1.0 
    
    # Scale both to safely sit inside [-0.4, 0.4] so they don't hit the bounding box edges
    max_dist = torch.max(torch.norm(gt_pts, dim=-1))
    if max_dist > 0.4:
        scale_factor = 0.4 / max_dist.item()
        gt_pts *= scale_factor
        partial_pts *= scale_factor

    for epoch in range(epochs):
        optimizer.zero_grad()
        
        shape_latent = encoder(partial_pts)
        
        # --- BOUNDARY-ENRICHED MIXED SAMPLING ---
        num_queries = 4096
        num_uniform = num_queries // 2
        num_boundary = num_queries - num_uniform
        
        # 1. 50% Uniform Space Queries
        q_uniform = (torch.rand(1, num_uniform, 3, device=device) - 0.5) # [-0.5, 0.5]
        
        # 2. 50% Near-Surface Queries (Pick random GT points and perturb them slightly)
        random_indices = torch.randint(0, gt_pts.shape[1], (num_boundary,), device=device)
        chosen_gt = gt_pts[:, random_indices, :] # [1, num_boundary, 3]
        
        # Add small gaussian noise (e.g., 1.5 cm standard deviation)
        q_boundary = chosen_gt + torch.randn(1, num_boundary, 3, device=device) * 0.015
        
        # Combine both sets into our final training batch
        query_coords = torch.cat([q_uniform, q_boundary], dim=1) # [1, 4096, 3]
        
        # Step C: Compute ground truth binary occupancy targets
        dists = torch.cdist(query_coords, gt_pts) 
        min_dists, _ = torch.min(dists, dim=-1)   
        target_occupancy = (min_dists < 0.025).float() # Using a 2.5cm threshold envelope
        
        # Step D: Run the implicit prediction pass
        pred_logits = decoder(query_coords, shape_latent) 
        
        loss = criterion(pred_logits, target_occupancy)
        loss.backward()
        optimizer.step()
        
        if epoch % 200 == 0 or epoch == epochs - 1:
            pos_ratio = target_occupancy.mean().item() * 100
            print(f"   [Epoch {epoch:03d}] Loss: {loss.item():.6f} | Positive Target Ratio: {pos_ratio:.1f}%")
            
            # Extract how the embedding is organizing its 3D coordinates right now
            current_points = extract_implicit_shape(encoder, decoder, partial_pts, device, resolution=64)
            
            if len(current_points) > 0:
                snap_path = f"diagnostics/snapshots/epoch_{epoch:03d}.ply"

                trimesh.points.PointCloud(
                    current_points.astype(np.float32)
                ).export(snap_path)

                print(
                    f"      📸 Snapshot written to: {snap_path} "
                    f"({len(current_points)} surface points)"
                )
            else:
                print("      ⚠️ Snapshot empty: Field boundaries are still chaotic.")

    # ==========================================
    # 4. DECODING TRICK (No Oracle Required)
    # ==========================================
    print("\n✅ Training Complete. Testing extraction grid resolution...")
    encoder.eval()
    decoder.eval()
    
    with torch.no_grad():
        # 1. Extract the final trained shape embedding
        final_latent = encoder(partial_pts)
        
        # 2. Build a completely standard, uniform 3D evaluation grid spanning [-0.5, 0.5]
        # Change this number to generate higher or lower density clouds out of thin air!
        grid_resolution = 64 
        linear_spaces = torch.linspace(-0.5, 0.5, grid_resolution, device=device)
        grid_x, grid_y, grid_z = torch.meshgrid(linear_spaces, linear_spaces, linear_spaces, indexing='ij')
        
        # Flatten into a raw list of 3D query coordinates [1, 262144, 3]
        eval_coords = torch.stack([grid_x, grid_y, grid_z], dim=-1).view(1, -1, 3)
        
        # 3. Chunk inference to prevent GPU memory spikes
        chunk_size = 50000
        pred_slices = []
        for i in range(0, eval_coords.shape[1], chunk_size):
            chunk = eval_coords[:, i:i+chunk_size, :]
            logits = decoder(chunk, final_latent)
            pred_slices.append(logits)
        
        total_logits = torch.cat(pred_slices, dim=1) # [1, 262144]
        probabilities = torch.sigmoid(total_logits).squeeze(0)
        
        # 4. Isolate coordinates where the network is confident an object exists
        reconstructed_mask = probabilities > 0.5
        reconstructed_points = eval_coords.squeeze(0)[reconstructed_mask].cpu().numpy()
        
    print(f"✨ Reconstructed {len(reconstructed_points)} points using coordinate inquiries.")

    # Save output point cloud for external 3D inspection
    if len(reconstructed_points) > 0:

        # === 🛠️ INVERSE TRANSFORMATION TO WORLD SPACE ===
        # Convert your tracking centroid tensor to a flat NumPy array [3]
        centroid_np = gt_centroid.squeeze().cpu().numpy()
        
        # Reverse the scaling first, then shift the center back to the original position
        reconstructed_points = (reconstructed_points / scale_factor) + centroid_np
        # =====================================================

        output_path = "diagnostics/implicit_reconstructed.ply"

        trimesh.points.PointCloud(
            reconstructed_points.astype(np.float32)
        ).export(output_path)

        print(f"✨ Interactive point cloud saved to: {output_path}")
        print("   ↳ Open in CloudCompare, MeshLab, or Open3D for inspection.")

    else:
        print("❌ Threshold target too high; no isosurface detected.")

print("\n🎉 Overfitting test complete! Check your 'diagnostics/snapshots/' directory to view the evolution.")