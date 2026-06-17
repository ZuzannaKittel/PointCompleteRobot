import os
import glob
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import trimesh

# Import our multi-modal layers
from models.implicit_network import MultiModalFeatureEncoder, ImplicitDecoder

# ==========================================
# PYTORCH DATASET WITH EXTRACTED FEATURE MAPPING
# ==========================================
class ScanNetppImplicitDataset(Dataset):
    def __init__(self, file_paths, num_input_pts=2048, num_queries=4096):
        self.file_paths = file_paths
        self.num_input_pts = num_input_pts
        self.num_queries = num_queries

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        # Load the packed dictionary from your data factory
        sample = torch.load(self.file_paths[idx], weights_only=False)
        
        # Up-cast half precision tensors (.half()) to float32 for downstream computation
        partial_pts = sample['partial_pts'].float()
        partial_feats = sample['partial_feats'].float()
        gt_pts = sample['gt_pts'].float()

        # Keep a clean record of the true world-space GT points for evaluation saving
        gt_pts_world = gt_pts.clone()

        # 1. Coordinate Space Centering and Scaling (To Canonical Bounds)
        centroid = gt_pts.mean(dim=0, keepdim=True)
        gt_pts = gt_pts - centroid
        partial_pts = partial_pts - centroid

        scale_factor = 1.0
        max_dist = torch.max(torch.norm(gt_pts, dim=-1))
        if max_dist > 0.4:
            scale_factor = 0.4 / max_dist.item()
            gt_pts *= scale_factor
            partial_pts *= scale_factor

        # 2. Slice/Pad point cloud sizes while maintaining strict feature alignment
        if partial_pts.shape[0] >= self.num_input_pts:
            indices = torch.randperm(partial_pts.shape[0])[:self.num_input_pts]
        else:
            indices = torch.randint(0, partial_pts.shape[0], (self.num_input_pts,))
            
        partial_pts_fixed = partial_pts[indices]
        partial_feats_fixed = partial_feats[indices]  # Crucial alignment index hook

        # Reconstruct the exact 2048 downsampled input points back to world space for visualization
        partial_pts_world = (partial_pts_fixed / scale_factor) + centroid

        # 3. Dynamic Query Field Generation
        num_uniform = self.num_queries // 2
        num_boundary = self.num_queries - num_uniform

        q_uniform = torch.rand(num_uniform, 3) - 0.5
        
        rand_idx = torch.randint(0, gt_pts.shape[0], (num_boundary,))
        chosen_gt = gt_pts[rand_idx, :]
        q_boundary = chosen_gt + torch.randn(num_boundary, 3) * 0.015

        query_coords = torch.cat([q_uniform, q_boundary], dim=0)

        # 4. Target Occupancy Evaluation
        dists = torch.cdist(query_coords.unsqueeze(0), gt_pts.unsqueeze(0))
        min_dists, _ = torch.min(dists, dim=-1)
        target_occupancy = (min_dists < 0.025).float().squeeze(0)

        return {
            'partial_feats': partial_feats_fixed,
            'query_coords': query_coords,
            'target_occupancy': target_occupancy,
            'centroid': centroid.squeeze(0),
            'scale_factor': torch.tensor(scale_factor, dtype=torch.float32),
            'partial_pts_world': partial_pts_world,
            'gt_pts_world': gt_pts_world
        }


# ==========================================
# INFERENCE GRID SAMPLER (Feature-Driven)
# ==========================================
def extract_implicit_shape(encoder, decoder, partial_feats, device, resolution=64):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Build multi-modal structural latent vector
        final_latent = encoder(partial_feats.unsqueeze(0))
        
        # Query discrete locations across the scalar matrix field
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
        # Compute quantiles for occupancy probabilities
        print(
            probabilities.quantile(torch.tensor([0.1,0.5,0.9], device=device))
        )

        # Determine the occupancy threshold for reconstruction
        thresholds = [0.5, 0.6, 0.7, 0.8]

        for thresh in thresholds:
            # Reconstruct the point cloud mask based on the threshold
            reconstructed_mask = probabilities > thresh
            pts = eval_coords.squeeze(0)[reconstructed_mask].cpu().numpy()

            trimesh.points.PointCloud(
                pts.astype(np.float32)
            ).export(
                f"recon_thresh_{thresh:.1f}.ply"
            )

        reconstructed_points = eval_coords.squeeze(0)[reconstructed_mask].cpu().numpy()
        
    encoder.train()
    decoder.train()
    return reconstructed_points


# ==========================================
# ENGINE RUNTIME PIPELINE
# ==========================================
if __name__ == "__main__":
    print("🚀 Initializing Multi-Modal Multi-Class Pipeline...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    data_files = glob.glob("data/geometric_pairs_dataset/train/*.pt")
    if not data_files:
        raise FileNotFoundError("Missing data packages inside 'data/geometric_pairs_dataset/train/'")
    print(f"📂 Discovered {len(data_files)} multi-modal data matrices.")

    torch.manual_seed(42)
    train_size = int(0.8 * len(data_files))
    val_size = len(data_files) - train_size
    train_files, val_files = random_split(data_files, [train_size, val_size])

    train_dataset = ScanNetppImplicitDataset(train_files)
    val_dataset = ScanNetppImplicitDataset(val_files)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    print(f"📊 Training Sets: {len(train_dataset)} shapes | Validation Sets: {len(val_dataset)} shapes")

    sample_data = torch.load(data_files[0], weights_only=False)
    detected_dim = sample_data['partial_feats'].shape[-1]
    print(f"🧬 Automatically detected feature dimension from factory output: {detected_dim}")

    encoder = MultiModalFeatureEncoder(
        input_feat_dim=detected_dim,
        latent_dim=512
    ).to(device)
    
    decoder = ImplicitDecoder(
        latent_dim=1024,
        hidden_dim=256
    ).to(device)
    
    optimizer = torch.optim.AdamW(list(encoder.parameters()) + list(decoder.parameters()), lr=1e-3)
    criterion = nn.BCEWithLogitsLoss()

    os.makedirs("runs/multimodal_baseline/snapshots", exist_ok=True)
    epochs = 50

    first_batch = next(iter(train_loader))

    print(first_batch['partial_feats'].shape)

    print("Feature dim:", detected_dim)
    print("Encoder dim:", detected_dim + 3)

    print("🔥 Starting training...")
    for epoch in range(epochs):
        encoder.train()
        decoder.train()
        running_loss = 0.0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            p_feats = batch['partial_feats'].to(device) # Multi-modal features (e.g., fused Utonia + DINOv2) for each point 
            q_coords = batch['query_coords'].to(device) # Continuous query coordinates for occupancy evaluation 
            targets = batch['target_occupancy'].to(device) # Binary occupancy labels for each query coordinate

            latents = encoder(p_feats)

            pred_logits = decoder(q_coords, latents)
            
            loss = criterion(pred_logits, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        epoch_loss = running_loss / len(train_loader)
        print(f"📈 [Epoch {epoch+1:02d}/{epochs}] Loss: {epoch_loss:.6f}")

        # --- EXHAUSTIVE TRIPLE VISUAL SNAPSHOT GENERATION ---
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            val_sample = val_dataset[0]  # Tracking index 0 of validation data split
            v_feats = val_sample['partial_feats'].to(device)

            recon_pts = extract_implicit_shape(
                encoder,
                decoder,
                v_feats,
                device,
                resolution=96
            )
            
            # Setup clean base naming conventions
            base_snap_path = f"runs/multimodal_baseline/snapshots/epoch_{epoch+1:02d}"
            
            # 1. Export the Ground Truth cloud (constant context reference)
            gt_pts_world = val_sample['gt_pts_world'].numpy()
            trimesh.points.PointCloud(gt_pts_world.astype(np.float32)).export(f"{base_snap_path}_gt.ply")
            
            # 2. Export the exact Partial Scan points fed to the encoder model pass
            partial_pts_world = val_sample['partial_pts_world'].numpy()
            trimesh.points.PointCloud(partial_pts_world.astype(np.float32)).export(f"{base_snap_path}_partial.ply")

            # 3. Export the predicted Continuous Reconstruction (if occupancy hits boundaries)
            if len(recon_pts) > 0:
                c_np = val_sample['centroid'].numpy()
                sf_np = val_sample['scale_factor'].item()
                recon_pts_world = (recon_pts / sf_np) + c_np

                trimesh.points.PointCloud(recon_pts_world.astype(np.float32)).export(f"{base_snap_path}_reconstructed.ply")
                print(f"   📸 Saved Triple Snapshot Group [GT / Partial / Reconstructed] at prefix: {base_snap_path}")
            else:
                print(f"   ⚠️ Reconstructed returned empty at epoch {epoch+1:02d}, skipped saving recon layer.")

    print("\n🏁 Framework routine finished. Run your evaluation snapshots through CloudCompare to see the improvements.")