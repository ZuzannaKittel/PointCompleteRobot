import numpy as np
import trimesh
import torch

from utils.evaluation import extract_implicit_shape

def train_model(encoder, decoder, train_loader, val_dataset, optimizer, criterion, device, epochs):
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

            recon_pts = extract_implicit_shape(encoder, decoder, v_feats, device, resolution=64, threshold=0.8)
            
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