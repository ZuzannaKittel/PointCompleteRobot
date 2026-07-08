import numpy as np
import trimesh
import torch

from utils.evaluation import extract_implicit_shape

def train_model(encoder, decoder, train_loader, val_loader, val_dataset, optimizer, criterion, device, epochs, start_epoch=0):
    # ------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------
    for epoch in range(start_epoch, epochs):
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

        # ======================================
        # VALIDATION PASS
        # ======================================
        encoder.eval()
        decoder.eval()
        val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:

                p_feats = batch['partial_feats'].to(device)
                q_coords = batch['query_coords'].to(device)
                targets = batch['target_occupancy'].to(device)

                latents = encoder(p_feats)

                pred_logits = decoder(q_coords, latents)

                loss = criterion(pred_logits, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)

        print(
            f"📈 [Epoch {epoch+1:02d}/{epochs}] "
            f"Train: {epoch_loss:.6f} | "
            f"Val: {val_loss:.6f}"
        )

        # --------------------------------------------------
        # CHECKPOINT + VISUALIZATION
        # --------------------------------------------------
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            # Save checkpoint
            ckpt_path = (f"runs/multimodal_doublelatent/checkpoints/"f"epoch_{epoch+1:02d}.pth")
            torch.save({
                "epoch": epoch + 1,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": epoch_loss,
            }, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

            # Visualize reconstruction for a few samples from the validation set
            tracked_indices = [0, 1, 2]
            for sample_idx in tracked_indices:
                val_sample = val_dataset[sample_idx]
                v_feats = val_sample['partial_feats'].to(device)

                # Extract implicit shape from the model
                recon_pts = extract_implicit_shape(encoder, decoder, v_feats, device, resolution=64, threshold=0.8)

                base_snap_path = (f"runs/multimodal_doublelatent/snapshots/"f"epoch_{epoch+1:02d}_obj{sample_idx}")
                
                # Save ground truth and partial point clouds for comparison
                gt_pts_world = val_sample['gt_pts_world'].numpy()
                trimesh.points.PointCloud(gt_pts_world.astype(np.float32)).export(f"{base_snap_path}_gt.ply")

                # Save partial point cloud
                partial_pts_world = val_sample['partial_pts_world'].numpy()
                trimesh.points.PointCloud(partial_pts_world.astype(np.float32)).export(f"{base_snap_path}_partial.ply")

                # Save reconstructed point cloud if available
                if len(recon_pts) > 0:
                    c_np = val_sample['centroid'].numpy()
                    sf_np = val_sample['scale_factor'].item()

                    # Transform reconstructed points back to world coordinates
                    recon_pts_world = (recon_pts / sf_np) + c_np
                    trimesh.points.PointCloud(recon_pts_world.astype(np.float32)).export(f"{base_snap_path}_reconstructed.ply")

                    print(f"   📸 Saved sample {sample_idx} "f"for epoch {epoch+1:02d}")