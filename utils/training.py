import numpy as np
import trimesh
import torch
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from configs.run_config import get_run_name, get_run_dir
from utils.evaluation import extract_implicit_shape

def train_model(encoder, decoder, train_loader, val_loader, val_dataset, optimizer, criterion, device, epochs, start_epoch=0, RUN_NAME=None, RUN_DIR=None):
    if RUN_NAME is None:
        RUN_NAME = get_run_name()
    if RUN_DIR is None:
        RUN_DIR = get_run_dir()
    Path(RUN_DIR).mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------

    # Initialize lists to track training and validation losses, as well as epoch times
    train_losses = []
    val_losses = []
    epoch_times = []

    best_val_loss = float("inf")

    config = {
            "input_features":"fusion",
            "decoder":"double_latent",
            "epochs":epochs,
            "batch_size":32,
            "learning_rate":1e-3,
            "latent_dim":1024,
            "hidden_dim":256,
            "threshold":0.8,
            "resolution_eval":128
        }

    pd.DataFrame([config]).to_csv(
        f"{RUN_DIR}/config.csv",
        index=False
    )

    for epoch in range(start_epoch, epochs):

        # Initialize epoch timer
        epoch_start = time.time()

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

        # Track the best validation loss and save the corresponding model checkpoint
        train_losses.append(epoch_loss)
        val_losses.append(val_loss)

        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)

        print(
            f"📈 [Epoch {epoch+1:02d}/{epochs}] "
            f"Train: {epoch_loss:.6f} | "
            f"Val: {val_loss:.6f}"
        )

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            torch.save({
                "epoch": epoch + 1,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
            },
            f"{RUN_DIR}/checkpoints/best_model.pth")

            print("⭐ New best model saved.")

        # --------------------------------------------------
        # CHECKPOINT + VISUALIZATION
        # --------------------------------------------------
        if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
            # Save checkpoint
            ckpt_path = (f"{RUN_DIR}/checkpoints/"f"epoch_{epoch+1:02d}.pth")
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
                recon_pts, inference_time = extract_implicit_shape(encoder, decoder, v_feats, device, resolution=64, threshold=0.8)

                base_snap_path = (f"{RUN_DIR}/snapshots/"f"epoch_{epoch+1:02d}_obj{sample_idx}")
                
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

    # --------------------------------------------------
    # Save training history to CSV for later analysis   
    # --------------------------------------------------
    history = pd.DataFrame({
        "epoch": np.arange(1, len(train_losses)+1),
        "train_loss": train_losses,
        "val_loss": val_losses,
        "epoch_time_sec": epoch_times
    })

    history["best_val_so_far"] = history["val_loss"].cummin()

    history.to_csv(
        f"{RUN_DIR}/training_history.csv",
        index=False
    )

    # Generate and save the training curve plot
    plt.figure(figsize=(8,5))

    plt.plot(history["epoch"],
            history["train_loss"],
            label="Train")

    plt.plot(history["epoch"],
            history["val_loss"],
            label="Validation")

    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        f"{RUN_DIR}/training_curve.png",
        dpi=300
    )

    plt.close()