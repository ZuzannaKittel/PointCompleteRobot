import numpy as np
import trimesh
import torch
import os
import time
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from models.dinocomplete import DinoCompleteBaselineEncoder

from configs.run_config import RUN_DIR, get_run_decoder, get_run_model, get_run_name, get_run_dir
from utils.evaluation import extract_implicit_shape

from utils.evaluation import THRESHOLD

def train_model(encoder, decoder, train_loader, val_loader, val_dataset, optimizer, criterion, device, epochs, start_epoch=0, RUN_NAME=None, RUN_DIR=None, resume_training=False, config=None):
    if RUN_NAME is None:
        RUN_NAME = get_run_name()
    if RUN_DIR is None:
        RUN_DIR = get_run_dir()
    Path(RUN_DIR).mkdir(parents=True, exist_ok=True)
    Path(f"{RUN_DIR}/checkpoints").mkdir(parents=True, exist_ok=True)
    Path(f"{RUN_DIR}/snapshots").mkdir(parents=True, exist_ok=True)
    # ------------------------------------------------
    # TRAINING LOOP
    # ------------------------------------------------

    # Initialize lists to track training and validation losses, as well as epoch times
    # or resume training history
    history_path = f"{RUN_DIR}/training_history.csv"
    history_tmp_path = f"{RUN_DIR}/training_history.csv.tmp"

    history_columns = [
        "epoch",
        "train_loss",
        "val_loss",
        "learning_rate",
        "epoch_time_sec",
        "best_val_so_far",
    ]

    if resume_training and os.path.exists(history_path):
        history = pd.read_csv(history_path)

        # Remove empty / malformed rows from interrupted runs.
        history = history.dropna(how="all")
        history = history.dropna(
            subset=["epoch", "train_loss", "val_loss", "learning_rate", "epoch_time_sec"],
            how="any",
        )

        history["epoch"] = history["epoch"].astype(int)
        history = history.sort_values("epoch").reset_index(drop=True)

        print(f"📂 Loaded {len(history)} previous epochs of history.")
    else:
        history = pd.DataFrame(columns=history_columns)

    patience = 10
    epochs_without_improvement = 0
    best_val_loss = float("inf")

    # Create the LR scheduler 
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
        threshold=1e-4,
        min_lr=1e-6,
    )

    if resume_training:
        best_ckpt_path = f"{RUN_DIR}/checkpoints/best_model.pth"
        latest_ckpt_path = f"{RUN_DIR}/checkpoints/latest_model.pth"

        if os.path.exists(latest_ckpt_path):
            checkpoint = torch.load(
                latest_ckpt_path,
                map_location=device,
            )

            if "scheduler_state_dict" in checkpoint:
                scheduler.load_state_dict(
                    checkpoint["scheduler_state_dict"]
                )

        if os.path.exists(best_ckpt_path):
            checkpoint = torch.load(
                best_ckpt_path,
                map_location=device,
            )

            best_val_loss = checkpoint["val_loss"]

            if os.path.exists(latest_ckpt_path):
                checkpoint_latest = torch.load(
                    latest_ckpt_path,
                    map_location=device,
                )
                epochs_without_improvement = checkpoint_latest.get(
                    "epochs_without_improvement",
                    0,
                )
            else:
                epochs_without_improvement = checkpoint.get(
                    "epochs_without_improvement",
                    0,
                )

            print(
                f"📂 Found previous best model "
                f"(best validation loss = {best_val_loss:.6f})"
            )

    if resume_training and len(history) > 0:
        last_logged_epoch = int(history["epoch"].max())

        if last_logged_epoch > start_epoch:
            print(
                f"⚠️ Trimming history from epoch {last_logged_epoch} "
                f"down to checkpoint epoch {start_epoch}."
            )
            history = history[history["epoch"] <= start_epoch].reset_index(drop=True)

    # Save experiment config to CSV for reproducibility
    if config is not None:
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
            p_pts = batch["partial_pts"].to(device) # Geometric points
            q_coords = batch['query_coords'].to(device) # Continuous query coordinates for occupancy evaluation 
            targets = batch['target_surface'].to(device) # Binary surface occupancylabels for each query coordinate

            # Print the positive occupancy ratio for the first batch of the first epoch to monitor class imbalance
            if epoch == 0 and running_loss == 0:
                print(
                    f"Positive surface occupancy ratio: "
                    f"{targets.float().mean().item():.4f}"
                )
            
            # Universal architecture
            latents = encoder(p_pts, p_feats)

            pred_logits = decoder(q_coords, latents)
            
            loss = criterion(pred_logits, targets)
            loss.backward()

            # Gradient clipping to prevent exploding gradients and stabilize training
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) +
                list(decoder.parameters()),
                max_norm=1.0
            )

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
                
                p_pts = batch["partial_pts"].to(device)
                p_feats = batch['partial_feats'].to(device)
                q_coords = batch['query_coords'].to(device)
                targets = batch['target_surface'].to(device)

                # Universal architecture
                latents = encoder(p_pts, p_feats)
                pred_logits = decoder(q_coords, latents)

                loss = criterion(pred_logits, targets)
                val_loss += loss.item()

        val_loss /= len(val_loader)
        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start
        current_lr = optimizer.param_groups[0]["lr"]

        new_row = pd.DataFrame([{
            "epoch": epoch + 1,
            "train_loss": epoch_loss,
            "val_loss": val_loss,
            "learning_rate": current_lr,
            "epoch_time_sec": epoch_time,
        }])

        history = pd.concat([history, new_row], ignore_index=True)
        history["best_val_so_far"] = history["val_loss"].cummin()

        # Atomic write so an interrupted job does not leave a broken CSV.
        history.to_csv(history_tmp_path, index=False)
        os.replace(history_tmp_path, history_path)

        print(
            f"📈 [Epoch {epoch+1:02d}/{epochs}] "
            f"Train: {epoch_loss:.6f} | "
            f"Val: {val_loss:.6f} | "
            f"LR: {current_lr:.2e}"
        )

        # Save the best model based on validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss

            epochs_without_improvement = 0

            torch.save({
                "epoch": epoch + 1,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "val_loss": val_loss,
                "epochs_without_improvement": epochs_without_improvement,
            },
            f"{RUN_DIR}/checkpoints/best_model.pth")

            print("⭐ New best model saved.")
        else:
            epochs_without_improvement += 1
            print(f"⏳ No improvement for {epochs_without_improvement} epoch(s).")

        # --------------------------------------------------
        # CHECKPOINT + VISUALIZATION
        # --------------------------------------------------
        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            # Save checkpoint
            ckpt_path = (f"{RUN_DIR}/checkpoints/latest_model.pth")
            torch.save({
                "epoch": epoch + 1,
                "encoder_state_dict": encoder.state_dict(),
                "decoder_state_dict": decoder.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "loss": epoch_loss,
                "val_loss": val_loss,
                "epochs_without_improvement": epochs_without_improvement,
            }, ckpt_path)
            print(f"💾 Checkpoint saved: {ckpt_path}")

            # Visualize reconstruction for a few samples from the validation set
            tracked_indices = [0, 10, 20]
            for sample_idx in tracked_indices:
                val_sample = val_dataset[sample_idx]
                v_pts = val_sample["partial_pts"].to(device)
                v_feats = val_sample['partial_feats'].to(device)

                # Extract implicit shape from the model
                recon_pts, inference_time = extract_implicit_shape(encoder, decoder, v_pts, v_feats, device, resolution=64, threshold=THRESHOLD, RUN_DIR=RUN_DIR)

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

        if epochs_without_improvement >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Final safety write.
    history = history.dropna(how="all")
    history = history.dropna(
        subset=["epoch", "train_loss", "val_loss", "learning_rate", "epoch_time_sec"],
        how="any",
    )
    history["best_val_so_far"] = history["val_loss"].cummin()
    history.to_csv(history_path, index=False)

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