import os
import numpy as np
import torch
from scipy.spatial import cKDTree
import trimesh
import matplotlib.pyplot as plt
import pandas as pd

# ==========================================
# Chamfer Distance Evaluation Metric
# ==========================================
def chamfer_distance(pred_pts, gt_pts):
    """
    Symmetric Chamfer Distance.
    Lower is better.
    """
    pred_tree = cKDTree(pred_pts)
    gt_tree = cKDTree(gt_pts)
    pred_to_gt, _ = gt_tree.query(pred_pts)
    gt_to_pred, _ = pred_tree.query(gt_pts)
    cd = (np.mean(pred_to_gt ** 2) + np.mean(gt_to_pred ** 2))
    return cd

# ==========================================
# INFERENCE GRID SAMPLER (Utonia+DINOv2-Feature-Driven)
# ==========================================
def extract_implicit_shape(encoder, decoder, partial_feats, device, resolution=64, threshold=0.8):
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
            f"Occupancy stats | "
            f"min={probabilities.min():.6f} "
            f"mean={probabilities.mean():.6f} "
            f"max={probabilities.max():.6f}"
        )

        """ # Determine the occupancy threshold for reconstruction
        thresholds = [0.5, 0.6, 0.7, 0.8]

        for thresh in thresholds:
            # Reconstruct the point cloud mask based on the threshold
            reconstructed_mask = probabilities > thresh
            pts = eval_coords.squeeze(0)[reconstructed_mask].cpu().numpy()

            trimesh.points.PointCloud(
                pts.astype(np.float32)
            ).export(
                f"recon_thresh_{thresh:.1f}.ply"
            )"""
    
        reconstructed_mask = probabilities > threshold
        reconstructed_points = eval_coords.squeeze(0)[reconstructed_mask].cpu().numpy()
        
    encoder.train()
    decoder.train()

    # Print the number of reconstructed points for debugging
    print(f"Reconstructed points: " f"{len(reconstructed_points)}")

    return reconstructed_points


# ==========================================
# TEST SET EVALUATION AND SNAPSHOT GENERATION
# ==========================================
def save_test_sample(sample_idx, prefix,
                     encoder, decoder,
                     test_dataset, device):

        sample = test_dataset[sample_idx]
        feats = sample['partial_feats'].to(device)
        recon_pts = extract_implicit_shape(encoder, decoder, feats, device, resolution=96, threshold=0.8)
        if len(recon_pts) == 0:
            print(f"Skipping {prefix}: empty reconstruction")
            return
        
        centroid = sample['centroid'].numpy()
        scale_factor = sample['scale_factor'].item()
        recon_world = (recon_pts / scale_factor) + centroid

        trimesh.points.PointCloud(sample['gt_pts_world'].numpy().astype(np.float32)).export(f"runs/test_debug/{prefix}_gt.ply")

        trimesh.points.PointCloud(sample['partial_pts_world'].numpy().astype(np.float32)).export(f"runs/test_debug/{prefix}_partial.ply")

        trimesh.points.PointCloud(recon_world.astype(np.float32)).export(f"runs/test_debug/{prefix}_reconstruction.ply")

        print(f"Saved {prefix}")

# ==========================================
# EVALUATION ON TEST SET
# ==========================================
def evaluate_test_set(encoder, decoder, test_dataset, device):
    print("\n🧪 Evaluating on test set...")
    all_cd = []
    encoder.eval()
    decoder.eval()
    results = []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            feats = sample['partial_feats'].to(device)
            recon_pts = extract_implicit_shape(encoder, decoder, feats, device, resolution=96, threshold=0.8)

            if len(recon_pts) == 0:
                continue

            centroid = sample['centroid'].numpy()
            scale_factor = sample['scale_factor'].item()
            recon_world = (recon_pts / scale_factor) + centroid
            gt_world = sample['gt_pts_world'].numpy()
            cd = chamfer_distance(recon_world, gt_world)
            
            results.append({
                "idx": idx,
                "file": os.path.basename(
                    test_dataset.file_paths[idx]
                ),
                "cd": float(cd)
            })

            all_cd.append(cd)
            if (idx + 1) % 25 == 0:
                print(f"Processed {idx+1}/{len(test_dataset)}")

    results = sorted(results, key=lambda x: x["cd"])

    df = pd.DataFrame(results)
    df.to_csv("runs/test_results.csv", index=False)

    best_idx = results[0]["idx"]
    median_idx = results[len(results)//2]["idx"]
    worst_idx = results[-1]["idx"]

    os.makedirs("runs/test_debug", exist_ok=True)

    save_test_sample(best_idx, "best", encoder, decoder, test_dataset, device)

    save_test_sample(median_idx, "median", encoder, decoder, test_dataset, device)

    save_test_sample(worst_idx, "worst", encoder, decoder, test_dataset, device)

    print("Best sample:", results[0])
    print("Worst sample:", results[-1])

    print(f"Std Chamfer: {np.std(all_cd):.6f}")

    print("\n==============================")
    print(f"Mean Chamfer:   {np.mean(all_cd):.6f}")
    print(f"Median Chamfer: {np.median(all_cd):.6f}")
    print(f"Best Chamfer:   {np.min(all_cd):.6f}")
    print(f"Worst Chamfer:  {np.max(all_cd):.6f}")
    print("==============================")

    plt.hist(all_cd, bins=30)
    plt.xlabel("Chamfer Distance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig("runs/chamfer_histogram.png")
    plt.close()