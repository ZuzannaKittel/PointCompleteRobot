import os
import numpy as np
import torch
from scipy.spatial import cKDTree
from configs.run_config import get_run_dir
import trimesh
import matplotlib.pyplot as plt
import pandas as pd
import time

from models.dinocomplete import DinoCompleteBaselineEncoder

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

def fscore(pred_pts,
           gt_pts,
           threshold=0.02):

    pred_tree = cKDTree(pred_pts)
    gt_tree = cKDTree(gt_pts)

    pred_to_gt,_ = gt_tree.query(pred_pts)
    gt_to_pred,_ = pred_tree.query(gt_pts)

    precision = np.mean(pred_to_gt < threshold)
    recall = np.mean(gt_to_pred < threshold)

    if precision + recall == 0:
        return 0.0

    return (
        2*precision*recall
        /
        (precision+recall)
    )

# ==========================================
# INFERENCE GRID SAMPLER (Utonia+DINOv2-Feature-Driven)
# ==========================================
def extract_implicit_shape(encoder, decoder, partial_pts, partial_feats, device, resolution=64, threshold=0.8):
    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        # Set the timer 
        start = time.time()

        # Build multi-modal structural latent vector
        if isinstance(encoder, DinoCompleteBaselineEncoder):
            final_latent = encoder(partial_pts.unsqueeze(0), partial_feats.unsqueeze(0))
        else:
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
        """print(
            f"Occupancy stats | "
            f"min={probabilities.min():.6f} "
            f"mean={probabilities.mean():.6f} "
            f"max={probabilities.max():.6f}"
        )"""

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
        
    #encoder.train()
    #decoder.train()

    # Print the number of reconstructed points for debugging
    #print(f"Reconstructed points: " f"{len(reconstructed_points)}")

    elapsed = time.time() - start

    return reconstructed_points, elapsed


# ==========================================
# TEST SET EVALUATION AND SNAPSHOT GENERATION
# ==========================================
def save_test_sample(sample_idx, prefix,
                     encoder, decoder,
                     test_dataset, device,
                     RUN_DIR=None):
    if RUN_DIR is None:
        RUN_DIR = get_run_dir()

    sample = test_dataset[sample_idx]
    pts = sample["partial_pts"].to(device)
    feats = sample['partial_feats'].to(device)
    recon_pts, inference_time = extract_implicit_shape(encoder, decoder, pts, feats, device, resolution=128, threshold=0.8)
    if len(recon_pts) == 0:
        print(f"Skipping {prefix}: empty reconstruction")
        return
    
    centroid = sample['centroid'].numpy()
    scale_factor = sample['scale_factor'].item()
    recon_world = (recon_pts / scale_factor) + centroid

    trimesh.points.PointCloud(sample['gt_pts_world'].numpy().astype(np.float32)).export(f"{RUN_DIR}/test_debug/{prefix}_gt.ply")

    trimesh.points.PointCloud(sample['partial_pts_world'].numpy().astype(np.float32)).export(f"{RUN_DIR}/test_debug/{prefix}_partial.ply")

    trimesh.points.PointCloud(recon_world.astype(np.float32)).export(f"{RUN_DIR}/test_debug/{prefix}_reconstruction.ply")

    print(f"Saved {prefix}")

# ==========================================
# EVALUATION ON TEST SET
# ==================================
def evaluate_test_set(encoder, decoder, test_dataset, device, RUN_DIR=None):
    if RUN_DIR is None:
        RUN_DIR = get_run_dir()
    print("\n🧪 Evaluating on test set...")
    all_cd = []
    encoder.eval()
    decoder.eval()
    results = []

    # Add a list to track the runtime and F-score for each sample
    all_runtime = []
    all_fscore = []

    with torch.no_grad():
        for idx in range(len(test_dataset)):
            sample = test_dataset[idx]
            pts = sample["partial_pts"].to(device)
            feats = sample['partial_feats'].to(device)
            recon_pts, inference_time = extract_implicit_shape(encoder, decoder, pts, feats, device, resolution=128, threshold=0.8)

            all_runtime.append(inference_time)

            if len(recon_pts) == 0:
                continue

            centroid = sample['centroid'].numpy()
            scale_factor = sample['scale_factor'].item()
            recon_world = (recon_pts / scale_factor) + centroid
            gt_world = sample['gt_pts_world'].numpy()
            cd = chamfer_distance(recon_world, gt_world)

            fs = fscore(recon_world,
                        gt_world)

            all_fscore.append(fs)
            
            results.append({
                "idx": idx,
                "file": os.path.basename(test_dataset.file_paths[idx]),
                "category": os.path.basename(test_dataset.file_paths[idx]).split("_")[-2],
                "cd": float(cd),
                "fscore": float(fs),
                "runtime": inference_time
            })

            all_cd.append(cd)
            if (idx + 1) % 25 == 0:
                print(f"Processed {idx+1}/{len(test_dataset)}")

    print(f"Mean runtime: {np.mean(all_runtime):.4f} s")
    print(f"Median runtime: {np.median(all_runtime):.4f} s")
    print(f"Std runtime: {np.std(all_runtime):.4f} s")

    runtime_df = pd.DataFrame({
        "mean_runtime_sec": [np.mean(all_runtime)],
        "median_runtime_sec": [np.median(all_runtime)],
        "std_runtime_sec": [np.std(all_runtime)]
    })

    runtime_df.to_csv(
        f"{RUN_DIR}/runtime_statistics.csv",
        index=False
    )

    results = sorted(results, key=lambda x: x["cd"])

    pd.DataFrame(results).to_csv(
        f"{RUN_DIR}/test_results.csv",
        index=False
    )

    if len(results) == 0:
        print("No valid reconstructions found.")
        return

    df = pd.DataFrame(results)
    category_df = (
        df
        .groupby("category")
        .agg(
            mean_cd=("cd","mean"),
            median_cd=("cd","median"),
            std_cd=("cd","std"),
            mean_fscore=("fscore","mean"),
            runtime=("runtime","mean"),
            samples=("cd","count")
        )
    )

    category_df.to_csv(
        f"{RUN_DIR}/category_results.csv"
    )

    best_idx = results[0]["idx"]
    median_idx = results[len(results)//2]["idx"]
    worst_idx = results[-1]["idx"]

    os.makedirs(f"{RUN_DIR}/test_debug", exist_ok=True)

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
    print(f"Mean F-score:   {np.mean(all_fscore):.4f}")
    print(f"Median F-score: {np.median(all_fscore):.4f}")
    print("==============================")

    plt.figure(figsize=(7,5))
    plt.hist(all_cd, bins=30)
    plt.grid(alpha=0.3)
    plt.xlabel("Chamfer Distance")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(
        f"{RUN_DIR}/chamfer_histogram.png",
        dpi=300,
        bbox_inches="tight"
    )
    plt.close()

    summary = pd.DataFrame({
        "mean_cd":[np.mean(all_cd)],
        "median_cd":[np.median(all_cd)],
        "best_cd":[np.min(all_cd)],
        "worst_cd":[np.max(all_cd)],
        "std_cd":[np.std(all_cd)],
        "mean_fscore":[np.mean(all_fscore)],
        "median_fscore":[np.median(all_fscore)]
    })

    summary.to_csv(
        f"{RUN_DIR}/evaluation_summary.csv",
        index=False
    )