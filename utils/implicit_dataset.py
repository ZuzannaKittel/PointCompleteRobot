import os

import torch
from torch.utils.data import Dataset

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

        # 1. Coordinate Space Normalization & Centering
        centroid = partial_pts.mean(dim=0, keepdim=True)
        gt_pts = gt_pts - centroid
        partial_pts = partial_pts - centroid

        # Estimate scale from the observed partial geometry.
        bbox = partial_pts.max(dim=0)[0] - partial_pts.min(dim=0)[0]
        max_extent = bbox.max().item()

        scale_factor = 1.0
        if max_extent > 0.4:
            scale_factor = 0.4 / max_extent

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
        q_boundary = chosen_gt + torch.randn(num_boundary, 3) * 0.025

        query_coords = torch.cat([q_uniform, q_boundary], dim=0)

        # 4. Target Occupancy Evaluation
        dists = torch.cdist(query_coords.unsqueeze(0), gt_pts.unsqueeze(0))
        min_dists, _ = torch.min(dists, dim=-1)
        target_occupancy = (min_dists < 0.015).float().squeeze(0)

        # =====================================================
        # DEBUG: save exactly what the network receives
        # =====================================================
        DEBUG = True

        if DEBUG and idx == 0:

            os.makedirs("debug_dataset", exist_ok=True)

            def save_ply(filename, pts):

                pts = pts.cpu().numpy()

                with open(filename, "w") as f:

                    f.write("ply\n")
                    f.write("format ascii 1.0\n")
                    f.write(f"element vertex {len(pts)}\n")
                    f.write("property float x\n")
                    f.write("property float y\n")
                    f.write("property float z\n")
                    f.write("end_header\n")

                    for p in pts:
                        f.write(f"{p[0]} {p[1]} {p[2]}\n")

            save_ply(
                "debug_dataset/partial_after_preprocessing.ply",
                partial_pts_fixed
            )

            save_ply(
                "debug_dataset/gt_after_preprocessing.ply",
                gt_pts
            )

            save_ply(
                "debug_dataset/query_points.ply",
                query_coords
            )

            positive_queries = query_coords[target_occupancy.bool()]

            save_ply(
                "debug_dataset/positive_queries.ply",
                positive_queries
            )

            print("\n==============================")
            print("DATASET DEBUG")
            print("==============================")

            print("Partial shape :", partial_pts_fixed.shape)
            print("GT shape      :", gt_pts.shape)
            print("Queries       :", query_coords.shape)

            print()

            print(f"Positive occupancy ratio : {target_occupancy.mean().item():.4f}")

            print()

            print("Centroid:")
            print(centroid.squeeze())

            print()

            print("Scale factor:", scale_factor)

            print()

            print("Partial bbox")
            print(partial_pts_fixed.min(dim=0)[0])
            print(partial_pts_fixed.max(dim=0)[0])

            print()

            print("GT bbox")
            print(gt_pts.min(dim=0)[0])
            print(gt_pts.max(dim=0)[0])

            print()

            print("Query bbox")
            print(query_coords.min(dim=0)[0])
            print(query_coords.max(dim=0)[0])

            print("==============================\n")

        return {
            "partial_pts": partial_pts_fixed,
            "partial_feats": partial_feats_fixed,
            "query_coords": query_coords,
            "target_occupancy": target_occupancy,
            "centroid": centroid.squeeze(0),
            "scale_factor": torch.tensor(scale_factor, dtype=torch.float32),
            "partial_pts_world": partial_pts_world,
            "gt_pts_world": gt_pts_world,
        }


if __name__ == "__main__":

    import glob

    files = glob.glob("./data/pairs_scannet/train/*.pt")

    dataset = ScanNetppImplicitDataset(files)

    sample = dataset[0]