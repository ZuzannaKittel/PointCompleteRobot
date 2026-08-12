import torch
from torch.utils.data import Dataset

class ScanNetppImplicitDataset(Dataset):
    def __init__(
        self,
        file_paths,
        num_input_pts=2048,
        num_queries=16384,
        occupancy_threshold=0.01,
        boundary_sigma=0.005,
        boundary_fraction = 0.25,
    ):
        self.file_paths = file_paths
        self.num_input_pts = num_input_pts
        self.num_queries = num_queries
        self.occupancy_threshold = occupancy_threshold
        self.boundary_sigma = boundary_sigma
        self.boundary_fraction = boundary_fraction

    def __len__(self):
        return len(self.file_paths)

    @staticmethod
    def _min_distance_to_gt(query_coords, gt_pts, chunk_size=1024):
        min_dists = []

        for start in range(0, query_coords.shape[0], chunk_size):
            q = query_coords[start:start + chunk_size]

            dists = torch.cdist(
                q.unsqueeze(0),
                gt_pts.unsqueeze(0),
            )

            min_dists.append(
                dists.min(dim=-1).values.squeeze(0)
            )

        return torch.cat(min_dists, dim=0)

    def __getitem__(self, idx):
        sample = torch.load(self.file_paths[idx], weights_only=False)

        partial_pts = sample["partial_pts"].float()
        partial_feats = sample["partial_feats"].float()
        gt_pts = sample["gt_pts"].float()

        gt_pts_world = gt_pts.clone()

        # Data generation already provides a shared canonical
        # coordinate system for partial and GT geometry.
        centroid = torch.zeros(3, dtype=partial_pts.dtype)
        scale_factor = 1.0

        # Keep point-feature correspondence exact.
        if partial_pts.shape[0] >= self.num_input_pts:
            indices = torch.randperm(partial_pts.shape[0])[:self.num_input_pts]
        else:
            indices = torch.randint(
                0,
                partial_pts.shape[0],
                (self.num_input_pts,),
            )

        partial_pts_fixed = partial_pts[indices]
        partial_feats_fixed = partial_feats[indices]

        # Already in canonical coordinates.
        partial_pts_world = partial_pts_fixed.clone()

        # Generate implicit-field queries based on the boundary_fraction
        num_boundary = int(self.num_queries * self.boundary_fraction)
        num_uniform = self.num_queries - num_boundary

        # Uniform queries cover the complete normalized scene/object volume.
        q_uniform = torch.rand(num_uniform, 3) - 0.5

        # Boundary queries are sampled close to the GT surface.
        rand_idx = torch.randint(
            0,
            gt_pts.shape[0],
            (num_boundary,),
        )

        chosen_gt = gt_pts[rand_idx]
        q_boundary = (
            chosen_gt
            + torch.randn_like(chosen_gt) * self.boundary_sigma
        )

        query_coords = torch.cat(
            [q_uniform, q_boundary],
            dim=0,
        )
        
        
        min_dists = self._min_distance_to_gt(
            query_coords,
            gt_pts,
        )

        target_surface = (
            min_dists < self.occupancy_threshold
        ).float()

        # DEBUG: Save the normalized partial and GT point clouds, as well as the query points.
        if idx == 0:
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
            positive_queries = query_coords[target_surface.bool()]
            save_ply(
                "debug_dataset/positive_queries.ply",
                positive_queries
            )
        
        return {
            "partial_pts": partial_pts_fixed,
            "partial_feats": partial_feats_fixed,
            "query_coords": query_coords,
            "target_surface": target_surface,
            "centroid": centroid,
            "scale_factor": torch.tensor(
                scale_factor,
                dtype=torch.float32,
            ),
            "partial_pts_world": partial_pts_world,
            "gt_pts_world": gt_pts_world,
        }


if __name__ == "__main__":
    import glob

    data_path = "../data/pairs_scannet/train/*.pt"
    files = glob.glob(data_path)

    if not files:
        raise FileNotFoundError(
            f"No .pt files found at {data_path}"
        )

    dataset = ScanNetppImplicitDataset(files)

    sample = dataset[0]

    print("\nData loaded successfully.")
    print("Partial:", sample["partial_pts"].shape)
    print("Features:", sample["partial_feats"].shape)
    print("Queries:", sample["query_coords"].shape)
    print("Occupancy:", sample["target_surface"].shape)
    print(
        "Positive ratio:",
        sample["target_surface"].mean().item(),
    )