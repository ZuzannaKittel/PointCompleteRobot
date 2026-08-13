import math
import torch
from torch.utils.data import Dataset

# Helper class for random rotation augmentation
def _random_rotation_matrix(max_angle_deg=15.0):
    """Small rotation about a random axis, via Rodrigues' formula."""
    max_angle_rad = math.radians(max_angle_deg)
    angle = (torch.rand(1).item() * 2 - 1) * max_angle_rad

    axis = torch.randn(3)
    axis = axis / axis.norm().clamp(min=1e-8)

    K = torch.tensor([
        [0.0, -axis[2].item(), axis[1].item()],
        [axis[2].item(), 0.0, -axis[0].item()],
        [-axis[1].item(), axis[0].item(), 0.0],
    ])
    I = torch.eye(3)
    R = I + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
    return R

class ScanNetppImplicitDataset(Dataset):
    def __init__(
        self,
        file_paths,
        num_input_pts=2048,
        num_queries=8192,
        occupancy_threshold=0.01,
        boundary_sigma=0.005,
        boundary_fraction=0.3,
        deterministic=False,
    ):
        self.file_paths = file_paths
        self.num_input_pts = num_input_pts
        self.num_queries = num_queries
        self.occupancy_threshold = occupancy_threshold
        self.boundary_sigma = boundary_sigma
        self.boundary_fraction = boundary_fraction
        self.deterministic = deterministic

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
        sample = torch.load(
            self.file_paths[idx],
            weights_only=False,
        )

        partial_pts = sample["partial_pts"].float()
        partial_feats = sample["partial_feats"].float()
        gt_pts = sample["gt_pts"].float()

        gt_pts_world = gt_pts.clone()

        # AUGMENTATION (train only). Gated on deterministic so val/test
        # stay reproducible. Must run before subsampling and query
        # generation so labels are computed against augmented geometry.
        if not self.deterministic:
            R = _random_rotation_matrix(max_angle_deg=15.0)
            partial_pts = partial_pts @ R.T
            gt_pts = gt_pts @ R.T

            scale_jitter = 1.0 + (torch.rand(1).item() * 2 - 1) * 0.05  # ±5%
            partial_pts = partial_pts * scale_jitter
            gt_pts = gt_pts * scale_jitter

            partial_pts = partial_pts + torch.randn_like(partial_pts) * 0.003

        # Data generation already provides a shared canonical
        # coordinate system for partial and GT geometry.
        centroid = torch.zeros(3, dtype=partial_pts.dtype,)
        scale_factor = 1.0

        # RANDOMNESS
        if self.deterministic:
            generator = torch.Generator()
            generator.manual_seed(idx)
        else:
            generator = None

        # INPUT POINT SAMPLING
        # Keep point-feature correspondence exact.
        if partial_pts.shape[0] >= self.num_input_pts:
            indices = torch.randperm(
                partial_pts.shape[0],
                generator=generator,
            )[:self.num_input_pts]
        else:
            indices = torch.randint(
                0,
                partial_pts.shape[0],
                (self.num_input_pts,),
                generator=generator,
            )

        partial_pts_fixed = partial_pts[indices]
        partial_feats_fixed = partial_feats[indices]

        # Already in canonical coordinates.
        partial_pts_world = partial_pts_fixed.clone()

        # QUERY SAMPLING
        num_boundary = int(self.num_queries * self.boundary_fraction)
        num_uniform = (self.num_queries - num_boundary)

        # Uniform queries cover the complete canonical volume.
        q_uniform = (
            torch.rand(
                num_uniform,
                3,
                generator=generator,
            ) - 0.5
        )

        # Boundary queries are sampled close to the GT surface.
        rand_idx = torch.randint(
            0,
            gt_pts.shape[0],
            (num_boundary,),
            generator=generator,
        )

        chosen_gt = gt_pts[rand_idx]

        noise = torch.randn(
            chosen_gt.shape,
            generator=generator,
            dtype=chosen_gt.dtype,
        )

        q_boundary = (chosen_gt + noise * self.boundary_sigma)

        query_coords = torch.cat([q_uniform, q_boundary], dim=0)

        # OCCUPANCY LABELS
        min_dists = self._min_distance_to_gt(query_coords, gt_pts)

        target_surface = (min_dists < self.occupancy_threshold).float()

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