import os

import numpy as np
import torch
import trimesh
from scipy.spatial import KDTree


class ShapeNetDataEngine:
    def __init__(self, utonia):
        self.uto = utonia

    # ShapeNet Ground Truth and Mesh Loader
    # No normalization since ShapeNet_preprocessed is already canonical
    def load_mesh_and_gt(self, mesh_path, num_points=16384):
        mesh = trimesh.load(mesh_path, force="mesh")
        gt = mesh.sample(num_points).astype(np.float32)
        return mesh, gt
    
    # Camera generation
    def create_camera_pose(self, radius=2.0):
        theta = np.random.uniform(0, 2 * np.pi)
        phi = np.random.uniform(np.pi / 6, np.pi / 2)

        cam = np.array([
            radius * np.sin(phi) * np.cos(theta),
            radius * np.sin(phi) * np.sin(theta),
            radius * np.cos(phi),
        ])

        forward = -cam / np.linalg.norm(cam)
        up = np.array([0, 0, 1])
        right = np.cross(up, forward)
        right /= np.linalg.norm(right)
        up = np.cross(forward, right)

        R = np.stack([right, up, forward], axis=1)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = cam

        return T

    # Synthetic partial observation
    def generate_partial_cloud(self, mesh):
        sampled = mesh.sample(30000)

        if isinstance(sampled, tuple):
            sampled = sampled[0]

        partial_views = []

        # Random number of viewpoints
        num_views = np.random.randint(3,6)

        for _ in range(num_views):
            pose = self.create_camera_pose(radius=np.random.uniform(1.7, 2.5))
            cam = pose[:3, 3]
            direction = -cam / np.linalg.norm(cam)

            vectors = sampled - cam
            visible = np.sum(vectors * direction, axis=1) > 0
            pts = sampled[visible]

            if len(pts) == 0:
                continue

            # Random visibility reduction
            keep_ratio = np.random.uniform(0.70,0.95)
            keep = np.random.choice(len(pts), int(len(pts) * keep_ratio), replace=False)
            pts = pts[keep]
            partial_views.append(pts)

        # Combine all partial views into a single point cloud
        # If no views were generated, return the original sampled points
        if len(partial_views) == 0:
            partial = sampled
        else:
            partial = np.concatenate(partial_views, axis=0)

        return partial.astype(np.float32)

    ############################################################
    # Utonia
    ############################################################
    def run_utonia(self, points, num_pts=2048):
        pts = np.asarray(points, dtype=np.float32)

        if len(pts) >= num_pts:
            idx = np.random.choice(len(pts), num_pts, replace=False)
        else:
            idx = np.random.choice(len(pts), num_pts, replace=True)

        sampled = pts[idx]

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        inp = torch.from_numpy(sampled).float().unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = self.uto(inp)

        # Extract sparse Utonia representation
        if isinstance(outputs, tuple):
            _, internal = outputs

            if isinstance(internal, tuple):
                sparse_coords, feats = internal
            else:
                feats = internal
                sparse_coords = sampled
        else:
            feats = outputs
            sparse_coords = sampled

        if hasattr(feats, "detach"):
            feats = feats.detach().cpu().numpy()

        if hasattr(sparse_coords, "detach"):
            sparse_coords = sparse_coords.detach().cpu().numpy()

        if feats.ndim == 3:
            feats = feats[0]

        if sparse_coords.ndim == 3:
            sparse_coords = sparse_coords[0]

        # Align sparse features back to 2048 points
        if len(feats) != len(sampled):
            tree = KDTree(sparse_coords[:, -3:])
            distances, indices = tree.query(sampled, k=min(4, len(sparse_coords)))

            weights = 1.0 / (distances + 1e-8)
            weights /= weights.sum(axis=1, keepdims=True)

            aligned_feats = np.sum(feats[indices] * weights[:, :, None], axis=1)
        else:
            aligned_feats = feats

        print("DEBUG Utonia aligned:", aligned_feats.shape)
        return sampled, aligned_feats.astype(np.float32)

    ############################################################
    # Main generation
    ############################################################
    def generate_pair(self, mesh_path):
        mesh, gt = self.load_mesh_and_gt(mesh_path)

        # Generate partial
        partial = self.generate_partial_cloud(mesh)
        
        # Random point dropout
        drop_ratio = np.random.uniform(0.02, 0.08)
        keep = np.random.choice(len(partial), int(len(partial) * (1 - drop_ratio)), replace=False)
        partial = partial[keep]

        # Add Gaussian noise to the partial point cloud
        noise_sigma = np.random.uniform(0.001, 0.003)
        partial += np.random.normal(0, noise_sigma, partial.shape).astype(np.float32)
        
        # Ensure the partial point cloud has at least 2048 points
        if len(partial) < 2048:
            ids = np.random.choice(len(partial), 2048, replace=True)
            partial = partial[ids]

        print(f"Partial size before Utonia: {len(partial)}")

        # Utonia feature extraction
        pts, feats = self.run_utonia(partial, num_pts=2048)
        fused = feats

        print("DEBUG partial:", type(pts), pts.shape)
        print("DEBUG features:", type(fused), fused.shape)
        print("DEBUG gt:", type(gt), gt.shape)

        return {
            "partial_pts": pts.astype(np.float32),
            "partial_feats": fused.astype(np.float32),
            "gt_pts": gt.astype(np.float32),
        }