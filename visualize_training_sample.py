import os
import glob
import torch
import numpy as np
import matplotlib.pyplot as plt

from utils.implicit_dataset2 import ScanNetppImplicitDataset


DATA_DIR = "data/pairs_scannet/train/*.pt"
OUTPUT = "training_sample_construction.png"


def set_equal_axes(ax, points):
    mins = points.min(axis=0)
    maxs = points.max(axis=0)

    center = (mins + maxs) / 2.0
    radius = np.max(maxs - mins) / 2.0

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


# Expand the glob pattern into a list of file paths
file_paths = glob.glob(DATA_DIR)
if not file_paths:
    raise FileNotFoundError(f"No files matching pattern: {DATA_DIR}")

dataset = ScanNetppImplicitDataset(
    file_paths,  # <-- Pass the resolved list here
    num_input_pts=2048,
    num_queries=8192,
    occupancy_threshold=0.01,
    boundary_sigma=0.005,
    boundary_fraction=0.3,
    deterministic=True,
)

sample = dataset[0]

partial = sample["partial_pts"].numpy()
gt = sample["gt_pts_world"].numpy()
queries = sample["query_coords"].numpy()
positive = queries[
    sample["target_surface"].bool().numpy()
]

print("Partial:", partial.shape)
print("GT:", gt.shape)
print("Queries:", queries.shape)
print("Positive queries:", positive.shape)
print(
    "Positive ratio:",
    len(positive) / len(queries),
)

# Use a common coordinate range for all panels.
all_points = np.concatenate(
    [partial, gt, queries],
    axis=0,
)

fig = plt.figure(figsize=(12, 10))

views = [
    (20, 35),
    (20, 35),
    (20, 35),
    (20, 35),
]

titles = [
    "Partial observation",
    "Ground-truth point cloud",
    "All query points",
    "Positive surface queries",
]

datasets = [
    partial,
    gt,
    queries,
    positive,
]

# Plot each representation separately.
for i, (ax_data, title, view) in enumerate(
    zip(datasets, titles, views)
):
    ax = fig.add_subplot(2, 2, i + 1, projection="3d")

    if i == 0:
        ax.scatter(
            ax_data[:, 0],
            ax_data[:, 1],
            ax_data[:, 2],
            s=2,
            alpha=0.8,
        )

    elif i == 1:
        ax.scatter(
            ax_data[:, 0],
            ax_data[:, 1],
            ax_data[:, 2],
            s=1.5,
            alpha=0.7,
        )

    elif i == 2:
        ax.scatter(
            ax_data[:, 0],
            ax_data[:, 1],
            ax_data[:, 2],
            s=1,
            alpha=0.15,
        )

    else:
        ax.scatter(
            ax_data[:, 0],
            ax_data[:, 1],
            ax_data[:, 2],
            s=3,
            alpha=0.8,
        )

    ax.set_title(title, fontsize=13)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")

    ax.view_init(
        elev=view[0],
        azim=view[1],
    )

    set_equal_axes(ax, all_points)

plt.tight_layout()

plt.savefig(
    OUTPUT,
    dpi=300,
    bbox_inches="tight",
)

plt.show()

print(f"Saved figure to: {OUTPUT}")