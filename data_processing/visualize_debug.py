import torch
import numpy as np
import matplotlib.pyplot as plt

data = torch.load(
    "debug_pipeline.pt",
    weights_only=False
)

stages = [
    ("Merged", data["merged"]),
    ("Local OBB", data["local"]),
    ("Voxel", data["voxel"]),
    ("Clean", data["clean"]),
    ("Canonical", data["canonical"]),
    ("2048 Sampled", data["sampled"])
]

fig = plt.figure(figsize=(18, 10))

for i, (name, pts) in enumerate(stages):

    ax = fig.add_subplot(2, 3, i + 1, projection="3d")

    ax.scatter(
        pts[:, 0],
        pts[:, 1],
        pts[:, 2],
        s=0.5,
        c=np.linalg.norm(pts, axis=1),
        cmap="viridis"
    )

    ax.set_title(f"{name}\n({len(pts)} pts)")
    ax.set_box_aspect([1, 1, 1])

    mins = pts.min(0)
    maxs = pts.max(0)
    center = (mins + maxs) / 2
    radius = np.max(maxs - mins) / 2

    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)

plt.tight_layout()
plt.show()