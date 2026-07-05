import torch
import numpy as np
import matplotlib.pyplot as plt

data = torch.load(
    "debug_multiview.pt",
    weights_only=False
)

frames = data["selected_frames"]

colors = [
    "red",
    "green",
    "blue",
    "orange",
    "purple",
    "cyan",
    "magenta",
    "yellow",
]

fig = plt.figure(figsize=(20,5))

ax1 = fig.add_subplot(141, projection="3d")
ax2 = fig.add_subplot(142, projection="3d")
ax3 = fig.add_subplot(143, projection="3d")
ax4 = fig.add_subplot(144, projection="3d")

merged = []

# ---------------------------------------------------
# Raw points
# ---------------------------------------------------
for i, frame in enumerate(frames):

    pts = frame["raw_points"]

    ax1.scatter(
        pts[:,0],
        pts[:,1],
        pts[:,2],
        s=0.3,
        color=colors[i % len(colors)],
        label=frame["frame"]
    )

    cam = frame["c2w"][:3,3]

    ax1.scatter(
        cam[0],
        cam[1],
        cam[2],
        marker="^",
        s=80,
        color=colors[i % len(colors)]
    )

ax1.set_title("Raw World Clouds")
ax1.legend(fontsize=6)


# ---------------------------------------------------
# OBB cropped
# ---------------------------------------------------
for i, frame in enumerate(frames):

    pts = frame["cropped_points"]

    ax2.scatter(
        pts[:,0],
        pts[:,1],
        pts[:,2],
        s=0.3,
        color=colors[i % len(colors)],
        label=frame["frame"]
    )

ax2.set_title("After OBB Crop")


# ---------------------------------------------------
# Verified
# ---------------------------------------------------
for i, frame in enumerate(frames):

    pts = frame["points"]

    ax3.scatter(
        pts[:,0],
        pts[:,1],
        pts[:,2],
        s=0.3,
        color=colors[i % len(colors)],
        label=frame["frame"]
    )

    merged.append(pts)

ax3.set_title("Verified Per Frame")


# ---------------------------------------------------
# Final merged cloud
# ---------------------------------------------------
merged = np.concatenate(merged, axis=0)

ax4.scatter(
    merged[:,0],
    merged[:,1],
    merged[:,2],
    s=0.3,
    c="black"
)

ax4.set_title("Merged Cloud")


# ---------------------------------------------------
# Keep identical axes
# ---------------------------------------------------
mins = merged.min(axis=0)
maxs = merged.max(axis=0)

center = (mins + maxs) / 2
radius = np.max(maxs - mins) / 2

for ax in [ax1, ax2, ax3, ax4]:

    ax.set_xlim(center[0]-radius, center[0]+radius)
    ax.set_ylim(center[1]-radius, center[1]+radius)
    ax.set_zlim(center[2]-radius, center[2]+radius)

    ax.set_box_aspect([1,1,1])

plt.tight_layout()
plt.show()