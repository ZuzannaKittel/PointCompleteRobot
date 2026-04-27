import matplotlib.pyplot as plt
import torch
import numpy as np

# Load your tensor
slat = torch.load('desk_slat_64.pt') # Shape should be (1, 8, 64, 64, 64) for a 64^3 grid with 8 channels of features

# Get the actual shape dynamically
batch_size, channels, depth, height, width = slat.shape
grid_res = depth  # Assuming cubic grid (depth == height == width)

print(f"Loaded tensor shape: {slat.shape} (grid resolution: {grid_res}x{grid_res}x{grid_res})")

# 1. Take the absolute sum across the channels to get "Intensity"
# 2. Max-Project along the vertical axis (Depth/Height)
# This creates a "Heatmap" of the desk as seen from above
intensity_map = slat[0].abs().sum(dim=0).detach().cpu().numpy()
top_down_view = np.max(intensity_map, axis=0) 

plt.imshow(top_down_view, cmap='magma')
plt.title("Object-X SLat: Top-Down Density Map")
plt.colorbar()
plt.show()