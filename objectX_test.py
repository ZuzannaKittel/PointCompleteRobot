import matplotlib.pyplot as plt
import torch
import numpy as np

# Load your tensor
slat = torch.load('desk_slat_embedding_4.pt')

# 1. Take the absolute sum across the 8 channels to get "Intensity"
# 2. Max-Project along the vertical axis (Depth/Height)
# This creates a "Heatmap" of the desk as seen from above
intensity_map = slat[0].abs().sum(dim=0).cpu().numpy()
top_down_view = np.max(intensity_map, axis=0) 

plt.imshow(top_down_view, cmap='magma')
plt.title("Object-X SLat: Top-Down Density Map")
plt.colorbar()
plt.show()