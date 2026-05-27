import torch
import numpy as np

def create_structured_latent(points, features, min_bound, max_bound, grid_res=64):
    """
    Voxelizes continuous points into a discrete 64^3 grid based on Oracle bounds.
    Handles Float16/Float32 conversions automatically.
    """
    if isinstance(points, np.ndarray):
        pts_t = torch.from_numpy(points).float().cuda()
        feats_t = torch.from_numpy(features).float().cuda()
    else:
        pts_t = points.float().cuda()
        feats_t = features.float().cuda()
        
    min_bound = min_bound.to(pts_t.device)
    max_bound = max_bound.to(pts_t.device)
    
    extent = max_bound - min_bound
    extent = torch.where(extent == 0, torch.ones_like(extent) * 1e-5, extent)
    
    # Map coordinates tightly to [0, 1] relative to the GT bounding box
    pts_norm = (pts_t - min_bound) / extent
    grid_coords = torch.clamp((pts_norm * grid_res).long(), 0, grid_res - 1)
    
    indices = grid_coords[:, 0] * (grid_res**2) + grid_coords[:, 1] * grid_res + grid_coords[:, 2]
    
    # Create the Feature Grid via max reduction
    C, V = feats_t.shape[1], grid_res**3
    flat_grid = torch.zeros((C, V), device=feats_t.device)
    flat_grid.index_reduce_(1, indices, feats_t.T, reduce='amax', include_self=False)
    voxel_volume = flat_grid.view(1, C, grid_res, grid_res, grid_res)
    
    # Create the Occupancy Mask
    mask_flat = torch.zeros(V, device=feats_t.device)
    mask_flat.scatter_(0, indices, 1.0) 
    occupancy_mask = mask_flat.view(1, 1, grid_res, grid_res, grid_res)
    
    return voxel_volume, occupancy_mask