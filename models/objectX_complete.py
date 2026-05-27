import torch
import torch.nn as nn
from .objectX import SLatCompressor, U3DGS_SpatialCompressor
from .decoder import U3DGS_Decoder
import torch.nn.functional as F

# --- SYSTEM CONTROLLER (WITH BOTTLENECK MASKING) ---
class ObjectX_System(nn.Module):
    def __init__(self, in_channels=1408):
        super().__init__()
        self.encoder_64 = SLatCompressor(in_channels=in_channels, out_channels=8)
        self.compressor_16 = U3DGS_SpatialCompressor()
        self.decoder = U3DGS_Decoder(bottleneck_channels=8, slat_channels=8)

    def forward(self, voxel_grid, mask_64):
        # 1. SLat Generation (64^3)
        slat_64 = self.encoder_64(voxel_grid) * mask_64
        
        # 2. Spatial Compression (64^3 -> 16^3)
        u3dgs_16 = self.compressor_16(slat_64)
        
        # --- Downsample high-res mask to eliminate "Purple Soup" background leak ---
        # A kernel/stride of 4 perfectly shrinks a 64^3 grid to a 16^3 grid
        mask_16 = F.max_pool3d(mask_64, kernel_size=4, stride=4)
        u3dgs_16 = u3dgs_16 * mask_16
        
        # 3. Reconstruction
        occ, offsets = self.decoder(u3dgs_16, slat_64)
        
        return {
            "slat": slat_64,
            "bottleneck": u3dgs_16,
            "occ": occ,
            "offsets": offsets
        }
    
    # --- Decoding Function (Un-warped, mathematically rigorous coordinate inversion) ---
    def decode_from_results(self, results, threshold=0.5, workspace_radius=1.0, grid_res=64):
        """
        Uses the output dictionary from forward() to generate the point cloud
        using mathematically rigorous, un-warped coordinate inversion.
        """
        occ = results['occ'].squeeze()        # [64, 64, 64]
        offsets = results['offsets']          # [1, 3, 64, 64, 64]
        
        active_voxels = (occ > threshold).nonzero() # [N, 3] -> Indices matching (X, Y, Z) array positions

        if len(active_voxels) == 0:
            return None

        x_idx, y_idx, z_idx = active_voxels[:, 0], active_voxels[:, 1], active_voxels[:, 2]
        
        # Gather predicted sub-voxel offsets [N, 3]
        point_offsets = offsets[0, :, x_idx, y_idx, z_idx].transpose(0, 1) 
        
        # --- Strict, un-warped inverse coordinate calculation ---
        min_bound = -workspace_radius
        max_bound = workspace_radius
        voxel_size = (max_bound - min_bound) / grid_res # Continuous voxel width
        
        # Find continuous real-world center coordinate for each active voxel
        voxel_centers = min_bound + (active_voxels.float() + 0.5) * voxel_size
        
        # Apply sub-voxel local offset adjustments
        final_pts = voxel_centers + (point_offsets * (voxel_size / 2.0))
        
        return final_pts.cpu().numpy()