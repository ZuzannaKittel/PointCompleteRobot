import torch
import torch.nn as nn
from .objectX import SLatCompressor, U3DGS_SpatialCompressor
from .decoder import U3DGS_Decoder

class ObjectX_System(nn.Module):
    def __init__(self, in_channels=1408):
        super().__init__()
        self.encoder_64 = SLatCompressor(in_channels=in_channels, out_channels=8)
        self.compressor_16 = U3DGS_SpatialCompressor()
        self.decoder = U3DGS_Decoder(bottleneck_channels=8, slat_channels=8)

    def forward(self, voxel_grid, mask_64):
        # 1. SLat Generation
        slat_64 = self.encoder_64(voxel_grid) * mask_64
        
        # 2. Spatial Compression
        u3dgs_16 = self.compressor_16(slat_64)
        
        # 3. Reconstruction
        occ, offsets = self.decoder(u3dgs_16, slat_64)
        
        return {
            "slat": slat_64,
            "bottleneck": u3dgs_16,
            "occ": occ,
            "offsets": offsets
        }
    
    def decode_from_results(self, results, threshold=0.5):
        """
        Uses the output dictionary from forward() to generate the point cloud.
        """
        occ = results['occ'].squeeze()        # [64, 64, 64]
        offsets = results['offsets']          # [1, 3, 64, 64, 64]
        
        active_voxels = (occ > threshold).nonzero() # [N, 3]

        if len(active_voxels) == 0:
            return None

        x_idx, y_idx, z_idx = active_voxels[:, 0], active_voxels[:, 1], active_voxels[:, 2]
        
        # Gather offsets [N, 3]
        point_offsets = offsets[0, :, x_idx, y_idx, z_idx].transpose(0, 1) 
        
        # Calculate final points
        norm_coords = (active_voxels.float() / 63.0) * 2.0 - 1.0
        voxel_size = 2.0 / 64.0
        final_pts = norm_coords + (point_offsets * (voxel_size / 2.0))
        
        return final_pts.cpu().numpy()