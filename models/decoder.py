import torch
import torch.nn as nn
import torch.nn.functional as F

class U3DGS_Decoder(nn.Module):
    def __init__(self, bottleneck_channels=8, slat_channels=8):
        super().__init__()
        
        # 1. Upsample 16^3 -> 32^3
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(bottleneck_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU()
        )
        
        # 2. Upsample 32^3 -> 64^3
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU()
        )
        
        # 3. Fusion Layer (Combines Upsampled 64^3 with the SLat 64^3 skip connection)
        # 16 channels from up2 + 8 channels from slat_64 = 24 channels
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(16 + slat_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU()
        )
        
        # 4. Output Heads
        # Head A: Occupancy (Is there a point here?)
        self.head_occupancy = nn.Sequential(
            nn.Conv3d(16, 1, kernel_size=1),
            nn.Sigmoid() # Bounds between 0 and 1
        )
        
        # Head B: XYZ Offsets (Sub-voxel refinement for smooth surfaces)
        self.head_offsets = nn.Sequential(
            nn.Conv3d(16, 3, kernel_size=1),
            nn.Tanh() # Bounds between -1 and 1
        )

    def forward(self, u3dgs_16, slat_64):
        """
        u3dgs_16: [B, 8, 16, 16, 16] (The compressed bottleneck)
        slat_64:  [B, 8, 64, 64, 64] (The high-res structured latent skip connection)
        """
        # Step 1: Inflate the bottleneck
        x_32 = self.up1(u3dgs_16) # [B, 32, 32, 32, 32]
        x_64 = self.up2(x_32)     # [B, 16, 64, 64, 64]
        
        # Step 2: Skip Connection Fusion
        # Concatenate the upsampled features with the original structured latent
        fused = torch.cat([x_64, slat_64], dim=1) # [B, 24, 64, 64, 64]
        
        # Step 3: Process the fused features
        out_features = self.fusion_conv(fused) # [B, 16, 64, 64, 64]
        
        # Step 4: Generate final geometry predictions
        occupancy = self.head_occupancy(out_features) # [B, 1, 64, 64, 64]
        offsets = self.head_offsets(out_features)     # [B, 3, 64, 64, 64]
        
        return occupancy, offsets
    
    def decode_to_points(self, u3dgs, slat, threshold=0.5):
        """
        Self-contained extraction: The model knows how to turn its 
        own latent output into a point cloud.
        """
        occ, offsets = self.forward(u3dgs, slat)
        occ = occ.squeeze() # [64, 64, 64]
        
        active_voxels = (occ > threshold).nonzero() # [N, 3]

        if len(active_voxels) == 0:
            return None

        # Extract indices
        x_idx, y_idx, z_idx = active_voxels[:, 0], active_voxels[:, 1], active_voxels[:, 2]
        
        # Gather offsets [N, 3]
        point_offsets = offsets[0, :, x_idx, y_idx, z_idx].transpose(0, 1) 
        
        # Calculate final points
        norm_coords = (active_voxels.float() / 63.0) * 2.0 - 1.0
        voxel_size = 2.0 / 64.0
        final_pts = norm_coords + (point_offsets * (voxel_size / 2.0))
        
        return final_pts.cpu().numpy()