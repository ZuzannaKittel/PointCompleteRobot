import torch
import torch.nn as nn
import torch.nn.functional as F

class SLatCompressor(nn.Module):
    def __init__(self, in_channels=1408, out_channels=8):
        super().__init__()
        leak = 0.2 
        
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(leak, inplace=True),
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(leak, inplace=True),
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.bottleneck = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(leak, inplace=True),
        )
        
        self.upsampler1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(leak, inplace=True),
        )
        
        self.upsampler2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = nn.Sequential(
            nn.Conv3d(256 + 128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(leak, inplace=True),
        )
        self.final = nn.Conv3d(256, out_channels, kernel_size=1)

    def forward(self, x):
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        bn = self.bottleneck(p2)
        
        u1 = self.upsampler1(bn)
        u1 = torch.cat([u1, e2], dim=1)
        d1 = self.dec1(u1)
        
        u2 = self.upsampler2(d1)
        u2 = torch.cat([u2, e1], dim=1)
        d2 = self.dec2(u2)
        return self.final(d2)


class U3DGS_SpatialCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2),
            nn.Conv3d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(8)
        )
    def forward(self, x):
        return self.encoder(x)


class U3DGS_Decoder(nn.Module):
    def __init__(self, bottleneck_channels=8, slat_channels=8):
        super().__init__()
        self.up1 = nn.Sequential(
            nn.ConvTranspose3d(bottleneck_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU()
        )
        self.up2 = nn.Sequential(
            nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU()
        )
        self.fusion_conv = nn.Sequential(
            nn.Conv3d(16 + slat_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm3d(32),
            nn.GELU(),
            nn.Conv3d(32, 16, kernel_size=3, padding=1),
            nn.BatchNorm3d(16),
            nn.GELU()
        )
        self.head_occupancy = nn.Sequential(
            nn.Conv3d(16, 1, kernel_size=1),
            nn.Sigmoid()
        )
        self.head_offsets = nn.Sequential(
            nn.Conv3d(16, 3, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, u3dgs_16, slat_64):
        x_32 = self.up1(u3dgs_16)
        x_64 = self.up2(x_32) 
        fused = torch.cat([x_64, slat_64], dim=1)
        out_features = self.fusion_conv(fused)
        return self.head_occupancy(out_features), self.head_offsets(out_features)


class ObjectX_System(nn.Module):
    def __init__(self, in_channels=1408):
        super().__init__()
        self.encoder_64 = SLatCompressor(in_channels=in_channels, out_channels=8)
        self.compressor_16 = U3DGS_SpatialCompressor()
        self.decoder = U3DGS_Decoder(bottleneck_channels=8, slat_channels=8)

    def forward(self, voxel_grid, mask_64):
        slat_64 = self.encoder_64(voxel_grid) * mask_64
        u3dgs_16 = self.compressor_16(slat_64)
        
        # Eliminate bottleneck background leaks
        mask_16 = F.max_pool3d(mask_64, kernel_size=4, stride=4)
        u3dgs_16 = u3dgs_16 * mask_16
        
        occ, offsets = self.decoder(u3dgs_16, slat_64)
        return {"slat": slat_64, "bottleneck": u3dgs_16, "occ": occ, "offsets": offsets}
    
    def decode_from_results_oracle(self, results, min_bound, max_bound, threshold=0.5, grid_res=64):
        """
        Translates spatial predictions back into metric space using verified Oracle bounds.
        """
        occ = results['occ'].squeeze()        
        offsets = results['offsets']          
        
        active_voxels = (occ > threshold).nonzero() 
        if len(active_voxels) == 0:
            return None

        x_idx, y_idx, z_idx = active_voxels[:, 0], active_voxels[:, 1], active_voxels[:, 2]
        point_offsets = offsets[0, :, x_idx, y_idx, z_idx].transpose(0, 1) 
        
        extent = max_bound - min_bound
        voxel_size = extent / grid_res
        
        voxel_centers = min_bound + (active_voxels.float() + 0.5) * voxel_size
        final_pts = voxel_centers + (point_offsets * (voxel_size / 2.0))
        return final_pts.cpu().numpy()