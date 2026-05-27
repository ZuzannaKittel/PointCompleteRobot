import torch
import torch.nn as nn

# --- 1. The Compression Network (3D U-Net with 2 downsampling layers) ---
class SLatCompressor(nn.Module):
    def __init__(self, in_channels=1408, out_channels=8):
        super().__init__()
        leak = 0.2 
        
        # Encoder (downsampling path)
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
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(leak, inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(leak, inplace=True),
        )
        
        # Decoder (upsampling path)
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
    
# --- 2. The Spatial Compressor (Simple 3D CNN) ---
class U3DGS_SpatialCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # 64^3 -> 32^3
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2),
            # 32^3 -> 16^3
            nn.Conv3d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(8)
        )
    def forward(self, x):
        return self.encoder(x)


# --- MATHEMATICALLY ALIGNED VOXELIZATION ---
def create_structured_latent(points, features, grid_res=64, workspace_radius=1.0):
    """
    Voxelizes points within a rigid, fixed 3D workspace.
    Uses standard uniform grid division to match the decoder perfectly.
    """
    pts_t = torch.from_numpy(points).float().cuda()
    feats_t = torch.from_numpy(features).float().cuda()
    
    min_bound = -workspace_radius
    max_bound = workspace_radius
    
    # Map coordinates strictly from [-workspace_radius, +workspace_radius] to [0, 1]
    pts_norm = (pts_t - min_bound) / (max_bound - min_bound)
    
    # ---  Use strict grid_res spacing rather than (grid_res - 1) ---
    grid_coords = torch.clamp((pts_norm * grid_res).long(), 0, grid_res - 1)
    
    indices = grid_coords[:, 0] * (grid_res**2) + grid_coords[:, 1] * grid_res + grid_coords[:, 2]
    
    # Create the Feature Grid
    C, V = feats_t.shape[1], grid_res**3
    flat_grid = torch.zeros((C, V), device=feats_t.device)
    flat_grid.index_reduce_(1, indices, feats_t.T, reduce='amax', include_self=False)
    voxel_volume = flat_grid.view(1, C, grid_res, grid_res, grid_res)
    
    # Create the Occupancy Mask
    mask_flat = torch.zeros(V, device=feats_t.device)
    mask_flat.scatter_(0, indices, 1.0) 
    occupancy_mask = mask_flat.view(1, 1, grid_res, grid_res, grid_res)
    
    return voxel_volume, occupancy_mask