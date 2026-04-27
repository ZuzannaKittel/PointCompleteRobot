import torch
import torch.nn as nn

# --- 1. The Compression Network (3D U-Net with 2 downsampling layers) ---
class SLatCompressor(nn.Module):
    def __init__(self, in_channels=1408, out_channels=8):
        super().__init__()
        
        # Use a negative slope of 0.2, which is standard for 3D sparse architectures
        leak = 0.2 
        
        # Encoder (downsampling path)
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(leak, inplace=True), # CHANGED
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(leak, inplace=True), # CHANGED
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(leak, inplace=True), # CHANGED
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(leak, inplace=True), # CHANGED
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(leak, inplace=True), # CHANGED
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.LeakyReLU(leak, inplace=True), # CHANGED
        )
        
        # Decoder (upsampling path)
        self.upsampler1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(leak, inplace=True), # CHANGED
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(leak, inplace=True), # CHANGED
        )
        
        self.upsampler2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = nn.Sequential(
            nn.Conv3d(256 + 128, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(leak, inplace=True), # CHANGED
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.LeakyReLU(leak, inplace=True), # CHANGED
        )
        
        # Final layer: NO LEAKY RELU HERE. 
        # Keep it linear so we can apply the occupancy mask effectively.
        self.final = nn.Conv3d(256, out_channels, kernel_size=1)

    def forward(self, x):
        # x expected shape: (Batch, Channels, Depth, Height, Width)
        
        # Encoder
        e1 = self.enc1(x)
        p1 = self.pool1(e1)
        
        e2 = self.enc2(p1)
        p2 = self.pool2(e2)
        
        # Bottleneck
        bn = self.bottleneck(p2)
        
        # Decoder with skip connections
        u1 = self.upsampler1(bn)
        u1 = torch.cat([u1, e2], dim=1)  # Skip connection
        d1 = self.dec1(u1)
        
        u2 = self.upsampler2(d1)
        u2 = torch.cat([u2, e1], dim=1)  # Skip connection
        d2 = self.dec2(u2)
        
        # Final compression to target dimension
        out = self.final(d2)
        return out
    

class U3DGS_SpatialCompressor(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            # 64 -> 32
            nn.Conv3d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(16),
            nn.LeakyReLU(0.2),
            # 32 -> 16
            nn.Conv3d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm3d(8)
        )
    def forward(self, x):
        return self.encoder(x)


# --- 2. The Voxelization Function ---
def create_structured_latent(points, features, grid_res=64):
    """
    Returns:
        voxel_volume: (1, 1408, 64, 64, 64)
        occupancy_mask: (1, 1, 64, 64, 64) - Binary mask (1 where points exist, 0 else)
    """
    print(f"🧊 Building {grid_res}^3 Structured Latent...")
    
    pts_t = torch.from_numpy(points).float().cuda()
    feats_t = torch.from_numpy(features).float().cuda()
    
    # Normalize and scale
    pts_min, pts_max = pts_t.min(0)[0], pts_t.max(0)[0]
    pts_norm = (pts_t - pts_min) / (pts_max - pts_min + 1e-8)
    grid_coords = torch.clamp((pts_norm * (grid_res - 1)).long(), 0, grid_res - 1)
    
    indices = grid_coords[:, 0] * (grid_res**2) + grid_coords[:, 1] * grid_res + grid_coords[:, 2]
    
    # Create the Feature Grid
    C, V = feats_t.shape[1], grid_res**3
    flat_grid = torch.zeros((C, V), device=feats_t.device)
    flat_grid.index_reduce_(1, indices, feats_t.T, reduce='amax', include_self=False)
    voxel_volume = flat_grid.view(1, C, grid_res, grid_res, grid_res)
    
    # --- NEW: Create the Occupancy Mask ---
    mask_flat = torch.zeros(V, device=feats_t.device)
    mask_flat.scatter_(0, indices, 1.0) # Mark occupied voxels as 1
    occupancy_mask = mask_flat.view(1, 1, grid_res, grid_res, grid_res)
    
    return voxel_volume, occupancy_mask