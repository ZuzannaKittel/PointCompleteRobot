import torch
import torch.nn as nn

# --- 1. The Compression Network (3D U-Net with 2 downsampling layers) ---
class SLatCompressor(nn.Module):
    def __init__(self, in_channels=1408, out_channels=8):
        """
        3D U-Net that compresses multi-modal features into lightweight latent space
        while preserving fine geometric details through multi-scale encoding.
        Architecture: 2 downsampling layers + bottleneck + 2 upsampling layers with skip connections.
        """
        super().__init__()
        
        # Encoder (downsampling path)
        self.enc1 = nn.Sequential(
            nn.Conv3d(in_channels, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        self.enc2 = nn.Sequential(
            nn.Conv3d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv3d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Conv3d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        
        # Decoder (upsampling path)
        self.upsampler1 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec1 = nn.Sequential(
            nn.Conv3d(128 + 64, 128, kernel_size=3, padding=1),  # Skip connection from enc2
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            nn.Conv3d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        
        self.upsampler2 = nn.Upsample(scale_factor=2, mode='nearest')
        self.dec2 = nn.Sequential(
            nn.Conv3d(256 + 128, 256, kernel_size=3, padding=1),  # Skip connection from enc1
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
            nn.Conv3d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        
        # Final layer: compress to output channels
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


# --- 2. The Voxelization Function ---
def create_structured_latent(points, features, grid_res=64):
    """
    Converts a point cloud with features into a 3D Voxel Grid (Structured Latent).
    """
    print(f"🧊 Building {grid_res}^3 Structured Latent (Object-X style)...")
    
    pts_t = torch.from_numpy(points).float().cuda()
    feats_t = torch.from_numpy(features).float().cuda()
    
    # 1. Normalize points to strictly [0, 1] bounding box
    pts_min = pts_t.min(dim=0)[0]
    pts_max = pts_t.max(dim=0)[0]
    pts_norm = (pts_t - pts_min) / (pts_max - pts_min + 1e-8)
    
    # 2. Scale to grid indices [0, 15]
    grid_coords = torch.clamp((pts_norm * grid_res).long(), 0, grid_res - 1)
    
    # 3. Flatten 3D coordinates into a 1D index array for scattering
    # index = x * (res^2) + y * res + z
    indices = grid_coords[:, 0] * (grid_res**2) + grid_coords[:, 1] * grid_res + grid_coords[:, 2]
    
    # 4. Initialize empty flat grid: Shape (Channels, Total_Voxels)
    C = feats_t.shape[1]
    V = grid_res**3
    flat_grid = torch.zeros((C, V), device=feats_t.device)
    
    # 5. Scatter Max Pooling
    # We transpose features to (C, N) to scatter them into the (C, V) grid based on indices
    # We use index_reduce_ (the beta function from your logs!) to take the max feature per voxel
    flat_grid.index_reduce_(1, indices, feats_t.T, reduce='amax', include_self=False)
    
    # 6. Reshape back into 3D volume: (Batch, Channels, D, H, W)
    voxel_volume = flat_grid.view(1, C, grid_res, grid_res, grid_res)
    
    return voxel_volume