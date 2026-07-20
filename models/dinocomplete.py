import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryEncoder(nn.Module):
    """
    Geometry branch.

    Input:
        xyz : [B, N, 3]

    Output:
        geom_features : [B, N, 512]
    """

    def __init__(self, hidden_dim=256, out_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, out_dim),
            nn.GELU(),
        )

    def forward(self, xyz):
        return self.net(xyz)


class SemanticEncoder(nn.Module):
    """
    Semantic branch.

    Receives ONLY DINO features.

    Input:
        dino : [B, N, 384]

    Output:
        semantic : [B, N, 512]
    """

    def __init__(self, input_dim=384, hidden_dim=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, dino):
        return self.net(dino)


class PointTransformerBlock(nn.Module):
    """
    Global contextual reasoning block.

    Point equivalent of DinoComplete's
    voxel state-space operator.

    Input:
        [B, N, C]

    Output:
        [B, N, C]
    """

    def __init__(self, embed_dim=512, num_heads=8, mlp_ratio=4, dropout=0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * mlp_ratio,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

    def forward(self, x):
        return self.layer(x)


class AttentionPooling(nn.Module):
    """
    Learns which point features are most informative
    when constructing the global latent representation.
    """

    def __init__(self, dim=512):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x):
        """
        x : [B, N, C]
        """
        weights = self.score(x)
        weights = torch.softmax(weights, dim=1)
        pooled = torch.sum(weights * x, dim=1)
        return pooled


class DinoCompleteBaselineEncoder(nn.Module):
    """
    DinoComplete-inspired baseline adapted to point clouds.

    Geometry:
        XYZ

    Semantics:
        DINO only

    Fusion:
        Equation (9) from DinoComplete

    Output:
        1024-dimensional latent vector.
    """

    def __init__(self):
        super().__init__()
        self.geometry_encoder = GeometryEncoder()
        self.semantic_encoder = SemanticEncoder()
        self.context = PointTransformerBlock(
            embed_dim=512,
            num_heads=8,
        )

        self.pool = AttentionPooling(512)

        self.projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
        )

        self.output_norm = nn.LayerNorm(1024)

    def forward(self, partial_pts, partial_feats):
        z_geo = self.geometry_encoder(partial_pts)
        z_geo_context = self.context(z_geo)

        dino = partial_feats[..., 1024:]
        z_sem = self.semantic_encoder(dino)
        z_sem_context = self.context(z_sem)

        fused = z_geo_context + z_sem_context + z_geo

        latent = self.pool(fused)
        latent = self.projection(latent)
        latent = F.normalize(latent, dim=-1)

        return latent