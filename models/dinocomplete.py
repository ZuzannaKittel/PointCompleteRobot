import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryEncoder(nn.Module):
    """
    Geometry branch for point-based DinoComplete proxy.
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
    Semantic branch for DINO descriptors.
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
    Lightweight contextual reasoning block.
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
    Attention pooling over point descriptors.
    """
    def __init__(self, dim=512):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, x):
        weights = self.score(x)
        weights = torch.softmax(weights, dim=1)
        return torch.sum(weights * x, dim=1)


class DinoCompleteBaselineEncoder(nn.Module):
    """
    Point-based DinoComplete-inspired encoder.

    This implementation preserves the principal ideas of
    DinoComplete:

        • separate geometry and semantic branches;
        • transformer-based contextual reasoning;
        • feature fusion;
        • global representation learning.

    Unlike the original DinoComplete architecture, the model
    operates directly on point clouds and produces a global
    latent representation that is subsequently used by an
    implicit occupancy decoder.
    """

    def __init__(self):
        super().__init__()

        # Input normalisation helps when geometry and DINO features have different scales.
        self.geo_input_norm = nn.LayerNorm(3)
        self.sem_input_norm = nn.LayerNorm(384)

        self.geometry_encoder = GeometryEncoder()
        self.semantic_encoder = SemanticEncoder()

        # Separate context blocks per branch, instead of one shared block.
        self.geo_context = PointTransformerBlock(embed_dim=512, num_heads=8)
        self.sem_context = PointTransformerBlock(embed_dim=512, num_heads=8)

        # Small fusion norm to stabilise branch combination.
        self.fusion_norm = nn.LayerNorm(512)

        self.pool = AttentionPooling(512)

        self.projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
        )

        # Use the norm you already defined.
        self.output_norm = nn.LayerNorm(1024)

    def forward(self, partial_pts, partial_feats):
        # Split features
        dino = partial_feats[..., 1024:]

        # Branch-specific input normalisation
        partial_pts = self.geo_input_norm(partial_pts)
        dino = self.sem_input_norm(dino)

        # Separate geometry and semantic encoders
        z_geo = self.geometry_encoder(partial_pts)
        z_sem = self.semantic_encoder(dino)

        # Residual contextual refinement per branch
        z_geo = z_geo + self.geo_context(z_geo)
        z_sem = z_sem + self.sem_context(z_sem)

        # Fuse the two modalities while keeping the representation compact
        fused = self.fusion_norm(z_geo + z_sem)

        # Global aggregation
        latent = self.pool(fused)
        latent = self.projection(latent)
        latent = self.output_norm(latent)
        latent = F.normalize(latent, dim=-1)

        return latent


import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoInspiredDecoder(nn.Module):
    """
    Simple implicit decoder for the DinoComplete-inspired baseline.

    Deliberately excludes:

        - Fourier features
        - residual blocks
        - latent reinjection
        - skip connections

    so that the proposed architecture can be evaluated fairly.
    """

    def __init__(
        self,
        latent_dim=1024,
        coord_dim=128,
        hidden_dim=512,
        dropout=0.1,
    ):
        super().__init__()

        self.coord_encoder = nn.Sequential(
            nn.Linear(3, coord_dim),
            nn.LayerNorm(coord_dim),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(coord_dim + latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, query_coords, latent):

        batch_size, num_queries, _ = query_coords.shape

        coord_features = self.coord_encoder(query_coords)

        latent = latent.unsqueeze(1)
        latent = latent.expand(-1, num_queries, -1)

        features = torch.cat(
            [coord_features, latent],
            dim=-1,
        )

        logits = self.decoder(features)

        return logits.squeeze(-1)