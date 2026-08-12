import torch
import torch.nn as nn
import torch.nn.functional as F


class GeometryEncoder(nn.Module):
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
    def __init__(self, input_dim=384, hidden_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

    def forward(self, x):
        return self.net(x)


class PointTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim=512,
        num_heads=8,
        mlp_ratio=4,
        dropout=0.1,
    ):
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


class DinoCompleteBaselineEncoder(nn.Module):

    def __init__(self):
        super().__init__()

        self.geo_input_norm = nn.LayerNorm(3)
        self.sem_input_norm = nn.LayerNorm(384)

        self.geometry_encoder = GeometryEncoder()
        self.semantic_encoder = SemanticEncoder()

        self.geo_context = PointTransformerBlock(
            embed_dim=512,
            num_heads=8,
        )

        self.sem_context = PointTransformerBlock(
            embed_dim=512,
            num_heads=8,
        )

        # Preserve both modalities instead of adding them.
        self.fusion = nn.Sequential(
            nn.Linear(1024, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
        )

        self.output_projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
        )

    def forward(self, partial_pts, partial_feats):

        if partial_feats.shape[-1] != 1408:
            raise ValueError(
                f"Expected 1408-D features, "
                f"got {partial_feats.shape[-1]}"
            )

        dino = partial_feats[..., 1024:1408]

        partial_pts = self.geo_input_norm(partial_pts)
        dino = self.sem_input_norm(dino)

        z_geo = self.geometry_encoder(partial_pts)
        z_sem = self.semantic_encoder(dino)

        z_geo = z_geo + self.geo_context(z_geo)
        z_sem = z_sem + self.sem_context(z_sem)

        # Concatenate instead of destroying modality information.
        fused = torch.cat(
            [z_geo, z_sem],
            dim=-1,
        )

        fused = self.fusion(fused)

        # Preserve both local statistics.
        mean_feat = fused.mean(dim=1)
        max_feat = fused.max(dim=1).values

        latent = torch.cat(
            [mean_feat, max_feat],
            dim=-1,
        )

        latent = self.output_projection(latent)

        return latent


import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoInspiredDecoder(nn.Module):

    def __init__(
        self,
        latent_dim=1024,
        coord_dim=128,
        hidden_dim=384,
    ):
        super().__init__()

        self.coord_encoder = nn.Sequential(
            nn.Linear(3, coord_dim),
            nn.GELU(),
            nn.Linear(coord_dim, coord_dim),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(coord_dim + latent_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),

            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, query_coords, latent):

        coord_features = self.coord_encoder(query_coords)

        latent = latent.unsqueeze(1)
        latent = latent.expand(
            -1,
            query_coords.shape[1],
            -1,
        )

        features = torch.cat(
            [coord_features, latent],
            dim=-1,
        )

        logits = self.decoder(features)

        return logits.squeeze(-1)