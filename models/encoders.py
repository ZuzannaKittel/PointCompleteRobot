import torch
import torch.nn as nn


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


class FeatureEncoder(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512):
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


class AblationEncoder(nn.Module):
    """
    Encoder used for modality ablations.

    coord_dino:
        XYZ + DINO

    utonia_dino:
        Utonia + DINO

    utonia:
        Utonia only

    All variants produce a 1024-D global latent using
    mean + max pooling followed by projection.
    """

    def __init__(self, encoder_type):
        super().__init__()

        valid_types = {
            "coord_dino",
            "utonia_dino",
            "utonia",
        }

        if encoder_type not in valid_types:
            raise ValueError(
                f"Unknown encoder type '{encoder_type}'. "
                f"Available: {sorted(valid_types)}"
            )

        self.encoder_type = encoder_type

        self.geo_input_norm = nn.LayerNorm(3)
        self.utonia_input_norm = nn.LayerNorm(1024)
        self.dino_input_norm = nn.LayerNorm(384)

        if encoder_type == "coord_dino":
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

            self.fusion = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.GELU(),
                nn.LayerNorm(1024),
            )

        elif encoder_type == "utonia_dino":
            self.utonia_encoder = FeatureEncoder(
                input_dim=1024,
                hidden_dim=512,
            )

            self.semantic_encoder = SemanticEncoder()

            self.utonia_context = PointTransformerBlock(
                embed_dim=512,
                num_heads=8,
            )

            self.sem_context = PointTransformerBlock(
                embed_dim=512,
                num_heads=8,
            )

            self.fusion = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.GELU(),
                nn.LayerNorm(1024),
            )

        elif encoder_type == "utonia":
            self.utonia_encoder = FeatureEncoder(
                input_dim=1024,
                hidden_dim=512,
            )

            self.context = PointTransformerBlock(
                embed_dim=512,
                num_heads=8,
            )

            self.fusion = nn.Identity()

        # Mean + max pooling gives 2048 dimensions.
        # Project back to the 1024-D latent expected by the decoder.
        self.global_projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
        )

    def forward(self, partial_pts, partial_feats):
        if partial_feats.shape[-1] != 1408:
            raise ValueError(
                f"{self.encoder_type} expects 1408-D ScanNet++ features, "
                f"got {partial_feats.shape[-1]}"
            )

        dino = partial_feats[..., 1024:1408]
        utonia = partial_feats[..., :1024]

        if self.encoder_type == "coord_dino":
            partial_pts = self.geo_input_norm(partial_pts)
            dino = self.dino_input_norm(dino)

            z_geo = self.geometry_encoder(partial_pts)
            z_sem = self.semantic_encoder(dino)

            z_geo = z_geo + self.geo_context(z_geo)
            z_sem = z_sem + self.sem_context(z_sem)

            fused = torch.cat([z_geo, z_sem], dim=-1)
            fused = self.fusion(fused)

        elif self.encoder_type == "utonia_dino":
            utonia = self.utonia_input_norm(utonia)
            dino = self.dino_input_norm(dino)

            z_utonia = self.utonia_encoder(utonia)
            z_sem = self.semantic_encoder(dino)

            z_utonia = z_utonia + self.utonia_context(z_utonia)
            z_sem = z_sem + self.sem_context(z_sem)

            fused = torch.cat(
                [z_utonia, z_sem],
                dim=-1,
            )

            fused = self.fusion(fused)

        else:
            utonia = self.utonia_input_norm(utonia)

            z_utonia = self.utonia_encoder(utonia)
            z_utonia = z_utonia + self.context(z_utonia)

            fused = self.fusion(z_utonia)

        mean_feat = fused.mean(dim=1)
        max_feat = fused.max(dim=1).values

        latent = torch.cat(
            [mean_feat, max_feat],
            dim=-1,
        )

        latent = self.global_projection(latent)

        return latent