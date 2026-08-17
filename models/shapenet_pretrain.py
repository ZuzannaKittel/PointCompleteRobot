import torch
import torch.nn as nn

from models.encoders import (
    GeometryEncoder,
    FeatureEncoder,
    PointTransformerBlock,
)


class ShapeNetBranchPretrainEncoder(nn.Module):
    """
    Lightweight ShapeNet pretraining encoder.

    The encoder reproduces the modality-specific branch used by
    the corresponding ScanNet AblationEncoder and is used only
    to pretrain transferable geometric features on ShapeNet.

    mode='utonia':
        Utonia -> FeatureEncoder -> contextual transformer

    mode='coords':
        XYZ -> GeometryEncoder -> contextual transformer

    The modality-specific branch is followed by a temporary
    projection so that the resulting representation is compatible
    with the 1024-D implicit decoder used during pretraining.
    """

    def __init__(self, mode):
        super().__init__()

        if mode not in {"utonia", "coords"}:
            raise ValueError(
                f"Unknown ShapeNet pretraining mode: {mode}"
            )

        self.mode = mode

        if mode == "utonia":
            self.utonia_input_norm = nn.LayerNorm(1024)

            # Exactly the same modules used by AblationEncoder.
            self.utonia_encoder = FeatureEncoder(
                input_dim=1024,
                hidden_dim=512,
            )

            self.utonia_context = PointTransformerBlock(
                embed_dim=512,
                num_heads=8,
                mlp_ratio=2,
                dropout=0.1,
            )

        else:
            # Exactly the same modules used by AblationEncoder.
            self.geometry_encoder = GeometryEncoder()

            self.geo_context = PointTransformerBlock(
                embed_dim=512,
                num_heads=8,
                mlp_ratio=2,
                dropout=0.1,
            )

        # ShapeNet-only projection from the single modality to
        # the 1024-D representation expected by the decoder.
        self.branch_projection = nn.Sequential(
            nn.Linear(512, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
        )

        # This module has exactly the same structure and dimensions
        # as the final projection in AblationEncoder and can therefore
        # be transferred to ScanNet.
        self.global_projection = nn.Sequential(
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.LayerNorm(1024),
        )

    def forward(self, partial_pts, partial_feats):
        if self.mode == "utonia":
            utonia = partial_feats[..., :1024]

            utonia = self.utonia_input_norm(utonia)

            z = self.utonia_encoder(utonia)
            z = self.utonia_context(z)

        else:
            z = self.geometry_encoder(partial_pts)
            z = self.geo_context(z)

        z = self.branch_projection(z)

        mean_feat = z.mean(dim=1)
        max_feat = z.max(dim=1).values

        latent = torch.cat(
            [mean_feat, max_feat],
            dim=-1,
        )

        latent = self.global_projection(latent)

        return latent