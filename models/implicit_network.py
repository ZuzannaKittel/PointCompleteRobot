import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalFeatureEncoder(nn.Module):
    """
    Multi-modal point feature encoder.

    The encoder projects Utonia (and optionally DINO) descriptors into a
    shared feature space, extracts higher-level point-wise representations
    using an MLP, and aggregates them into a single global latent vector
    through mean and max pooling.

    ShapeNet:
        Utonia (1024)

    ScanNet++:
        Utonia (1024) + DINO (384)

    Output:
        Global latent vector (1024-D)
    """

    def __init__(self, input_feat_dim, latent_dim=512):
        super().__init__()
        self.input_feat_dim = input_feat_dim
        if input_feat_dim not in (1024, 1408):
            raise ValueError(f"Unsupported feature dimension: {input_feat_dim}")
        
        self.latent_dim = latent_dim

        # Normalize fused multi-modal descriptors before relational reasoning.
        self.fusion_norm = nn.LayerNorm(1024)

        if input_feat_dim == 1024:
            # ShapeNet pretraining
            self.utonia_projection = nn.Sequential(
                nn.Linear(1024,1024),
                nn.GELU(),
            )
            self.dino_projection = None

        elif input_feat_dim == 1408:
            # ScanNet++ finetuning
            self.utonia_projection = nn.Sequential(
                nn.Linear(1024,1024),
                nn.GELU(),
            )

            self.dino_projection = nn.Sequential(
                nn.Linear(384,1024),
                nn.GELU(),
            )

            self.feature_fusion = nn.Sequential(
                nn.Linear(2048, 1024),
                nn.LayerNorm(1024),
                nn.GELU(),
            )

        # Reduce dimensionality before self-attention.
        # This keeps the transformer lightweight while still allowing
        # relational reasoning between visible points.
        self.pre_transformer = nn.Sequential(
            nn.Linear(1024, 256),
            nn.LayerNorm(256),
            nn.GELU(),
        )

        # Self-attention enables every observed point to exchange information
        # with the remaining visible geometry, allowing the network to capture
        # long-range structural relationships.
        self.transformer = nn.TransformerEncoder(

            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                dim_feedforward=512,
                dropout=0.1,
                activation="gelu",
                batch_first=True,
                norm_first=True,
            ),

            num_layers=1,
        )

        self.transformer_dropout = nn.Dropout(0.1)

        # Refine contextualized descriptors and expand them into the
        # final point-wise embedding used for global pooling.
        self.encoder = nn.Sequential(

            nn.Linear(256,512),
            nn.LayerNorm(512),
            nn.GELU(),

            nn.Linear(512,512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        # Final projection producing the global latent representation
        # consumed by the implicit decoder.
        self.global_projection = nn.LayerNorm(1024)

        self.output_dim = latent_dim * 2

    def forward(self, partial_feats):

        # [B,N,input_dim]
        if self.input_feat_dim == 1024:
            x = self.utonia_projection(partial_feats)
        else:
            utonia = partial_feats[..., :1024]
            dino = partial_feats[..., 1024:]

            utonia = self.utonia_projection(utonia)
            dino = self.dino_projection(dino)

            x = torch.cat((utonia, dino), dim=-1)

            x = self.feature_fusion(x)

        # Normalize fused descriptors.
        x = self.fusion_norm(x)

        # Compress descriptors before relational reasoning.
        x = self.pre_transformer(x)

        # Self-attention enables every visible point to attend to every
        # other observed point, capturing long-range geometric relationships
        # before global aggregation.
        x = self.transformer(x)

        # Help reduce overfitting
        x = self.transformer_dropout(x)

        # Refine the contextualized point descriptors after attention.
        x = self.encoder(x)

        # Aggregate local point descriptors into a global object representation
        # using complementary average and maximum pooling.
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values

        # Fuse complementary mean- and max-pooled statistics into the final
        # global shape embedding consumed by the implicit occupancy decoder.
        latent = torch.cat((mean_pool, max_pool), dim=-1)
        latent = self.global_projection(latent)

        return latent


class ImplicitDecoderBasic(nn.Module):
    """Implicit occupancy decoder."""

    def __init__(self, latent_dim=1024, hidden_dim=256):
        super().__init__()

        self.coord_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, query_pts, latent_vector):
        _, Q, _ = query_pts.shape

        coord_feats = self.coord_encoder(query_pts)
        latent = latent_vector.unsqueeze(1).expand(-1, Q, -1)
        x = torch.cat([coord_feats, latent], dim=-1)
        logits = self.decoder(x)

        return logits.squeeze(-1)