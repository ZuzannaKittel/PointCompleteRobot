import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalFeatureEncoder(nn.Module):
    """
    Multi-modal point feature encoder.

    The encoder projects Utonia (and optionally DINO) descriptors into a
    shared feature space, extracts higher-level point-wise representations
    using an MLP, and aggregates them into a single global latent vector
    through attention pooling and max pooling.

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
                nn.Linear(1024, 1024),
                nn.GELU(),
            )
            self.dino_projection = None

        elif input_feat_dim == 1408:
            # ScanNet++ finetuning
            self.utonia_projection = nn.Sequential(
                nn.Linear(1024, 1024),
                nn.GELU(),
            )

            self.dino_projection = nn.Sequential(
                nn.Linear(384, 1024),
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
            nn.Linear(256, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.LayerNorm(512),
            nn.GELU(),
        )

        self.output_dim = latent_dim * 2

    def forward(self, partial_pts, partial_feats):

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

        # Preserve pre-attention point descriptors for residual learning.
        residual = x

        # Self-attention enables every visible point to attend to every
        # other observed point, capturing long-range geometric relationships
        # before global aggregation.
        x = self.transformer(x)

        # Residual connection improves optimization stability and preserves
        # local point information alongside contextual features.
        x = x + residual

        # Help reduce overfitting.
        x = self.transformer_dropout(x)

        # Refine the contextualized point descriptors after attention.
        x = self.encoder(x)

        # Aggregate contextualized point descriptors into a global object
        # representation using complementary average and maximum pooling.
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values

        latent = torch.cat((mean_pool, max_pool), dim=-1)
        return latent



class FourierEncoding(nn.Module):
    """
    Fourier positional encoding for continuous 3D coordinates.

    Input:
        [B, Q, 3]

    Output:
        [B, Q, 3 + 2 * 3 * num_bands]

    For num_bands=10:
        3 + 2*3*10 = 63 dimensions.
    """

    def __init__(self, num_bands=10):
        super().__init__()

        frequencies = (
            2.0 ** torch.arange(num_bands, dtype=torch.float32)
        ) * torch.pi

        self.register_buffer("freq", frequencies)

    def forward(self, x):
        outputs = [x]

        for f in self.freq:
            outputs.append(torch.sin(f * x))
            outputs.append(torch.cos(f * x))

        return torch.cat(outputs, dim=-1)


class ResidualBlock(nn.Module):
    """
    Residual MLP block used in the final ablation stage.
    """

    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x

        x = self.fc1(x)
        x = self.norm1(x)
        x = F.gelu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = self.norm2(x)

        x = x + residual

        return F.gelu(x)


class ImplicitDecoderBasic(nn.Module):
    """
    Configurable implicit occupancy decoder.

    Ablation stages:

        1. Baseline
           hidden_dim=256
           Fourier=False
           Residual=False

        2. Baseline + hidden_dim=384
           hidden_dim=384
           Fourier=False
           Residual=False

        3. Baseline + hidden_dim=384 + Fourier
           hidden_dim=384
           Fourier=True
           Residual=False

        4. Baseline + hidden_dim=384 + Fourier + Residual
           hidden_dim=384
           Fourier=True
           Residual=True
    """

    def __init__(
        self,
        latent_dim=1024,
        hidden_dim=256,
        use_fourier=False,
        use_residual=False,
        num_fourier_bands=10,
        num_residual_blocks=4,
        dropout=0.1,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim
        self.use_fourier = use_fourier
        self.use_residual = use_residual

        if use_fourier:
            self.positional_encoding = FourierEncoding(
                num_bands=num_fourier_bands
            )
            coord_dim = 3 + 2 * 3 * num_fourier_bands
        else:
            self.positional_encoding = nn.Identity()
            coord_dim = 3

        # Coordinate encoder: same structure as the original decoder.
        self.coord_encoder = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        if not use_residual:
            # Original GitHub decoder.
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim + latent_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )

        else:
            # Residual decoder used in the final ablation.
            self.fc1 = nn.Linear(
                hidden_dim + latent_dim,
                hidden_dim
            )

            self.blocks = nn.ModuleList([
                ResidualBlock(
                    hidden_dim,
                    dropout=dropout
                )
                for _ in range(num_residual_blocks)
            ])

            self.fc_reinject = nn.Linear(
                hidden_dim + latent_dim,
                hidden_dim
            )

            self.reinject_dropout = nn.Dropout(dropout)

            self.final_block = ResidualBlock(
                hidden_dim,
                dropout=dropout
            )

            self.out = nn.Linear(hidden_dim, 1)

            nn.init.normal_(
                self.out.weight,
                mean=0.0,
                std=0.01
            )
            nn.init.constant_(self.out.bias, 0.0)

    def forward(self, query_pts, latent_vector):
        _, Q, _ = query_pts.shape

        query_pts = self.positional_encoding(query_pts)
        coord_feats = self.coord_encoder(query_pts)

        latent = latent_vector.unsqueeze(1).expand(
            -1, Q, -1
        )

        if not self.use_residual:
            # Baseline, hidden_dim, and Fourier experiments.
            x = torch.cat(
                [coord_feats, latent],
                dim=-1
            )

            logits = self.decoder(x)

        else:
            # Final residual experiment.
            x = torch.cat(
                [coord_feats, latent],
                dim=-1
            )

            x = F.gelu(self.fc1(x))

            for block in self.blocks:
                x = block(x)

            # Midway latent reinjection.
            x = torch.cat(
                [x, latent],
                dim=-1
            )

            x = F.gelu(self.fc_reinject(x))
            x = self.reinject_dropout(x)

            # Final residual refinement.
            x = self.final_block(x)

            logits = self.out(x)

        return logits.squeeze(-1)