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

        # Preserve pre-attention point descriptors for residual learning.
        residual = x

        # Self-attention enables every visible point to attend to every
        # other observed point, capturing long-range geometric relationships
        # before global aggregation.
        x = self.transformer(x)

        # Residual connection improves optimization stability and preserves
        # local point information alongside contextual features.
        x = x + residual

        # Help reduce overfitting
        x = self.transformer_dropout(x)

        # Refine the contextualized point descriptors after attention.
        x = self.encoder(x)

        # Aggregate contextualized point descriptors into a global object
        # representation using complementary average and maximum pooling.
        mean_pool = x.mean(dim=1)
        max_pool = x.max(dim=1).values

        latent = torch.cat((mean_pool, max_pool), dim=-1)

        return latent

class ResidualBlock(nn.Module):

    def __init__(self, hidden_dim):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.dropout = nn.Dropout(0.1)

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

class FourierEncoding(nn.Module):
    """
    Fourier positional encoding for continuous 3D coordinates.

    Input:
        [B, Q, 3]

    Output:
        [B, Q, 3 + 2 * 3 * num_bands]
    """

    def __init__(self, num_bands=10):
        super().__init__()

        frequencies = (2.0 ** torch.arange(num_bands, dtype=torch.float32)) * torch.pi
        self.register_buffer("freq", frequencies)

    def forward(self, x):

        out = [x]

        for f in self.freq:
            out.append(torch.sin(f * x))
            out.append(torch.cos(f * x))

        return torch.cat(out, dim=-1)


class ImplicitDecoderBasic(nn.Module):
    """
    Implicit occupancy decoder.

    The decoder predicts the occupancy probability of arbitrary 3D query
    coordinates conditioned on the global latent shape representation.
    A second latent injection is used midway through the network to
    reinforce global shape information during occupancy prediction.

    --- ABLATION study aka EXPERIMENTS ---
        a) change hidden_dim to 384
        b) Fourier positional encoding
        c) residual blocks
    """

    def __init__(self, latent_dim=1024, hidden_dim=384):
        super().__init__()

        num_bands = 10

        # Multi-scale positional encoding of query coordinates.
        self.positional_encoding = FourierEncoding(num_bands=num_bands)

        coord_dim = 3 + 2 * 3 * num_bands

        # Encode Fourier-enhanced query coordinates.
        self.coord_encoder = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),

            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # Initial conditioning on the global latent representation.
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)

        # Residual refinement blocks.
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
            ResidualBlock(hidden_dim),
        ])

        # Midway latent reinjection.
        self.fc_reinject = nn.Linear(
            hidden_dim + latent_dim,
            hidden_dim,
        )

        self.reinject_dropout = nn.Dropout(0.1)

        # Additional refinement.
        self.final_block = ResidualBlock(hidden_dim)

        # Occupancy prediction.
        self.out = nn.Linear(hidden_dim, 1)

        nn.init.normal_(self.out.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.out.bias, 0.0)

    def forward(self, query_pts, latent_vector):
        _, Q, _ = query_pts.shape

        # Apply Fourier positional encoding before coordinate encoding.
        query_pts = self.positional_encoding(query_pts)

        # Encode Fourier-enhanced query coordinates.
        coord_feats = self.coord_encoder(query_pts)

        # Broadcast the global latent representation to every query point.
        latent = latent_vector.unsqueeze(1).expand(-1, Q, -1)

        # Initial latent conditioning.
        x = torch.cat([coord_feats, latent], dim=-1)
        x = F.gelu(self.fc1(x))

        # Residual refinement.
        for block in self.blocks:
            x = block(x)

        # Inject the global shape descriptor again.
        x = torch.cat([x, latent], dim=-1)
        x = F.gelu(self.fc_reinject(x))
        x = self.reinject_dropout(x)

        # Final refinement.
        x = self.final_block(x)

        # Predict occupancies.
        logits = self.out(x)

        return logits.squeeze(-1)