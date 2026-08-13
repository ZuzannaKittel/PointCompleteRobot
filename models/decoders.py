import torch
import torch.nn as nn
import torch.nn.functional as F

class FourierEncoding(nn.Module):
    """
    Fourier positional encoding for continuous 3D coordinates.

    Input:
        [B, Q, 3]

    Output:
        [B, Q, 3 + 2 * 3 * num_bands]

    For num_bands=6:
        3 + 2*3*6 = 39 dimensions.
    """
    def __init__(self, num_bands=6):
        super().__init__()
        frequencies = (
            2.0 ** torch.arange(
                num_bands,
                dtype=torch.float32,
            )
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
    Residual MLP block used in the final decoder ablation.
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
    Configurable shape-conditioned implicit decoder.

    hidden_dim is fixed at 384 across all ablation stages so it isn't a
    confound -- each stage only adds one architectural component.

    Ablation stages:

        1. baseline
           Fourier=False
           Residual=False

        2. fourier
           Fourier=True
           Residual=False

        3. fourier_residual
           Fourier=True
           Residual=True

    All variants use the same global-latent conditioning mechanism.
    """
    def __init__(
        self,
        latent_dim=1024,
        hidden_dim=384,
        use_fourier=False,
        use_residual=False,
        num_fourier_bands=6,
        num_residual_blocks=2,
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

        # Encode each query coordinate into the decoder hidden space.
        self.coord_encoder = nn.Sequential(
            nn.Linear(coord_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
        )

        # Project the global object latent into conditioning parameters
        # for the coordinate representation.
        self.latent_condition = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim * 2),
        )

        if not use_residual:
            self.decoder = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Linear(hidden_dim, 1),
            )
        else:
            # Residual refinement is added only in the final ablation.
            self.fc1 = nn.Linear(
                hidden_dim,
                hidden_dim,
            )
            self.blocks = nn.ModuleList([
                ResidualBlock(
                    hidden_dim,
                    dropout=dropout,
                )
                for _ in range(num_residual_blocks)
            ])
            self.out = nn.Linear(
                hidden_dim,
                1,
            )

    def forward(self, query_pts, latent_vector):
        _, Q, _ = query_pts.shape

        query_pts = self.positional_encoding(query_pts)
        coord_feats = self.coord_encoder(query_pts)

        # Generate shape-dependent scale and bias.
        conditioning = self.latent_condition(latent_vector)

        gamma, beta = conditioning.chunk(
            2,
            dim=-1,
        )

        gamma = gamma.unsqueeze(1)
        beta = beta.unsqueeze(1)

        # Shape-conditioned coordinate representation.
        coord_feats = (
            (1.0 + gamma) * coord_feats
            + beta
        )

        if not self.use_residual:
            logits = self.decoder(coord_feats)

        else:
            x = F.gelu(self.fc1(coord_feats))

            for block in self.blocks:
                x = block(x)

            logits = self.out(x)

        return logits.squeeze(-1)