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