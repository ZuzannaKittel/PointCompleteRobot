import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiModalFeatureEncoder(nn.Module):
    """
    Encodes point-wise features (Utonia or Utonia+DINO)
    into a global latent representation.

    ShapeNet:
        1024 -> 1024 -> 512

    ScanNet++:
        1408 -> 1024 -> 512

    The only thing that changes is input_feat_dim.
    """

    def __init__(self, input_feat_dim, latent_dim=512):
        super().__init__()
        self.input_feat_dim = input_feat_dim
        self.latent_dim = latent_dim

        self.input_projection = nn.Sequential(
            nn.Linear(input_feat_dim, 1024),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        self.encoder = nn.Sequential(
            nn.Linear(1024, latent_dim),
            nn.GELU(),
            nn.Dropout(0.15),
        )

        self.output_dim = latent_dim * 2

    def forward(self, partial_feats):

        # [B,N,input_dim]
        x = self.input_projection(partial_feats)

        # [B,N,512]
        x = self.encoder(x)

        max_pool = torch.max(x, dim=1)[0]
        mean_pool = torch.mean(x, dim=1)
        latent = torch.cat([max_pool, mean_pool], dim=-1)
        latent = F.normalize(latent, dim=-1)

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