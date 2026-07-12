import torch
import torch.nn as nn

class MultiModalFeatureEncoder(nn.Module):
    """
    Takes pre-computed multi-modal point features (e.g. fused Utonia + DINOv2),
    projects them to a latent space, and pools them into a global shape embedding.
    """
    def __init__(self, input_feat_dim=2048, latent_dim=512):
        super().__init__()
        # NOTE: If your data factory fused_pointwise feature dimension is different 
        # (e.g., 768 or 1280), adjust 'input_feat_dim' to match it when instantiating.
        
        self.mlp = nn.Sequential(
            nn.Linear(input_feat_dim, 512),
            nn.GELU(),
            nn.Linear(512, latent_dim),
            nn.GELU()
        )

        self.latent_dim = latent_dim
        self.output_dim = 2 * latent_dim

        # Layer normalization to stabilize training and improve convergence
        self.input_norm = nn.LayerNorm(input_feat_dim)

    def forward(self, partial_feats):
        """
        Args:
            partial_feats: [B, N, input_feat_dim] tensor of point-wise features
        Returns:
            global_latent: [B, 2 * latent_dim] tensor representing the global shape embedding
        """
        # Step 1: Map complex fused features to latent shape space
        # Apply layer normalization to stabilize training and improve convergence
        x = self.mlp(self.input_norm(partial_feats))
        #x = self.mlp(partial_feats)  # [B, N, latent_dim]
        
        # Step 2: Extract global shape priors via symmetric Max Pooling
        max_pool = torch.max(x, dim=1)[0]
        mean_pool = torch.mean(x, dim=1)

        # Step 3: Concatenate pooled features to form a multi-modal global descriptor
        global_latent = torch.cat(
            [max_pool, mean_pool],
            dim=-1
        )

        return global_latent


class ImplicitDecoderDoubleLatent(nn.Module):
    """
    Decodes a continuous coordinate space conditioned on a multi-modal global latent shape vector.
    This version incorporates a second fusion step to enhance the interaction between the latent vector and the coordinate
    features, which can improve the model's ability to capture complex shape details.
    """
    def __init__(self, latent_dim=1024, hidden_dim=256):
        super().__init__()

        self.coord_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )

        # FIRST FUSION
        self.fc1 = nn.Linear(hidden_dim + latent_dim, hidden_dim)

        # SECOND FUSION (IMPORTANT CHANGE)
        self.fc2 = nn.Linear(hidden_dim + latent_dim, hidden_dim)

        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out = nn.Linear(hidden_dim, 1)

    def forward(self, query_pts, latent_vector):
        B, Q, _ = query_pts.shape

        coord_feats = self.coord_encoder(query_pts)

        latent = latent_vector.unsqueeze(1).expand(-1, Q, -1)

        # ---- FIRST FUSION ----
        x = torch.cat([coord_feats, latent], dim=-1)
        x = torch.nn.functional.gelu(self.fc1(x))

        # ---- SECOND FUSION (KEY IMPROVEMENT) ----
        x = torch.cat([x, latent], dim=-1)
        x = torch.nn.functional.gelu(self.fc2(x))

        # ---- FINAL MLP ----
        x = torch.nn.functional.gelu(self.fc3(x))
        logits = self.fc_out(x)

        return logits.squeeze(-1)
    

class ImplicitDecoderBasic(nn.Module):
    """
    Decodes a continuous coordinate space conditioned on a multi-modal global latent shape vector.
    This is a simpler version of the decoder that uses a single fusion step between the coordinate features and the latent vector.
    """
    def __init__(self, latent_dim=1024, hidden_dim=256):
        super().__init__()
        self.coord_encoder = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim + latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, query_pts, latent_vector):
        coord_feats = self.coord_encoder(query_pts) # [B, Q, hidden_dim]

        # Broadcast the global descriptor to align with individual queries
        latent_expanded = latent_vector.unsqueeze(1).expand(-1, Q, -1) # [B, Q, latent_dim]

        # Concatenate features and query continuous scalar fields
        combined = torch.cat([coord_feats, latent_expanded], dim=-1)