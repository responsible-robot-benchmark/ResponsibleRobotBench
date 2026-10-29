import torch
import torch.nn as nn


class FiLM(nn.Module):
    """Feature-wise Linear Modulation"""
    def __init__(self, out_feat_dim: int, cond_feat_dim: int):
        'Args:\n    out_feat_dim: Output feature dimension\n    cond_feat_dim: Condition feature dimension'
        super().__init__()
        self.out_feat_dim = out_feat_dim
        self.cond_feat_dim = cond_feat_dim
        
        # MLP predicts gamma and beta
        self.mlp = nn.Sequential(
            nn.Linear(cond_feat_dim, cond_feat_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cond_feat_dim, out_feat_dim * 2)
        )

    def forward(self, out_feats: torch.Tensor, cond_feats: torch.Tensor) -> torch.Tensor:
        'Args:\n    out_feats: [B, T, out_feat_dim] Output features\n    cond_feats: [B, T, cond_feat_dim] Condition features\nReturns:\n    modulated: [B, T, out_feat_dim] Modulated output features'
        B, T, D = out_feats.shape
        assert D == self.out_feat_dim
        
        # Predicts gamma and beta
        gb = self.mlp(cond_feats)  # [B, T, 2*D]
        gb = gb.view(B, T, 2, D)   # [B, T, 2, D]
        gamma = gb[:, :, 0]        # [B, T, D]
        beta = gb[:, :, 1]         # [B, T, D]
        
        # FiLM: (1 + gamma) * x + beta
        out = out_feats * (1.0 + gamma) + beta
        return out
