import torch
from torch import nn
import einops

__all__ = ['AdaptiveSpatialAdditiveModule']

class AdaptiveSpatialAdditiveModule(nn.Module):
    def __init__(self, in_dims=512, token_dim=256, num_heads=1):
        super().__init__()
        self.in_dims = in_dims
        self.token_dim = in_dims
        self.num_heads = num_heads

        self.to_query = nn.Linear(in_dims, self.token_dim * num_heads)
        self.to_key = nn.Linear(in_dims, self.token_dim * num_heads)

        self.w_g = nn.Parameter(torch.randn(self.token_dim * num_heads, 1))
        self.scale_factor = self.token_dim ** -0.5
        self.Proj = nn.Linear(self.token_dim * num_heads, self.token_dim * num_heads)
        self.final = nn.Linear(self.token_dim * num_heads, self.token_dim)

        self.spatial_conv = nn.Conv2d(in_dims, in_dims, 3, padding=1, groups=in_dims)
        self.spatial_gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_dims, max(32, in_dims // 8), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(32, in_dims // 8), in_dims, 1),
            nn.Sigmoid()
        )

    def forward(self, x_4d):
        B, C, H, W = x_4d.size()

        spatial_features = self.spatial_conv(x_4d)
        spatial_weights = self.spatial_gate(x_4d)
        enhanced_x = x_4d + spatial_features * spatial_weights

        x_flat = enhanced_x.flatten(2).transpose(1, 2)
        query = self.to_query(x_flat)
        key = self.to_key(x_flat)

        query = torch.nn.functional.normalize(query, dim=-1)
        key = torch.nn.functional.normalize(key, dim=-1)

        query_weight = query @ self.w_g
        A = query_weight * self.scale_factor
        A = torch.nn.functional.normalize(A, dim=1)

        G = torch.sum(A * query, dim=1)
        G = einops.repeat(G, "b d -> b repeat d", repeat=key.shape[1])

        out = self.Proj(G * key) + query
        out = self.final(out)

        out_4d = out.transpose(1, 2).reshape((B, C, H, W))

        return out_4d