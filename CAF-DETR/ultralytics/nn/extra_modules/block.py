import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from ..modules.conv import Conv
from ..modules.block import  C2f

from timm.layers import  DropPath

__all__ = ['AIFB', 'CAFM','SPDConv']

class SPDConv(nn.Module):
    def __init__(self, inc, ouc, dimension=1):
        super().__init__()
        self.d = dimension
        self.conv = Conv(inc * 4, ouc, k=3)

    def forward(self, x):
        x = torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)
        x = self.conv(x)
        return x


class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)
        self.fc1 = nn.Conv2d(in_features, hidden_features * 2, 1)
        self.dwconv = nn.Sequential(
            nn.Conv2d(hidden_features, hidden_features, kernel_size=3, stride=1, padding=1, bias=True, groups=hidden_features),
            act_layer()
        )
        self.fc2 = nn.Conv2d(hidden_features, out_features, 1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x_shortcut = x
        x, v = self.fc1(x).chunk(2, dim=1)
        x = self.dwconv(x) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x_shortcut + x

class MultiscaleDynamicConvolutionModule(nn.Module):

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=11):
        super().__init__()
        self.dwconv = nn.ModuleList([
            nn.Conv2d(in_channels, in_channels, square_kernel_size, padding=square_kernel_size//2, groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(1, band_kernel_size), padding=(0, band_kernel_size//2), groups=in_channels),
            nn.Conv2d(in_channels, in_channels, kernel_size=(band_kernel_size, 1), padding=(band_kernel_size//2, 0), groups=in_channels)
        ])

        self.bn = nn.BatchNorm2d(in_channels)
        self.act = nn.SiLU()

        self.dkw = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels * 3, 1)
        )

    def forward(self, x):
        x_dkw = rearrange(self.dkw(x), 'bs (g ch) h w -> g bs ch h w', g=3)
        x_dkw = F.softmax(x_dkw, dim=0)
        x = torch.stack([self.dwconv[i](x) * x_dkw[i] for i in range(len(self.dwconv))]).sum(0)
        return self.act(self.bn(x))

class AdaptiveInceptionFusionModule(nn.Module):
    def __init__(self, channel=256, kernels=[3, 5]):
        super().__init__()
        self.groups = len(kernels)
        min_ch = channel // 2

        self.convs = nn.ModuleList([])
        for ks in kernels:
            self.convs.append(MultiscaleDynamicConvolutionModule(min_ch, ks, ks * 3 + 2))
        self.conv_1x1 = Conv(channel, channel, k=1)

    def forward(self, x):
        _, c, _, _ = x.size()
        x_group = torch.split(x, [c // 2, c // 2], dim=1)
        x_group = torch.cat([self.convs[i](x_group[i]) for i in range(len(self.convs))], dim=1)
        x = self.conv_1x1(x_group)
        return x

class AdaptiveInceptionFusionBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0):
        super().__init__()
        self.norm1 = nn.BatchNorm2d(dim)
        self.norm2 = nn.BatchNorm2d(dim)
        self.mixer = AdaptiveInceptionFusionModule(dim)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp = ConvolutionalGLU(dim)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.layer_scale_2 = nn.Parameter(
            layer_scale_init_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        x = x + self.drop_path(self.layer_scale_1.unsqueeze(-1).unsqueeze(-1) * self.mixer(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2.unsqueeze(-1).unsqueeze(-1) * self.mlp(self.norm2(x)))
        return x

class AIFB(C2f):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        self.m = nn.ModuleList(AdaptiveInceptionFusionBlock(self.c) for _ in range(n))

class CrosslayerAttentionModule(nn.Module):

    def __init__(self, dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim = dim // num_heads
        self.all_head_dim = all_head_dim = head_dim * self.num_heads

        self.q = Conv(dim, all_head_dim, 1, act=False)
        self.kv = Conv(dim, all_head_dim * 2, 1, act=False)
        self.proj = Conv(all_head_dim, dim, 1, act=False)
        self.pe = Conv(all_head_dim, dim, 7, 1, 3, g=dim, act=False)

    def forward(self, x, upper_feat):
        B, C, H, W = x.shape
        N = H * W
        _, _, H_up, W_up = upper_feat.shape

        q = self.q(x).view(B, self.num_heads, self.head_dim, N).permute(0, 1, 3, 2)

        kv = self.kv(upper_feat).view(B, self.num_heads, 2 * self.head_dim, H_up * W_up).permute(0, 1, 3, 2)
        k, v = kv.split(self.head_dim, dim=3)

        sim = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = sim.softmax(dim=-1)
        attention_output = (attn @ v)

        # Reshape and apply positional encoding
        attention_output = attention_output.transpose(2, 3).reshape(B, self.all_head_dim, H,
                                                                    W)
        v_reshaped = v.transpose(2, 3).reshape(B, self.all_head_dim, H_up, W_up)
        v_pe = self.pe(v_reshaped)
        v_pe = F.interpolate(v_pe, size=(H, W), mode='bilinear', align_corners=False)
        attention_output = attention_output + v_pe

        return self.proj(attention_output)


class CrosslayerAttentiveFusionModule(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=2):
        super().__init__()
        self.attn = CrosslayerAttentionModule(dim, num_heads=num_heads)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            Conv(dim, mlp_hidden_dim, 1),
            Conv(mlp_hidden_dim, dim, 1, act=False)
        )

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x, upper_feat):
        x = x + self.attn(x, upper_feat)
        return x + self.mlp(x)

class CAFM(nn.Module):

    def __init__(self, c1, c_up, c2, n=1, mlp_ratio=2.0, e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        assert c_ % 32 == 0, "Hidden channels must be a multiple of 32."

        self.cv1 = Conv(c1, c_, 1, 1)
        self.cvup = Conv(c_up, c_, 1, 1)
        self.cv2 = Conv((1 + n) * c_, c2, 1)

        self.num_layers = n
        for i in range(n):
            layer = CrosslayerAttentiveFusionModule(c_, c_ // 32, mlp_ratio)
            self.add_module(f"attnlayer_{i}", layer)

    def forward(self, x):
        upper_feat = x[1]
        x = self.cv1(x[0])
        upper_feat = self.cvup(upper_feat)
        y = [x]
        for i in range(self.num_layers):
            layer = getattr(self, f"attnlayer_{i}")
            attened = layer(y[-1], upper_feat)
            y.append(attened)

        y = self.cv2(torch.cat(y, 1))
        return y
