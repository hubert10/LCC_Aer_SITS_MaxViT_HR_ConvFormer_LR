"""
Link: https://arxiv.org/abs/2203.04838
FFM: Feature Fusion Module:

Feature Fusion Module (FFM) with two stages of (1) information exchange and (2) fusion

Inputs are: 
    (1): SITS feature maps:   512x16x16
    (2): Aerial feature maps: 512x16x16
Outpus are:
     Fused Features:   512x16x16

"""

import torch
import torch.nn as nn
from timm.layers import trunc_normal_
import math


# Stage 1
class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None):
        super(CrossAttention, self).__init__()
        assert (
            dim % num_heads == 0
        ), f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5
        self.kv1 = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.kv2 = nn.Linear(dim, dim * 2, bias=qkv_bias)

    def forward(self, x1, x2):
        B, N, C = x1.shape
        q1 = (
            x1.reshape(B, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        q2 = (
            x2.reshape(B, -1, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
            .contiguous()
        )
        k1, v1 = (
            self.kv1(x1)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )
        k2, v2 = (
            self.kv2(x2)
            .reshape(B, -1, 2, self.num_heads, C // self.num_heads)
            .permute(2, 0, 3, 1, 4)
            .contiguous()
        )

        # Adopted from Efficient Attention: Attention with Linear Complexities
        # https://arxiv.org/abs/1812.01243

        # NOTE: The order of matrix multiplication in self-attention
        # primarily affects computation complexity, but it does not
        # reduce the number of parameters.

        # The number of floating-point operations (FLOPs) in matrix multiplication
        # refers to the number of arithmetic operations (additions
        # and multiplications) required to perform the multiplication
        # between two matrices. Specifically, it measures how computationally
        # expensive the multiplication is in terms of the number of
        # floating-point calculations needed to generate the result.

        ctx1 = (k1.transpose(-2, -1) @ v1) * self.scale
        ctx1 = ctx1.softmax(dim=-2)
        ctx2 = (k2.transpose(-2, -1) @ v2) * self.scale
        ctx2 = ctx2.softmax(dim=-2)

        x1 = (q1 @ ctx2).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()
        x2 = (q2 @ ctx1).permute(0, 2, 1, 3).reshape(B, N, C).contiguous()

        return x1, x2


class CrossPath(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.LayerNorm):
        super().__init__()
        self.channel_proj1 = nn.Linear(dim, dim // reduction * 2)
        self.channel_proj2 = nn.Linear(dim, dim // reduction * 2)
        self.act1 = nn.ReLU(inplace=True)
        self.act2 = nn.ReLU(inplace=True)
        self.cross_attn = CrossAttention(dim // reduction, num_heads=num_heads)
        self.end_proj1 = nn.Linear(dim // reduction * 2, dim)
        self.end_proj2 = nn.Linear(dim // reduction * 2, dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        y1, u1 = self.act1(self.channel_proj1(x1)).chunk(2, dim=-1)
        y2, u2 = self.act2(self.channel_proj2(x2)).chunk(2, dim=-1)
        v1, v2 = self.cross_attn(u1, u2)
        y1 = torch.cat((y1, v1), dim=-1)
        y2 = torch.cat((y2, v2), dim=-1)
        out_x1 = self.norm1(x1 + self.end_proj1(y1))
        out_x2 = self.norm2(x2 + self.end_proj2(y2))
        return out_x1, out_x2


# Stage 2
class ChannelEmbed(nn.Module):
    def __init__(
        self, in_channels, out_channels, reduction=1, norm_layer=nn.BatchNorm2d
    ):
        super(ChannelEmbed, self).__init__()
        self.out_channels = out_channels
        self.residual = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.channel_embed = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // reduction, kernel_size=1, bias=True),
            nn.Conv2d(
                out_channels // reduction,
                out_channels // reduction,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=True,
                groups=out_channels // reduction,
            ),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels // reduction, out_channels, kernel_size=1, bias=True
            ),
            norm_layer(out_channels),
        )
        self.norm = norm_layer(out_channels)

    def forward(self, x, H, W):
        # print("x forward:", x.shape)
        B, N, _C = x.shape
        x = x.permute(0, 2, 1).reshape(B, _C, H, W).contiguous()
        residual = self.residual(x)
        x = self.channel_embed(x)
        out = self.norm(residual + x)
        # print("out:", out.shape)
        return out


class FeatureFusionModule(nn.Module):
    def __init__(self, dim, reduction=1, num_heads=None, norm_layer=nn.BatchNorm2d):
        super().__init__()
        self.cross = CrossPath(dim=dim, reduction=reduction, num_heads=num_heads)
        self.channel_emb = ChannelEmbed(
            in_channels=dim * 2,
            out_channels=dim,
            reduction=reduction,
            norm_layer=norm_layer,
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2):
        # print("x1 forward:", x1.shape)
        # print("x2 forward:", x2.shape)

        B, C, H, W = x1.shape
        x1 = x1.flatten(2).transpose(1, 2)
        x2 = x2.flatten(2).transpose(1, 2)

        # print("x1 flatten:", x1.shape)
        # print("x2 flatten:", x2.shape)
        # x1 flatten: torch.Size([4, 256, 512])
        # x2 flatten: torch.Size([4, 256, 512])

        x1, x2 = self.cross(x1, x2)
        merge = torch.cat((x1, x2), dim=-1)  # torch.Size([4, 256, 1024])
        # print("merge cat:", merge.shape)
        merge = self.channel_emb(merge, H, W)  # torch.Size([4, 512, 16, 16])
        # print("merge:", merge.shape)
        return merge


# Note: We can also proceed with the method used in the paper:
# MFTransNet: A Multi-Modal Fusion with CNN-Transformer
# Network for Semantic Segmentation of HSR Remote
# Sensing Images (https://www.mdpi.com/2227-7390/11/3/722)

# page 10, from formulae (9) to (13) in which a channel and spatial
# attention is applied to each feature maps in the final stages
# of the Fusion
