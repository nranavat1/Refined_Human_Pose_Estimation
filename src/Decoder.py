import torch
from torch.nn import (
    Module,
    ModuleList,
    Linear,
    Conv2d,
    Upsample,
    ReLU,
    Parameter
)

import backbone_models.LayerNorm as LN
import backbone_models.TransformerBlock as TransformerBlock
import backbone_models.FeaturePyramidNetwork as FPN

class Decoder(Module):
    # Transformer-based Decoder with optional FPN refinement

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int = 1,
        num_keypoints: int = 17,
        upsample_stages: int = 2,
        dropout: float = 0.1,
        use_fpn: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_keypoints = num_keypoints
        self.upsample_stages = upsample_stages
        self.dropout = dropout
        self.use_fpn = use_fpn

        self.transformer_layers = ModuleList()
        for _ in range(self.num_layers):
            self.transformer_layers.append(
                TransformerBlock(embed_dim, hidden_dim, num_heads, dropout)
            )

        if self.use_fpn:
            self.fpn = FPN(in_channels=embed_dim)

        self.upsample_blocks = ModuleList()
        for _ in range(self.upsample_stages):
            self.upsample_blocks.append(
                torch.nn.Sequential(
                    Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                    Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
                    ReLU(inplace=True),
                )
            )

        self.head = Conv2d(embed_dim, num_keypoints, kernel_size=1)

    def forward(self, x):
        for layer in self.transformer_layers:
            x, _ = layer(x)

        x = x[:, 1:, :]  # shape (N, P, D)
        B, P, D = x.shape
        H = W = int(P ** 0.5)
        x = x.permute(0, 2, 1).reshape(B, D, H, W)  # reshape to feature map

        if self.use_fpn:
            x = self.fpn(x)

        for block in self.upsample_blocks:
            x = block(x)

        heatmaps = self.head(x)
        return heatmaps
