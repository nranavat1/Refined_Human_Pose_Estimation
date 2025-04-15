import torch
from torch.nn import (Linear, 
                        ModuleList,
                        Module, 
                        Parameter
                        )

import backbone_models.LayerNorm as LN
import backbone_models.TransformerBlock as TransformerBlock
import backbone_models.FeaturePyramidNetwork as FPN

class Encoder(Module):
    #vanilla ViT Encoder + FPN

    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_layers: int,
        num_heads: int=1,
        input_dims: tuple=(3,32,32),
        num_keypoints: int=17, #17 for COCO
        downsampling_ratio: int=16,
        dropout: float=0.1,
        dtype: torch.dtype=torch.float,
        loss_fn: str='softmax_loss',
        ):
        super().__init__()

        assert num_heads>0, "Number of heads must be greater than zero"

        C, H, W = input_dims
        assert H%downsampling_ratio==0, "Height must be divisible by patch_size"
        assert W%downsampling_ratio==0, "Width must be divisible by patch_size"

        self.patch_size = (H//downsampling_ratio, W//downsampling_ratio)
        self.num_patches = (H//self.patch_size[0]) * (W//self.patch_size[1])
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.input_dims = input_dims
        self.num_keypoints = num_keypoints
        self.downsampling_ratio = downsampling_ratio
        self.dropout = dropout
        self.dtype = dtype
        self.loss_fn = loss_fn

        self.cls_token = Parameter(torch.randn(1, 1, self.embed_dim))
        self.pos_embedding = Parameter(torch.randn(1, 1+self.num_patches, self.embed_dim))

        self.in_proj = None
        self.transformer_layers = ModuleList()

        self.in_proj = Linear(self.patch_size[0] * self.patch_size[1] * C, self.embed_dim) # needs to be flattened patches
        for i in range(self.num_layers):
            self.transformer_layers.append(TransformerBlock(self.embed_dim, self.hidden_dim, self.num_heads, self.dropout))

        self.fpn = FPN(in_channels = self.embed_dim) #if we use FPN at the end of the encoder blocks 

        self.keypoints_head = torch.nn.Conv2d(
            in_channels=self.embed_dim,
            out_channels=self.num_keypoints,  
            kernel_size=1
        )
        self.params = self.parameters()
    
    def forward(self, x):
        """
        Calculate the ViT model output.

        Inputs:
        - x: patches

        Returns:
        - keypoints: Tensor of shape (N, C) giving the attention output after final projection
        - out_tokens: Tensor of shape (N, S, D_v) giving the attention tokens of last transformer layer
        - out_attention_maps: List of tensors giving the attention probability tensors from *every* transformer layer
        """
        out_tokens, out_attention_maps = None, None
        N,H_W , P, C = x.shape #(N, H*W/patchsize , patch area, C)

        patch_embeddings = self.in_proj(x)
        
        cls_tokens = self.cls_token.repeat(N,1,1)
        
        cls_patch_embeddings = torch.cat((cls_tokens, patch_embeddings),dim=1)
        embeddings = cls_patch_embeddings + self.pos_embedding
    
        out_attention_maps = []
        out_tokens = embeddings
        for layer in range(self.num_layers):
            out_tokens, attention_map = self.transformer_layers[layer](out_tokens)
            out_attention_maps.append(attention_map)
            
        feature_maps = self.fpn(out_tokens) #extra feature pyramid network

        keypoints = self.keypoints_head(feature_maps)  # shape: (4, B, num_keypoints, H, W)

        return keypoints, feature_maps, out_tokens, out_attention_maps



    def softmax_loss(self, X, y=None):
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        scores, tokens, attention = self.forward(X)

        if y is None:
            return scores
        loss_function = torch.nn.CrossEntropyLoss()
        loss, _ = loss_function(scores, y)

        return loss

    def loss(self, X, y=None):
        if self.loss_fn=='softmax_loss':
            return self.softmax_loss(X, y)
        else:
            raise NotImplementedError