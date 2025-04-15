import torch
from torch.nn import (Linear, 
                        Softmax, 
                        Dropout, 
                        GELU, 
                        Sequential, 
                        ModuleList,
                        Module, 
                        Parameter
                        )

import FeedForwardNetwork
import LayerNorm
import MultiHeadAttention 

class EncoderBlock(Module):
    def __init__(
        self,
        embed_dim: int,
        hidden_dim: int,
        num_heads: int=1,
        dropout: float=0.1
        ):
        super().__init__()

        assert num_heads>0, "Number of heads must be greater than zero"

        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.dropout = dropout

        self.norm1 = None
        self.norm2 = None
        self.attn = None
        self.mlp = None

        self.norm1 = LayerNorm(self.embed_dim)
        self.norm2 = LayerNorm(self.embed_dim)
        self.attn = MultiHeadAttention(self.embed_dim, self.hidden_dim, self.num_heads, self.dropout)
        

        self.ffn = FeedForwardNetwork(self.embed_dim, self.hidden_dim)


    def forward(self, x):
        out, attention = None, None
        
        layernorm1 = self.norm1(x)
        
        multihead_attn, attention = self.attn(layernorm1, layernorm1, layernorm1) 
        multihead_res = multihead_attn + x #residual connection
        
        layernorm2 = self.norm2(multihead_res)
        ffn_out = self.ffn(layernorm2)
        
        out = ffn_out + multihead_res        

        return out, attention