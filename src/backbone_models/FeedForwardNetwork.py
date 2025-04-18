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

class FeedForwardNetwork(Module):

    def __init__(self, embed_dim, hidden_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        mlp_linear1 = Linear(self.embed_dim, self.hidden_dim)
        mlp_linear2 = Linear(self.hidden_dim, self.embed_dim)

        self.ffn = Sequential(
            mlp_linear1, 
            GELU(),
            mlp_linear2)

    def forward(self, x):

        out = self.ffn(x)
        return out