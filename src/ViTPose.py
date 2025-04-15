import torch
from torch.nn import (
                        Module, 
                        Parameter
                        )
import Encoder
from utils import patchify

class ViTPose(Module):
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
        self.encoder = Encoder(self.embed_dim, self.hidden_dim, self.num_layers, self.num_heads, self.input_dims,
                               self.num_keypoints, self.downsampling_ratio, self.dropout, self.dtype, self.loss_fn)

    def save(self, path):
        checkpoint = {
        'num_patches': self.num_patches,
        'embed_dim': self.embed_dim,
        'hidden_dim': self.hidden_dim,
        'num_layers': self.num_layers,
        'num_heads': self.num_heads,
        'input_dims': self.input_dims,
        'num_clases': self.num_clases,
        'downsampling_ratio': self.downsampling_ratio,
        'dropout': self.dropout,
        'dtype': self.dtype,
        'loss_fn': self.loss_fn,
        'state_dict': self.state_dict(),
        }
        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.num_patches = checkpoint['num_patches']
        self.embed_dim = checkpoint['embed_dim']
        self.hidden_dim = checkpoint['hidden_dim']
        self.num_layers = checkpoint['num_layers']
        self.num_heads = checkpoint['num_heads']
        self.input_dims = checkpoint['input_dims']
        self.num_clases = checkpoint['num_clases']
        self.downsampling_ratio = checkpoint['downsampling_ratio']
        self.dropout = checkpoint['dropout']
        self.dtype = checkpoint['dtype']
        self.loss_fn = checkpoint['loss_fn']

        self.load_state_dict(checkpoint['state_dict'])

        print("load checkpoint file: {}".format(path))
    
    def forward(self, x):
        """
        Calculate the ViT model output.

        Inputs:
        - x: Input image data, of shape (N, D, H, W)

        Returns:
        - out_cls: Tensor of shape (N, C) giving the attention output after final projection
        - out_tokens: Tensor of shape (N, S, D_v) giving the attention tokens of last transformer layer
        - out_attention_maps: List of tensors giving the attention probability tensors from *every* transformer layer
        """
        out_cls, out_tokens, out_attention_maps = None, None, None
        N, C, H, W = x.shape

        patches = patchify(x, patch_size = self.patch_size)

        keypoints, feature_maps, out_tokens, out_attention_maps = self.encoder(patches)
        #decoder block can come in here



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