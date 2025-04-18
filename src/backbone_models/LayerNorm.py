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


class LayerNorm_fn(object):


    @staticmethod
    def forward(x, gamma, beta, ln_param):
        """
        Forward pass for layer normalization.

        During both training and testing, the mean and variance
        are computed from the features of each data point.

        Inputs:
        - x: Data of shape (*, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift parameter of shape (D,)
        - ln_param: Dictionary with the following keys:
          - eps: Constant for numeric stability

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
        eps = ln_param.get('eps', 1e-5)

        D = x.shape[-1]

        out, cache = None, None
        
        mu = x.mean(dim=-1,keepdim=True)
        var = x.var(dim=-1,keepdim=True,unbiased=False) 
        #we want biased variance because it is norm over feature dimension, each sample should have it's own variance
        diff = x - mu
        vareps = var + eps
        sqrt = torch.sqrt(vareps)
        isqrt = 1/sqrt
        x_hat = diff/sqrt
        
        out = x_hat * gamma.unsqueeze(0) + beta.unsqueeze(0)
        cache = (x_hat, gamma, beta, mu, var, diff, vareps, sqrt, isqrt, x_hat) #not sure what isqrt is
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for layer normalization.

        For this implementation, you should write out a
        computation graph for layer normalization to understand the
        gradients flow.

        Inputs:
        - dout: Upstream derivatives, of shape (*, D)
        - cache: Variable of intermediates from LayerNorm.forward.

        Returns a tuple of:
        - dx: Gradient with respect to inputs x, of shape (*, D)
        - dgamma: Gradient with respect to scale parameter gamma,
          of shape (D,)
        - dbeta: Gradient with respect to shift parameter beta,
          of shape (D,)
        """
        x, gamma, beta, mean, var, diff, vareps, sqrt, isqrt, div = cache
        D = x.shape[-1]
        
        dydx = dout * gamma
        dx = (dydx - dydx.mean(-1,keepdim=True)- (div * (dydx * div).mean(-1, keepdim=True)))* isqrt
      
        dgamma = (dout*div).sum(dim = tuple(range(dout.dim()-1)))#same here
        
        dbeta = dout.sum(dim = tuple(range(dout.dim()-1))) #dbeta is 1 but we need to condense dout to size (D,) 


        return dx, dgamma, dbeta



class LayerNorm(Module):

    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

        self.gamma = Parameter(torch.ones(self.embed_dim))
        self.beta = Parameter(torch.zeros(self.embed_dim))

    def forward(self, x):
        return LayerNorm_fn.forward(x, self.gamma, self.beta, {})[0]