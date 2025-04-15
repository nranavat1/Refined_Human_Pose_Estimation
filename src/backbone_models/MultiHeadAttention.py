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

class Attention(Module):
    """
    Attention Layer: This model takes in batches of input embeddings 
    with shape `(N, *, D_{k,v})` and applies a single layer of scaled dot-product 
    attention in a sequential fashion of the form:

        Attention(q,k,v) = dropout( softmax(q * k.T / sqrt(D)) * v )
    
    See Equation (1) in Attention paper (https://arxiv.org/abs/1706.03762).

    """

    def __init__(self, embed_dim: int, hidden_dim: int, dropout: float=0.1):
        """
        Construct a new attention layer that applies each projection in sequence.

        Input:
        - embed_dim: Size of the expected input dimension
        - hidden_dim: Size of the dimension used within the self-attention
        - dropout: Dropout mask probability
        """
        super().__init__()

        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.scale = torch.sqrt(torch.tensor(self.hidden_dim))
        self.query = Linear(self.embed_dim, self.hidden_dim)
        self.key = Linear(self.embed_dim, self.hidden_dim)
        self.value = Linear(self.embed_dim, self.hidden_dim)
        self.attend = Softmax(dim=-1)
        self.dropout = Dropout(p=dropout)
        self.out_proj = Linear(self.hidden_dim, self.embed_dim)
        
    
    def forward(self, query, key, value):
        """
        Calculate the attention output.

        Inputs:
        - query: Input query data, of shape (N, S, D_k)
        - key: Input key data, of shape (N, T, D_k)
        - value: Input value data to be used as the value, of shape (N, T, D_v)
        
        Assumes: 
        - D_k==D_v

        Returns:
        - out: Tensor of shape (N, S, D_v) giving the attention output after final projection
        - attention: Tensor of shape (N, S, D_v) giving the attention probability maps
        """
        out, attention = None, None
        N, S, D = query.shape
        N, T, D = key.shape
        assert key.shape==value.shape
        
        matmul= torch.matmul(self.query(query) ,self.key(key).transpose(-2,-1))
        attention = self.attend(matmul/self.scale)
        dropout = self.dropout(torch.matmul(attention, self.value(value)))
        out = self.out_proj(dropout)

        return out, attention


class MultiHeadAttention(Module):
    """
    MultiHeadAttention Layer: This model takes in batches of input embeddings 
    with shape `(N, *, D_{k,v})` and applies a single layer of multi-head scaled 
    dot-product attention in a **parallel** fashion of the form:

        MultiHeadAttention(q,k,v) = dropout( softmax((q*W_i^Q) * (k*W_i^K).T / sqrt(D/h)) * v*W_i^V )
                                    for i in range(Num_Heads)
    
    See Section 3.2.2 in Attention paper (https://arxiv.org/abs/1706.03762).

    """
    def __init__(self, embed_dim: int, hidden_dim: int, num_heads: int, dropout: float=0.1):
        super().__init__()
        """
        Construct a new multi-head attention layer that applies each projection head in parallel.

        Input:
        - embed_dim: Size of the expected input dimension
        - hidden_dim: Size of the dimension used within the self-attention
        - dropout: Dropout mask probability
        """
        
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.H=num_heads

        assert embed_dim%num_heads==0, "MHSA requires embedding dimension divisible by number of heads"


        self.scale = torch.sqrt(torch.tensor(self.hidden_dim/self.H))
        self.query = Linear(self.embed_dim, self.hidden_dim)
        self.key = Linear(self.embed_dim, self.hidden_dim)
        self.value = Linear(self.embed_dim, self.hidden_dim)
        self.attend = Softmax(dim=-1)
        self.dropout = Dropout(p=dropout)
        self.out_proj = Linear(self.hidden_dim, self.embed_dim)

    def forward(self, query, key, value, attn_mask=None):
        """
        Calculate the attention output.

        Inputs:
        - query: Input query data, of shape (N, S, D_k)
        - key: Input key data, of shape (N, T, D_k)
        - value: Input value data to be used as the value, of shape (N, T, D_v)
        
        Assumes: 
        - D_k==D_v

        Returns:
        - out: Tensor of shape (N, S, D_v) giving the attention output after final projection
        - attention: Tensor of shape (N, S, D_v) giving the attention probability maps
        """
        out, attention = None, None
        N, S, D = query.shape
        N, T, D = key.shape
        assert key.shape==value.shape
        
        #query is (N,S,self.hidden_dim) size after linear layer make it split amongst heads
        # (N,S,heads, hidden_dim//heads)
        #but for batch matrix multiplication it (N,heads, ...) since that is same amongst all of them
        #just in case S and num_heads are same using transpose instead
        query = self.query(query).view(N, S, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
        key = self.key(key).view(N, T, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
        value = self.value(value).view(N, T, self.num_heads, self.hidden_dim // self.num_heads).transpose(1, 2)
        
        attention = self.attend(torch.matmul(query,key.transpose(-2,-1))/self.scale) #(N,num_heads, S,T)
        attend_v = torch.matmul(attention, value) #(N,heads,S,T) * (N, heads, T, hidden/heads) -> (N, heads, S, hidden/heads)
        dropout = self.dropout(attend_v)
        dropout = dropout.transpose(1, 2).reshape(N, S, self.hidden_dim)
        out = self.out_proj(dropout)

        return out, attention