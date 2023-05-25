import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weight_norm
import numpy as np

class GaussianFourierProjection(nn.Module):

    def __init__(self, dim_embed, scale=16):
        super().__init__()
        half_dim = dim_embed // 2
        self.W = nn.Parameter(torch.randn(half_dim) * scale, requires_grad=False)

    def forward(self, t):
        W = self.W.to(t.device)
        t_proj = t[:, None] * W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1)

class ScoreNet(nn.Module):

    def __init__(self, args, marginal_prob):
        """time-dependent score-based network."""
        super().__init__()
        self.sde = args.sde
        dim_embed = args.dim_embedding
        dim_hidden = args.dim_hidden
        dim_input = args.dim
        num_layers = args.num_layers
        kind = args.score_model
        self.marginal_prob = marginal_prob

        self.projection = GaussianFourierProjection(dim_embed=dim_embed)

        self.time_embedding = MLP(dim_in=dim_embed, 
                             hidden_layers=[dim_hidden], 
                             dim_out=dim_hidden,
                             activation='LeakyReLU',
                             final_activation='LeakyReLU'
                            ).to(args.device)

        self.embedding = MLP(dim_in=dim_input, 
                            hidden_layers=[dim_hidden], 
                            dim_out=dim_hidden,
                            activation='LeakyReLU',
                            final_activation='LeakyReLU'
                  ).to(args.device)

        if kind=='MLP':
            
            dims = 2 * dim_hidden

            self.dense = MLP(dim_in=dims, 
                             hidden_layers=[dims]*num_layers, 
                             dim_out=dims,
                             activation='LeakyReLU',
                             final_activation='LeakyReLU',
                             device = args.device
                              )

            self.dense_last = MLP(dim_in=dims, 
                                  hidden_layers=[dims], 
                                  dim_out=dim_input,
                                  activation='LeakyReLU',
                                  final_activation=None,
                                  device = args.device
                                  )

        if kind=='Attention':

            dims = 2 * dim_hidden
            
            self.dense = SelfAttention(dim_in=dims, 
                                       dim_hidden=dims, 
                                       dim_out=dims, 
                                       num_heads=1, 
                                       num_transformer=num_layers,                                
                                       device = args.device)

            self.dense_last = MLP(dim_in=dims, 
                                 hidden_layers=[dims], 
                                 dim_out=dim_input,
                                 activation='GELU',
                                 final_activation=None,
                                 device = args.device)

    def forward(self, x, t):
        time = self.time_embedding(self.projection(t))
        x = self.embedding(x)
        xt = torch.cat((x, time), dim=1)
        h = self.dense(xt)
        score = self.dense_last(xt+h)
        if 'Exploding' in self.sde:    
            _ , std = self.marginal_prob(x, t)
            score = score / std
        return score



class MLP(nn.Module):
    """
    Simple multi-layer perceptron.

    Example:
    >>> net = MLP(2, [64, 64], 1)
    >>> net(torch.randn(1, 2))
    tensor([[-0.0132]], grad_fn=<AddmmBackward>)

    """
    def __init__(self, dim_in, hidden_layers, dim_out, activation='LeakyReLU',
                 final_activation=None, wrapper_func=None, device='cpu', **kwargs):
        super().__init__()

        if not wrapper_func: wrapper_func = lambda x: x

        hidden_layers = hidden_layers[:]
        hidden_layers.append(dim_out)
        layers = [nn.Linear(dim_in, hidden_layers[0])]

        for i in range(len(hidden_layers) - 1):
            layers.append(getattr(nn, activation)())
            layers.append(wrapper_func(nn.Linear(hidden_layers[i], hidden_layers[i+1])))
        layers[-1].bias.data.fill_(0.0)

        if final_activation is not None:
            layers.append(getattr(nn, final_activation)())

        self.net = nn.Sequential(*layers).to(device)

    def forward(self, x, **kwargs):
        return self.net(x)



class SelfAttention(nn.Module):

    """
    Attention layer.

    Example:
    >>> net = Attention(3, [64, 64], 8, num_heads=4)
    >>> x = torch.randn(32, 10, 3)
    >>> net(x, x, x).shape
    torch.Size([32, 10, 8])

    Args:
        input_dim (int): Input size
        layers_dim (List[int]): Hidden dimensions of embedding network, last dimension is used as an embedding size for queries, keys and values
        output_dim (int): Output size
        num_heads (int, optional): Number of attention heads, must divide last element of `layers_dim`. Default: 1
    """

    def __init__(self, dim_in, dim_hidden, dim_out, num_heads=1, num_transformer=4, device= 'cpu', **kwargs):
        super().__init__()

        self.embedding_key   = MLP(dim_in, [dim_hidden], dim_out, 'LeakyReLU', wrapper_func=weight_norm, device=device)
        self.embedding_query = MLP(dim_in, [dim_hidden], dim_out, 'LeakyReLU', wrapper_func=weight_norm, device=device)
        self.embedding_value = MLP(dim_in, [dim_hidden], dim_out, 'LeakyReLU', wrapper_func=weight_norm, device=device)
        self.num_heads = num_heads
        self.num_transformer = num_transformer
        self.dim_out = dim_out
        self.device = device

    def attention_block(self, Q, K, V):

        *query_shape, _ = Q.shape
        *value_shape, D = V.shape

        Q = Q.view(*query_shape, self.num_heads, D // self.num_heads).transpose(-2, -3)
        V = V.view(*value_shape, self.num_heads, D // self.num_heads).transpose(-2, -3)
        K = K.view(*value_shape, self.num_heads, D // self.num_heads).transpose(-2, -3)
        
        d = K.shape[-1]
        A = torch.softmax( Q @ K.transpose(-1, -2) * (1 / d)**0.5, dim=-1) @ V
        A = A.transpose(-2, -3).reshape(*query_shape, -1)

        return A

    def forward(self, x, **kwargs):

        for _ in range(self.num_transformer):
            h1 = self.attention_block(self.embedding_query(x), self.embedding_key(x), self.embedding_value(x))
            h2 = h1 + x 
            h3 = MLP(self.dim_out, [self.dim_out], self.dim_out, 'GELU', wrapper_func=weight_norm, device=self.device)(h2)

        return x + h3 


