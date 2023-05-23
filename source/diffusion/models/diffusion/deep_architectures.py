import torch
import torch.nn as nn
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
    """A time-dependent score-based model built upon a simplified architecture."""

    def __init__(self, args, marginal_prob):
        """Initialize a time-dependent score-based network.

        Args:
            marginal_prob_std: A function that takes time t and gives the standard
                deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
            dim_hidden: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()

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
                             final_activation='LeakyReLU'
                              ).to(args.device)

        self.dense_last = MLP(dim_in=2*dim_hidden, 
                         hidden_layers=[2 * dim_hidden], 
                         dim_out=dim_input,
                         activation='LeakyReLU',
                         final_activation=None
                        ).to(args.device)

    def forward(self, x, t):
        time = self.time_embedding(self.projection(t))
        x = self.embedding(x)
        xt = torch.cat((x, time), dim=1)
        h = self.dense(xt)
        score = self.dense_last(xt+h)
        _, std = self.marginal_prob(t)
        return score / std



# class SelfAttentionRatioEstimatorPerTime(DynamicRatioEstimator):
#     name_="dynamic_ratio_estimator_self_attention"
#     def __init__(self, **kwargs):
#         super(SelfAttentionRatioEstimatorPerTime, self).__init__(**kwargs)

#         self.number_of_spins = kwargs.get("number_of_spins")
#         self.hidden_1 = kwargs.get("hidden_1")
#         self.time_embedding_dim = kwargs.get("time_embedding_dim", 10)
#         self.num_heads = kwargs.get("num_heads", 2)

#         self.self_attention = SelfAttention(self.number_of_spins, [self.hidden_1], self.hidden_1, self.num_heads)
#         self.f = nn.Linear(self.hidden_1+self.time_embedding_dim,self.number_of_spins)

#     def forward_states_and_times(self, states, times):
#         time_embbedings = get_timestep_embedding(times.squeeze(),
#                                                  time_embedding_dim=self.time_embedding_dim)

#         step_attn = self.self_attention(states, dim=1)
#         step_two = torch.concat([step_attn, time_embbedings],dim=1)
#         ratio_estimator = self.f(step_two)
        
#         return softplus(ratio_estimator)

#     @classmethod
#     def get_parameters(self) -> dict:
#         kwargs = super().get_parameters()
#         kwargs.update({"hidden_1": 14, "num_heads": 2})
#         return kwargs


class MLP(nn.Module):
    """
    Simple multi-layer perceptron.

    Example:
    >>> net = MLP(2, [64, 64], 1)
    >>> net(torch.randn(1, 2))
    tensor([[-0.0132]], grad_fn=<AddmmBackward>)

    """
    def __init__(self, dim_in, hidden_layers, dim_out, activation='LeakyReLU',
                 final_activation=None, wrapper_func=None, **kwargs):
        super().__init__()

        if not wrapper_func:
            wrapper_func = lambda x: x

        hidden_layers = hidden_layers[:]
        hidden_layers.append(dim_out)
        layers = [nn.Linear(dim_in, hidden_layers[0])]

        for i in range(len(hidden_layers) - 1):
            layers.append(getattr(nn, activation)())
            layers.append(wrapper_func(nn.Linear(hidden_layers[i], hidden_layers[i+1])))
        layers[-1].bias.data.fill_(0.0)

        if final_activation is not None:
            layers.append(getattr(nn, final_activation)())

        self.net = nn.Sequential(*layers)

    def forward(self, x, **kwargs):
        return self.net(x)



class Attention(nn.Module):

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

    def __init__(self, dim_in, hidden_layers, dim_out, activation='LeakyReLU', final_activation=None, num_heads=1, **kwargs):
        super().__init__()

        self.embedding_key = MLP(dim_in, hidden_layers, dim_out, activation, final_activation)
        self.embedding_query = MLP(dim_in, hidden_layers, dim_out, activation, final_activation)
        self.embedding_value = MLP(dim_in, hidden_layers, dim_out, activation, final_activation)
        
        self.num_heads = num_heads

    def attention(self, Q, K, V):

        """
        Multihead attention layer.
        "Attention Is All You Need" (https://arxiv.org/abs/1706.03762)

        Args:
            Q (tensor): Query matrix (..., N, D)
            K (tensor): Key matrix (..., N, D)
            V (tensor): Value matrix (..., N, D)
            num_heads (int, optional): Number of attention heads, must divide D. Default: 1

        Returns:
            Attention (tensor): Result of multihead attention with shape (..., N, D)
        """
        *query_shape, _ = Q.shape
        *value_shape, D = V.shape

        Q = Q.view(*query_shape, self.num_heads, D // self.num_heads).transpose(-2, -3)
        V = V.view(*value_shape, self.num_heads, D // self.num_heads).transpose(-2, -3)
        K = K.view(*value_shape, self.num_heads, D // self.num_heads).transpose(-2, -3)
        
        d = K.shape[-1]
        A = torch.softmax( Q @ K.transpose(-1, -2) * (1 / d)**0.5, dim=-1) @ V
        A = A.transpose(-2, -3).reshape(*query_shape, -1)

        return A

    def forward(self, Q, K, V, **kwargs):

        return self.attention(self.embedding_query(Q), 
                              self.embedding_key(K), 
                              self.embedding_value(V))


class SelfAttention(Attention):

    def __init__(self, dim_in, hidden_layers, dim_out, activation='LeakyReLU', final_activation=None, num_heads=1, **kwargs):
        super().__init__(dim_in, hidden_layers, dim_out, activation, final_activation, num_heads)

    def forward(self, X, **kwargs):
        return super().forward(X, X, X)


