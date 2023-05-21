import torch
import torch.nn as nn
import numpy as np

class GaussianFourierProjection(nn.Module):
  """Gaussian random features for encoding time steps."""  
  def __init__(self, dim_embed, scale=30.):
    super().__init__()
    # Randomly sample weights during initialization. These weights are fixed 
    # during optimization and are not trainable.
    self.W = nn.Parameter(torch.randn(dim_embed // 2) * scale, requires_grad=False)
  def forward(self, x):
    x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
    return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
  """A fully connected layer that reshapes outputs to feature maps."""
  def __init__(self, input_dim, output_dim):
    super().__init__()
    self.dense = nn.Linear(input_dim, output_dim)
  def forward(self, x):
    return self.dense(x)[..., None, None]


class MlpScore(nn.Module):
    """A time-dependent score-based model built upon a simplified architecture."""

    def __init__(self, args, marginal_prob_std):
        """Initialize a time-dependent score-based network.

        Args:
            marginal_prob_std: A function that takes time t and gives the standard
                deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
            dim_embed: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()

        dim_embed= args.dim_embed
        dim_input = args.dim
        self.std = marginal_prob_std

        self.embedding = nn.Sequential(
            GaussianFourierProjection(dim_embed=dim_embed),
            nn.Linear(dim_embed, dim_embed)
        )

        self.layers = nn.Sequential(
            Dense(dim_input, dim_embed), nn.LeakyReLU(0.01),
			Dense(dim_embed, dim_embed), nn.LeakyReLU(0.01),
			Dense(dim_embed, dim_embed), nn.LeakyReLU(0.01),
			Dense(dim_embed, dim_embed), nn.LeakyReLU(0.01)
        	)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)

    def forward(self, x, t):
        time_embedding = self.act(self.embedding(t))
        h = Dense(x)
        h += Dense(time_embedding)
        h = self.layers(h)
        h = self.act(h) 
        h = h / self.std(t)[:, None]
        return h
