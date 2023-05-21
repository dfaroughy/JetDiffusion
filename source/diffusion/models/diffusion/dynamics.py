
import torch
import numpy as np
import abc

class SDE(abc.ABC):
  """SDE abstract class. Functions are designed for a mini-batch of inputs."""

  def __init__(self, args):
    """Construct an SDE.

    Args:
      N: number of discretization time steps.
    """
    super().__init__()
    self.args= args

  @property
  @abc.abstractmethod
  def T(self):
    """End time of the SDE."""
    pass

  @abc.abstractmethod
  def sde(self, x, t):
    pass

  @abc.abstractmethod
  def marginal_prob(self, x, t):
    """Parameters to determine the marginal distribution of the SDE, $p_t(x)$."""
    pass

  def discretize(self, x, t):
    """Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.

    Useful for reverse diffusion sampling and probabiliy flow sampling.
    Defaults to Euler-Maruyama discretization.

    Args:
      x: a torch tensor
      t: a torch float representing the time step (from 0 to `self.T`)

    Returns:
      f, G
    """
    dt = 1 / self.args.N
    drift, diffusion = self.sde(x, t)
    f = drift * dt
    G = diffusion * torch.sqrt(torch.tensor(dt, device=t.device))
    return f, G




class driftlessSDE(SDE):

  def __init__(self, args):
    super().__init__(args)
    self.args = args

  @property
  def T(self):
    return 1

  def sde(self, t):
    drift = 0.
    diffusion = self.args.sigma**t
    return drift, diffusion

  def marginal_prob(self, t):
    t = torch.tensor(t, device=self.args.device)
    mean = 0.
    std = torch.sqrt((self.args.sigma**(2 * t) - 1.) / 2. / np.log(self.args.sigma))
    return mean, std

  def prior(self, shape):
    return torch.randn(*shape)