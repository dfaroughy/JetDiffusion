
import torch
import numpy as np
import abc
from tqdm.auto import tqdm


class Predictor(abc.ABC):
	"""The abstract class for a predictor algorithm."""
	def __init__(self, sde, n_steps):
		super().__init__()
		self.sde = sde
		self.n_steps = n_steps

	@abc.abstractmethod
	def update(self, x, t):
		"""One update of the predictor.
		Args:
		  x: A PyTorch tensor representing the current state
		  t: A Pytorch tensor representing the current time step.
		Returns:
		  x: A PyTorch tensor of the next state.
		  x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
		"""
		pass


class Corrector(abc.ABC):
	"""The abstract class for a corrector algorithm."""
	def __init__(self, sde, score, n_steps, snr, variance_preserving):
		super().__init__()
		self.sde = sde
		self.score = score
		self.n_steps = n_steps
		self.snr = snr
		self.variance_preserving = variance_preserving

	@abc.abstractmethod
	def update(self, x, t):
		"""One update of the corrector.
		Args:
		  x: A PyTorch tensor representing the current state
		  t: A PyTorch tensor representing the current time step.
		Returns:
		  x: A PyTorch tensor of the next state.
		  x_mean: A PyTorch tensor. The next state without random noise. Useful for denoising.
		"""
		pass


class EulerMaruyamaPredictor(Predictor):

	def __init__(self, sde,  n_steps):
		super().__init__(sde, n_steps)

	def update(self, x, t):
		dt = -1. / self.n_steps
		z = torch.randn_like(x)
		drift, diffusion = self.sde(x, t)
		mean = x + drift * dt
		x = mean + diffusion * np.sqrt(-dt) * z
		return x, mean

class LangevinCorrector(Corrector):
	def __init__(self, sde, score, n_steps, snr, variance_preserving):
		super().__init__(sde, score, n_steps, snr, variance_preserving)

	def update(self, x, t):
		sde = self.sde
		score = self.score
		n_steps = self.n_steps
		target_snr = self.snr	
		timestep = (t * (n_steps - 1) / 1.0).long()
		alpha = sde.alphas.to(t.device)[timestep] if self.variance_preserving else torch.ones_like(t)

		for i in range(n_steps):
			grad = score(x, t)
			noise = torch.randn_like(x)
			grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
			noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
			step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
			mean = x + step_size[:, None] * grad
			x = mean + torch.sqrt(step_size * 2)[:, None] * noise

		return x, mean
