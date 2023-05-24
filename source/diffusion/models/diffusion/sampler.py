
import torch
import numpy as np
import abc
from tqdm.auto import tqdm


class Predictor(abc.ABC):
	"""The abstract class for a predictor algorithm."""
	def __init__(self, sde, num_time_steps):
		super().__init__()
		self.sde = sde
		self.num_time_steps = num_time_steps

	@abc.abstractmethod
	def update(self, x, t):
		"""One update of the predictor"""
		pass

class Corrector(abc.ABC):
	"""The abstract class for a corrector algorithm."""
	def __init__(self, sde, score, m_corrector_steps, snr, variance_preserving):
		super().__init__()
		self.sde = sde
		self.score = score
		self.m_corrector_steps = m_corrector_steps
		self.snr = snr
		self.variance_preserving = variance_preserving

	@abc.abstractmethod
	def update(self, x, t):
		"""One update of the corrector"""
		pass


class EulerMaruyamaPredictor(Predictor):

	def __init__(self, sde,  num_time_steps, denoise_last_step=False):
		super().__init__(sde, num_time_steps)
		self.denoise = denoise_last_step

	def update(self, x, t):
		dt = -1. / self.num_time_steps
		drift, diffusion = self.sde(x, t)
		x_mean = x + drift * dt
		if self.denoise: 
			x_mean = x_mean + diffusion * torch.randn_like(x) * np.sqrt(-dt)
		return x_mean

class LangevinCorrector(Corrector):

	'''Algorithm 4,5 in 20011.13456'''

	def __init__(self, sde, score, m_corrector_steps, snr, variance_preserving):
		super().__init__(sde, score, m_corrector_steps, snr, variance_preserving)

	def update(self, x, t):
		timestep = (t * (self.m_corrector_steps - 1) / 1.0).long()
		alpha = self.sde.alphas.to(t.device)[timestep] if self.variance_preserving else torch.ones_like(t)

		for i in range(self.m_corrector_steps):
			g, z = self.score(x, t), torch.randn_like(x)
			g_norm = torch.norm(g.reshape(g.shape[0], -1), dim=-1).mean()
			z_norm = torch.norm(z.reshape(z.shape[0], -1), dim=-1).mean()
			epsilon = 2 * alpha * (self.snr * z_norm / g_norm)**2 
			x_mean = x + g * epsilon[:, None]  + z * torch.sqrt(2*epsilon[:, None])

		return x_mean
