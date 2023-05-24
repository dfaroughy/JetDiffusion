
import torch
import numpy as np
import abc
from tqdm.auto import tqdm


class Predictor(abc.ABC):
	"""The abstract class for a predictor algorithm."""
	def __init__(self, rsde, score, N):
		super().__init__()
		self.rsde = rsde
		self.N = N
		self.score = score
	@abc.abstractmethod
	def update(self, x, t):
		"""One update of the predictor"""
		pass

class Corrector(abc.ABC):
	"""The abstract class for a corrector algorithm."""
	def __init__(self, alpha_fn, score, M, snr):
		super().__init__()
		self.score = score
		self.M = M # num corrector steps
		self.alpha_fn = alpha_fn
		self.snr = snr
	@abc.abstractmethod
	def update(self, x, t):
		"""One update of the corrector"""
		pass

class EulerMaruyamaPredictor(Predictor):

	def __init__(self, rsde,  score, N, denoise_last_step=False):
		super().__init__(rsde, score, N)
		self.denoise = denoise_last_step
	
	def update(self, x, t):
		dt = -1. / self.N
		drift, diffusion = self.rsde(x, t, self.score)
		x_mean = x + drift * dt
		if self.denoise: 
			x_mean = x_mean + diffusion * torch.randn_like(x) * np.sqrt(-dt)
		return x_mean

class LangevinCorrector(Corrector):

	'''Algorithm 4,5 in 20011.13456'''

	def __init__(self, alpha_fn, score, M, snr):
		super().__init__(alpha_fn, score, M, snr)

	def update(self, x, t):
		for i in range(self.M):
			g, z = self.score(x, t), torch.randn_like(x)
			g_norm = torch.norm(g.reshape(g.shape[0], -1), dim=-1).mean()
			z_norm = torch.norm(z.reshape(z.shape[0], -1), dim=-1).mean()
			epsilon = 2 * self.alpha_fn(t) * (self.snr * z_norm / g_norm)**2 
			x_mean = x + g * epsilon[:, None]  + z * torch.sqrt(2*epsilon[:, None])

		return x_mean
