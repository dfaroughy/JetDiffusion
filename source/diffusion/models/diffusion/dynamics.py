
import torch
import numpy as np
import abc
from tqdm.auto import tqdm

from diffusion.models.diffusion.sampler import EulerMaruyamaPredictor, LangevinCorrector

class SDE(abc.ABC):
	"""SDE abstract class."""

	def __init__(self, args):
		"""Construct an SDE."""
		super().__init__()
		self.args = args
		self.N = args.num_time_steps
		self.device = args.device
		self.dim = args.dim
		self.snr = args.sig_to_noise

	@property
	@abc.abstractmethod
	def T(self):
		"""End time of the SDE."""
		pass
	@abc.abstractmethod
	def sde(self, x, t):
		pass
	@abc.abstractmethod
	def backward_sde(self, x, t, score):
		pass
	@abc.abstractmethod
	def marginal_prob(self, x, t):
		"""Parameters to determine the marginal distribution (perturbation kernel) of the SDE, $p_0t(x(t)|x(0))$."""
		pass
	@abc.abstractmethod
	def prior_sampling(self, shape):
		"""Generate one sample from the prior distribution, $p_T(x)$."""
		pass
	@abc.abstractmethod
	def sampler(self, score, eps):
		"""Generate samples from model with predictor-corrector sampler."""
		pass


class VariancePreservingSDE(SDE):

	def __init__(self, args, beta_min=0.1, beta_max=20.):
		super().__init__(args)
		self.beta_min = beta_min
		self.beta_max = beta_max
		self.discrete_betas = torch.linspace(self.beta_min / self.N, self.beta_max / self.N, self.N)
		self.alphas = 1. - self.discrete_betas
		self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
		self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
		self.sqrt_1m_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

	@property
	def T(self):
		return 1.0

	def sde(self, x, t):
		beta_t = self.beta_min + t * (self.beta_max - self.beta_min)
		drift = -0.5 * beta_t[:, None] * x
		diffusion = torch.sqrt(beta_t[:, None])
		return drift, diffusion 

	def backward_sde(self, x, t, score):
		"""Create the drift and diffusion functions for the reverse SDE/ODE."""
		drift, diffusion = self.sde(x, t)
		drift = drift - diffusion ** 2 * score(x, t) 
		return drift, diffusion

	def marginal_prob(self, x, t):
		log_mean_coeff = -0.25 * t**2 * (self.beta_max - self.beta_min) - 0.5 * t * self.beta_min
		mean = torch.exp(log_mean_coeff[:, None]) * x
		std = torch.sqrt(1. - torch.exp(2. * log_mean_coeff[:, None]))
		return mean, std

	def prior_sampling(self, shape):
		return torch.randn(*shape, device=self.device)

	@torch.no_grad()
	def sampler(self, 
				score,
				num_gen,
				num_corrector_steps, 
				eps=1e-3):

		def alpha_fn(t, M=num_corrector_steps):
			timestep = (t * (M - 1) / self.T).long()	
			return self.alphas.to(t.device)[timestep]

		predictor = EulerMaruyamaPredictor(rsde=self.backward_sde, score=score, N=self.N)
		corrector = LangevinCorrector(alpha_fn=alpha_fn, score=score, M=num_corrector_steps, snr=self.snr)
		x = self.prior_sampling((num_gen, self.dim))
		timesteps = torch.linspace(self.T, eps, self.N, device=self.device)

		for i in tqdm(range(self.N), desc=" PC sampling"):
			t = torch.ones(num_gen, device=self.device) * timesteps[i]
			x = corrector.update(x, t)
			x = predictor.update(x, t)
		return x


class VarianceExplodingSDE(SDE):

	def __init__(self, args, beta_min=0.01, beta_max=50.):
		super().__init__(args)

		self.sigma_min = beta_min
		self.sigma_max = beta_max
		self.discrete_sigmas = torch.exp(torch.linspace(np.log(self.sigma_min), np.log(self.sigma_max), self.N))

	@property
	def T(self):
		return 1.0

	def sde(self, x, t):
		sigma = self.sigma_min * (self.sigma_max / self.sigma_min)**t
		drift = torch.zeros_like(x)
		diffusion =  sigma * torch.sqrt(torch.tensor(2 * (np.log(self.sigma_max) - np.log(self.sigma_min))))
		return drift, diffusion[:, None]

	def backward_sde(self, x, t, score):
		"""Create the drift and diffusion functions for the reverse SDE/ODE."""
		drift, diffusion = self.sde(x, t)
		drift = drift - diffusion ** 2 * score(x, t) 
		return drift, diffusion

	def marginal_prob(self, x, t):
		mean = x
		std = self.sigma_min * (self.sigma_max / self.sigma_min)**t 
		return mean, std[:, None]

	def prior_sampling(self, shape):
		return torch.randn(*shape, device=self.device)

	@torch.no_grad()
	def sampler(self, 
				score,
				num_gen,
				num_corrector_steps, 
				eps=1e-5):

		def alpha_fn(t):
			return torch.ones_like(t)

		predictor = EulerMaruyamaPredictor(rsde=self.backward_sde, score=score, N=self.N)
		corrector = LangevinCorrector(alpha_fn=alpha_fn, score=score, M=num_corrector_steps, snr=self.snr)
		x = self.prior_sampling((num_gen, self.dim))
		timesteps = torch.linspace(self.T, eps, self.N, device=self.device)

		for i in tqdm(range(self.N), desc=" PC sampling"):
			t = torch.ones(num_gen, device=self.device) * timesteps[i]
			x = corrector.update(x, t)
			x = predictor.update(x, t)
		return x



