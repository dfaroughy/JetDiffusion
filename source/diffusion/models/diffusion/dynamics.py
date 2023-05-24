
import torch
import numpy as np
import abc
from tqdm.auto import tqdm

from diffusion.models.diffusion.sampler import EulerMaruyamaPredictor, LangevinCorrector

class SDE(abc.ABC):
	"""SDE abstract class."""

	def __init__(self, args):
		"""Construct an SDE.
		"""
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
				eps=1e-3):

		predictor = EulerMaruyamaPredictor(sde=self.sde, num_time_steps=self.N)
		corrector = LangevinCorrector(sde=self.sde, score=score, snr=self.snr, m_corrector_steps=50, variance_preserving=True)
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
				eps=1e-5):

		predictor = EulerMaruyamaPredictor(sde=self.sde, num_time_steps=self.N)
		corrector = LangevinCorrector(sde=self.sde, score=score, snr=self.snr, m_corrector_steps=50, variance_preserving=False)
		x = self.prior_sampling((num_gen, self.dim))
		timesteps = torch.linspace(self.T, eps, self.N, device=self.device)

		for i in tqdm(range(self.N), desc=" PC sampling"):
			t = torch.ones(num_gen, device=self.device) * timesteps[i]
			x = corrector.update(x, t)
			x = predictor.update(x, t)
		return x


	# def sampler(self, 
	# 			score_model, 
	# 			eps=1e-3,
	# 			num_samples=None,
	# 			context=None):

	# 	if not num_samples: num_samples = args.num_gen

	# 	t = torch.ones(num_samples, device=self.device)
	# 	mean, std = self.marginal_prob(t)

	# 	init_x = self.prior_sampling((num_samples, self.dim)) * std
	# 	time_steps = np.linspace(self.T, eps, self.N)
	# 	step_size = time_steps[0] - time_steps[1]
	# 	x = init_x

	# 	for time_step in tqdm(time_steps, desc=' predictor-corrector sampling'):   

	# 		batch_time_step = torch.ones(num_samples, device=self.device) * time_step
			
	# 		# Corrector step (Langevin MCMC)
	# 		grad = score_model(x, batch_time_step)
	# 		grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
	# 		noise_norm = np.sqrt(np.prod(x.shape[1:]))
	# 		langevin_step_size = 2 * (self.snr * noise_norm / grad_norm)**2
	# 		x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

	# 		# Predictor step (Euler-Maruyama)
	# 		_ , g = self.sde(batch_time_step)
	# 		x_mean = x + (g**2) * score_model(x, batch_time_step) * step_size
	# 		x = x_mean + torch.sqrt(g**2 * step_size) * torch.randn_like(x)      
			
	# 		# The last step does not include any noise
	# 		return x_mean





