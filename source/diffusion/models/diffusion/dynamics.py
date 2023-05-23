
import torch
import numpy as np
import abc
from tqdm.auto import tqdm

class Foward_SDE(abc.ABC):
	"""SDE abstract class."""

	def __init__(self, args):
		"""Construct an SDE.
		"""
		super().__init__()
		self.args = args

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

	@abc.abstractmethod
	def prior_sampling(self, shape):
		"""Generate one sample from the prior distribution, $p_T(x)$."""
		pass

	@abc.abstractmethod
	def sampler(self, score_model, eps, context):
		"""Generate samples from model with predictor-corrector sampler."""
		pass


	# def discretize(self, x, t):
	# 	"""Discretize the SDE in the form: x_{i+1} = x_i + f_i(x_i) + G_i z_i.
	# 	"""
	# 	dt = 1 / self.num_steps
	# 	drift, diffusion = self.sde(x, t)
	# 	f = drift * dt
	# 	G = diffusion * torch.sqrt(torch.tensor(dt))
	# 	return f, G

	# def reverse(self, score, probability_flow=False):
	# 	"""Create the reverse-time SDE/ODE.
	# 	"""
	# 	num_steps = self.num_steps
	# 	T = self.T
	# 	sde_fwd = self.sde
	# 	discretization = self.discretize

	# 	# Build the class for reverse-time SDE.
	# 	class Backward_SDE(self.__class__):
	# 		def __init__(self):
	# 		self.num_steps = num_steps
	# 		self.probability_flow = probability_flow

	# 		@property
	# 		def T(self):
	# 			return T

	# 		def sde(self, x, t):
	# 			"""Create the drift and diffusion functions for the reverse SDE/ODE."""
	# 			drift, diffusion = sde_fwd(x, t)
	# 			drift = drift - diffusion[:, None] ** 2 * score(x, t) * (0.5 if self.probability_flow else 1.)
	# 			# Set the diffusion function to zero for ODEs.
	# 			diffusion = 0. if self.probability_flow else diffusion
	# 			return drift, diffusion

	# 		def discretize(self, x, t):
	# 			"""Create discretized iteration rules for the reverse diffusion sampler."""
	# 			f, G = discretization(x, t)
	# 			rev_f = f - G[:, None]**2 * score(x, t) * (0.5 if self.probability_flow else 1.)
	# 			rev_G = torch.zeros_like(G) if self.probability_flow else G
	# 			return rev_f, rev_G

	# 	return Backward_SDE()


class driftlessSDE(Foward_SDE):

	def __init__(self, args):
		super().__init__(args)

		self.sigma = args.sigma
		self.num_steps = args.num_time_steps
		self.num_samples = args.num_gen
		self.device = args.device
		self.dim = args.dim
		self.snr = args.R_sig_to_noise

	@property
	def T(self):
		return 1.0

	def sde(self, t):
		drift = None
		diffusion = self.sigma**t
		return drift, diffusion[:, None]

	def marginal_prob(self, t):
		mean = None
		std = torch.sqrt((self.sigma**(2 * t) - 1.) / 2. / np.log(self.sigma))
		return mean, std[:, None]

	def prior_sampling(self, shape):
		return torch.randn(*shape, device=self.device)

	def sampler(self, 
				score_model, 
				eps=1e-3,
				num_samples=None,
				context=None):

		"""Generate samples from score-based models with Predictor-Corrector method.

		Args:
			score_model: A PyTorch model that represents the time-dependent score-based model.
			marginal_prob: A function that gives the standard deviation of the perturbation kernel.
			diffusion_coeff: A function that gives the diffusion coefficient of the SDE.
			num_samples: The number of samplers to generate by calling this function once.
			num_steps: The number of sampling steps. 
			Equivalent to the number of discretized time steps.    
			device: 'cuda' for running on GPUs, and 'cpu' for running on CPUs.
			eps: The smallest time step for numerical stability.
		
		Returns: 
			Samples.
		"""

		if not num_samples: num_samples = self.num_samples

		t = torch.ones(num_samples, device=self.device)
		_ , std = self.marginal_prob(t)
		init_x = self.prior_sampling((num_samples, self.dim)) * std
		time_steps = np.linspace(self.T, eps, self.num_steps)
		step_size = time_steps[0] - time_steps[1]
		x = init_x

		for time_step in tqdm(time_steps, desc=' predictor-corrector sampling'):   

			batch_time_step = torch.ones(num_samples, device=self.device) * time_step
			
			# Corrector step (Langevin MCMC)
			grad = score_model(x, batch_time_step)
			grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
			noise_norm = np.sqrt(np.prod(x.shape[1:]))
			langevin_step_size = 2 * (self.snr * noise_norm / grad_norm)**2
			x = x + langevin_step_size * grad + torch.sqrt(2 * langevin_step_size) * torch.randn_like(x)      

			# Predictor step (Euler-Maruyama)
			_ , g = self.sde(batch_time_step)
			x_mean = x + (g**2) * score_model(x, batch_time_step) * step_size
			x = x_mean + torch.sqrt(g**2 * step_size) * torch.randn_like(x)      
			
			# The last step does not include any noise
			return x_mean





