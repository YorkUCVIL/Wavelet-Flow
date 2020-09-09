import tensorflow as tf
from tf_components import *
import numpy as np
import math
from util import *
import tensorflow_probability as tfp

from models.shared.Normalizing_flow import Normalizing_flow

class Single_scale_flow(Normalizing_flow):
	# --------------------------------
	# forward: data -> latent
	# backward: latent -> data
	# --------------------------------
	def __init__(self,shape,n_steps,coupling_width,spatial_bias,conditional=False,name='Norm_flow'):
		h = shape[0]
		w = shape[1]
		c = shape[2]
		self.shape = shape
		super().__init__([None,h,w,c], 0) # we don't squeeze split, multi-scale handled by wavelets

		# create scope for variables
		with tf.variable_scope(None,default_name=name) as scope:
			self.variable_scope = scope

		# helpers
		self.conditional = conditional
		self.spatial_bias = spatial_bias

		# generate flow layers here
		with tf.variable_scope(self.variable_scope) as scope:
			self.step = bijector.Multi_step(
					n_steps,
					coupling_width,
					edge_bias=self.spatial_bias,
					conditional=conditional,
					name="flow_steps",
			)
			# base measure tforms
			if self.spatial_bias:
				self.base_tform = bijector.Element_shift_scale()
			else:
				self.base_tform = bijector.Act_norm()

	def data_to_latent_and_log_density(self, data, conditioning=None, init=False):
		'''
		forward pass
		'''
		with tf.variable_scope(None,default_name='data_to_latent_and_log_density') as scope:
			# forward passes
			assert (self.conditional) == (conditioning is not None), 'Single_scale_flow must have conditioning tensor on forward when set as conditional'
			latent, ldj = self.step.forward(data,init=init,conditioning=conditioning)

			# base measure tform
			latent, base_ldj = self.base_tform.forward(latent)

			# compute densities
			ld = self.latent_log_density(latent)

			# sum densities/det
			log_density = ldj + ld + base_ldj

			return latent, log_density

	def latent_to_data(self, latent_in, conditioning=None):
		'''
		inverse pass
		'''
		with tf.variable_scope(None,default_name='latent_to_data') as scope:
			# base measure tforms
			latent, ldj_base = self.base_tform.inverse(latent_in[0])

			# main body of flow
			assert (self.conditional) == (conditioning is not None), 'Single_scale_flow must have conditioning tensor on forward when set as conditional'
			inverted, ldj_flow = self.step.inverse(latent,conditioning=conditioning)

			# compute likelihood and components
			ld_base = self.latent_log_density(latent_in[0])
			ldj =  - ldj_base - ldj_flow
			ld = ld_base + ldj

			out = to_attributes({})
			out.ld = ld
			out.ld_base = ld_base
			out.ldj = ldj
			out.data = inverted
			return out

	def sample_latent_mcmc(self,
		conditioning=None,n_batch=1,temperature=1.0,
		step_size=0.01,adaptation_steps=1,warmup_steps=3):

		gamma = 1.0/(temperature**2.0)
		seed = config.random_seed.tensorflow

		def prob_f(state):
			latent = [tf.reshape(state,[n_batch]+self.shape)]

			sampled_data = self.latent_to_data(latent,conditioning=conditioning)
			log_prob = gamma*sampled_data.ld_base + (gamma-1)*sampled_data.ldj

			return log_prob

		init_state = self.sample_latents(n_batch=n_batch,temperature=temperature)
		init_state = tf.reshape(init_state,[n_batch,-1])
		sampler_nuts = tfp.mcmc.NoUTurnSampler(prob_f,step_size,10,seed=seed)
		sampler = tfp.mcmc.DualAveragingStepSizeAdaptation(
			inner_kernel=sampler_nuts,
			step_size_setter_fn=lambda pkr, new_step_size: pkr._replace(step_size=new_step_size),
			step_size_getter_fn=lambda pkr: pkr.step_size,
			log_accept_prob_getter_fn=lambda pkr: pkr.log_accept_ratio,
			num_adaptation_steps=adaptation_steps,
			target_accept_prob=0.60
		)
		samples, trace = tfp.mcmc.sample_chain(num_results=adaptation_steps+warmup_steps,current_state=init_state,kernel=sampler,parallel_iterations=1)

		return samples, trace

	def get_variables(self, filter=None):
		'''
		gets variables
		'''
		vars = []

		# collect layer vars
		vars += self.step.get_variables()
		vars += self.base_tform.get_variables()

		return vars
