import tensorflow as tf
import numpy as np

class Normalizing_flow:
	def __init__(self, data_shape, n_squeezes):
		self.data_shape = data_shape
		self.n_squeezes = n_squeezes

	def data_to_latent_and_log_density(self, data):
		'''
		forward pass, must implement
		'''
		latents = []
		log_density = 0

		return latents, log_density

	def latent_to_data(self, latents): # inverse pass
		'''
		inverse pass, must implement
		'''
		pass

	def get_variables(self):
		'''
		get variables for training, must implement
		'''
		pass


	def compute_latent_shapes(self, data_shape, n_squeezes):
		'''
		computes shapes of latents given data shape
		assumes we split the latent in half every squeeze except last
		'''
		h = data_shape[1]
		w = data_shape[2]
		c = data_shape[3]
		n_resolutions = n_squeezes
		latent_shapes = [] # ordered closest to data to furthest

		cur_h = h
		cur_w = w
		cur_c = c

		for res_level in range(n_resolutions):
			cur_h //= 2
			cur_w //= 2
			cur_c *= 2
			latent_shapes.append([cur_h,cur_w,cur_c])

		# correct last shape becuase no split
		if n_resolutions == 0:
			latent_shapes.append([cur_h,cur_w,cur_c])
		else:
			latent_shapes[-1][-1] = latent_shapes[-1][-1]*2

		return latent_shapes

	def sample_latents(self, n_batch=1, temperature=1.0, truncate=False):
		'''
		samples all latents required to generate data(s)
		'''
		shapes = self.compute_latent_shapes(self.data_shape, self.n_squeezes)
		latents = []

		if not type(temperature) == list:
			temperature = [temperature]*len(shapes)

		for shape,temp in zip(shapes,temperature):
			latents.append(self.sample_latent([n_batch]+shape, temp, truncate))

		return latents

	def sample_latent(self, shape, temperature=1.0, truncate=False):
		if truncate:
			latent = tf.random.truncated_normal(shape=shape)*temperature
		else:
			latent = tf.random.normal(shape=shape)*temperature

		return latent

	def latent_log_density(self,latent):
		'''
		latent variable is diagonal unit gaussian
		'''
		with tf.variable_scope(None,default_name='latent_to_ldensity') as scope:
			element_log_density = -0.5 * tf.log(2*np.pi)
			element_log_density -= 0.5 * latent**2
			total_log_density = tf.reduce_sum(element_log_density,axis=[1,2,3])

			return total_log_density
