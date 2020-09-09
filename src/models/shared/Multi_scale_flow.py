import tensorflow as tf
from tf_components import *
from models.shared.Single_scale_flow import *
import numpy as np

class Multi_scale_flow:
	def __init__(self,conditioning_network,partial_level=-1):
		# create scope for variables
		with tf.variable_scope(None,default_name='multi_scale_flow') as scope:
			self.variable_scope = scope

		# helpers
		n_steps = config.model.steps_per_resolution
		coupling_widths = config.model.conv_widths
		spatial_biasing = config.model.spatial_biasing
		self.n_levels = config.model.n_levels
		self.base_level = config.model.base_level
		self.partial_level = partial_level

		with tf.variable_scope(self.variable_scope) as scope:
			self.haar_squeeze_split = bijector.Haar_squeeze_split(compensate=True)

			# create base flow if builing it specifically or building whole flow
			if partial_level == -1 or partial_level == self.base_level:
				base_size = 2**self.base_level # assume shape is always square
				self.base_flow = Single_scale_flow(
					[base_size,base_size,3],
					n_steps[self.base_level],
					coupling_widths[self.base_level],
					spatial_biasing[self.base_level],
					name='Single_scale_flow_base'
				)
			else:
				self.base_flow = None

			start_flow_padding = [None]*self.base_level # add padding, since base may not be 0
			self.sub_flows = start_flow_padding + [self.base_flow] # append base
			for level in range(self.base_level+1,self.n_levels+1):
				if partial_level != -1 and partial_level != level:
					self.sub_flows.append(None)
				else:
					h = 2**(level-1) # assume shape is always square
					w = 2**(level-1)

					# assume original image is 3 channels, magic number
					self.sub_flows.append(Single_scale_flow([h,w,9],n_steps[level],coupling_widths[level],spatial_biasing[level],conditional=True,name='Single_scale_flow_{}'.format(level)))

			self.conditioning_network = conditioning_network

	def data_to_latent_and_log_density(self, data, init=False, partial_level=-1):
		'''
		forward, for training and validation
		'''
		with tf.variable_scope(None,default_name='data_to_latent_and_log_density') as scope:
			latents = []
			log_density = 0.0
			base = data # maybe call it "last_downsampled"
			for level in range(self.n_levels,self.base_level-1,-1):
				# compute flow
				if level == partial_level or partial_level == -1:
					if level == self.base_level: # base level doesn't need to extract details
						flow = self.base_flow
						latent, ld = flow.data_to_latent_and_log_density(base,init=init)
					else:
						# decompose base
						haar = self.haar_squeeze_split.forward(base)
						details = haar.details
						base = haar.base

						# condition
						conditioning = self.conditioning_network.encoder_list[level](base)
						flow = self.sub_flows[level]
						latent, ld = flow.data_to_latent_and_log_density(details,init=init,conditioning=conditioning)

					latents.append(latent)
					haar_ldj = flow.shape[0]*flow.shape[1]*flow.shape[2]*tf.log(0.5)*(self.n_levels-level)
					log_density += ld + haar_ldj # need custom haar_ldj because of partial

					if partial_level != -1:
						break # stop of we are doing partial
				else:
					# decompose base
					if self.partial_level <= 8 and level > 8: # i think this is where we use pre downsampled data
						pass
					else: # perform dowsampling, but don't build flow
						haar = self.haar_squeeze_split.forward(base)
						base = haar.base

					latents.append(None)

			out = to_attributes({})
			out.latents = latents # should we pad the skipped latents in?
			out.log_density = log_density
			return out

	def latent_to_data(self, latents):
		'''
		inverse pass
		'''
		with tf.variable_scope(None,default_name='latent_to_data') as scope:
			base_data = self.base_flow.latent_to_data(latents[self.base_level])
			base = base_data.data

			start_padding = [None]*self.base_level
			reconstructions = start_padding+[base]
			details = start_padding+[None]
			ld = base_data.ld
			ld_base = base_data.ld_base
			ldj = base_data.ldj
			for level in range(self.base_level+1,self.n_levels+1):
				latent = latents[level]
				base = reconstructions[-1]
				super_res = self.latent_to_super_res(latent,level,base)

				ld += super_res.ld
				ld_base += super_res.ld_base
				ldj += super_res.ldj
				reconstructions.append(super_res.reconstruction)
				details.append(super_res.details)

			out = to_attributes({})
			out.ld = ld
			out.ld_base = ld_base
			out.ldj = ldj
			out.reconstructions = reconstructions
			out.details = details
			return out

	def latent_to_super_res(self,latent,level,base):
		with tf.variable_scope(None,default_name='latent_to_super_res') as scope:
			flow = self.sub_flows[level]
			conditioning = self.conditioning_network.encoder_list[level](base)
			level_sample = flow.latent_to_data(latent,conditioning=conditioning)
			detail = level_sample.data
			recon, ldj_haar = self.haar_squeeze_split.inverse(base,detail)

			out = to_attributes({})
			out.ld = level_sample.ld - ldj_haar
			out.ld_base = level_sample.ld_base
			out.ldj = level_sample.ldj - ldj_haar
			out.reconstruction = recon
			out.details = detail
			return out

	def sample_latents(self,n_batch=1,temperature=1.0):
		latents = [None]*self.base_level

		for flow in self.sub_flows:
			latents.append(flow.sample_latents(n_batch=n_batch,temperature=temperature))

		return latents


	def get_variables(self,filter=None):
		vars = []
		vars += self.conditioning_network.get_variables()

		if filter is None:
			for flow in self.sub_flows:
				vars += flow.get_variables()
		elif filter and filter.startswith('partial_'):
			idx = int(filter[filter.find('_')+1:])
			vars += self.sub_flows[idx].get_variables()

		return vars
