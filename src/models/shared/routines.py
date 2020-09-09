from models.shared.solver import *
from util import *
from tf_components import *

def sample(net,n_batch=1,temperature=1.0):
	tlog('Building sampling graph','note')
	with tf.variable_scope(None,default_name='generator_sampling'):
		latent = net.sample_latents(n_batch=n_batch,temperature=temperature)
		sampled_data = net.latent_to_data(latent)

		return sampled_data

def validate(validation_data,net):
	n_bins = 2.0**config.model.data.n_bits
	data_shape = [
		config.model.data.dimensions.h,
		config.model.data.dimensions.w,
		config.model.data.dimensions.c
	]

	tlog('Building validation graph','note')
	with tf.variable_scope(None,default_name='validation'):
		latents_densities = net.data_to_latent_and_log_density(validation_data.im)
		val_log_prob = uniform_dequant_bound(data_shape,latents_densities.log_density,n_bins)
		val_bpd = bpd_metric(data_shape,val_log_prob)

		return val_bpd

def train_partial(training_data,net,global_placeholders, partial_level):
	assert partial_level >= 0 and partial_level <= net.n_levels, 'level must be >= 0 and <= n_levels'
	n_bins = 2.0**config.model.data.n_bits
	data_shape = net.sub_flows[partial_level].shape
	full_data_shape = [ # used for non-cropped bpd computation
		config.model.data.dimensions.h,
		config.model.data.dimensions.w,
		config.model.data.dimensions.c
	]

	tlog('Building training graph','note')
	with tf.variable_scope(None,default_name='training'):
		latents_densities = net.data_to_latent_and_log_density(training_data.im, partial_level=partial_level)

		log_prob = uniform_dequant_bound(data_shape,latents_densities.log_density,n_bins)
		train_bpd = bpd_metric(data_shape,log_prob)
		train_bpd_mean = tf.reduce_mean(train_bpd) # this is in respect for current dimensions, with crop
		loss = train_bpd_mean

		params = net.get_variables('partial_{}'.format(partial_level))
		train_solver = solver(params,loss,config, global_placeholders)

		tf.summary.scalar('train_bpd',train_bpd_mean,collections=['frequent_summaries'])

		# this is the additive component of the bpd for the whole model
		# sum of this over all levels is bpd of model
		train_contrib_bpd = tf.reduce_mean(bpd_metric(full_data_shape,log_prob))
		tf.summary.scalar('train_contrib_bpd',train_contrib_bpd,collections=['frequent_summaries'])

		return train_solver, train_bpd_mean

def ddi_partial(training_data, net, partial_level):
	assert partial_level >= 0 and partial_level <= net.n_levels, 'level must be >= 0 and <= n_levels'
	tlog('Building ddi graph','note')
	with tf.variable_scope(None,default_name='training_ddi'):
		latents_densities = net.data_to_latent_and_log_density(training_data.ddi_im, partial_level=partial_level, init=True)
		valid_latents = []
		for l in latents_densities.latents:
			if l is not None:
				valid_latents.append(l)
		run_op = tf.group(valid_latents)
		return run_op

def validate_partial(validation_data,net,partial_level):
	assert partial_level >= 0 and partial_level <= net.n_levels, 'level must be >= 0 and <= n_levels'
	n_bins = 2.0**config.model.data.n_bits
	data_shape = net.sub_flows[partial_level].shape
	full_data_shape = [
		config.model.data.dimensions.h,
		config.model.data.dimensions.w,
		config.model.data.dimensions.c
	]

	tlog('Building validation graph','note')
	with tf.variable_scope(None,default_name='validation'):
		latents_densities = net.data_to_latent_and_log_density(validation_data.im,partial_level=partial_level)
		val_log_prob = uniform_dequant_bound(data_shape,latents_densities.log_density,n_bins)
		val_bpd = bpd_metric(data_shape,val_log_prob)

		tf.summary.scalar('val_bpd',tf.reduce_mean(val_bpd),collections=['infrequent_summaries'])

		# look at train partial for explanation
		val_contrib_bpd = tf.reduce_mean(bpd_metric(full_data_shape,val_log_prob))
		tf.summary.scalar('val_contrib_bpd',val_contrib_bpd,collections=['infrequent_summaries'])

		return val_bpd

def sample_super_res(net,base,temperature=1.0):
	tlog('Building sampling graph','note')
	with tf.variable_scope(None,default_name='generator_sampling'):
		n_batch = base.get_shape()[0].value
		base_h = float(base.get_shape()[1].value)
		base_level = int(round(np.log(base_h)/np.log(2.0)))

		reconstructions = [None]*(base_level+1)
		latents = net.sample_latents(n_batch=n_batch,temperature=temperature)
		for level in range(base_level+1,net.n_levels+1):
			latent = latents[level]
			sampled_data = net.latent_to_super_res(latent,level,base)
			base = sampled_data.reconstruction
			reconstructions.append(base)

		return reconstructions
