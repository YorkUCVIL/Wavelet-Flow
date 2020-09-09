import tensorflow as tf

def uniform_dequant_bound(data_shape, log_density, n_bins):
	'''
	converts a continuous log density to a lower bound on the log probability
	in respect to uniform dequant, assumes values are compressed within [0,1]
	'''
	with tf.variable_scope(None,default_name='uniform_dequant_bound'):
		data_h = data_shape[0]
		data_w = data_shape[1]
		data_c = data_shape[2]
		n_dim = data_h*data_w*data_c

		log_prob = log_density - (tf.log(n_bins)*n_dim)
		return log_prob
