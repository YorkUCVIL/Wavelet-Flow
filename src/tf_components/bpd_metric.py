import tensorflow as tf
import numpy as np

def bpd_metric(data_shape, log_likelihood):
	'''
	bits per dimension metric
	'''
	with tf.variable_scope(None,default_name='bpd_metric'):
		data_h = data_shape[0]
		data_w = data_shape[1]
		data_c = data_shape[2]
		n_dim = data_h*data_w*data_c

		bits = -log_likelihood/np.log(2.0)

		bpd = bits/n_dim
		return bpd
