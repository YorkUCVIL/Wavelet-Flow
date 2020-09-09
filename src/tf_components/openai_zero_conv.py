import tensorflow as tf
from tf_components.Layer import *
from tf_components.edge_bias import *
from util import *

class Openai_zero_conv(Layer):
	'''
	open ai style conv
	'''
	def __init__(self,k_size,n_out,stride,edge_bias,
	dtype=tf.float32,collection=None,name="openai_zero_conv"):
		# setup naming and scoping
		super().__init__(collection=collection,name=name)

		self.w = None
		self.b = None
		self.log_scale = None

		self.k_size = k_size
		self.n_out = n_out
		self.stride = [1,stride,stride,1]
		self.dtype = dtype
		self.edge_bias = edge_bias
		self.log_scale_factor = config.model.openai_conv.zero_logscale_factor
		self.use_logscale = config.model.openai_conv.zero_use_logscale

	def __call__(self,x):
		with tf.variable_scope(self.scope):
			# setup edge bias
			padding = 'SAME'
			if self.edge_bias:
				x = edge_bias(x,self.k_size)
				padding = 'VALID'

			# setup variables if needed
			if self.w is None:
				x_shape = x.get_shape()
				c_in = x_shape[3].value
				filter_shape = [self.k_size,self.k_size,c_in,self.n_out]
				self.w = tf.get_variable('w',shape=filter_shape,initializer=tf.zeros_initializer(),dtype=self.dtype)
				self.b = tf.get_variable('b',shape=[1,1,1,self.n_out],initializer=tf.zeros_initializer(),dtype=self.dtype)
				if self.use_logscale:
					self.log_scale = tf.get_variable('log_scale',shape=[1,1,1,self.n_out],initializer=tf.zeros_initializer(),dtype=self.dtype)



			z = tf.nn.conv2d(x,self.w,self.stride,padding)
			z += self.b
			if self.use_logscale:
				z *= tf.exp(self.log_scale * self.log_scale_factor)
			return z

	def get_variables(self,filter=None):
		assert self.w is not None, 'Variables not initialized'

		vars = []

		vars += [self.w,self.b]
		if self.use_logscale:
			vars += [self.log_scale]

		return vars
