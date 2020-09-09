import tensorflow as tf
from tf_components.Layer import *
from tf_components.edge_bias import *
from tf_components.bijector.Act_norm import Act_norm
from util import *

class Openai_conv(Layer):
	'''
	open ai style conv
	act norm will be included in the future
	'''
	def __init__(self,k_size,n_out,stride,edge_bias,activation='relu',
	dtype=tf.float32,collection=None,name="openai_conv"):
		# setup naming and scoping
		super().__init__(collection=collection,name=name)

		self.w = None
		self.b = None
		self.actnorm = Act_norm()

		self.k_size = k_size
		self.n_out = n_out
		self.stride = [1,stride,stride,1]
		self.dtype = dtype
		self.edge_bias = edge_bias
		self.normalize = config.model.openai_conv.normalized
		self.activation = activation

	def __call__(self,x,init=False):
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
				initializer = tf.random_normal_initializer(0.0,0.05)
				filter_shape = [self.k_size,self.k_size,c_in,self.n_out]
				self.w = tf.get_variable('w',shape=filter_shape,initializer=initializer,dtype=self.dtype)
				if not self.normalize:
					self.b = tf.get_variable('b',shape=[1,1,1,self.n_out],initializer=tf.zeros_initializer(),dtype=self.dtype)

			z = tf.nn.conv2d(x,self.w,self.stride,padding)

			if self.normalize: # add param later
				z,_ = self.actnorm.forward(z,init=init)
			else:
				z += self.b

			if self.activation == 'relu':
				z = tf.nn.relu(z)

			return z

	def get_variables(self,filter=None):
		assert self.w is not None, 'Variables not initialized'

		vars = []

		vars += [self.w]
		if not self.normalize:
			vars += [self.b]
		else:
			vars += self.actnorm.get_variables(filter)

		return vars
