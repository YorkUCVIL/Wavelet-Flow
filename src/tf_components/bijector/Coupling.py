import tensorflow as tf
from util import *
from tf_components.bijector.Bijector import *
from tf_components.bijector.Act_norm import *
from tf_components.openai_conv import Openai_conv
from tf_components.openai_zero_conv import *

class Coupling_res_block(Layer):
	def __init__(self,k_size,width,edge_bias,
	dtype=tf.float32,collection=None,name="coupling_res_block"):
		# setup naming and scoping
		super().__init__(collection=collection,name=name)

		self.conv1 = Openai_conv(k_size,width,1,edge_bias=edge_bias)
		self.conv2 = Openai_conv(1,width,1,edge_bias=edge_bias,activation='none')

	def __call__(self,x,init=False):
		with tf.variable_scope(self.scope):
			conv1 = self.conv1(x,init=init)
			conv2 = self.conv2(conv1,init=init)
			conv2 = tf.nn.relu(x+conv2)

			return conv2

	def get_variables(self,filter=None):
		vars = []
		vars += self.conv1.get_variables()
		vars += self.conv2.get_variables()
		return vars

class Coupling(Bijector):
	def __init__(self,
	k,
	width,
	edge_bias,
	conditional=False,
	collection=None, name='coupling'):
		super().__init__(collection=None, name=name)

		self.conditional = conditional
		self.k = k
		self.width = width # config.model.conv_width
		self.split_scheme = 'channel'
		self.edge_bias = edge_bias

		n_res_blocks = config.model.n_res_blocks

		with tf.variable_scope(self.scope):
			self.conv_in = Openai_conv(1,self.width,1,edge_bias=self.edge_bias)

			self.res_blocks = []
			for n in range(n_res_blocks):
				self.res_blocks.append(Coupling_res_block(self.k,self.width,self.edge_bias))

			self.conv_out = None

	def st(self,x,out_size,init=False,conditioning=None):
		assert (self.conditional) == (conditioning is not None), 'Coupling must have conditioning tensor on forward when set as conditional'
		x_shape = x.get_shape()
		c = x_shape[3].value
		out_size2 = out_size*2

		with tf.variable_scope(self.scope):
			# must create here because is data dependent
			if self.conv_out is None:
				self.conv_out = Openai_zero_conv(self.k,out_size2,1,edge_bias=self.edge_bias)

		with tf.variable_scope(None,default_name='scale_translate') as scope:
			network_input = tf.concat([x,self.adapt_conditioning(conditioning)],axis=-1) if self.conditional else x

			conv_in = self.conv_in(network_input,init=init)

			res = conv_in
			for block in self.res_blocks:
				res = block(res,init=init)

			conv_out = self.conv_out(res)

			s = conv_out[:,:,:,:out_size]
			t = conv_out[:,:,:,out_size:]

			return s,t

	def forward(self,x,init=False,conditioning=None):
		with tf.variable_scope(None,default_name=self.name+'_forward'):
			# split input along channels
			x1,x2 = self.split_features(x)
			x2_c = x2.get_shape()[3]

			# neural network of t,s
			s,t = self.st(x1,out_size=x2_c,init=init,conditioning=conditioning)

			# compute ys
			y1 = x1
			tanh = tf.tanh(s)
			scale = tf.exp(tanh)
			y2 = (x2 + t) * scale

			# recombine y values
			y = self.concat_features(y1,y2)

			# ldj
			log_scale = tanh
			ldj = tf.reduce_sum(log_scale,axis=[1,2,3])

			# check numerics
			if config.debug.check_numerics:
				y = tf.debugging.check_numerics(y,'bad_numerics')
				ldj = tf.debugging.check_numerics(ldj,'bad_numerics')

			return y,ldj

	def inverse(self,y,conditioning=None):
		with tf.variable_scope(None,default_name=self.name+'_inverse'):
			# split input along channels
			y1,y2 = self.split_features(y)
			y2_c = y2.get_shape()[3]

			# neural network of t,s
			s,t = self.st(y1,out_size=y2_c,conditioning=conditioning)

			# compute ys
			x1 = y1
			tanh = tf.tanh(s)
			scale_inv = tf.exp(-tanh)
			x2 = y2*scale_inv - t

			# recombine y values
			x = self.concat_features(x1,x2)

			# ldj
			log_scale = tanh
			ldj = -tf.reduce_sum(log_scale,axis=[1,2,3])

			return x,ldj

	def adapt_conditioning(self,conditioning):
		if self.split_scheme == 'channel':
			cond = conditioning
		else:
			assert False, 'unknown split scheme: {}'.format(self.split_scheme)
		return cond

	def split_features(self,x):
		x_shape = x.get_shape()
		h = x_shape[1].value
		w = x_shape[2].value
		c = x_shape[3].value
		if self.split_scheme == 'channel':
			channel_center = c//2
			x1 = x[:,:,:,:channel_center]
			x2 = x[:,:,:,channel_center:]
		else:
			assert False, 'unknown split scheme: {}'.format(self.split_scheme)

		return x1,x2

	def concat_features(self,x1,x2):
		x1_shape = x1.get_shape()
		x2_shape = x2.get_shape()
		if self.split_scheme == 'channel':
			x = tf.concat([x1,x2],axis=3)
		else:
			assert False, 'unknown split scheme: {}'.format(self.split_scheme)

		return x

	def get_variables(self, filter=None):
		vars = []
		vars += self.conv_in.get_variables()
		for block in self.res_blocks:
			vars += block.get_variables()
		vars += self.conv_out.get_variables()
		return vars
