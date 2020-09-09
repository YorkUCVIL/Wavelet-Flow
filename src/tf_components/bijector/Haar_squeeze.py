import tensorflow as tf
# from op_blocks import *
from tf_components.bijector.Bijector import *
import numpy as np

class Haar_squeeze(Bijector):
	'''
	does haar decomposition, x4 channels, 1/2 spatial resolution
	'''

	def __init__(self,collection=None,name='haar_squeeze'):
		super().__init__(collection=None, name=name)

		self.cached_kernel = None
		self.cached_conv_kernel = None

	def get_haar_kernel(self, n_channels):
		'''
		generates kernel for haar wavelet downsampling
		kernel should be inverse of itself
		'''
		if self.cached_kernel is not None:
			return self.cached_kernel

		kernel = []
		for n in range(n_channels):
			front_padding = [0.0,0.0,0.0,0.0]*n
			back_padding = [0.0,0.0,0.0,0.0]*(n_channels-n-1)

			row = front_padding + [0.5,0.5,0.5,0.5] + back_padding
			kernel.append(row)
			row = front_padding + [0.5,-0.5,0.5,-0.5] + back_padding
			kernel.append(row)
			row = front_padding + [0.5,0.5,-0.5,-.5] + back_padding
			kernel.append(row)
			row = front_padding + [0.5,-0.5,-0.5,0.5] + back_padding
			kernel.append(row)

		# invert to prepare for conversion to 1x1 conv kernel
		kernel = tf.transpose(kernel,[0,1])

		# expand to valid 1x1 conv kernel
		kernel = tf.expand_dims(kernel,0)
		kernel = tf.expand_dims(kernel,0)

		# cache
		self.cached_kernel = kernel

		return self.cached_kernel

	def get_conv_haar_kernel(self,n_channels):
		if self.cached_conv_kernel is not None:
			return self.cached_conv_kernel

		k = np.zeros(shape=[2,2,n_channels,4*n_channels])

		for i in range(n_channels):
			k[:,:,i,i*4+0] = [[0.5,0.5],[0.5,0.5]]
			k[:,:,i,i*4+1] = [[0.5,-0.5],[0.5,-0.5]]
			k[:,:,i,i*4+2] = [[0.5,0.5],[-0.5,-0.5]]
			k[:,:,i,i*4+3] = [[0.5,-0.5],[-0.5,0.5]]

		k = tf.constant(k,dtype=tf.float32)

		self.cached_conv_kernel = k

		return self.cached_conv_kernel

	def forward(self,x):
		return self.forward_conv(x)
		# return self.forward_reshape(x)

	def forward_reshape(self,x):
		with tf.variable_scope(None,default_name='haar_squeeze_forward') as scope:
			factor = 2
			x_shape = x.get_shape()
			h = x_shape[1].value
			w = x_shape[2].value
			c = x_shape[3].value

			# make sure we can divide nicely
			assert h % factor == 0 and w % factor == 0, '({},{}) not dividing by {} nicely'.format(h,w,factor)

			# get kernel for haar
			haar_kernel = self.get_haar_kernel(c)

			# reshape to add two auxillary dimensions
			y = tf.reshape(x, [-1,
				h//factor, factor,
				w//factor, factor,
				c])

			# transpose to move auxillary dimentions near the channel dimensions
			y = tf.transpose(y, [0, 1, 3, 5, 2, 4])

			# collapse auxillary dimensions into channel dimension
			y = tf.reshape(y, [-1,
				h//factor,
				w//factor,
				c*factor*factor])

			# apply haar downsampling
			y = tf.nn.conv2d(y,haar_kernel,[1,1,1,1],'SAME')

			return y,0

	def forward_conv(self,x):
		with tf.variable_scope(None,default_name='haar_squeeze_forward') as scope:
			factor = 2
			x_shape = x.get_shape()
			h = x_shape[1].value
			w = x_shape[2].value
			c = x_shape[3].value

			# make sure we can divide nicely
			assert h % factor == 0 and w % factor == 0, '({},{}) not dividing by {} nicely'.format(h,w,factor)

			# get kernel for haar
			haar_kernel = self.get_conv_haar_kernel(c)

			y = tf.nn.conv2d(x,haar_kernel,[1,2,2,1],'VALID')

			return y,0

	def inverse(self,y):
		with tf.variable_scope(None,default_name='haar_squeeze_inverse') as scope:
			factor = 2
			y_shape = y.get_shape()
			h = y_shape[1].value
			w = y_shape[2].value
			c = y_shape[3].value

			# make sure n channels is divisible by 4
			assert c >= 4 and c % 4 == 0, '({}) channels must be divisible by 4'.format(c)

			# get kernel for haar
			haar_kernel = self.get_haar_kernel(c//4)

			# apply haar downsampling inverse
			y = tf.nn.conv2d(y,haar_kernel,[1,1,1,1],'SAME')

			x = tf.reshape(y, [-1, h, w,
				int(c/factor**2), factor, factor])
			x = tf.transpose(x, [0, 1, 4, 2, 5, 3])
			x = tf.reshape(x, [-1,
				int(h*factor),
				int(w*factor),
				int(c/(factor*factor))])

			return x

	def get_variables(self, t):
		return []
