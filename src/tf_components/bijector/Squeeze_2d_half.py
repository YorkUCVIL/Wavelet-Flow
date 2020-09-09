import tensorflow as tf
# from tf_components import *
from tf_components.bijector.Bijector import *

class Squeeze_2d_half(Bijector):
	def __init__(self,collection=None,name='squeeze'):
		super().__init__(collection=None, name=name)

	def forward(self,x):
		with tf.variable_scope(None,default_name='squeeze_forward') as scope:
			factor = 2
			x_shape = x.get_shape()
			h = x_shape[1].value
			w = x_shape[2].value
			c = x_shape[3].value

			# make sure we can divide nicely
			assert h % factor == 0 and w % factor == 0, "({},{}) not dividing by {} nicely".format(h,w,factor)

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

			return y,0

	def inverse(self,y):
		with tf.variable_scope(None,default_name='squeeze_inverse') as scope:
			factor = 2
			y_shape = y.get_shape()
			h = y_shape[1].value
			w = y_shape[2].value
			c = y_shape[3].value

			# make sure n channels is divisible by 4
			assert c >= 4 and c % 4 == 0, "({}) channels must be divisible by 4".format(c)

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
