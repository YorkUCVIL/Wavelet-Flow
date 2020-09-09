import tensorflow as tf
from tf_components.bijector.Bijector import *

class Haar_split(Bijector):
	'''
	seperates out low pass and detail coefficients from haar
	'''

	def __init__(self,collection=None,name='haar_split'):
		super().__init__(collection=None, name=name)

	def forward(self,x):
		with tf.variable_scope(None,default_name='haar_squeeze_forward') as scope:
			x_shape = x.get_shape()
			c = x_shape[3].value

			assert c%4 == 0, 'channels must be divisible by 4'

			n_actual = c//4

			averages = []
			details = []
			for it in range(n_actual):
				idx = it*4
				averages.append(x[:,:,:,idx:idx+1])
				details.append(x[:,:,:,idx+1:idx+4])

			averages = tf.concat(averages,axis=-1)
			details = tf.concat(details,axis=-1)

			return averages,details

	def inverse(self,averages,details):
		with tf.variable_scope(None,default_name='haar_squeeze_inverse') as scope:
			n_actual = averages.get_shape()[3].value

			assert n_actual == details.get_shape()[3].value/3, 'n channels mismatched'

			slices = []
			for it in range(n_actual):
				threes = it*3
				slices.append(averages[:,:,:,it:it+1])
				slices.append(details[:,:,:,threes:threes+3])

			reconstructed = tf.concat(slices,axis=-1)
			return reconstructed


	def get_variables(self, t):
		return []
