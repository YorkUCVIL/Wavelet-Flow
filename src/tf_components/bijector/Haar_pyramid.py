import tensorflow as tf
from util import *
from tf_components.bijector.Bijector import *
from tf_components.bijector.Haar_squeeze_split import *

class Haar_pyramid(Bijector):
	'''
	used stand alone, wavelet flow has its own internal pyramid
	'''

	def __init__(self,
	levels,
	collection=None, name='haar_pyramid'):
		super().__init__(collection=None, name=name)

		self.levels = levels

		with tf.variable_scope(self.scope):
			self.haar_squeeze_split = Haar_squeeze_split(compensate=True)

	def forward(self,x):
		with tf.variable_scope(None,default_name=self.name+'_forward'):

			base = x
			details = []
			downsampled = [] # currently here for debug
			ldj = 0
			for it in range(self.levels):
				haar = self.haar_squeeze_split.forward(base)
				details.append(haar.details)
				downsampled.append(haar.base)
				base = haar.base
				ldj += haar.ldj

			# flip, order from lowest res to highest
			details = list(reversed(details))
			downsampled = list(reversed(downsampled))

			haar_representation = to_attributes({})
			haar_representation.details = details
			haar_representation.base = base
			haar_representation.downsampled = downsampled
			haar_representation.ldj = ldj

			return haar_representation

	def inverse(self,haar_representation):
		with tf.variable_scope(None,default_name=self.name+'_inverse'):

			base = haar_representation.base
			downsampled = []
			for it in range(self.levels):
				downsampled.append(base)
				base = self.haar_squeeze_split.inverse(base,haar_representation.details[it])

			haar_representation = to_attributes({})
			haar_representation.reconstruction = base
			haar_representation.downsampled = downsampled

			return haar_representation

	def get_variables(self, t):
		vars = []
		return vars
