import tensorflow as tf
from util import *
from tf_components.bijector.Bijector import *
from tf_components.bijector.Haar_squeeze import *
from tf_components.bijector.Haar_split import *

class Haar_squeeze_split(Bijector):
	'''
	combined haar decomp and then spit low-pass from detais
	'''

	def __init__(self,
	compensate=False,
	collection=None, name='haar_squeeze_split'):
		super().__init__(collection=None, name=name)

		self.compensate = compensate

		with tf.variable_scope(self.scope):
			self.split = Haar_split()
			self.haar_squeeze = Haar_squeeze()

	def forward(self,full_res):
		with tf.variable_scope(None,default_name=self.name+'_forward'):
			haar_squeeze, _ = self.haar_squeeze.forward(full_res)
			base,details = self.split.forward(haar_squeeze)
			ldj = 0

			if self.compensate:
				base_shape = base.get_shape()
				h = base_shape[1]
				w = base_shape[2]
				c = base_shape[3]
				n_dim = tf.cast(h*w*c,tf.float32)

				base = base*0.5 # haar base is 2*average
				ldj = tf.log(0.5)*n_dim

			haar_representation = to_attributes({})
			haar_representation.details = details
			haar_representation.base = base
			haar_representation.ldj = ldj

			return haar_representation

	def inverse(self,base,detail):
		with tf.variable_scope(None,default_name=self.name+'_inverse'):
			ldj = 0
			if self.compensate:
				base = base*2.0
				base_shape = base.get_shape()
				h = base_shape[1]
				w = base_shape[2]
				c = base_shape[3]
				n_dim = tf.cast(h*w*c,tf.float32)
				ldj = tf.log(0.5)*n_dim

			haar = self.split.inverse(base,detail)
			reconstructed = self.haar_squeeze.inverse(haar)

			return reconstructed, ldj

	def get_variables(self, t):
		vars = []
		return vars
