import tensorflow as tf
from tf_components.bijector.Act_norm import *
from tf_components.bijector.Invertible_1x1 import *
from tf_components.bijector.Coupling import *
from tf_components.bijector.Bijector import *

class Step(Bijector):
	def __init__(self,
	width,
	edge_bias,
	conditional=False,
	collection=None,name='step'):
		super().__init__(collection=None, name=name)

		self.conditional = conditional

		with tf.variable_scope(self.scope) as scope:
			self.act_norm = Act_norm()
			self.invertible_1x1 = Invertible_1x1()
			self.coupling = Coupling(
				k=3,
				width=width,
				edge_bias=edge_bias,
				conditional=conditional
			)

	def forward(self,x,init=False,conditioning=None):
		with tf.variable_scope(None,default_name=self.name+'_forward'):
			assert (self.conditional) == (conditioning is not None), 'Step must have conditioning tensor on forward when set as conditional'

			y1,ldj1 = self.act_norm.forward(x,init=init)
			y2,ldj2 = self.invertible_1x1.forward(y1)
			y3,ldj3 = self.coupling.forward(y2,init=init,conditioning=conditioning)

			# sum ldj
			with tf.variable_scope(None,default_name='sum_ldj'):
				ldj = ldj1 + ldj2 + ldj3

			# checkpoint for memory saving
			tf.add_to_collection('checkpoints',y3)
			tf.add_to_collection('checkpoints',ldj1)
			tf.add_to_collection('checkpoints',ldj2)
			tf.add_to_collection('checkpoints',ldj3)

			return y3, ldj

	def inverse(self,y,conditioning=None):
		with tf.variable_scope(None,default_name=self.name+'_inverse'):
			assert (self.conditional) == (conditioning is not None), 'Step must have conditioning tensor on forward when set as conditional'

			x1, ldj1 = self.coupling.inverse(y,conditioning=conditioning)
			x2, ldj2 = self.invertible_1x1.inverse(x1)
			x3, ldj3 = self.act_norm.inverse(x2)

			ldj = ldj1+ldj2+ldj3
			return x3, ldj

	def get_variables(self, filter=None):
		vars = []

		vars += self.act_norm.get_variables()
		vars += self.invertible_1x1.get_variables()
		vars += self.coupling.get_variables()

		return vars
