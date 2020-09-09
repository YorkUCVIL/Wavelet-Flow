import tensorflow as tf
from tf_components.bijector.Squeeze_2d_half import *
from tf_components.bijector.Multi_step import *
# from tf_components.bijector.Split_Latent import *

class Squeeze_step(Bijector):
	def __init__(self,n_steps,width,split,conditional=False,collection=None,name='mutlistep',**remaining_kwargs):
		super().__init__(collection=None, name=name)

		self.split_last = split
		self.conditional = conditional

		with tf.variable_scope(self.scope) as scope:
			self.squeeze = Squeeze_2d_half()
			self.multi_step = Multi_step(n_steps, width,conditional=conditional,**remaining_kwargs)
			if self.split_last:
				self.split = Split_Latent()

	def forward(self,x,conditioning=None,init=False):
		with tf.variable_scope(None,default_name=self.name+'_forward'):
			# make sure we have conditioning tensor if this is conditional
			assert (self.conditional) == (conditioning is not None), 'Squeeze_step must have conditioning tensor on forward when set as conditional'

			y, _ = self.squeeze.forward(x)
			y, ldj = self.multi_step.forward(y,conditioning=conditioning,init=init)

			latent = None
			if self.split_last:
				y, _, latent = self.split.forward(y)

			return y, ldj, latent

	def inverse(self,y,conditioning=None,latent=None):
		with tf.variable_scope(None,default_name=self.name+'_inverse'):
			# make sure we have conditioning tensor if this is conditional
			assert (self.conditional) == (conditioning is not None), 'Squeeze_step must have conditioning tensor on inverse when set as conditional'

			if self.split_last:
				x = self.split.inverse(y,latent)
			else:
				x = y

			x = self.multi_step.inverse(x,conditioning=conditioning)
			x = self.squeeze.inverse(x)

			return x

	def get_variables(self, t):
		return self.multi_step.get_variables(t)
