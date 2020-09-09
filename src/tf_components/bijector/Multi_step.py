import tensorflow as tf
from tf_components.bijector.Step import *
# from tf_components.bijector.Conditioning import *
from tf_components.bijector.Bijector import *
from util import *

class Multi_step(Bijector):
	def __init__(self,n_steps,width,edge_bias,conditional=False,collection=None,name='mutlistep',**remaining_kwargs):
		super().__init__(collection=None, name=name)

		self.conditional = conditional
		CONDITIONING_POSITIONS = [4,8] # legacy

		self.steps = []
		with tf.variable_scope(self.scope) as scope:
			for n in range(n_steps):
				self.steps.append(Step(width,edge_bias,conditional=conditional,name='step_'+str(n),**remaining_kwargs))

	def forward(self,x,conditioning=None,init=False):
		with tf.variable_scope(None,default_name=self.name+'_forward'):
			# make sure we have conditioning tensor if this is conditional
			assert (self.conditional) == (conditioning is not None), 'Multi_step must have conditioning tensor on forward when set as conditional'

			ldjs = []
			y = x
			for step in self.steps:
				cond = conditioning if step.conditional else None
				y, ldj_part = step.forward(y,conditioning=cond,init=init)
				ldjs.append(ldj_part)

			# sum ldj
			ldj = 0
			for x in ldjs:
				ldj += x

			return y, ldj

	def inverse(self,y,conditioning=None):
		with tf.variable_scope(None,default_name=self.name+'_inverse'):
			# make sure we have conditioning tensor if this is conditional
			assert (self.conditional) == (conditioning is not None), 'Multi_step must have conditioning tensor on inverse when set as conditional'

			x = y
			ldj = 0
			for step in reversed(self.steps):
				cond = conditioning if step.conditional else None
				x, ldj_part = step.inverse(x,conditioning=cond)
				ldj += ldj_part

			return x, ldj

	def get_variables(self, filter=None):
		vars = []
		for step in self.steps:
			vars += step.get_variables()
		return vars
