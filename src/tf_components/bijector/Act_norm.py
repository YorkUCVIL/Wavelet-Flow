import tensorflow as tf
from tf_components.bijector.Bijector import *
from util import *

class Act_norm(Bijector):
	def __init__(self, dtype=tf.float32, collection=None, name='actNorm'):
		super().__init__(collection=None, name=name)

		self.dtype = dtype
		self.epsilon = 0.00001
		self.std_log_scale_factor = config.model.actnorm.logscale

		self.mean = None
		self.log_std = None

	def get_mean_log_var(self, channels):
		# we need this to init based on input tensor shape

		if self.mean == None:
			with tf.variable_scope(self.scope) as scope:
				self.mean = tf.get_variable('mean',shape=[1,1,1,channels],initializer=tf.constant_initializer(0),dtype=self.dtype)
				self.log_std = tf.get_variable('variance',shape=[1,1,1,channels],initializer=tf.constant_initializer(0),dtype=self.dtype)

		#log_std_scaled = self.log_std*self.std_log_scale_factor
		return self.mean, self.log_std

	def forward(self,x,init=False):
		with tf.variable_scope(None,default_name=self.name+'_forward') as scope:
			e_mean, e_var = tf.nn.moments(x,[0,1,2],keep_dims=True)

			mean, log_std = self.get_mean_log_var(x.get_shape()[3].value)
			if init: # do ddi if doing init
				mean = mean.assign(e_mean)
				log_std_init = tf.log(tf.sqrt(e_var))/self.std_log_scale_factor
				log_std = log_std.assign(log_std_init)

			log_std = log_std*self.std_log_scale_factor
			inv_std = tf.exp(-log_std)

			# normalize
			y = (x-mean)*inv_std

			#ldj
			h = x.get_shape()[1].value
			w = x.get_shape()[2].value
			ldj = h*w * -tf.reduce_sum(log_std,axis=[1,2,3])

			# check numerics
			if config.debug.check_numerics:
				y = tf.debugging.check_numerics(y,'bad_numerics')
				ldj = tf.debugging.check_numerics(ldj,'bad_numerics')

			return y, ldj

	def inverse(self,y):
		with tf.variable_scope(None,default_name=self.name+'_inverse') as scope:
			mean, log_std = self.get_mean_log_var(y.get_shape()[3].value)
			log_std = log_std*self.std_log_scale_factor
			x = (y*tf.exp(log_std)) + mean

			h = x.get_shape()[1].value
			w = x.get_shape()[2].value
			ldj = h*w * tf.reduce_sum(log_std,axis=[1,2,3])
			return x, ldj

	def get_variables(self, filter=None):
		assert self.mean is not None, 'Variables not initialized'

		vars = [self.mean,self.log_std]
		return vars
