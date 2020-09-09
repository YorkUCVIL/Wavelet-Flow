import tensorflow as tf
from util import *
from tf_components.bijector.Bijector import *

class Element_shift_scale(Bijector):
	def __init__(self,std_log_scale_factor=3.0, dtype=tf.float32, collection=None, name='Element_shift_scale'):
		super().__init__(collection=None, name=name)

		self.dtype = dtype
		self.epsilon = 0.00001
		self.std_log_scale_factor = std_log_scale_factor

		self.mean = None
		self.log_std = None

	def get_mean_log_var(self, shape):
		# we need this to init based on input tensor shape
		h = shape[1]
		w = shape[2]
		channels = shape[3]

		if self.mean == None:
			with tf.variable_scope(self.scope) as scope:
				self.mean = tf.get_variable('mean',shape=[1,h,w,channels],initializer=tf.constant_initializer(0),dtype=self.dtype)
				self.log_std = tf.get_variable('variance',shape=[1,h,w,channels],initializer=tf.constant_initializer(0),dtype=self.dtype)

		#log_std_scaled = self.log_std*self.std_log_scale_factor
		return self.mean, self.log_std

	def forward(self,x):
		with tf.variable_scope(None,default_name=self.name+'_forward') as scope:
			mean, log_std = self.get_mean_log_var(x.get_shape())

			log_std = log_std*self.std_log_scale_factor
			inv_std = tf.exp(-log_std)

			# normalize
			y = (x-mean)*inv_std

			#ldj
			ldj = -tf.reduce_sum(log_std,axis=[1,2,3])

			# check numerics
			if config.debug.check_numerics:
				y = tf.debugging.check_numerics(y,'bad_numerics')
				ldj = tf.debugging.check_numerics(ldj,'bad_numerics')

			return y, ldj

	def inverse(self,y):
		with tf.variable_scope(None,default_name=self.name+'_inverse') as scope:
			mean, log_std = self.get_mean_log_var(y.get_shape())
			log_std = log_std*self.std_log_scale_factor
			x = (y*tf.exp(log_std)) + mean

			ldj = tf.reduce_sum(log_std,axis=[1,2,3])
			return x, ldj

	def get_variables(self, filter=None):
		assert self.mean is not None, 'Variables not initialized'

		vars = [self.mean,self.log_std]
		return vars
