import tensorflow as tf
import numpy as np
from util import *
from tf_components.bijector.Bijector import *
import scipy

def Invertible_1x1(use_lu=False,*args,**kwargs):
	if use_lu:
		return Invertible_1x1_lu(*args,**kwargs)
	else:
		return Invertible_1x1_matrix(*args,**kwargs)


class Invertible_1x1_lu(Bijector):
	def __init__(self,dtype = tf.float32, collection=None, name='invertible_1x1_lu'):
		super().__init__(collection=None, name=name)
		self.dtype = dtype
		self.k = None
		self.k_size = None

	def get_kernel(self,x):
		with tf.variable_scope(self.scope) as scope:
			if self.k is None:
				c = x.get_shape()[3].value
				self.k_size = c

				# init with random rotation
				# init and factor into components
				init = np.linalg.qr(np.random.randn(c,c))[0].astype('float32')
				p_init, l_init, u_init = scipy.linalg.lu(init,False)
				s_sign_init = np.sign(np.diag(u_init))# we need to keep the sign
				log_s_init = np.log(abs(np.diag(u_init))) # keep log s, force invertibility
				u_init = np.triu(u_init,1) # remove s from u

				# create our vars
				p = tf.constant(p_init)
				l = tf.get_variable("l",initializer=l_init,dtype=self.dtype)
				u = tf.get_variable("u",initializer=u_init,dtype=self.dtype)
				s_sign = tf.constant(s_sign_init)
				log_s = tf.get_variable("log_s",initializer=log_s_init,dtype=self.dtype)

				# mask to force LU decomp forever
				u_mask = np.triu(np.ones([c,c]),1)
				l_mask = np.tril(np.ones([c,c]),-1)
				u_masked = u * u_mask
				l_masked = l * l_mask + np.eye(c) # mask and replace diags

				# recombine into original matrix
				u_full = tf.diag(tf.exp(log_s)*s_sign) + u_masked
				k = tf.matmul(p,tf.matmul(l_masked,u_full))

				# save
				self.k = k
				self.log_s = log_s
				self.l = l
				self.u = u


			return self.k, self.log_s

	def forward(self,x):
		with tf.variable_scope(None,default_name=self.name+'_forward') as scope:
			k_m, log_s = self.get_kernel(x)
			k = tf.reshape(k_m,[1,1,self.k_size,self.k_size])
			y = tf.nn.conv2d(x,k,[1,1,1,1],'SAME')

			#ldj
			x_shape = x.get_shape()
			h = x_shape[1].value
			w = x_shape[2].value

			ldj = h * w * tf.reduce_sum(log_s)

			# check numerics
			if config.debug.check_numerics:
				y = tf.debugging.check_numerics(y,'bad_numerics')
				ldj = tf.debugging.check_numerics(ldj,'bad_numerics')

			return y,ldj

	def inverse(self,y):
		'''
		NOT TAKING ADVANTAGE OF LU TO INVERT
		'''
		with tf.variable_scope(None,default_name=self.name+'_inverse') as scope:
			k, log_s = self.get_kernel(y) # y should be same size as x
			k = tf.matrix_inverse(k)
			k = tf.reshape(k,[1,1,self.k_size,self.k_size])
			x = tf.nn.conv2d(y,k,[1,1,1,1],'SAME')
			return x

	def get_variables(self, t):
		if t == 'all_trainable':
			vars = [self.l,self.u,self.log_s]
			return vars
		else:
			return []

class Invertible_1x1_matrix(Bijector):
	def __init__(self,dtype = tf.float32, collection=None, name='invertible_1x1_matrix'):
		super().__init__(collection=None, name=name)
		self.dtype = dtype
		self.k = None
		self.k_size = None


	def get_kernel(self,x):
		with tf.variable_scope(self.scope) as scope:
			if self.k is None:
				c = x.get_shape()[3].value
				self.k_size = c

				# init with random rotation
				init = np.linalg.qr(np.random.randn(c,c))[0]
				init = np.asarray(init,np.float32)
				self.k = tf.get_variable('kernel',initializer=init,dtype=self.dtype)

			return self.k

	def forward(self,x):
		with tf.variable_scope(None,default_name=self.name+'_forward') as scope:
			k_m = self.get_kernel(x)
			k = tf.reshape(k_m,[1,1,self.k_size,self.k_size])
			y = tf.nn.conv2d(x,k,[1,1,1,1],'SAME')

			#ldj
			x_shape = x.get_shape()
			h = x_shape[1].value
			w = x_shape[2].value

			_, det = tf.linalg.slogdet(tf.cast(k_m,tf.float64))
			det = tf.cast(det,tf.float32)
			det *= h * w
			ldj = det

			return y,ldj

	def inverse(self,y):
		with tf.variable_scope(None,default_name=self.name+'_inverse') as scope:
			k_m = self.get_kernel(y) # y should be same size as x
			k_m_inv = tf.matrix_inverse(k_m)
			k = tf.reshape(k_m_inv,[1,1,self.k_size,self.k_size])
			x = tf.nn.conv2d(y,k,[1,1,1,1],'SAME')

			# ldj
			x_shape = x.get_shape()
			h = x_shape[1].value
			w = x_shape[2].value
			_, det = tf.linalg.slogdet(tf.cast(k_m,tf.float64))
			det = tf.cast(det,tf.float32)
			det *= h * w
			ldj = -det
			return x, ldj

	def get_variables(self, filter=None):
		assert self.k is not None, 'Variables not initialized'

		vars = [self.k]
		return vars
