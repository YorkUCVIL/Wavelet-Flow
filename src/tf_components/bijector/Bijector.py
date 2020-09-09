import tensorflow as tf
from tf_components.Layer import *

class Bijector(Layer):
	def __init__(self,collection=None,name="Layer"):
		super().__init__(collection=collection,name=name)

	def __call__(self,*args,**kwargs):
		'''
		default to call forward
		'''
		return self.forward(*args,**kwargs)

	def forward(self):
		'''
		implement forward pass
		'''
		raise NotImplementedError

	def inverse(self):
		'''
		implement inverse pass
		'''
		raise NotImplementedError
