import tensorflow as tf

class Layer:
	'''
	This is a shared class for layers with parameters
	should aid transition to TF 2.0
	'''
	def __init__(self,collection=None,name="Layer"):
		'''
		auto scoping for TF 1.0 compatibility
		collection mechanism for simple layer tracking
		'''
		# setup naming and scoping
		self.name = name
		with tf.variable_scope(None,default_name=self.name) as scope:
			self.scope = scope

		# append self to collection
		if collection is not None:
			collection.append(self)

	def __call__(self):
		'''
		implement simple way to run layers
		'''
		raise NotImplementedError

	def get_variables(self,filter=None):
		'''
		implement for shared method to collect variables
		'''
		raise NotImplementedError
