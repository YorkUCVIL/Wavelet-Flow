
def to_attributes(var_in):
	'''
	convert dictionary to object with keys as attributes
	should implement with a class inheriting dict but would cause leak in 2.7
	ignores non dict objects
	'''
	class Container:
		def __str__(self):
			return str(self.__dict__)
		def update(self,up_dict):
			self.__dict__.update(up_dict.__dict__)

	if type(var_in) == dict:
		container = Container()
		container.__dict__ = {k: to_attributes(v) for k, v in var_in.items()}
		return container
	else:
		return var_in
