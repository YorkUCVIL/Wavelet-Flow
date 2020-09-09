import tensorflow as tf
from util import *

class Conditioning_network:
	def __init__(self):
		self.encoder_list = [
			None,
			self.encode_1,
			self.encode_2,
			self.encode_3,
			self.encode_4,
			self.encode_5,
		]

	def encode(self,downsampled_stack):
		'''
		these networks must be manually specified,
		not setup procedurally via config
		'''
		conditioning_tensors = []
		for it,encoder in enumerate(self.encoder_list):
			conditioning_tensors.append(encoder(downsampled_stack[it]))

		return conditioning_tensors

	def encode_1(self,base):
		return base

	def encode_2(self,base):
		return base

	def encode_3(self,base):
		return base

	def encode_4(self,base):
		return base

	def encode_5(self,base):
		return base

	def get_variables(self,filter=None):
		vars = []

		return vars
