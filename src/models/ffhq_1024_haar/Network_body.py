import tensorflow as tf
from tf_components import *
from models.ffhq_1024_haar.Conditioning_network import *
import numpy as np

from models.shared.Multi_scale_flow import Multi_scale_flow

class Network_body(Multi_scale_flow):
	def __init__(self,partial_level=-1):
		conditioning_network = Conditioning_network()
		super().__init__(conditioning_network,partial_level=partial_level)
