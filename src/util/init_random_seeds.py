import tensorflow as tf
import numpy as np

from util import *

def init_random_seeds():
	tf.set_random_seed(config.random_seed.tensorflow)
	np.random.seed(config.random_seed.numpy)
