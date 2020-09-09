import tensorflow as tf

def session_setup(cmd_args):
	"""
	container for tensorflow config
	"""

	# config
	config = tf.ConfigProto()
	config.gpu_options.allow_growth = True
	config.allow_soft_placement = True

	return tf.Session(config=config)
