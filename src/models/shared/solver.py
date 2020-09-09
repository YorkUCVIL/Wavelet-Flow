import tensorflow as tf
from tf_components import *
from models.shared.learning_rate_schedule import *

def solver(params, loss, config, global_placeholders):
	with tf.variable_scope(None,default_name='solver'):
		config_opt = config.training.optimizer
		if config_opt == "adamax":
			optimizer_op = optimizer.openai_adamax
		else:
			raise Exception('Unknown optimizer: {}'.format(config_opt))

		# lr schedule
		lr = learning_rate_schedule(config, global_placeholders)
		tf.summary.scalar('learning_rate',lr,collections=['frequent_summaries'])

		# gradients
		gradients = tf.gradients(loss,params)

		return optimizer_op(params, gradients, lr, config)
