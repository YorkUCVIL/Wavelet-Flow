import tensorflow as tf

def learning_rate_schedule(config,global_placeholders):
	with tf.variable_scope(None,default_name='learning_rate_schedule'):
		base_lr = config.training.base_learning_rate
		ramp_up_iterations = config.training.ramp_up_iterations
		it = global_placeholders.current_iteration

		scale = tf.minimum(1.0,it/(ramp_up_iterations))
		return base_lr * scale
