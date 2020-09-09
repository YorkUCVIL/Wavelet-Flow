import tensorflow as tf

'''
modified from glow
'''

def polyak(params, beta):
	ema = tf.train.ExponentialMovingAverage(decay=beta, zero_debias=True)
	avg_op = tf.group(ema.apply(params))
	# Swapping op
	updates = []
	for i in range(len(params)):
		p = params[i]
		avg = ema.average(p)
		tmp = 0. + avg * 1.
		with tf.control_dependencies([tmp]):
			update1 = avg.assign(p)
			with tf.control_dependencies([update1]):
				update2 = p.assign(tmp)
				updates += [update1, update2]
	swap_op = tf.group(*updates)
	return avg_op, swap_op, ema

def openai_adamax(params, cost_or_grads, lr, config, epsilon=1e-8):
	beta1 = config.optimizer.adamax.beta1
	weight_decay = config.optimizer.adamax.weight_decay
	polyak_epochs = config.optimizer.adamax.polyak_epochs # set to constant
	train_its = config.optimizer.adamax.train_its # used to set beta2
	# ---------- this was taken from openai's glow code, params were not tuned heavly ------------

	updates = []
	if type(cost_or_grads) is not list: # for future if need to use mem saving
		gs = tf.gradients(cost_or_grads, params)
	else:
		gs = cost_or_grads

	beta2 = 1-1./(train_its* polyak_epochs)

	# all-reduce
	grads = [g for g in gs] # use hvd size 1 so is identity

	t = tf.Variable(1., 'adam_t') # no idea what this is
	alpha_t = lr * tf.sqrt((1. - tf.pow(beta2, t))) / \
		(1. - tf.pow(beta1, t))
	updates.append(t.assign_add(1))

	for w, g in zip(params, grads):
		mom2 = tf.Variable(tf.zeros(w.get_shape()), w.name + '_adam_m2')
		if beta1 > 0:
			mom1 = tf.Variable(tf.zeros(w.get_shape()), w.name + '_adam_m1')
			mom1_new = beta1 * mom1 + (1. - beta1) * g
			updates.append(mom1.assign(mom1_new))
		else:
			mom1_new = g
		m2_new = tf.maximum(beta2 * mom2, abs(g))
		delta_t = mom1_new / (m2_new + epsilon)
		w_new = weight_decay * w - alpha_t * delta_t
		updates.append(mom2.assign(m2_new))
		updates.append(w.assign(w_new))

	# Polyak averaging
	polyak_avg_op, polyak_swap_op, ema = polyak(params, beta2)
	train_op = tf.group(polyak_avg_op, *updates)

	return train_op
