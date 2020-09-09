import tensorflow as tf

def edge_bias(x, k_size):
	'''
	injects a mask into a tensor for 2d convs to indicate padding locations
	'''
	with tf.variable_scope(None,default_name='edge_bias'):
		pad_size = k_size//2

		# manually pad data for conv
		x_padded = tf.pad(x,[
			[0, 0],
			[pad_size, pad_size],
			[pad_size, pad_size],
			[0, 0]
		])

		# generate mask to indicated padded pixels
		# x_mask_inner = tf.zeros(shape=tf.shape(x))
		x_mask_inner = tf.zeros_like(x)
		x_mask_inner = x_mask_inner[:,:,:,0:1]
		x_mask = tf.pad(x_mask_inner,[
			[0, 0],
			[pad_size, pad_size],
			[pad_size, pad_size],
			[0, 0]
		], constant_values=1.0)

		# combine into 1 tensor
		x_augmented = tf.concat([x_padded, x_mask], axis=-1)

		return x_augmented
