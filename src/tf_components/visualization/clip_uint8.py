import tensorflow as tf

def clip_uint8(im_in):
	'''
	clips a float value between 0 and 255 and casts to uint8
	'''
	with tf.variable_scope(None,default_name='clip_uint8'):
		im = tf.clip_by_value(im_in, 0, 255)
		return tf.cast(im, tf.uint8)
