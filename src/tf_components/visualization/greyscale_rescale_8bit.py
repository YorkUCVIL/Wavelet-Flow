import tensorflow as tf

def greyscale_rescale_8bit(im,min_max=None, use_std=True):
	"""
	if min_max is none, auto range per image
	"""
	if min_max is None:
		if use_std:
			mean,var = tf.nn.moments(im,axes=[1,2])
			std = tf.sqrt(var)
			min_max = [ mean - std*3, mean + std*3 ]
		else:
			min_max = [
				tf.reduce_min(im,axis=[1,2],keepdims=True),
				tf.reduce_max(im,axis=[1,2],keepdims=True)
			]

	diff = min_max[1] - min_max[0]
	im_scaled = (im-min_max[0])/diff
	im_scaled = tf.clip_by_value(im_scaled,0,1)
	im_8bit = tf.cast(im_scaled*255,tf.uint8)


	return im_8bit
