import tensorflow as tf
from tf_components.bijector.Haar_squeeze_split import *
from tf_components.visualization.clip_uint8 import *

def haar_visualization(details):
	squeeze_split = Haar_squeeze_split()

	shape = tf.shape(details)
	n = shape[0]
	h = shape[1]
	w = shape[2]
	base = tf.zeros(shape=[n,h,w,3])
	viz = squeeze_split.inverse(base,details)
	return viz
