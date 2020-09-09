import tensorflow as tf
import os
from util import *

class Validation_data:
	def __init__(self,batch_override=None,shuffle_repeat=True,partial_level=0):
		with tf.variable_scope(None,default_name='validation_data'):
			self.crop_factor = config.validation.partial_training_crops[partial_level]
			datasetRoot = config.validation.data.root_path
			data_list_path = os.path.join(datasetRoot,config.validation.data.path)
			n_batch = batch_override or config.validation.n_batch[partial_level]

			# read in datalist and create dataset
			with open(data_list_path) as f:
				data_path_list = [datasetRoot + x[:-1] for x in f.readlines()]
				n_data = len(data_path_list)
			dataset = tf.data.Dataset.from_tensor_slices(data_path_list)
			if shuffle_repeat:
				dataset = dataset.shuffle(n_data).repeat()
			dataset = dataset.map(self.data_map)

			# validation
			validation_dataset = dataset.batch(n_batch).prefetch(4)
			validation_iterator = validation_dataset.make_one_shot_iterator()
			validation_batch = validation_iterator.get_next()

			# post processing
			im = self.post_process(validation_batch)

			self.im = im
			self.n_data = n_data

	def data_map(self, img_path):
		n_bits = config.model.data.n_bits
		n_bins = 2**n_bits
		rgb = tf.image.decode_png(tf.read_file(img_path), channels=3, dtype=tf.uint8)

		h = config.model.data.dimensions.h
		w = config.model.data.dimensions.w
		c = config.model.data.dimensions.c
		rgb.set_shape([h,w,c])

		# random crops
		crop_h = h//self.crop_factor
		crop_w = w//self.crop_factor
		rgb = tf.image.random_crop(rgb,size=[crop_h,crop_w,c])

		# cast, bit conversion, compress domain, center
		rgb = tf.cast(rgb, tf.float32)
		if n_bits < 8:
			rgb = tf.floor(rgb/(2**(8-n_bits)))
		rgb = rgb/(n_bins) - 0.5

		return rgb

	def post_process(self, rgb, add_dequantization_noise=True):
		n_bits = config.model.data.n_bits
		n_bins = 2**n_bits

		rgb_out = rgb

		# discretization noise
		if add_dequantization_noise:
			shape = tf.shape(rgb_out)
			rgb_out += tf.random_uniform(shape=shape)*(1/n_bins)

		return rgb_out
