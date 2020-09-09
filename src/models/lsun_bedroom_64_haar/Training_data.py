import tensorflow as tf
import os
from util import *

class Training_data:
	def __init__(self,partial_level=0):
		with tf.variable_scope(None,default_name='training_data'):
			self.crop_factor = config.training.partial_training_crops[partial_level]
			datasetRoot = config.training.data.root_path
			data_list_path = os.path.join(datasetRoot,config.training.data.path)
			n_batch = config.training.n_batch[partial_level]
			n_ddi_batch = config.training.n_ddi_batch[partial_level]

			# read in datalist and create dataset
			with open(data_list_path) as f:
				data_path_list = [datasetRoot + x[:-1] for x in f.readlines()]
				n_data = len(data_path_list)
			dataset = tf.data.Dataset.from_tensor_slices(data_path_list)
			dataset = dataset.shuffle(n_data).repeat()
			dataset = dataset.map(self.data_map,num_parallel_calls=8)

			# training
			training_dataset = dataset.batch(n_batch).prefetch(64)
			training_iterator = training_dataset.make_one_shot_iterator()
			training_batch = training_iterator.get_next()

			# ddi
			ddi_dataset = dataset.batch(n_ddi_batch)
			ddi_batch = ddi_dataset.make_one_shot_iterator().get_next()

			# post processing
			im = self.post_process(training_batch)
			ddi_im = self.post_process(ddi_batch)

			self.im = im
			self.ddi_im = ddi_im

	def data_map(self, img_path):
		n_bits = config.model.data.n_bits
		n_bins = 2**n_bits
		rgb = tf.image.decode_png(tf.read_file(img_path), channels=3, dtype=tf.uint8)

		h = config.model.data.dimensions.h
		w = config.model.data.dimensions.w
		c = config.model.data.dimensions.c
		# rgb.set_shape([h,w,c]) # don't set because going to crop anyway

		# crop for lsun 96, see realnvp and glow for specifics
		rgb = tf.image.random_crop(rgb,size=[h,w,c])

		# crop for patch training
		crop_h = h//self.crop_factor
		crop_w = w//self.crop_factor
		rgb = tf.image.random_crop(rgb,size=[crop_h,crop_w,c])

		# random left-right flops
		rgb = tf.image.random_flip_left_right(rgb)

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
