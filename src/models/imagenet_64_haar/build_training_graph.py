import tensorflow as tf
from tf_components import *
from models.imagenet_64_haar import *
# from util import *

def build_training_graph_partial(config,partial_level):
	tlog('Started building graph','note')
	assert partial_level >= 0 and partial_level <= config.model.n_levels, 'level must be >= 0 and <= n_levels'

	# data input
	training_data = Training_data(partial_level=partial_level)
	validation_data = Validation_data(partial_level=partial_level)

	# create global placeholders
	global_placeholders = to_attributes
	global_placeholders.current_iteration = tf.placeholder(shape=[],dtype=tf.float32)

	# build network
	net = Network_body(partial_level)

	# =========================== training ===========================
	train_solver, train_bpd = routines.train_partial(training_data,net,global_placeholders, partial_level)

	# =========================== training ddi init ===========================
	ddi_op = routines.ddi_partial(training_data, net, partial_level)

	# =========================== validation ===========================
	val_bpd = routines.validate_partial(validation_data, net, partial_level)

	# return important tensors
	ret = to_attributes({})
	ret.train_solver = train_solver
	ret.train_bpd = train_bpd
	ret.global_placeholders = global_placeholders
	ret.ddi_op = ddi_op
	ret.params = net.get_variables('partial_{}'.format(partial_level))

	tlog('Done building graph','note')
	return ret
