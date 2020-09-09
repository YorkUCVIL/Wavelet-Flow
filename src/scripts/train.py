# import libraries
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import sys
import argparse
import time
import datetime

# add parent dir to path
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
instance_dir = os.path.abspath(os.path.join(script_dir,'..'))
sys.path.append(instance_dir)
os.chdir(instance_dir)

# grab cmd args
argParser = argparse.ArgumentParser(description="start/resume training")
argParser.add_argument("-g","--gpu",dest="gpu",action="store",default=0,type=int)
argParser.add_argument("-i","--iterations",dest="iterations",action="store",default=0,type=int) # setup!
argParser.add_argument("-p","--partial",dest="partial",action="store",default=0,type=int)
argParser.add_argument("-t","--test",dest="test",action='store_true')
argParser.add_argument("-d","--dataset",dest="dataset",action='store',type=str,default='NOT_SPECIFIED')
cmd_args = argParser.parse_args()

# load our modules
from util import *
if cmd_args.dataset == 'imagenet_64':
	from models.imagenet_64_haar import *
elif cmd_args.dataset == 'imagenet_32':
	from models.imagenet_32_haar import *
elif cmd_args.dataset == 'celeba_1024':
	from models.celeba_1024_haar import *
elif cmd_args.dataset == 'ffhq_1024':
	from models.ffhq_1024_haar import *
elif cmd_args.dataset == 'lsun_bedroom_64':
	from models.lsun_bedroom_64_haar import *
elif cmd_args.dataset == 'lsun_tower_64':
	from models.lsun_tower_64_haar import *
elif cmd_args.dataset == 'lsun_church_64':
	from models.lsun_church_64_haar import *
else:
	print('invalid dataset: {}'.format(cmd_args.dataset))
	exit()
from tf_components import *

# set cuda visible if not already defined
if "CUDA_VISIBLE_DEVICES" not in os.environ.keys():
	os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
	os.environ["CUDA_VISIBLE_DEVICES"] = str(cmd_args.gpu)
	tlog('CUDA_VISIBLE_DEVICES set to: {}'.format(cmd_args.gpu), 'note')
else:
	tlog('CUDA_VISIBLE_DEVICES already set, ignoring manual gpu selection', 'note')

# load config
load_config(instance_dir,model_config_path)
config.random_seed.tensorflow += cmd_args.partial
config.random_seed.numpy += cmd_args.partial
init_random_seeds()

# override for testing mode, trains for a few its and makes checkpoints
if cmd_args.test:
	config.checkpoints.frequency = 10
	config.validation.frequency = 10
	config.training.iterations = 20
	config.training.n_batch = [2]*(config.model.n_levels+1)

# modify config for partial subdirs
config.checkpoints.path += 'partial_{}/'.format(cmd_args.partial)
config.logging.path += 'partial_{}/'.format(cmd_args.partial)

# check for previous runs, logs, checkpoints
resume_state = check_resume(config)

# ----------------- build components -----------------
graph = build_training_graph_partial(config,cmd_args.partial)

train_bpd = graph.train_bpd
train_solver = graph.train_solver
current_iteration_placeholder = graph.global_placeholders.current_iteration
ddi_op = graph.ddi_op

frequent_summaries = tf.summary.merge_all('frequent_summaries')
infrequent_summaries = tf.summary.merge_all('infrequent_summaries')
saver = tf.train.Saver(max_to_keep=0)

# ----------------- training -----------------
with session_setup(cmd_args) as sess:
	# init
	if resume_state.resume:
		tlog('Loading iteration {}'.format(resume_state.resume_iteration),'note')
		resume_checkpoint_path = resume_state.resume_iteration_path
		saver.restore(sess,resume_checkpoint_path)
	else:
		tlog('Training from scratch','note')
		sess.run(tf.global_variables_initializer())

	# start summary writer
	summary_writer = tf.summary.FileWriter(config.logging.path, None, flush_secs=10)

	# ddi actnorms
	if not resume_state.resume:
		tlog('Starting ActNorm ddi','note')
		sess.run(ddi_op)

	# train
	tlog('Starting training','note')
	start_iteration = resume_state.resume_iteration + 1 if resume_state.resume else 1
	for it in range(start_iteration,config.training.iterations+1):
		feed_dict = {
			current_iteration_placeholder: it # do this
		}
		iteration_start = time.time()
		r = sess.run([train_solver,train_bpd,frequent_summaries],feed_dict=feed_dict)
		iteration_duration = time.time() - iteration_start

		# compute ETA
		its_per_sec = 1/iteration_duration
		remaining_its = config.training.iterations - it - 1
		eta_sec = remaining_its * iteration_duration
		eta_min = eta_sec//60
		eta = str(datetime.timedelta(minutes=eta_min))

		# print status
		if it % config.logging.frequency == 0:
			iteration_print(it,[
				('bpd',r[1]),
				('its/s',its_per_sec),
				('ETA',eta),
			])

			# record frequent summary
			summary_writer.add_summary(r[-1],it)

		# record infrequent summary
		if it % config.validation.frequency == 0:
			tlog('Performing validation','note')
			r = sess.run([infrequent_summaries])
			summary_writer.add_summary(r[0],it)

		# checkpointting
		if it % config.checkpoints.frequency == 0:
			checkpoint_path = get_checkpoint_path(config.checkpoints.path,it)
			tlog('Checkpointing','note')
			saver.save(sess,checkpoint_path,write_meta_graph=False)
			tlog('Checkpoint saved: {}'.format(checkpoint_path),'note')

	tlog('Training complete','note')
