# import libraries
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import sys
import argparse
import time
import datetime
from tqdm import tqdm
import math

# add parent dir to path
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
instance_dir = os.path.abspath(os.path.join(script_dir,'..'))
sys.path.append(instance_dir)
os.chdir(instance_dir)

# grab cmd args
argParser = argparse.ArgumentParser(description="validate BPD")
argParser.add_argument("-g","--gpu",dest="gpu",action="store",default=0,type=int)
argParser.add_argument("-i","--iteration",dest="iteration",action="store",default='',type=str)
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

# check for previous runs, logs, checkpoints
checkpoint_path = config.checkpoints.path
n_checkpoints = config.model.n_levels + 1
latest_iters = []
latest_checkpoint_paths = []

for idx in range(0,n_checkpoints):
	partial_path = '{}partial_{}'.format(checkpoint_path,idx)
	checkpoints_exist, checkpoint_dir_contents = get_checkpoints(partial_path)
	if not checkpoints_exist:
		tlog('Missing checkpoints: {}, exiting'.format(partial_path),'error')
		exit()
	latest_iter = get_latest_checkpoint_iter(partial_path,checkpoint_dir_contents)
	latest_checkpoint_path = get_checkpoint_path(partial_path,latest_iter)

	latest_iters.append(latest_iter)
	latest_checkpoint_paths.append(latest_checkpoint_path)

# set latest iter as max
latest_iter = latest_iters[0]
for iter in latest_iters:
	if iter < latest_iter:
		latest_iter = iter

# ----------------- build components -----------------
net = Network_body()
validation_data = Validation_data(shuffle_repeat=False)

val_bpd = routines.validate(validation_data,net)

savers = []
for it in range(net.n_levels+1):
	params = net.get_variables('partial_{}'.format(it))
	savers.append(tf.train.Saver(var_list=params, max_to_keep=0))

# ----------------- validation -----------------
with session_setup(cmd_args) as sess:
	# init
	tlog('Loading iteration {}'.format(latest_iter),'note')
	for idx,saver in enumerate(savers):
		saver.restore(sess,latest_checkpoint_paths[idx])

	# compute required loop
	n_batch = config.validation.n_batch[0]
	n_data = validation_data.n_data
	its_total = math.ceil(n_data/n_batch)

	# validation
	tlog('Starting validation','note')
	bits_per_dimension = []
	for it in tqdm(range(its_total)):
		r = sess.run([val_bpd])
		bits_per_dimension += r[0].tolist()

	# convert back to numpy array to take mean
	bits_per_dimension = np.asarray(bits_per_dimension)
	print('Average bpd: {}'.format(np.mean(bits_per_dimension)))

	tlog('Validation complete','note')
