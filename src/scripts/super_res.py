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
argParser.add_argument("-i","--iteration",dest="iteration",action="store",default='',type=str)
argParser.add_argument("-n","--n_samples",dest="n_samples",action="store",default=10,type=int)
argParser.add_argument("-t","--temperature",dest="temperature",action="store",default=1.0,type=float)
argParser.add_argument("-l","--level",dest="level",action="store",default=-1,type=int)
argParser.add_argument("-b","--base_level",dest="base_level",action="store",default=0,type=int)
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
init_random_seeds()

# set level default if needed
if cmd_args.level == -1:
	cmd_args.level = config.model.n_levels

# check for previous runs, logs, checkpoints
assert cmd_args.base_level < cmd_args.level, 'base level must be < level'
checkpoint_path = config.checkpoints.path
n_checkpoints = cmd_args.level + 1
latest_iters = []
latest_checkpoint_paths = []

for idx in range(cmd_args.base_level+1,n_checkpoints):
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
n_samples = cmd_args.n_samples**2
validation_data = Validation_data(batch_override=n_samples,shuffle_repeat=True)
pyramid = bijector.Haar_pyramid(config.model.n_levels)
net = Network_body()

haar = pyramid.forward(validation_data.im)
all_scales = haar.downsampled + [validation_data.im]
base = all_scales[cmd_args.base_level]
base_h = base.get_shape()[1].value
base_w = base.get_shape()[2].value
base_c = base.get_shape()[3].value
base.set_shape([n_samples,base_h,base_w,base_c])
reconstructions = routines.sample_super_res(net,base,temperature=cmd_args.temperature)

reconstruction = reconstructions[cmd_args.level]
out_inspect = visualization.clip_uint8((reconstruction+0.5)*255)

savers = []
for it in range(cmd_args.base_level+1,cmd_args.level+1):
	params = net.get_variables('partial_{}'.format(it))
	savers.append(tf.train.Saver(var_list=params, max_to_keep=0))

# ----------------- training -----------------
with session_setup(cmd_args) as sess:
	# init
	tlog('Loading iteration {}'.format(latest_iter),'note')
	# sess.run(tf.global_variables_initializer())
	for path,saver in zip(latest_checkpoint_paths,savers):
		saver.restore(sess,path)

	# sampling
	tlog('Starting sampling','note')

	im_grid = []
	count = 0
	r = sess.run([out_inspect])
	for it in range(cmd_args.n_samples):
		row = []
		for it2 in range(cmd_args.n_samples):
			arr = r[0][count,:,:,:]
			row.append(arr)
			count += 1
		im_grid.append(row)

	out_width = 2**cmd_args.level
	base_width = 2**cmd_args.base_level
	grid_image_path = '{}super-{}-{}_iter-{:08d}_temp-{}.png'.format(config.offline_sampling.sample_path,base_width,out_width,latest_iter,cmd_args.temperature)
	save_grid_image(im_grid,grid_image_path,scale=1)

	tlog('Sampling complete','note')
