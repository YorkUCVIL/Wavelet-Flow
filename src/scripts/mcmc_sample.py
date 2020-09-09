# import libraries
import tensorflow as tf
import tensorflow_probability as tfp
from PIL import Image
import numpy as np
import os
import sys
import argparse
import time
import datetime
from tqdm import tqdm as tqdm

# add parent dir to path
script_dir = os.path.dirname(os.path.realpath(sys.argv[0]))
instance_dir = os.path.abspath(os.path.join(script_dir,'..'))
sys.path.append(instance_dir)
os.chdir(instance_dir)

# grab cmd args
argParser = argparse.ArgumentParser(description="unconditional sampling")
argParser.add_argument("-g","--gpu",dest="gpu",action="store",default=0,type=int)
argParser.add_argument("-i","--iteration",dest="iteration",action="store",default='',type=str)
argParser.add_argument("-n","--n_samples",dest="n_samples",action="store",default=10,type=int)
argParser.add_argument("-t","--temperature",dest="temperature",action="store",default=1.0,type=float)
argParser.add_argument("-l","--level",dest="level",action="store",default=-1,type=int) # currently ignored
argParser.add_argument("-w","--warmup",dest="warmup",action="store",default=30,type=int)
argParser.add_argument("-a","--adaptation",dest="adaptation",action="store",default=10,type=int)
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
seed = config.random_seed.tensorflow
init_random_seeds()

# set level default if needed
if cmd_args.level == -1:
	cmd_args.level = config.model.n_levels

# check for previous runs, logs, checkpoints
checkpoint_path = config.checkpoints.path
n_checkpoints = cmd_args.level + 1
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

# sample latents with mcmc, depends on conditioning
n_batch_samples = 1
temperature = cmd_args.temperature
adaptation_steps = cmd_args.adaptation
warmup_steps = cmd_args.warmup
base_inputs = []
mcmc_samples = []
mcmc_traces = []
for level in range(net.n_levels+1):
	if level >= net.base_level:
		level_shape = net.sub_flows[level].shape
		if level == net.base_level:
			base_inputs.append(None)
		else:
			base_inputs.append(tf.placeholder(shape=[n_batch_samples]+level_shape[:-1]+[3],dtype=tf.float32))

		# ns, nt = nuts_sample(net,level,base_inputs[-1])
		ns, nt = net.sub_flows[level].sample_latent_mcmc(
			conditioning=base_inputs[-1],n_batch=n_batch_samples,temperature=temperature,
			step_size=0.01,adaptation_steps=adaptation_steps,warmup_steps=warmup_steps)
		mcmc_samples.append(ns)
		mcmc_traces.append(nt)
	else:
		base_inputs.append(None)
		mcmc_samples.append(None)
		mcmc_traces.append(None)

# reconstruct using base and latent
latent_in = []
reconstructions = []
visualizations = []
for level in range(net.n_levels+1):
	if level == net.base_level:
		latent_in.append(tf.placeholder(shape=[n_batch_samples]+net.base_flow.shape,dtype=tf.float32))

		recon = net.base_flow.latent_to_data([latent_in[-1]])
		reconstructions.append(recon.data)
		visualizations.append(visualization.clip_uint8((recon.data+0.5)*255))
	elif level > net.base_level:
		latent_in.append(tf.placeholder(shape=[n_batch_samples]+net.sub_flows[level].shape,dtype=tf.float32))
		recon = net.latent_to_super_res([latent_in[-1]],level,base_inputs[level])
		reconstructions.append(recon.reconstruction)
		visualizations.append(visualization.clip_uint8((recon.reconstruction+0.5)*255))
	else:
		latent_in.append(None)
		reconstructions.append(None)
		visualizations.append(None)

# ----------------- savers -----------------
savers = []
for it in range(cmd_args.level+1):
	params = net.get_variables('partial_{}'.format(it))
	savers.append(tf.train.Saver(var_list=params, max_to_keep=0))

# ----------------- sampling -----------------
with session_setup(cmd_args) as sess:
	# init
	tlog('Loading iteration {}'.format(latest_iter),'note')
	# sess.run(tf.global_variables_initializer())
	for idx,saver in enumerate(savers):
		saver.restore(sess,latest_checkpoint_paths[idx])

	# sampling
	tlog('Starting sampling','note')

	bases = []
	for level in range(net.n_levels+1):
		tlog(level)
		bases.append([])
		if level < net.base_level:
			continue

		im_grid = []
		count = 0
		with tqdm(total=cmd_args.n_samples**2) as pbar:
			for it in range(cmd_args.n_samples):
				row = []
				for it2 in range(cmd_args.n_samples):
					if level == net.base_level:
						base_feed = {}
					else:
						base_feed = {base_inputs[level]:bases[-2][count]}

					# sample latent
					trace_inner = mcmc_traces[level].inner_results
					r = sess.run([mcmc_samples[level],trace_inner.is_accepted,trace_inner.step_size,trace_inner.leapfrogs_taken],feed_dict=base_feed)
					acceptance = r[1]
					# tlog(r[3])

					# filter latent by acceptance
					accepted_proposal_idx = None
					for proposal_idx in reversed(range(acceptance.shape[0])):
						if acceptance[proposal_idx,0]:
							accepted_proposal_idx = proposal_idx
							break
					latent = r[0][accepted_proposal_idx,:,:]
					cur_flow_shape = net.sub_flows[level].shape
					latent = np.reshape(latent,[n_batch_samples]+cur_flow_shape)

					# reconstruct
					feed_dict = {latent_in[level]:latent}
					feed_dict.update(base_feed)
					r = sess.run([reconstructions[level],visualizations[level]],feed_dict=feed_dict)
					base = r[0]
					bases[-1].append(base)
					viz = r[1]

					row.append(viz[0,:,:,:])
					count += 1
					pbar.update(1)

				im_grid.append(row)

		out_width = 2**level
		grid_image_path = '{}mcmc-samples-{}_iter-{:08d}_temp-{}.png'.format(config.offline_sampling.sample_path,out_width,latest_iter,cmd_args.temperature)
		save_grid_image(im_grid,grid_image_path,scale=1)

	tlog('Sampling complete','note')
