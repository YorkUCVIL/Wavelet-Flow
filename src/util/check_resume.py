import os
from util.to_attributes import *

def clear_logs(log_path,log_dir_contents):
	for log in log_dir_contents:
		full_path = os.path.join(log_path,log)
		os.remove(full_path)

def clear_checkpoints(checkpoint_path,checkpoint_dir_contents):
	for checkpoint in checkpoint_dir_contents:
		full_path = os.path.join(checkpoint_path,checkpoint)
		os.remove(full_path)

def get_latest_checkpoint_iter(checkpoint_path,checkpoint_dir_contents):
	checkpoint_indices = [x for x in checkpoint_dir_contents if x.endswith('.ckpt.index')]
	checkpoint_indices.sort()
	latest = checkpoint_indices[-1].strip('iter_').strip('.ckpt.index')
	latest = int(latest)
	return latest

def get_logs(path):
	log_exists = False
	try:
		log_dir_contents = [x for x in os.listdir(path) if x.startswith('events.out.tfevents')]
	except:
		log_dir_contents = []
	log_exists = len(log_dir_contents) > 0
	return log_exists, log_dir_contents

def get_checkpoints(path):
	try:
		checkpoint_dir_contents = [x for x in os.listdir(path) if x.startswith('iter_') or x == 'checkpoint' ]
	except:
		checkpoint_dir_contents = []
	checkpoints_exist = len(checkpoint_dir_contents) > 0
	return checkpoints_exist, checkpoint_dir_contents

def get_checkpoint_path(path, iter):
	checkpoint_name = 'iter_{}.ckpt'.format(str(iter).zfill(16))
	checkpoint_path = os.path.join(path,checkpoint_name)
	return checkpoint_path

def check_resume(config):
	'''
	detects/manages files created during previous runs
	assumes log file is always created if checkpoints exist
	future: add option to always train fresh
	'''
	log_path = config.logging.path
	checkpoint_path = config.checkpoints.path
	resume_state = to_attributes({})
	resume_state.resume = False
	resume_state.resume_iteration = 0 # default 0 to start at beginning
	resume_state.resume_iteration_path = ''

	# check for logs
	log_exists, log_dir_contents = get_logs(log_path)

	# check for checkpoints
	checkpoints_exist, checkpoint_dir_contents = get_checkpoints(checkpoint_path)

	def ask_user_bool(prompt):
		user_choice = ''
		while user_choice != 'y' and user_choice != 'n':
			user_choice = input(prompt)
		return user_choice == 'y'

	if log_exists and checkpoints_exist:
			respond_y = ask_user_bool('Logs and checkpoints detected! Resume training (y) or clear old files and train from 0 (n): ')
			if respond_y:
				resume_state.resume = True
				resume_state.resume_iteration = get_latest_checkpoint_iter(checkpoint_path,checkpoint_dir_contents)
				resume_state.resume_iteration_path = get_checkpoint_path(checkpoint_path,resume_state.resume_iteration)
			else:
				clear_logs(log_path,log_dir_contents)
				clear_checkpoints(checkpoint_path,checkpoint_dir_contents)
	elif log_exists and not checkpoints_exist:
		respond_y = ask_user_bool('Only logs detected! Keep log (y) or clear old files and train from 0? (y/n): ')
		if respond_y:
			print('Nothing done, exiting')
			exit()
		else:
			# remove log
			clear_logs(log_path,log_dir_contents)
	elif not log_exists and checkpoints_exist:
		respond_y = ask_user_bool('Only checkpoints detected! Resume training (y) or clear old files and train from 0 (n): ')
		if respond_y:
			resume_state.resume = True
			resume_state.resume_iteration = get_latest_checkpoint_iter(checkpoint_path,checkpoint_dir_contents)
			resume_state.resume_iteration_path = get_checkpoint_path(checkpoint_path,resume_state.resume_iteration)
		else:
			clear_checkpoints(checkpoint_path,checkpoint_dir_contents)
	else:
		print('No logs or checkpoints detected, training from scratch')

	return resume_state
