import hjson
import json
import os
import tensorflow as tf
import numpy as np
import util

def load_config(instance_dir,relative_path):
	path = os.path.join(instance_dir,relative_path)
	with open(path) as f:
		config = hjson.load(f)

		# convert to regular json
		config = hjson.dumpsJSON(config)
		config = json.loads(config)

		# add instance path into config
		config['instance_dir'] = instance_dir

		# return to_attributes(config)
		util.config.update(util.to_attributes(config))
