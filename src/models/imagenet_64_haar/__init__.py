from models.imagenet_64_haar.Training_data import *
from models.imagenet_64_haar.Validation_data import *
from models.imagenet_64_haar.Network_body import *
from models.imagenet_64_haar.Conditioning_network import *
import models.shared.routines as routines
from models.imagenet_64_haar.build_training_graph import *

model_config_path = 'data/imagenet_64_haar/config.hjson'
