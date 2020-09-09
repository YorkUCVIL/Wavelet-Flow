from models.ffhq_1024_haar.Training_data import *
from models.ffhq_1024_haar.Validation_data import *
from models.ffhq_1024_haar.Network_body import *
from models.ffhq_1024_haar.Conditioning_network import *
import models.shared.routines as routines
from models.ffhq_1024_haar.build_training_graph import *

model_config_path = 'data/ffhq_1024_haar/config.hjson'
