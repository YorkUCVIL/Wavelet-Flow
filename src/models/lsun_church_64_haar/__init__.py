from models.lsun_bedroom_64_haar.Training_data import *
from models.lsun_bedroom_64_haar.Validation_data import *
from models.lsun_bedroom_64_haar.Network_body import *
from models.lsun_bedroom_64_haar.Conditioning_network import *
import models.shared.routines as routines
from models.lsun_bedroom_64_haar.build_training_graph import *

model_config_path = 'data/lsun_church_64_haar/config.hjson'
