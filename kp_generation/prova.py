import sys
#sys.path.insert(1, '../')
sys.path.insert(1, './kp_match/')
from siamese_network import SiameseNetwork
from transformers import AutoModel
import torch
import os

#print(os.listdir())
model_type='roberta-large'
model = SiameseNetwork(model_type=AutoModel.from_pretrained(model_type))
model.load_state_dict(torch.load("./models/task1/model_82"))
print('caricato')