import torch
from pytorch_metric_learning import losses
from challenge_metrics import load_kpm_data
from siamese_network import SiameseNetwork, train, test
from matching_utils import grid_search
from transformers import BertModel, BertTokenizer, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import itertools as it
import os

from ray import tune
import ray
from ray import air
from ray.air import session

import sys
sys.path.insert(1, "../")
import data_handler

os.environ["CUDA_VISIBLE_DEVICES"]="3"
device = torch.device(0)

ray.shutdown()
ray.init(num_gpus=1) 

df_train, df_val, _ = data_handler.load(path="../dataset/", filename_train="train.csv", filename_dev="dev.csv", sep_char='#')

# Concatenate topics and keypoints, as stated in the paper
df_train = data_handler.concatenate_topics(df_train)
df_val = data_handler.concatenate_topics(df_val)

model_type = 'bert-base-uncased'

# Load our model's (bert-base-uncased) tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_type, do_lower_case=True)

max_length = 60

params = {
    'tokenizer': tokenizer,
    'max_length': max_length,
    'batch_size': 8,
    'loss': torch.nn.MSELoss(),
    'optimizer': 'sgd',
    'lr': 1e-3,
    'eps': 'null',
    'epochs': 2,
    'warmup_steps': tune.grid_search([0]),
    'weight_decay': tune.grid_search([0]),
    'momentum': tune.grid_search([0]),
    'nesterov': False
}


results = grid_search(df_train[:8], df_val[:8], model_type, params, ['accuracy', 'precision', 'recall', 'f1'], device)


