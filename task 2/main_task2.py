import torch
import os
import pandas as pd
from ray import tune
import ray
from ray import air
from ray.air import session

from transformers import AutoTokenizer

from task2_utils import grid_search

import sys
sys.path.insert(1, '../')
import data_handler

ray.shutdown()
ray.init(num_gpus=1) 

os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device(0)

df_train, df_val, _ = data_handler.load_full_dataset('../dataset/', get_train=True, get_dev=True, get_test=False)

# Concatenate topics and keypoints, as stated in the paper
df_train = data_handler.concatenate_topics(df_train, input_col='argument', output_col='argument')
df_val = data_handler.concatenate_topics(df_val, input_col='argument', output_col='argument')

model_type = 'Robert-Spo'

tokenizer = AutoTokenizer.from_pretrained('google/pegasus-xsum')

max_length = 60

params = {
    'tokenizer': tokenizer,
    'max_length': max_length,
    'batch_size': 8,
    'loss': 'null',
    'optimizer': 'adamW',
    'lr': 1e-3,
    'eps': 1e-8,
    'epochs': 2,
    'warmup_steps': 0,
    'weight_decay': 0,
    'momentum': 'null',
    'nesterov': False
}


results = grid_search(df_train[:100], df_val[:100], model_type, params, ['rouge'], device)