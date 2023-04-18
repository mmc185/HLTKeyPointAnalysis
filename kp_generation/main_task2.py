import torch
import os
from ray import tune
import ray

from transformers import AutoTokenizer
from task2_utils import grid_search

import sys
sys.path.insert(1, '../')
import data_handler

os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device(0)

ray.shutdown()
ray.init(num_gpus=1) 

df_train, df_val, _ = data_handler.load_full_dataset('../dataset/', get_train=True, get_dev=True, get_test=False)

# Concatenate topics and keypoints, as stated in the paper
df_train = data_handler.concatenate_topics(df_train, input_col='argument', output_col='argument')
df_val = data_handler.concatenate_topics(df_val, input_col='argument', output_col='argument')

#model_type = 'google/pegasus-xsum'
model_type = 'google/pegasus-large'

if model_type == 'google/pegasus-large':
    tokenizer = AutoTokenizer.from_pretrained('google/pegasus-xsum')
else:
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    

max_length = 100

params = {
    'tokenizer': tokenizer,
    'max_length': max_length,
    'loss': 'null',
    'batch_size': 8,
    'optimizer': 'adamW',
    'lr': tune.grid_search([2e-4, 5e-4, 4e-4, 3e-4, 1e-4]),
    'eps': tune.grid_search([1e-8]),
    'epochs': tune.grid_search([2, 3]),
    'warmup_steps': tune.grid_search([1e2, 1e3]),
    'weight_decay': tune.grid_search([1e-6, 1e-8]),
    'mode': 'null',
    'match_model_type': 'null'
}

results = grid_search(df_train, df_val, model_type, params, ['rouge'], device)
