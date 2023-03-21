import torch
import os
import pandas as pd
from ray import tune
import ray
from ray import air
from ray.air import session

from transformers import AutoTokenizer, T5Tokenizer

from task2_utils import grid_search

import sys
sys.path.insert(1, '../')
import data_handler

os.environ["CUDA_VISIBLE_DEVICES"]="3"
device = torch.device(0)

ray.shutdown()
ray.init(num_gpus=1) 

df_train, df_val, _ = data_handler.load_full_dataset('../dataset/', get_train=True, get_dev=True, get_test=False)

# Concatenate topics and keypoints, as stated in the paper
df_train = data_handler.concatenate_topics(df_train, input_col='argument', output_col='argument')
df_val = data_handler.concatenate_topics(df_val, input_col='argument', output_col='argument')

model_type = 'google/pegasus-xsum'

#t5-small
tokenizer = AutoTokenizer.from_pretrained(model_type)

max_length = 100


params = {
    'tokenizer': tokenizer,
    'max_length': max_length,
    'loss': 'null',
    'batch_size': 8,
    'optimizer': 'adamW',
    'lr': tune.grid_search([1e-5, 1e-7]),
    'eps': tune.grid_search([1e-8, 1e-3]),
    'epochs': 1,
    'warmup_steps': tune.grid_search([0, 1e2]),
    'weight_decay': 1e-8
}

# lr: 1e-3 con 1e-3 tutti e 1e-8 con 1e2 warmup

results = grid_search(df_train, df_val, model_type, params, ['rouge'], device)
