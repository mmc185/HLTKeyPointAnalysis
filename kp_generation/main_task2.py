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
    'batch_size': 8,
    'loss': 'null',
    'optimizer': 'adamW',
    'lr': 1e-3,
    'eps': 1e-8,
    'epochs': 1,
    'warmup_steps': 0,
    'weight_decay': 0,
    'momentum': 'null',
    'nesterov': False
}


results = grid_search(df_train[:5], df_val[:5], model_type, params, ['rouge'], device)