import torch
import os
from ray import tune
import ray

from transformers import AutoTokenizer
from gen_utils import grid_search, concat_tag


import sys
sys.path.insert(1, '../')
import data_handler

os.environ["CUDA_VISIBLE_DEVICES"]="2"
device = torch.device(0)

ray.shutdown()
ray.init(num_gpus=1) 

df_train, df_val, _ = data_handler.load_full_dataset('../dataset/', get_train=True, get_dev=True, get_test=False)
print(len(df_train))

# Concatenate topics and keypoints, as stated in the paper
df_train = data_handler.concatenate_topics(df_train, input_col='argument', output_col='argument')
df_val = data_handler.concatenate_topics(df_val, input_col='argument', output_col='argument')

#model_type = 'google/pegasus-xsum'
model_type = 't5-large'

if model_type == 'google/pegasus-large':
    tokenizer = AutoTokenizer.from_pretrained('google/pegasus-xsum')
else:
    tokenizer = AutoTokenizer.from_pretrained(model_type)
    
if model_type == "t5-small" or model_type == "t5-base" or model_type == "t5-large":
    #Only for T5
    df_tr = df_train
    df_v = df_val
    df_train = concat_tag(df_tr, 'argument')
    df_val = concat_tag(df_v, 'argument')
    

max_length = 100

params = {
    'tokenizer': tokenizer,
    'max_length': max_length,
    'loss': 'null',
    'batch_size': 4,
    'optimizer': 'adamW',
    'lr': tune.grid_search([2e-5]),
    'eps': tune.grid_search([1e-8]),
    'epochs': tune.grid_search([2]),
    'warmup_steps': tune.grid_search([1e2, 3e2, 1e3]),
    'weight_decay': tune.grid_search([0, 1e-8]),
    'mode': 'null',
    'match_model_type': 'null'
}

results = grid_search(df_train, df_val, model_type, params, ['rouge'], device)
