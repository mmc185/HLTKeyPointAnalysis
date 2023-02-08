import torch
from pytorch_metric_learning import losses
import data_handler
from challenge_metrics import load_kpm_data
from siamese_network import SiameseNetwork, train, test
from task1_utils import grid_search
from transformers import BertModel, BertTokenizer, AutoTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from custom_losses import ContrastiveLoss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import itertools as it
import os

os.environ["CUDA_VISIBLE_DEVICES"]="1"
device = torch.device(0)

df_train, df_val, _ = data_handler.load(path="dataset/", filename_train="train.csv", filename_dev="dev.csv", sep_char='#')

# Concatenate topics and keypoints, as stated in the paper
df_train = data_handler.concatenate_topics(df_train)
df_val = data_handler.concatenate_topics(df_val)

model_type = 'bert-base-uncased'

# Load our model's (bert-base-uncased) tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_type, do_lower_case=True)

max_length = 60
# Tokenize data
columns_list = ['argument', 'key_points', 'label']
tokenized_tr = data_handler.tokenize_df(df_train[columns_list], tokenizer, max_length=max_length)
tokenized_val = data_handler.tokenize_df(df_val[columns_list], tokenizer, max_length=max_length)

"""
params = {
    'batch_size': [8],
    'loss': [torch.nn.MSELoss()],
    'optimizer': ['sgd'],
    'lr': [1e-3, 1e-5, 1e-7],
    'eps': ['null'],
    'epochs': [3],
    'warmup_steps': [0, 1e1, 1e2],
    'weight_decay': [0, 1e-1, 1e-5],
    'momentum': [0, 2e-1, 6e-1],
    'nesterov': [True, False]
}
"""
params = {
    'batch_size': [8],
    'loss': [torch.nn.MSELoss()],
    'optimizer': ['sgd'],
    'lr': [1e-5],
    'eps': ['null'],
    'epochs': [3],
    'warmup_steps': [1e1],
    'weight_decay': [1e-5],
    'momentum': [6e-1],
    'nesterov': [True]
}


results = grid_search(tokenized_tr, tokenized_val, model_type, params, ['accuracy', 'precision', 'recall', 'f1'], device)


