import torch
from pytorch_metric_learning import losses
import data_handler
from challenge_metrics import load_kpm_data
from siamese_network import SiameseNetwork, train, test
from task1_utils import grid_search
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from custom_losses import ContrastiveLoss
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import itertools as it

device = torch.device(2)

df_train, df_val, _ = data_handler.load(path="dataset/", filename_train="train.csv", filename_dev="dev.csv", sep_char='#')

# Concatenate topics and keypoints, as stated in the paper
df_train = data_handler.concatenate_topics(df_train)
df_val = data_handler.concatenate_topics(df_val)

# Load our model's (bert-base-uncased) tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

max_length = 60
# Tokenize data
columns_list = ['argument', 'key_points', 'label']
tokenized_tr = data_handler.tokenize_df(df_train[columns_list], tokenizer, max_length=max_length)
tokenized_val = data_handler.tokenize_df(df_val[columns_list], tokenizer, max_length=max_length)

params = {
    'batch_size': [4, 8],
    'loss': [torch.nn.MSELoss()],
    'optimizer': ['adam'],
    'lr': [1e-3, 1e-5, 1e-7],
    'eps': [1e-8, 1e-6],
    'epochs': [4],
    'warmup_steps': [0,1e1,1e2],
    'weight_decay': [1e-1, 1e-5]
}

results = grid_search(tokenized_tr, tokenized_val, 'bert-base-uncased', params, ['accuracy', 'precision', 'recall', 'f1'], device)

results_dict = {
    'params':[],
    'train_metrics':[],
    'train_challenge_metrics': [],
    'val_metrics': [],
    'val_challenge_metrics': []
}

for i, res in enumerate(results):
    params_string = f"batch_size {res['batch_size']}, loss {res['loss']}, optimizer {res['optimizer']}, lr {res['lr']}, eps {res['eps']}, epochs {res['epochs']}, warmup_steps {res['warmup_steps']}, weight_decay {res['weight_decay']}"
    results_dict['params'].append(params_string)
    results_dict['train_metrics'].append(res['train_metrics'])
    results_dict['train_challenge_metrics'].append(res['train_challenge_metrics'])
    results_dict['val_metrics'].append(res['val_metrics'])
    results_dict['val_challenge_metrics'].append(res['val_challenge_metrics'])
    
df=pd.DataFrame(results_dict)

df.to_csv('task1_grid_results.csv', sep='#')