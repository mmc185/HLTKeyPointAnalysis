import torch
from torch import nn
from transformers import BertModel
from sentence_transformers import util
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from siamese_network import SiameseNetwork, train, test
import data_handler
from challenge_metrics import load_kpm_data
from transformers import BertModel, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm
import itertools as it
from challenge_metrics import get_predictions, evaluate_predictions
from os import path
import gc

import ray
from ray import air
from ray import tune
from ray.air import session


def compute_metrics(predicted, expected, metrics):
    
    if torch.is_tensor(predicted):
        predicted = predicted.cpu().data.numpy()
        pred = predicted.copy().T
        
    if torch.is_tensor(expected):
        expected = expected.cpu().data.numpy()
        expected = expected.T
    
    metric_results = {}
    
    # Threshold
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
        
    if "accuracy" in metrics:
        metric_results['accuracy'] = accuracy_score(expected, pred)
        
    if "precision" in metrics:
        metric_results['precision'] = precision_score(expected, pred)
    
    if "recall" in metrics:
        metric_results['recall'] = recall_score(expected, pred)
        
    if "f1" in metrics:
        metric_results['f1'] = f1_score(expected, pred)
    
    return metric_results

def extract_challenge_metrics(predictions, labels_df, arg_df, kp_df):
    
    np_predicted = predictions.cpu().data.numpy()
    np_predicted = np_predicted.T
    
    pred_dict = {}
    # Dict example {
    #                "arg_15_0": {
    #                    "kp_15_0": 0.8282181024551392, 
    #                    "kp_15_2": 0.9438725709915161
    #                }, 
    #               "arg_15_1": {
    #                    "kp_15_0": 0.9994438290596008, 
    #                    "kp_15_2":0
    #                }
    #              }
    for idx, pred in enumerate(np_predicted):
            row = labels_df.iloc[idx]
            arg_id = row['arg_id']
            kp_id = row['key_point_id']
            if arg_id not in pred_dict.keys():
                pred_dict[arg_id] = {}
            pred_dict[arg_id][kp_id] = pred
    
    merged_df = get_predictions(pred_dict, labels_df, arg_df, kp_df)
    return evaluate_predictions(merged_df)
    
def grid_search(train_data, val_data, model_type, params, metrics, device):
    
    params['train_data'] = train_data
    params['val_data'] = val_data
    params['model_type'] = model_type
    
    #train_arg_df, train_kp_df, train_labels_df = load_kpm_data("dataset/", subset="train")
    #val_arg_df, val_kp_df, val_labels_df = load_kpm_data("dataset/", subset="dev")
    
    params['train_kpm_data'] = load_kpm_data("dataset/", subset="train")
    params['val_kpm_data'] = load_kpm_data("dataset/", subset="dev")
    
    params['device'] = device
    
    params['metrics'] = metrics
        
    tuner = tune.Tuner(tune.with_resources(trainable,
                                          {"cpu":2, "gpu":1}), 
                       param_space = params, 
                       run_config=air.RunConfig(name=params['optimizer'], verbose=1))
    results = tuner.fit()

    # Get a dataframe for the last reported results of all of the trials 
    df = results.get_dataframe()


    
def trainable(config_dict):
    
    model = SiameseNetwork(bert_type=BertModel.from_pretrained(config_dict['model_type']))
    model.to(config_dict['device'])
    
    #Tokenize data
    columns_list = ['argument', 'key_points', 'label']
    tokenized_tr = data_handler.tokenize_df(config_dict['train_data'][columns_list], config_dict['tokenizer'], max_length=config_dict['max_length'])
    tokenized_val = data_handler.tokenize_df(config_dict['val_data'][columns_list], config_dict['tokenizer'], max_length=config_dict['max_length'])
    
    train_loader = DataLoader(tokenized_tr, shuffle=True, batch_size=config_dict['batch_size'], pin_memory=True)

    optimizer=config_dict['optimizer']

    if(optimizer == 'adamW'):
        optimizer= torch.optim.AdamW(model.parameters(),
              lr = config_dict['lr'], 
              eps = config_dict['eps'],
              weight_decay = config_dict['weight_decay']
    )
    elif (optimizer == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(),
                                   lr = config_dict['lr'],
                                   momentum = config_dict['momentum'],
                                   nesterov = config_dict['nesterov']
    )
    elif (optimizer == 'adam'):
        optimizer= torch.optim.Adam(model.parameters(),
                  lr = config_dict['lr'], 
                  eps = config_dict['eps'],
                  weight_decay = config_dict['weight_decay']
        )    

    # Total number of training steps is [number of batches] x [number of epochs]. 
    # (Note that this is not the same as the number of training samples).
    total_steps = len(train_loader) * config_dict['epochs']

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                        num_warmup_steps = config_dict['warmup_steps'],
                                        num_training_steps = total_steps)

    train_res = train(model, config_dict['device'], train_loader, config_dict['loss'], optimizer, config_dict['epochs'], scheduler, verbose=False)

    config_dict['train'] = train_res

    config_dict['train_metrics'] = [None] * len(train_res['predicted'])
    config_dict['train_challenge_metrics'] = [None] * len(train_res['predicted'])

    for i, elem in enumerate(train_res['predicted']):
        config_dict['train_metrics'][i] = compute_metrics(elem, train_res['labels'], config_dict['metrics'])
        config_dict['train_challenge_metrics'][i] = extract_challenge_metrics(elem, config_dict['train_kpm_data'][2], config_dict['train_kpm_data'][0], config_dict['train_kpm_data'][1])



    val_loader = DataLoader(tokenized_val, pin_memory=True)
    val_res = test(model, config_dict['device'], val_loader, config_dict['loss'])

    config_dict['val'] = val_res

    config_dict['val_metrics'] = compute_metrics(val_res['predicted'].T, val_res['labels'].T, config_dict['metrics'])
    config_dict['val_challenge_metrics'] = extract_challenge_metrics(val_res['predicted'].T, config_dict['val_kpm_data'][2], config_dict['val_kpm_data'][0], config_dict['val_kpm_data'][1])
    
    config_dict.pop('train_data')
    config_dict.pop('val_data')
    config_dict.pop('train_kpm_data')
    config_dict.pop('val_kpm_data')
    config_dict.pop('tokenizer')
    config_dict.pop('device')
    config_dict.pop('metrics')
    
    for key, value in config_dict.items():
        config_dict[key] = [config_dict[key]]
    
    df=pd.DataFrame(config_dict)

    df.to_csv('../../../HLTKeyPointAnalysis/task1_grid_results_with_ray.csv', mode='a', sep='#', index=False, header=False if path.exists("../../../HLTKeyPointAnalysis/task1_grid_results_with_ray.csv") else True)
