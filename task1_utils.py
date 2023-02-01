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
    
    train_arg_df, train_kp_df, train_labels_df = load_kpm_data("dataset/", subset="train")
    val_arg_df, val_kp_df, val_labels_df = load_kpm_data("dataset/", subset="dev")
    
    keys, values = zip(*params.items())
    combo_list = list(it.product(*(values)))

    
    res_vec = []
    
    for i in tqdm(range(len(combo_list))):
        
        res_dict = {
            'batch_size': combo_list[i][0],
            'loss': combo_list[i][1],
            'optimizer': combo_list[i][2],
            'lr': combo_list[i][3],
            'eps': combo_list[i][4],
            'epochs': combo_list[i][5],
            'warmup_steps': combo_list[i][6],
            'weight_decay' : combo_list[i][7]
        }
        
        print(res_dict)
        
        model = SiameseNetwork(bert_type=BertModel.from_pretrained(model_type))
        model.to(device)
        
        train_loader = DataLoader(train_data, batch_size=res_dict['batch_size'], pin_memory=True)
        
        optimizer=res_dict['optimizer']
        
        if(optimizer == 'adam'):
            optimizer= torch.optim.AdamW(model.parameters(),
                  lr = res_dict['lr'], 
                  eps = res_dict['eps'] 
        )
            
        # Total number of training steps is [number of batches] x [number of epochs]. 
        # (Note that this is not the same as the number of training samples).
        total_steps = len(train_loader) * res_dict['epochs']

        # Create the learning rate scheduler.
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = res_dict['warmup_steps'],
                                            num_training_steps = total_steps)
        
        train_res = train(model, device, train_loader, res_dict['loss'], optimizer, res_dict['epochs'], scheduler, verbose=False)
        
        res_dict['train'] = train_res
        
        res_dict['train_metrics'] = [None] * len(train_res['predicted'])
        res_dict['train_challenge_metrics'] = [None] * len(train_res['predicted'])
        
        for i, elem in enumerate(train_res['predicted']):
            res_dict['train_metrics'][i] = compute_metrics(elem, train_res['labels'], metrics)
            res_dict['train_challenge_metrics'][i] = extract_challenge_metrics(elem, train_labels_df, train_arg_df, train_kp_df)
        
            
            
        val_loader = DataLoader(val_data, pin_memory=True)
        val_res = test(model, device, val_loader, res_dict['loss'])
        
        res_dict['val'] = val_res
        
        res_dict['val_metrics'] = compute_metrics(val_res['predicted'].T, val_res['labels'].T, metrics)
        res_dict['val_challenge_metrics'] = extract_challenge_metrics(val_res['predicted'].T, val_labels_df, val_arg_df, val_kp_df)
        
        
        res_vec.append(res_dict)
        
        results_dict = {
            'params':[],
            'train_metrics':[],
            'train_challenge_metrics': [],
            'val_metrics': [],
            'val_challenge_metrics': []
        }
        
        params_string = f"batch_size {res_vec['batch_size']}, loss {res_vec['loss']}, optimizer {res_vec['optimizer']}, lr {res_vec['lr']}, eps {res_vec['eps']}, epochs {res_vec['epochs']}, warmup_steps {res_vec['warmup_steps']}, weight_decay {res_vec['weight_decay']}"
        
        results_dict['params'].append(params_string)
        results_dict['train_metrics'].append(res_vec['train_metrics'])
        results_dict['train_challenge_metrics'].append(res_vec['train_challenge_metrics'])
        results_dict['val_metrics'].append(res_vec['val_metrics'])
        results_dict['val_challenge_metrics'].append(res_vec['val_challenge_metrics'])

        df=pd.DataFrame(results_dict)

        df.to_csv('task1_grid_results.csv', mode='a', sep='#', index=False, header=False if path.exists("task1_grid_results.csv") else True)
        
        
    return res_vec   
        