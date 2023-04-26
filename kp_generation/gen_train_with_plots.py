import pandas as pd
import numpy as np
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq
from generative_model import GenerativeModel, train, test, validate
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup
from task2_utils import tokenize_df_gen, decode_data, compute_metrics, concat_tag

import sys
import os
sys.path.insert(1, '../')
import data_handler
sys.path.insert(1, '../kp_match')
import warnings
warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt

def plot_loss(loss_arr, model_type, same_plot=False):
    """ Plot loss over mini-batches
        and saves it to a local file
    Parameters
    ----------
    loss_arr: array-like
        Contains loss value for every epoch for every each mini-batch
    model_type: string
        Name of model
    same_plot: bool, default=False
        If true, plots all losses on the same plot and saves it
        otherwise it saves n different files (n = # of epochs)
    """
    # For every epoch
    for i in range(0,len(loss_arr)):
        if not same_plot:
            plt.clf() # Clear plot
        plt.plot(range(1, len(loss_arr[i])+1), loss_arr[i].detach().numpy().reshape(len(loss_arr[i])))
        plt.xlabel("Mini-Batch")
        plt.ylabel("Loss")
        plt.title(f"Loss for epoch {i}")
        
        # Save plot
        if not same_plot:
            plt.savefig(f"./{model_type}_loss_epoch_{i}.png")
        else:
            plt.savefig(f"./{model_type}_loss.png")
        
def plot_fmeasure(train_score, val_score, epochs, model_type):
    """ Plot F1 score over epochs for training and validation sets
    Parameters
    ----------
    train_score: array-like
        Contains F1 scores on training set for each epoch
    val_score: array-like
        Contains F1 scores on validation set for each epoch
    epochs: int
        Number of epochs to train the model
    model_type: string
        Name of model
    """
    plt.plot( range(1,epochs+1), [train_score[i]['rouge']['rouge1']['fmeasure'] for i in range(0, len(train_score))],
            label='Train') 
    plt.plot( range(1,epochs+1), [val_score[i]['rouge']['rouge1']['fmeasure'] for i in range(0, len(val_score))],
            label='Val') 
    plt.ylim([0, 1])
    plt.xlabel('epochs')
    plt.ylabel('F1 score')
    plt.xticks(range(1, epochs+1))
    plt.legend()
    
    # Save plot
    plt.savefig(f"./{model_type}_fmeasure.png")

def train_with_plots(device, df_train, df_val, config, loss_dict, max_length, metrics):
    """ Trains Generative model and outputs
        arrays useful for plotting loss and metrics
    Parameters
    ----------
    device: torch device
        Selected device on which to perform the grid search 
        (usually a GPU)
    df_train: pd.DataFrame
        Training Data
    df_val: pd.DataFrame
        Validation data
    config: dict
        Contains hyper-parameters to train the model
    loss_dict: dictionary
        Dict containing informations about the
        loss function to use
    max_length: int
        Maximum number of tokens
    metrics: array-like
        Contains metrics to compute
    Returns
    -------
    results: dict
        Contains, for each epoch, the loss for every mini-batch for each epoch,
        the F1 score for training set and validation set for each epoch
    """
    
    epochs = config['epochs']
    
    # Load the best model's tokenizer
    if config['model_type'] == 'google/pegasus-large':
        tokenizer = AutoTokenizer.from_pretrained('google/pegasus-xsum')
    else:
        tokenizer = AutoTokenizer.from_pretrained(config['model_type'])
    
    #Tokenize data
    tokenized_tr = tokenize_df_gen(df_train, tokenizer, max_length=max_length)
    tokenized_val = tokenize_df_gen(df_val, tokenizer, max_length=max_length)
    
    train_loader = DataLoader(tokenized_tr, batch_size = config['batch_size'], shuffle = True, pin_memory=True)
    val_loader = DataLoader(tokenized_val, pin_memory=True)
 
    # Create model and move it on the desired device
    model = GenerativeModel(config['model_type'])
    model.to(device)
    model.train()
    
    total_steps = len(train_loader) * config['epochs']
    
    # Create optimizer and scheduler
    if config['optimizer'] == 'adamW':
        optimizer= torch.optim.AdamW(model.parameters(),
                  lr = config['lr'], 
                  eps = config['eps'],
                  weight_decay = config['weight_decay'])
 
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                        num_warmup_steps = config['warmup_steps'],
                                        num_training_steps = total_steps)
    
    # Create results structure
    results = {'loss': torch.zeros([epochs, len(train_loader), 1]),
               'train_score': [None] * epochs, 
               'val_score': [None] * epochs 
              }
 
    for epoch in range(0, epochs):
         
        print(f"Epoch: {epoch+1}/{epochs}")
        # Set model in "train mode"
        model.train()
    
        # Create intermediate structure for each epoch
        epoch_results = {'loss': torch.zeros([len(train_loader),1])}
 
        idx_start = 0
        idx_end = 0
 
        for batch_idx, (encodings) in enumerate(train_loader):
 
            # Extract arguments, key_points and labels all from the same batch
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
 
            decoder_input_ids = encodings['decoder_input_ids'].to(device)
            decoder_attention_mask = encodings['decoder_attention_mask'].to(device)
 
            labels = encodings['labels'].to(device)
            optimizer.zero_grad()
 
            if loss_dict is None:
                outs = model(input_ids, attention_mask, 
                         decoder_input_ids, decoder_attention_mask, 
                         labels) # Perform feed-forward pass
 
                loss = outs.loss
            else:
                loss_function = loss_dict['loss_function']
                # Generate summaries to use as loss
                generated_summaries = model.generate(input_args=input_ids, attention_masks=attention_mask)
 
                loss = loss_function({'input_ids':input_ids, 'attention_masks':attention_mask}, generated_summaries, loss_dict['gen_tokenizer'], loss_dict['match_model'], loss_dict['match_tokenizer'], device, loss_dict['mode'], max_length)
 
            epoch_results['loss'][batch_idx] = loss.cpu()
 
            # Performs a backward pass
            loss.backward()
 
            # Clip norm of gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
 
            optimizer.step()
 
            scheduler.step()
 
            # Generate and save predictions
            idx_start = batch_idx*encodings['input_ids'].shape[0]
            if (batch_idx+1)*encodings['input_ids'].shape[0] > len(train_loader.dataset):
                idx_end = len(train_loader.dataset)
            else:
                idx_end = (batch_idx+1)*encodings['input_ids'].shape[0]
 
            generated_summaries = model.generate(input_args=input_ids, attention_masks=attention_mask)
        
            gen_len = generated_summaries.shape[1]
            labels_len = labels.shape[1]
            
            dec_pred, dec_exp = decode_data(generated_summaries.cpu(), labels.cpu(), tokenizer)
            results['train_score'][epoch] = compute_metrics(dec_pred, dec_exp, metrics)
 
        
        # Perform validation step for each epoch to create a plot
        model.eval()
        
        val_res = validate(model, device, val_loader, max_length)
        
        # Compute metrics
        dec_pred, dec_exp = decode_data(val_res['predicted'].cpu(), val_res['labels'].cpu(), tokenizer)
        validation_scores = compute_metrics(dec_pred, dec_exp, metrics)
        
 
        #loss_len = len(epoch_results['loss'])
        # Save average loss over the whole epoch
        results['loss'][epoch] = epoch_results['loss'] 
        # Save validation score at each epoch
        results['val_score'][epoch] = validation_scores
 
    return results

os.environ["CUDA_VISIBLE_DEVICES"]="3"
device = torch.device(0)

df_train, df_val, df_test = data_handler.load_full_dataset('../dataset/', get_train=True, get_dev=True, get_test=True)

# Concatenate topics and keypoints, as stated in the paper
df_train = data_handler.concatenate_topics(df_train, input_col='argument', output_col='argument')
df_val = data_handler.concatenate_topics(df_val, input_col='argument', output_col='argument')
df_test = data_handler.concatenate_topics(df_test, input_col='argument', output_col='argument')

config = {}
config['model_type'] = 'google/pegasus-large'
config['epochs'] = 5
config['lr'] = 5e-4
config['eps'] = 1e-8
config['weight_decay'] = 1e-6
config['warmup_steps'] = 1e3
config['batch_size'] = 8
config['optimizer'] = 'adamW'

metrics = ['rouge'] 
res = train_with_plots(device, df_train, df_val, config, None, 100, metrics)

print("Train score:")
print(res['train_score'])

print("Val score:")
print(res['val_score'])

plot_loss(res['loss'], "pegasus_large")

plot_loss(res['loss'], "pegasus_large", same_plot=True)

epochs = config['epochs']

plot_fmeasure(res['train_score'], res['val_score'], epochs, "pegasus_large")