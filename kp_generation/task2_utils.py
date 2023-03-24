import ray
from ray import air
from ray import tune
from ray.air import session

from torch.utils.data import DataLoader
import pandas as pd
import torch
import sys
from os import path

from transformers import AutoTokenizer, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup
from datasets import load_metric

from generative_model import GenerativeModel, train, test

sys.path.insert(1, '../')
import data_handler
from data_handler import tokenization

def decode_data(pred, exp, tokenizer):
    
    # Cast Tensors elements to Int
    if torch.is_tensor(pred):
        pred = pred.type(torch.IntTensor).cpu().data.numpy()
    if torch.is_tensor(exp):
        exp = exp.type(torch.IntTensor).cpu().data.numpy()
    
    # Decode predictions and labels
    dec_pred = tokenizer.batch_decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    dec_exp = tokenizer.batch_decode(exp, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return dec_pred, dec_exp

def compute_metrics(predicted, expected, metrics):
    
    metric_results = {}
        
    if "rouge" in metrics:
        metric = load_metric("rouge")
        res = metric.compute(predictions = predicted, references = expected) 
        
        keys_vec = ['precision', 'recall', 'fmeasure']
        metric_results['rouge'] = {}
        for k,v in res.items():
            metric_results['rouge'][k] = {}
            for metric in keys_vec:
                metric_results['rouge'][k][metric] = getattr(v.mid, metric)
        
    return metric_results


def tokenization_target(sentences, tokenizer, max_length=512):
     # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []
        labels = []

      # For every sentence...
        with tokenizer.as_target_tokenizer():
            for sent in sentences:
                encoding = tokenizer(sent, max_length = max_length, 
                                         return_attention_mask = True,
                                         pad_to_max_length = True
                                )
                
                labels.append(encoding["input_ids"][1:])

              # Add the encoded sentence to the list.    
                input_ids.append(encoding['input_ids'][:-1])

              # And its attention mask (simply differentiates padding from non-padding).
                attention_masks.append(encoding['attention_mask'][:-1])

      # Convert the lists into tensors.
        input_ids = torch.as_tensor(input_ids)
        attention_masks = torch.as_tensor(attention_masks)
        labels = torch.as_tensor(labels)
        return input_ids, attention_masks, labels
    
    
def tokenize_df_gen(df, tokenizer, max_length=512):
    input_id_args, attention_masks_args = tokenization(df['argument'], tokenizer, max_length=max_length)
    input_id_kps, attention_masks_kps, labels = tokenization_target(df['key_point'], tokenizer, max_length=max_length)
    
    tokenized = [ { #'id': i,
        'input_ids': input_id_args[i],
        'attention_mask' : attention_masks_args[i], 
        'decoder_input_ids': input_id_kps[i],
         'decoder_attention_mask' : attention_masks_kps[i],
        'labels': labels[i]
        } for i in range(len(input_id_args)) ]

    return tokenized


def grid_search(train_data, val_data, model_type, params, metrics, device):
    
    params['train_data'] = train_data
    params['val_data'] = val_data
    params['model_type'] = model_type
    
    params['device'] = device
    
    params['metrics'] = metrics
    
    reporter = tune.CLIReporter(max_report_frequency=30)
    tuner = tune.Tuner(tune.with_resources(trainable,
                                          {"cpu":2, "gpu":1}), 
                       param_space = params, 
                       tune_config = tune.tune_config.TuneConfig(reuse_actors = False),
                       run_config=air.RunConfig(name='gen_'+params['optimizer'], verbose=1, progress_reporter=reporter))
    results = tuner.fit()
    
    # Get a dataframe for the last reported results of all of the trials 
    df = results.get_dataframe()
    
    
def trainable(config_dict):
    
    torch.cuda.empty_cache()
    
    model = GenerativeModel(config_dict['model_type'])
    model.to(config_dict['device'])
    
    tokenizer = config_dict['tokenizer']
    config_dict.pop('tokenizer')
    tokenized_tr = tokenize_df_gen(config_dict['train_data'], tokenizer, max_length=config_dict['max_length'])
    tokenized_val = tokenize_df_gen(config_dict['val_data'], tokenizer, max_length=config_dict['max_length'])
    
    config_dict.pop('train_data')
    config_dict.pop('val_data')
    
    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, max_length=config_dict['max_length'])
    
    train_loader = DataLoader(
        tokenized_tr, # dataset di validazione
        batch_size=config_dict['batch_size'], # dimensione del batch
        collate_fn=seq2seq_data_collator, # data collator
        shuffle=True,
        pin_memory=True
    )
    
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
    train_res = train(model, config_dict['device'], train_loader, optimizer, config_dict['epochs'], None, scheduler, config_dict['max_length'], verbose=True)
    
    # Evaluation of train predictions
    config_dict['train_metrics'] = [None] * len(train_res['predicted'])
    
    for i, elem in enumerate(train_res['predicted']):
        dec_pred, dec_exp = decode_data(elem, train_res['labels'][i], tokenizer)
        # Compute metrics
        config_dict['train_metrics'][i] = compute_metrics(dec_pred, dec_exp, config_dict['metrics'])
        
    
    del(train_res)
    torch.cuda.empty_cache()
    
    val_loader = DataLoader(
        tokenized_val, # dataset di validazione
        batch_size=1,
        collate_fn=seq2seq_data_collator, # data collator
        shuffle=True,
        pin_memory=True
    )
    
    val_res = test(model, config_dict['device'], val_loader, max_length=config_dict['max_length'])
    
    config_dict['validation_metrics'] = [None] * len(val_res['predicted'])
    dec_pred, dec_exp = decode_data(val_res['predicted'], val_res['labels'], tokenizer)
    # Compute metrics
    config_dict['validation_metrics'] = compute_metrics(dec_pred, dec_exp, config_dict['metrics'])
        
    config_dict.pop('device')
    config_dict.pop('metrics')
    
    for key, value in config_dict.items():
        config_dict[key] = [config_dict[key]]
    
    df=pd.DataFrame(config_dict)

    df.to_csv('../../../HLTKeyPointAnalysis/task 2/task2_grid_results.csv', mode='a', sep='#', index=False, header=False if path.exists("../../../HLTKeyPointAnalysis/task 2/task2_grid_results.csv") else True)