import ray
from ray import air
from ray import tune
from ray.air import session

from torch.utils.data import DataLoader
import pandas as pd
import torch
import sys
from os import path

from transformers import AutoTokenizer, AutoModel, DataCollatorForSeq2Seq, get_linear_schedule_with_warmup
from datasets import load_metric

from generative_model import GenerativeModel, train, test, validate

from custom_loss import compute_match_score

sys.path.insert(1, '../')
import data_handler
from data_handler import tokenization

sys.path.insert(1, '../kp_match')
from siamese_network import SiameseNetwork

def concat_tag(df, attribute):
    df[attribute] = df[attribute].apply(lambda x : "summarize:"+x)
    return df

def decode_data(pred, exp, tokenizer):
    """ Uses the tokenizer to decoded the predicted sentence
        and the expected sentence
    Parameters
    ----------
    pred: array-like
        Predicted tokens
    exp: array-like
        Target tokens
    tokenizer: Tokenizer object
        Tokenizer to perform decoding
    Returns
    -------
    dec_pred: string
        Predicted sentence
    dec_exp: string
        Target sentence
    """
    
    """
    If the inputs are tensors they must be converted to CPU
    numpy arrays of Integers
    """
    if torch.is_tensor(pred):
        pred = pred.type(torch.IntTensor).cpu().data.numpy()
    if torch.is_tensor(exp):
        exp = exp.type(torch.IntTensor).cpu().data.numpy()
    
    # Decode predictions and labels
    dec_pred = tokenizer.batch_decode(pred, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    dec_exp = tokenizer.batch_decode(exp, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    return dec_pred, dec_exp

def compute_metrics(predicted, expected, metrics):
    """ Compute selected metrics given predictions and targets
    Parameters
    ----------
    predicted: array-like
        Predicted tokens of the model
    expected: array-like
        Target tokens
    metrics: array of strings
        Name of metrics to compute
    Returns
    -------
    metric_results : dict
        For every selected metric, stores the result of the
        computation
    """
    
    metric_results = {}
        
    if "rouge" in metrics:
        metric = load_metric("rouge")
        res = metric.compute(predictions = predicted, references = expected) 
        
        keys_vec = ['precision', 'recall', 'fmeasure']
        metric_results['rouge'] = {}
        
        """
        For every type of rouge metric (rouge-1, rouge-2, etc.)
        save precision, recall and fmeasure scores in a dict
        
        Dict example:
        {'rouge': 
            {
            'rouge1': {'precision': 0.19, 'recall': 0.20, 'fmeasure': 0.18}, 
            'rouge2': {'precision': 0.10, 'recall': 0.10, 'fmeasure': 0.10}, 
            'rougeL': {'precision': 0.18, 'recall': 0.19, 'fmeasure': 0.17}, 
            'rougeLsum': {'precision': 0.18, 'recall': 0.19, 'fmeasure': 0.17}
            }
        }
        """
        for k,v in res.items():
            metric_results['rouge'][k] = {}
            for metric in keys_vec:
                metric_results['rouge'][k][metric] = getattr(v.mid, metric)
        
    return metric_results


def tokenization_target(sentences, tokenizer, max_length=512):
    """ Tokenize target sentences
    Parameters
    ----------
    sentences: array of strings
        Sentences to be tokenized
    tokenizer: Tokenizer object
        Tokenizer to perform tokenization
    max_length: int, default='512'
        Maximum length of tokenization
    Returns
    -------
    input_ids: array-like
        Input IDs of tokenized sentences
    attention_masks: array-like
        Attention masks of tokenized sentences
    labels: array-like
        Target tokens of data
    """
    
    input_ids = []
    attention_masks = []
    labels = []

    """
    Tokenize as target to perform teacher forcing
    """
    with tokenizer.as_target_tokenizer():
        for sent in sentences:
            encoding = tokenizer(sent, max_length = max_length, 
                                     return_attention_mask = True,
                                     pad_to_max_length = True
                            )

            """
            Targets are input IDs shifted by one 
            (starting from the second position)
            """
            labels.append(encoding["input_ids"][1:])

            # Store encoding input ID and attention mask
            
            input_ids.append(encoding['input_ids'][:-1])

            attention_masks.append(encoding['attention_mask'][:-1])

    # Convert the lists into tensors.
    input_ids = torch.as_tensor(input_ids)
    attention_masks = torch.as_tensor(attention_masks)
    labels = torch.as_tensor(labels)
    
    return input_ids, attention_masks, labels
    
    
def tokenize_df_gen(df, tokenizer, max_length=512, key_points_on=True):
    """ Tokenize a dataframe of sentences
        for a generation task
    Parameters
    ----------
    df: pd.Dataframe
        Data to be tokenized
    tokenizer: Tokenizer object
        Tokenizer to perform tokenization
    max_length: int, default='512'
        Maximum length of tokenization
    key_points_on: bool, default=True
        Changes structure of output dict,
        depends on the presence of key_points (True)
        or their absence (False)
    Returns
    -------
    tokenized: array-like
        List of dictionaries containing 
        each pair of tokenized argument 
        and eventual key-point
    """
    
    input_id_args, attention_masks_args = tokenization(df['argument'], tokenizer, max_length=max_length)
    if key_points_on:
        input_id_kps, attention_masks_kps, labels = tokenization_target(df['key_point'], tokenizer, max_length=max_length)

        tokenized = [ { #'id': i,
            'input_ids': input_id_args[i],
            'attention_mask' : attention_masks_args[i], 
            'decoder_input_ids': input_id_kps[i],
             'decoder_attention_mask' : attention_masks_kps[i],
            'labels': labels[i]
            } for i in range(len(input_id_args)) ]
    else:
        tokenized = [ { #'id': i,
            'input_ids': input_id_args[i],
            'attention_mask' : attention_masks_args[i]
            } for i in range(len(input_id_args)) ]

    return tokenized


def grid_search(train_data, val_data, model_type, params, metrics, device):
    """ Perform a grid search, given a set of configurations of hyper-parameters.
    Parameters
    ----------
    train_data: pd.DataFrame
        Data on which training is performed
    val_data: pd.DataFrame
        Data on which validation is performed
    model_type: string
        Name of model to be trained
    params: dict
        Configurations of hyper-parameters to test
    metrics: array of strings
        Name of metrics to compute
    device: torch device
        Selected device on which to perform the grid search 
        (usually a GPU)
    """
    
    # Add data to configuration
    params['train_data'] = train_data
    params['val_data'] = val_data
    params['model_type'] = model_type
    
    params['device'] = device
    
    params['metrics'] = metrics
    
    # Set logs to be shown on the Command Line Interface every 30 seconds
    reporter = tune.CLIReporter(max_report_frequency=30)
    
    # Starts grid search using RayTune
    tuner = tune.Tuner(tune.with_resources(trainable,
                                          {"cpu":2, "gpu":1}), 
                       param_space = params, 
                       tune_config = tune.tune_config.TuneConfig(reuse_actors = False),
                       run_config=air.RunConfig(name='gen_'+params['optimizer'], verbose=1, progress_reporter=reporter))
    results = tuner.fit()
    
    # Get a dataframe for the last reported results of all of the trials 
    df = results.get_dataframe()
    
    
def trainable(config_dict):
    """ Performs training on a single configuration of hyper-parameters.
    The results are stored in a .csv file.
    Parameters
    ----------
    config_dict: dict
        Data needed to perform training and validation
        (data, hyper-parameters, metrics, etc.)
    """
    
    # Empty GPU cache
    torch.cuda.empty_cache()
    
    # Load Generative model with the defined model_type and move it to the desired device
    model = GenerativeModel(config_dict['model_type'])
    model.to(config_dict['device'])
    
    #Tokenize data (both training and validation)
    tokenizer = config_dict['tokenizer']
    config_dict.pop('tokenizer')
    tokenized_tr = tokenize_df_gen(config_dict['train_data'], tokenizer, max_length=config_dict['max_length'])
    tokenized_val = tokenize_df_gen(config_dict['val_data'], tokenizer, max_length=config_dict['max_length'])
    
    # Remove useless data
    config_dict.pop('train_data')
    config_dict.pop('val_data')
    
    # Data is set for seq2seq tasks
    seq2seq_data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding=True, max_length=config_dict['max_length'])
    
    """
    Create DataLoader object for training data to feed it to the model.
    The data is shuffled at each epoch, it is divided in mini-batches with the batch size selected
    in the hyper-parameters configuration and it is pinned to memory for efficiency.
    Additionally, it uses the pre-defined seq2seq collator
    """
    train_loader = DataLoader(
        tokenized_tr, 
        batch_size=config_dict['batch_size'], 
        collate_fn=seq2seq_data_collator, 
        shuffle=True,
        pin_memory=True
    )
    
    optimizer=config_dict['optimizer']
    
    # Load the selected optimizer with the given hyper-parameters
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
        
    # Total number of training steps
    total_steps = len(train_loader) * config_dict['epochs']
    
    # Scheduler for the learning rate
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                        num_warmup_steps = config_dict['warmup_steps'],
                                        num_training_steps = total_steps)
    
    loss_dict = None
    # If custom loss is set
    if config_dict['match_model_type'] != 'null':
        
        # Load Siamese Network from file
        match_model = SiameseNetwork(model_type=AutoModel.from_pretrained(config_dict['match_model_type']))
        match_model.load_state_dict(torch.load("../../../HLTKeyPointAnalysis/kp_match/models/model_82"))
        
        # Move the siamese network to the desired device
        match_model.to(config_dict['device'])
        
        # Load corresponding tokenizer
        match_tokenizer = AutoTokenizer.from_pretrained(config_dict['match_model_type'])
        
        # Save custom loss configuration in dict
        loss_dict = {'gen_tokenizer': tokenizer, 'match_tokenizer': match_tokenizer, 'match_model': match_model, 'mode': config_dict['mode'], 'loss_function': compute_match_score}
    
    # Train model
    train_res = train(model, config_dict['device'], train_loader, optimizer, config_dict['epochs'], loss_dict, scheduler, config_dict['max_length'], verbose=True)
    
    # Evaluation of train predictions
    config_dict['train_metrics'] = [None] * len(train_res['predicted'])
    
    # For every epoch, compute selected metrics and store them
    for i, elem in enumerate(train_res['predicted']):
        dec_pred, dec_exp = decode_data(elem, train_res['labels'][i], tokenizer)
        config_dict['train_metrics'][i] = compute_metrics(dec_pred, dec_exp, config_dict['metrics'])
    
    """
    Create DataLoader object for validation data, it is pinned to memory for efficiency, its batch size is set to 1.
    Like for the training dataset, the seq2seq collator is used
    """
    val_loader = DataLoader(
        tokenized_val, 
        batch_size=1,
        collate_fn=seq2seq_data_collator, 
        shuffle=True,
        pin_memory=True
    )
    
    # Perform evaluation
    val_res = validate(model, config_dict['device'], val_loader, max_length=config_dict['max_length'])
    
    # Compute metrics
    config_dict['validation_metrics'] = [None] * len(val_res['predicted'])
    dec_pred, dec_exp = decode_data(val_res['predicted'], val_res['labels'], tokenizer)
    config_dict['validation_metrics'] = compute_metrics(dec_pred, dec_exp, config_dict['metrics'])
        
    # Remove useless data
    config_dict.pop('device')
    config_dict.pop('metrics')
    
    # Create a pd.DataFrame of the config with its results
    for key, value in config_dict.items():
        config_dict[key] = [config_dict[key]]
    
    df=pd.DataFrame(config_dict)

    # Store results (if file already exists, append the results otherwise create the .csv file)
    df.to_csv('../../../HLTKeyPointAnalysis/task2_grid_results.csv', mode='a', sep='#', index=False, header=False if path.exists("../../../HLTKeyPointAnalysis/task2_grid_results.csv") else True)