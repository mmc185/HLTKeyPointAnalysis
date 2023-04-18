import torch
from torch.utils.data import DataLoader
from transformers import get_linear_schedule_with_warmup, AutoModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from challenge_metrics import load_kpm_data, get_predictions, evaluate_predictions
from siamese_network import SiameseNetwork, train, test
import pandas as pd
from os import path

from ray import air
from ray import tune

import sys
sys.path.insert(1, "../")
import data_handler

def compute_metrics(predicted, expected, metrics):
    """ Compute selected metrics given predictions and targets
    Parameters
    ----------
    predicted: array-like
        Predicted labels of the model
    expected: array-like
        Target labels
    metrics: array of strings
        Name of metrics to compute
    Returns
    -------
    metric_results : dict
        For every selected metric, stores the result of the
        computation
    """
    
    """
    If the inputs are tensors they must be converted to CPU
    numpy arrays and transposed
    """
    if torch.is_tensor(predicted):
        predicted = predicted.cpu().data.numpy()
        pred = predicted.copy().T
        
    if torch.is_tensor(expected):
        expected = expected.cpu().data.numpy()
        expected = expected.T
    
    metric_results = {}
    
    # Threshold for labels, continuous targets are not handled by metrics
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
        
    """
    If the given metric is present in the selected ones, 
    compute it and store it
    """
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
    """ Compute challenge metrics given predictions and targets
    Parameters
    ----------
    predictions: array-like
        Predicted labels of the model
    labels_df: pd.DataFrame
        Target labels
    arg_df: pd.DataFrame
        Arguments used by the model to perform predictions
    kp_df: pd.DataFrame
        Key-points used by the model to perform predictions
    Returns
    -------
    (mAP_strict, mAP_relaxed): tuple
        Mean Average Precision results
    """
    
    # Cast predictions to numpy array and transpose it
    np_predicted = predictions.cpu().data.numpy()
    np_predicted = np_predicted.T
    
    """
    Create predictions dictionary as in the configuration stated 
    in the technical details of the challenge.
    For every sample, the corresponding argument and key-points are
    retrieved and the results stored in the dictionary.
    """
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
    
    # Add challenge data to compute challenge metrics
    params['train_kpm_data'] = load_kpm_data("../dataset/", subset="train")
    params['val_kpm_data'] = load_kpm_data("../dataset/", subset="dev")
    
    params['device'] = device
    
    params['metrics'] = metrics
      
    # Set logs to be shown on the Command Line Interface every 30 seconds
    reporter = tune.CLIReporter(max_report_frequency=30)
    
    # Starts grid search using RayTune
    tuner = tune.Tuner(tune.with_resources(trainable,
                                          {"cpu":2, "gpu":1}), 
                       param_space = params, 
                       tune_config = tune.tune_config.TuneConfig(reuse_actors = False),
                       run_config=air.RunConfig(name=params['optimizer'], verbose=1, progress_reporter=reporter))
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
    
    # Load Siamese model with the defined Transformer and move it to the desired device
    model = SiameseNetwork(model_type=AutoModel.from_pretrained(config_dict['model_type']))
    model.to(config_dict['device'])
    
    #Tokenize data (both training and validation)
    columns_list = ['argument', 'key_points', 'label']
    tokenized_tr = data_handler.tokenize_df(config_dict['train_data'][columns_list], config_dict['tokenizer'], max_length=config_dict['max_length'])
    tokenized_val = data_handler.tokenize_df(config_dict['val_data'][columns_list], config_dict['tokenizer'], max_length=config_dict['max_length'])
    
    """
    Create DataLoader object for training data to feed it to the model.
    The data is shuffled at each epoch, it is divided in mini-batches with the batch size selected
    in the hyper-parameters configuration and it is pinned to memory for efficiency
    """
    train_loader = DataLoader(tokenized_tr, shuffle=True, batch_size=config_dict['batch_size'], pin_memory=True)

    # Load the selected optimizer with the given hyper-parameters
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

    # Total number of training steps
    total_steps = len(train_loader) * config_dict['epochs']

    # Scheduler for the learning rate
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                        num_warmup_steps = config_dict['warmup_steps'],
                                        num_training_steps = total_steps)

    # Train model
    train_res = train(model, config_dict['device'], train_loader, config_dict['loss'], optimizer, config_dict['epochs'], scheduler, verbose=False)

    config_dict['train'] = train_res
    config_dict['train_metrics'] = [None] * len(train_res['predicted'])
    config_dict['train_challenge_metrics'] = [None] * len(train_res['predicted'])

    # For every epoch, compute both selected and challenge metrics and store them
    for i, elem in enumerate(train_res['predicted']):
        config_dict['train_metrics'][i] = compute_metrics(elem, train_res['labels'], config_dict['metrics'])
        config_dict['train_challenge_metrics'][i] = extract_challenge_metrics(elem, config_dict['train_kpm_data'][2], config_dict['train_kpm_data'][0], config_dict['train_kpm_data'][1])

    """
    Create DataLoader object for validation data, it is pinned to memory for efficiency, its batch size is set to 1
    """
    val_loader = DataLoader(tokenized_val, pin_memory=True)
    
    # Perform evaluation
    val_res = test(model, config_dict['device'], val_loader, config_dict['loss'])

    config_dict['val'] = val_res

    # Compute selected metrics and challenge metrics on validation results
    config_dict['val_metrics'] = compute_metrics(val_res['predicted'].T, val_res['labels'].T, config_dict['metrics'])
    config_dict['val_challenge_metrics'] = extract_challenge_metrics(val_res['predicted'].T, config_dict['val_kpm_data'][2], config_dict['val_kpm_data'][0], config_dict['val_kpm_data'][1])
    
    # Remove useless data
    config_dict.pop('train_data')
    config_dict.pop('val_data')
    config_dict.pop('train_kpm_data')
    config_dict.pop('val_kpm_data')
    config_dict.pop('tokenizer')
    config_dict.pop('device')
    config_dict.pop('metrics')
    
    # Create a pd.DataFrame of the config with its results
    for key, value in config_dict.items():
        config_dict[key] = [config_dict[key]]
    
    df=pd.DataFrame(config_dict)

    # Store results (if file already exists, append the results otherwise create the .csv file)
    df.to_csv('../../../HLTKeyPointAnalysis/kp_match/results/task1_grid_results_with_ray.csv', mode='a', sep='#', index=False, header=False if path.exists("../../../HLTKeyPointAnalysis/kp_match/results/task1_grid_results_with_ray.csv") else True)
    
