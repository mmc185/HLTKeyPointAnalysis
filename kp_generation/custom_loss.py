from sklearn.preprocessing import MinMaxScaler

import numpy as np
import sys
import torch

sys.path.insert(1, '../')
import data_handler


def compute_match_score(arguments, summaries, gen_tokenizer, match_model, match_tokenizer, device, mode = "null", max_length=512):
    """ Compute loss using the model trained for the matching task
    Parameters
    ----------
    arguments: array of strings
        Tokens representing the argument (input)
    summaries: array of strings
        Tokens representing the key_points (output predictions)   
    gen_tokenizer: Tokenizer object
        Tokenizer of generative model, used to perform decoding
    match_model: SiameseNetwork object
        Model trained for the matching task
    match_tokenizer: Tokenizer object
        Tokenizer of matching model, used to encode sentences
        for matching model
    device: torch device
        Selected device on which to perform the grid search 
    mode: string, default="null"
        Defines if some time of transformation must be applied
        to matching model output (e.g. "scaled")
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
    
    # Decode arguments and key_points with Generative Model Tokenizer
    dec_args = gen_tokenizer.batch_decode(arguments['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    dec_sums = gen_tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
    # Encode arguments and key_points with Matching Model Tokenizer
    enc_args = data_handler.tokenization(dec_args, match_tokenizer, max_length)
    enc_sums = data_handler.tokenization(dec_sums, match_tokenizer, max_length)
    
    args = {
        "input_ids" : enc_args[0],
        "attention_masks" : enc_args[1]
    }
    sums = {
        "input_ids" : enc_sums[0],
        "attention_masks" : enc_sums[1]
    }
    
    # Move arguments and key_points on the desired device
    args = {k:v.to(device) for k,v in args.items()}
    sums = {k:v.to(device) for k,v in sums.items()}
    
    # Perform prediction with matching model (matching vs. non-matching)
    score = match_model(args, sums)
    
    # Perform transformations on output score
    if mode == "scaled":
        # Scales data between 0 and 1
        score = score.cpu().data.numpy()
        score = (score - np.min(score)) / (np.max(score) - np.min(score))
        score = 1 - score
    
    score = torch.tensor(score, requires_grad=True)
    labels = torch.ones(score.shape[0]).to(device)
        
    # Computes MSE over predicted labels and targets
    loss_func = torch.nn.MSELoss()
    loss = loss_func(score.float(), labels.float())
        
    return loss