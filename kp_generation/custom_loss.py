from sklearn.preprocessing import MinMaxScaler

import numpy as np
import sys
import torch

sys.path.insert(1, '../')
import data_handler


def compute_match_score(arguments, summaries, gen_tokenizer, match_model, match_tokenizer, device, mode = "null", max_length=512):
    
    dec_args = gen_tokenizer.batch_decode(arguments['input_ids'], skip_special_tokens=True, clean_up_tokenization_spaces=True)
    dec_sums = gen_tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    
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
    
    args = {k:v.to(device) for k,v in args.items()}
    sums = {k:v.to(device) for k,v in sums.items()}
    
    score = match_model(args, sums)
    
    if mode == "scaled":
        score = score.cpu().data.numpy()
        score = (score - np.min(score)) / (np.max(score) - np.min(score))
        score = 1 - score
    
    score = torch.tensor(score, requires_grad=True)
        
        
    return score.mean()