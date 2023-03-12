import torch
import sys
sys.path.insert(1, '../')
import data_handler
from data_handler import tokenization

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

    '''tokenized = [ { 'id': i,
        'argument':{
            'input_ids': input_id_args[i],
            'attention_masks' : attention_masks_args[i]
            }, 
            'kp':{
                'input_ids': input_id_kps[i],
                'attention_masks' : attention_masks_kps[i]
            }} for i in range(len(input_id_args)) ]'''
    
    
    '''tokenized = [ { 'id': i,
        'argument':{
            'input_ids': input_id_args[i],
            'attention_masks' : attention_masks_args[i]
            }, 
            'kp':{
                'input_ids': input_id_kps[i],
                'attention_masks' : attention_masks_kps[i],
                'labels': labels
            }} for i in range(len(input_id_args)) ]
'''
    
    tokenized = [ { #'id': i,
        'input_ids': input_id_args[i],
        'attention_mask' : attention_masks_args[i], 
        'decoder_input_ids': input_id_kps[i],
         'decoder_attention_mask' : attention_masks_kps[i],
        'labels': labels[i]
        } for i in range(len(input_id_args)) ]

    return tokenized