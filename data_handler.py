import pandas as pd
import torch

'''
Load data from a csv file
Params:
    path: foder path where the file is saved
    filename_train: filename to the training set (could be empty)
    filename_test: filename to the test set (could be empty)
    sep_char: separator char in the indicated csv
Returns:
    train: training dataframe
    test: test dataframe
'''
def load(path = "", filename_train = "", filename_test = "", sep_char=';'):

    train = None
    test = None

    if filename_train != "":
        train = pd.read_csv(path+filename_train, sep=sep_char)
        train = __get_dataset(train)
    if filename_test != "":
        test = pd.read_csv(path+filename_test, sep=sep_char)
        test = __get_dataset(test)

    return train, test

'''
Private method used to clear the dataset from all the useless columns and to shuffle data
Params:
    df: dataframe to edit
Returns:
    df: edited dataframe
'''
def __get_dataset(df):
    # Drop all useless columns
    df.drop(['arg_id', 'key_point_id', 'stance'], axis=1, inplace=True)
    # Cast labels in float type
    df['label'] = df['label'].astype('float')
    # Shuffle the examples
    df = df.sample(frac=1, random_state=1).reset_index(drop=True)

    return df

'''
Add topic before keypoint
Params:
    df: dataframe to edit
Returns:
    df: edited dataframe
'''
def concatenate_topics(df):
    # No duplicates code
    input_args = df['argument'].tolist()

    # creating a list of keypoints for each topic
    input_kp = df['key_point'].tolist()
    topics = df['topic'].tolist()

    # appending topic to each vector
    for i, _ in enumerate(input_kp):
        input_kp[i] = topics[i] + " " + input_kp[i]

    # create a list of labels
    expected_res = df['label'].tolist()

    df = pd.DataFrame({'args': input_args,
        'key_points': input_kp,
        'labels': expected_res
    })

    return df

'''
Split the training data in train and validation set following the given percentage of split
Params:
    df: dataframe to edit
    perc_split: percentage of data that will be moved to the validation set
Returns:
    train: training set dataframe
    test: test set dataframe
'''
def split_train_data(df, perc_split=0.8):
    counts = df.label.value_counts()

    zero_train = round((counts[0] * perc_split))
    print(f"zero_train: ", zero_train)
    one_train = round((counts[1] * perc_split))
    print(f"one_train: ", one_train)
    zero_val = counts[0] - zero_train
    print(f"zero_val: ", zero_val)
    one_val = counts[1] - one_train
    print(f"one_val: ", one_val)

    train = df[df['label'] == 0][:zero_train]
    # train = train.append()
    train = pd.concat([train, df[df['label'] == 1][:one_train]])
    val = df[df['label'] == 0][zero_train:zero_train + zero_val]
    val = pd.concat([val, df[df['label'] == 1][one_train : one_train+one_val]])

    train = train[['argument', 'key_point', 'topic', 'label']]
    val = val[['argument', 'key_point', 'topic', 'label']]

    return train, val

'''
Given a set of sentences and labels, it makes a tokenization element-wise.
Params:
    sentences: sentences to be tokenized
    tokenizer: type of tokenizer to use
    labels: labels to be tokenized
Returns:
    input_ids: input_ids of the tokenized sentences
    attention_masks: attention masks of the tokenized sentences
'''
def tokenization(sentences, tokenizer, labels=None):
     # Tokenize all of the sentences and map the tokens to thier word IDs.
        input_ids = []
        attention_masks = []

      # For every sentence...
        for sent in sentences:
          # `encode_plus` will:
          #   (1) Tokenize the sentence.
          #   (2) Prepend the `[CLS]` token to the start.
          #   (3) Append the `[SEP]` token to the end.
          #   (4) Map tokens to their IDs.
          #   (5) Pad or truncate the sentence to `max_length`
          #   (6) Create attention masks for [PAD] tokens.
            encoding = tokenizer.encode_plus(
                              sent,                      # Sentence to encode.
                              add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                              max_length = 512,           # Pad & truncate all sentences.
                              pad_to_max_length = True,
                              return_attention_mask = True,   # Construct attn. masks.
                              return_tensors = 'pt',     # Return pytorch tensors.
                              truncation=True
                        )
          
          # Add the encoded sentence to the list.    
            input_ids.append(encoding['input_ids'])
          
          # And its attention mask (simply differentiates padding from non-padding).
            attention_masks.append(encoding['attention_mask'])

      # Convert the lists into tensors.
        input_ids = torch.cat(input_ids, dim=0)
        attention_masks = torch.cat(attention_masks, dim=0)

        if labels is not None:
            labels = torch.tensor(labels)
            return input_ids, attention_masks, labels
        else:
            return input_ids, attention_masks

'''
Tokenize a dataframe of sentences with labels
Param:
    df: dataframe to tokenize
    tokenizer: type of tokenizer to use
Returns:
    tokenized: a list composed by tokenized arguments (dict), tokenized key points (dict) and tokenized labels (list)
'''
def tokenize_df(df, tokenizer):
    input_id_args, attention_masks_args, labels = tokenization(df['args'], tokenizer, labels = df['labels'])
    input_id_kps, attention_masks_kps = tokenization(df['key_points'], tokenizer)

    tokenized = [ { 'arg':{
            'input_ids': input_id_args[i],
            'attention_masks' : attention_masks_args[i]
            }, 
            'kp':{
                'input_ids': input_id_kps[i],
                'attention_masks' : attention_masks_kps[i]
            }, 
            'label':labels[i] } for i in range(len(input_id_args)) ]

    return tokenized
