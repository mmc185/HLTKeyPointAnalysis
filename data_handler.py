import pandas as pd
import torch

def load_full_dataset(path, get_train=False, get_dev=False, get_test=False, sep_char='#'):
    """ Load dataset for generation task
    Parameters
    ----------
    path: string
        Path in which to find the files
    get_train: bool, default=False
        Return training data
    get_dev: bool, default=False
        Return validation data
    get_test: bool, default=False
        Return test data
    sep_char: char, default='#'
        Separator for .csv files
    Returns
    -------
    train: pd.DataFrame
        Training data
    dev: pd.DataFrame
        Validation data
    test: pd.DataFrame
        Test data
    """
    
    train = None
    dev = None
    test = None

    """
    For the generation task, the training set is composed of
    the original training + validation sets,
    the validation set is composed of the original test set
    and the test set has the IBM test phrases.
    """
    if get_train:
        train1 = pd.read_csv(path+'train.csv', sep=sep_char)
        train2 = pd.read_csv(path+'dev.csv', sep=sep_char)
        train = pd.concat([train1, train2])
        """
        Only select matching pairs of key-points and arguments,
        otherwise the model would be trained to generate 
        key-points which don't match the semantics of the argument
        """
        train = train[train['label'] == 1.0]
        train.drop(columns=['arg_id', 'key_point_id'], inplace=True)
    if get_dev:
        dev = pd.read_csv(path+'test.csv', sep=sep_char)
        dev = dev[dev['label'] == 1.0]
        dev.drop(columns=['arg_id', 'key_point_id'], inplace=True)
    if get_test:
        test = pd.read_csv(path+'test_IBM.csv', sep=sep_char)
    
    return train, dev, test


def load(path = "", filename_train = "", filename_dev = "", filename_test = "", sep_char=';', shuffle = False):
    """
    Load dataset for the matching task
    Parameters
    ----------
        path: string
            Path in which to find the files
        filename_train: string, default=""
            Filename of the training set
        filename_dev: string, default=""
            Filename of the validation set
        filename_test: string, default=""
            Filename of the test set
        sep_char: char, default=';'
            Separator for .csv files
        shuffle: bool, default=False
            If true, data is shuffled
    Returns
    -------
    train: pd.DataFrame
        Training data
    dev: pd.DataFrame
        Validation data
    test: pd.DataFrame
        Test data
    """
    train = None
    dev = None
    test = None

    # If string is not empty we want to return that specific set of data
    if filename_train != "":
        train = pd.read_csv(path+filename_train, sep=sep_char)
        train = __get_dataset(train, shuffle)
    if filename_dev != "":
        dev = pd.read_csv(path+filename_dev, sep=sep_char)
        dev = __get_dataset(dev, shuffle)
    if filename_test != "":
        test = pd.read_csv(path+filename_test, sep=sep_char)
        test = __get_dataset(test, shuffle)

    return train, dev, test


def __get_dataset(df, shuffle=False):
    """ Edits data to make it compatible for training
    Parameters
    ----------
    df: pd.Dataframe
        Data to edit
    shuffle: bool, default=False
            If true, data is shuffled
    Returns
    -------
    df: pd.DataFrame
        Edited data
    """

    # Cast labels in float type
    df['label'] = df['label'].astype('float')
    
    # Shuffle the examples
    if shuffle:
        df = df.sample(frac=1, random_state=1).reset_index(drop=True)
        
    return df


def concatenate_topics(df, input_col='key_point', output_col='key_points'):
    """ Concatenates each topic to the corresponding value in the 
    "input_col" column
    Parameters
    ----------
    df: pd.Dataframe
        Data containing the columns to concatenate
    input_col: string, default='key_point'
        Name of column of data that is going to be concatenated
        with the topics
    output_col: string, default='key_points'
        Name of new column in which 
        the concatenation is going to be stored
    Returns
    -------
    df: pd.DataFrame
        Edited data
    """

    # Transform data into lists
    input_data = df[input_col].tolist()
    topics = df['topic'].tolist()

    # Appending topic to each corresponding element
    for i, _ in enumerate(input_data):
        input_data[i] = topics[i] + " " + input_data[i]
    
    # Drops columns used to concatenate
    df.drop(columns=[input_col, 'topic'], inplace=True)
    # Stores concatenation in a new column
    df[output_col] = input_data
    
    df.reset_index(inplace=True)
    df.drop(columns=['index'], inplace=True)
    
    return df


def tokenization(sentences, tokenizer, max_length=512, labels=None):
    """ Tokenize sentences
    Parameters
    ----------
    sentences: array of strings
        Sentences to be tokenized
    tokenizer: Tokenizer object
        Tokenizer to perform tokenization
    max_length: int, default='512'
        Maximum length of tokenization
    labels: array-like, default=None
        Target labels of data
    Returns
    -------
    input_ids: array-like
        Input IDs of tokenized sentences
    attention_masks: array-like
        Attention masks of tokenized sentences
    labels: array-like
        Target labels of data
    """
    input_ids = []
    attention_masks = []

    for sent in sentences:
        """
        Tokenize sentence adding special tokens, truncating sentences to
        max_length and adding eventual padding
        """
        encoding = tokenizer.encode_plus(
                          sent,                      
                          add_special_tokens = True, 
                          max_length = max_length,   
                          pad_to_max_length = True,
                          return_attention_mask = True,   
                          return_tensors = 'pt',   
                          truncation=True
                    )


        # Store encoding input ID and attention mask
        input_ids.append(encoding['input_ids'])

        attention_masks.append(encoding['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    if labels is not None:
        # Convert labels to tensors
        labels = torch.tensor(labels)
        return input_ids, attention_masks, labels
    else:
        return input_ids, attention_masks

def tokenize_df(df, tokenizer, max_length=512):
    """ Tokenize a dataframe of sentences
    Parameters
    ----------
    df: pd.Dataframe
        Data to be tokenized
    tokenizer: Tokenizer object
        Tokenizer to perform tokenization
    max_length: int, default='512'
        Maximum length of tokenization
    Returns
    -------
    tokenized: array-like
        List of dictionaries containing 
        each pair of tokenized argument 
        and key-point along with an ID
    """
    # Tokenize arguments, saving labels
    input_id_args, attention_masks_args, labels = tokenization(df['argument'], tokenizer, labels = df['label'], max_length=max_length)
    # Tokenize key-points, labels are not saved again
    input_id_kps, attention_masks_kps = tokenization(df['key_points'], tokenizer, max_length=max_length)

    # Create structure for every pair of tokenized argument and key-point
    tokenized = [ { 'id': i,
        'argument':{
            'input_ids': input_id_args[i],
            'attention_masks' : attention_masks_args[i]
            }, 
            'kp':{
                'input_ids': input_id_kps[i],
                'attention_masks' : attention_masks_kps[i]
            }, 
            'label':labels[i] } for i in range(len(input_id_args)) ]

    return tokenized
