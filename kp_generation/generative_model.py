import torch
from torch import nn
from transformers import PegasusForConditionalGeneration, T5ForConditionalGeneration
 
class GenerativeModel(nn.Module):
    """
    Generative model class.
    It is used as wrapper for different generative models such as Pegasus and T5
    """
    def __init__(self, model_type=None):
        """ Initialize object of type GenerativeModel
        Parameters
        ----------
        model_type: string, default=None
            Name of model to be loaded
        """
        super(GenerativeModel, self).__init__()
 
        if model_type is None:
            self.model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum", num_labels = 2)
        elif model_type == "google/pegasus-xsum" or model_type == 'google/pegasus-large':
            self.model = PegasusForConditionalGeneration.from_pretrained(model_type, num_labels = 2)
        elif model_type == "t5-small" or model_type == "t5-base" or model_type == "t5-large":
            self.model = T5ForConditionalGeneration.from_pretrained(model_type, num_labels = 2)
 
    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels):
        """ Performs feed-forward pass
        Parameters
        ----------
        input_ids: array-like
            Input IDs of input string of encoder
        attention_masks: array-like
            Attention mask of input string of encoder
        decoder_input_ids: array-like
            Input IDs of input string of encoder
        decoder_attention_masks: array-like
            Attention mask of input string of encoder
        labels: 
        Returns
        -------
        outputs: Seq2SeqLMOutput object
            Object containing loss of model and other
            useful data
        """
        
        # Perform feed-forward given the input
        outputs = self.model(input_ids, attention_mask = attention_mask,
                  decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask,
                            labels = labels)
 
        return outputs
 
    def generate(self, input_args, attention_masks):
        """ Generates summary given input argument
        Parameters
        ----------
        input_args: array-like
            Input IDs of input argument
        attention_masks: array-like
            Attention mask of input argument
        Returns
        -------
        out_gen: array-like
            Input IDs of generated summary
        """
        
        out_gen = self.model.generate(input_ids = input_args, attention_mask = attention_masks, min_length=3, max_length=35)
 
        return out_gen
 
 
 
def train(model, device, train_loader, optimizer, epochs, loss_dict, scheduler, max_length, verbose=False):
    """ Train Generative model
    Parameters
    ----------
    model: GenerativeModel object
        Generative model to train
    device: torch device
        Selected device on which to perform the grid search 
        (usually a GPU)
    train_loader: DataLoader object
        Training Data already divided into mini-batches
    optimizer: Optimizer object
        Optimizer for the model
    epochs: int
        Number of epochs to train the model
    loss_dict: dictionary
        Dict containing informations about the
        loss function to use
    scheduler: Scheduler object
        Scheduler for the learning rate
    max_length: int
        Maximum number of tokens
    verbose: bool, default=False
        If true, it prints information every 10 mini-batches
    Returns
    -------
    results: dict
        Contains, for each epoch, the average loss over the epoch,
        the predictions and labels
    """
    
    # Set model in "train mode"
    model.train()
 
    # Create results structure
    results = {'loss': torch.zeros([epochs,1]),
              'predicted': torch.zeros([epochs,len(train_loader.dataset), max_length]),
              'labels': torch.zeros([epochs,len(train_loader.dataset), max_length])
              }
 
    for epoch in range(0, epochs):
 
        # Create intermediate structure for each epoch
        epoch_results = {'loss': torch.zeros([len(train_loader),1]),
              'predicted': torch.zeros([len(train_loader.dataset), max_length]),
              'labels': torch.zeros([len(train_loader.dataset), max_length])
        }
 
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
 
            # Perform optimizer step
            optimizer.step()
 
            # Update the learning rate
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
 
            epoch_results['predicted'][idx_start:idx_end, :gen_len] = generated_summaries.cpu()
 
            epoch_results['labels'][idx_start:idx_end, :labels_len] = labels.cpu()
 
            # Print current information
            if verbose:
                if batch_idx % 10 == 0:
                    print(f'Train Epoch:', epoch, 'batch:',
                        batch_idx, 'loss:',
                        loss.mean())
 
        # Save average loss over the whole epoch
        results['loss'][epoch] = torch.mean(epoch_results['loss'], 0)
        # Save predictions for each epoch
        results['predicted'][epoch] = epoch_results['predicted']        
        # Save labels
        results['labels'][epoch] = epoch_results['labels']
 
    return results
 
def validate(model, device, test_loader, max_length=100):
    """ Evaluate a Generative model
    Parameters
    ----------
    model: GenerativeModel object
        Generative model
    device: torch device
        Selected device on which to perform the grid search 
        (usually a GPU)
    test_loader: DataLoader object
        Data to perform evaluation on
    max_length: int, default=100
        Maximum number of tokens
    Returns
    -------
    results: dict
        Contains, for each mini-batch 
        (or sample if batch size is 1),
        the predictions and labels
    """
    
    # Set model in "evaluation/test mode"
    model.eval()
 
    # Create structure to save results
    results = {
            'predicted': torch.zeros([len(test_loader.dataset), max_length]),
            'labels': torch.zeros([(len(test_loader.dataset)),max_length])
          }
 
    with torch.no_grad():
 
        for batch_idx, (encodings) in enumerate(test_loader):
 
            idx_start = batch_idx*encodings['labels'].shape[0]
            idx_end = (batch_idx+1)*encodings['labels'].shape[0]
            if idx_end > len(test_loader.dataset):
                idx_end = -1
 
            # Extract arguments, key_points and labels all from the same batch
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
 
            labels = encodings['labels'].to(device)
 
            # Generate summaries
            outp = model.generate(input_args = input_ids, attention_masks = attention_mask)
            labels_length = labels.shape[1]
            results['labels'][idx_start:idx_end, :labels_length] = labels.cpu()
            pred_length = outp.shape[1]
            
            # Store results for each mini-batch
            results['predicted'][idx_start:idx_end, :pred_length] = outp.cpu()
 
    return results
 
def test(model, device, test_loader, max_length=100):
    """ Test a Generative model
    Parameters
    ----------
    model: GenerativeModel object
        Generative model
    device: torch device
        Selected device on which to perform the grid search 
        (usually a GPU)
    test_loader: DataLoader object
        Data to perform evaluation on
    max_length: int, default=100
        Maximum number of tokens
    Returns
    -------
    results: dict
        Contains, for each mini-batch 
        (or sample if batch size is 1),
        the predictions
    """
    
    # Set model in "evaluation/test mode"
    model.eval()
 
    # Create structure to save results
    results = {
            'predicted': torch.zeros([len(test_loader.dataset), max_length])
          }
 
    with torch.no_grad():
 
        for batch_idx, (encodings) in enumerate(test_loader):
 
            idx_start = batch_idx*encodings['attention_mask'].shape[0]
            idx_end = (batch_idx+1)*encodings['attention_mask'].shape[0]
            if idx_end > len(test_loader.dataset):
                idx_end = -1
 
            # Extract arguments, key_points and labels all from the same batch
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
 
            # Generate summaries
            outp = model.generate(input_args = input_ids, attention_masks = attention_mask)
            pred_length = outp.shape[1]
            
            # Store predictions
            results['predicted'][idx_start:idx_end, :pred_length] = outp.cpu()
 
    return results