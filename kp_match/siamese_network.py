import torch
from torch import nn
from torch.optim.lr_scheduler import StepLR
from transformers import BertModel, AutoModel
from sentence_transformers import util
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler
from challenge_metrics import get_predictions, evaluate_predictions

class SiameseNetwork(nn.Module):
    """
        Siamese network model class.
        This model is composed of two identical tails, each one is fed an input.
        The two tails share a common head where an output function is computed.
        In this case the two tails will contain the same model, a Transformer which
        will create embeddings of each input string and as output function 
        the cosine similarity of the two embeddings.
        The result is given as output of the siamese network.
    """
    def __init__(self, model_type=None):
        """ Initialize object of type SiameseNetwork
        Parameters
        ----------
        model_type: string, default=None
            Name of model to be inserted in each tail
        """
        super(SiameseNetwork, self).__init__()

        # Initialize model inside the two tails
        if model_type is None:
            self.model = AutoModel.from_pretrained("bert-base-uncased",
                                          num_labels = 2)
        else:
            self.model = model_type

        # Select Cosine Similarity as output function
        self.output_fun = torch.nn.CosineSimilarity()
    
    def forward_once(self, input_ids, attention_masks):
        """ Performs feed-forward pass over a single input
        Parameters
        ----------
        input_ids: array-like
            Input IDs of input string
        attention_masks: array-like
            Attention mask of input string
        Returns
        -------
        last_hidden_states: array-like
            Hidden state of the model 
            after feed-forward on input string
        """
        
        # Perform feed-forward given the input
        outputs = self.model(input_ids,
                            attention_mask = attention_masks)

        # Extract hidden states
        last_hidden_states = outputs.last_hidden_state

        return last_hidden_states

    def forward(self, input1, input2):
        """ Given two input strings, perform a feed-forward pass
        for each tail of the Siamese Network
        Parameters
        ----------
        input1: dict
            Contains Input IDs and Attention mask of the first input string
        input2: dict
            Contains Input IDs and Attention mask of the second input string
        Returns
        -------
        out: float
            Cosine similarity of the two strings
        """
        output1 = self.forward_once(input1['input_ids'], input1['attention_masks'])
        output2 = self.forward_once(input2['input_ids'], input2['attention_masks'])
        
        # Average pooling of every token
        output1 = torch.mean(output1, 1)
        output2 = torch.mean(output2, 1)

        out = self.output_fun(output1, output2)

        return out


def train(model, device, train_loader, loss_function, optimizer, epochs, scheduler, verbose=False):
    """ Train Siamese Network model
    Parameters
    ----------
    model: SiameseNetwork object
        Siamese Network
    device: torch device
        Selected device on which to perform the grid search 
        (usually a GPU)
    train_loader: DataLoader object
        Training Data already divided into mini-batches
    loss_function: function
        Loss function to be computed at each step
    optimizer: Optimizer object
        Optimizer for the model
    epochs: int
        Number of epochs to train the model
    scheduler: Scheduler object
        Scheduler for the learning rate
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
    
    loss_function.to(device) # Move loss to device
    
    # Create results structure
    results = {'loss': torch.zeros([epochs,1]),
              'predicted': torch.zeros([epochs,len(train_loader.dataset)]),
              'labels': torch.zeros([epochs,len(train_loader.dataset)])
              }
    
    # Move results to device
    results = {k:v.to(device) for k,v in results.items()}
    
    for epoch in range(0, epochs):
        
        # Create intermediate structure for each epoch
        epoch_results = {'loss': torch.zeros([len(train_loader),1]),
              'predicted': torch.zeros([len(train_loader.dataset)]),
              'labels': torch.zeros([len(train_loader.dataset)])
        }
        
        epoch_results = {k:v.to(device) for k,v in epoch_results.items()}
        
        for batch_idx, (encodings) in enumerate(train_loader):

            # Extract arguments, key_points and labels all from the same batch
            args = {k:v.to(device) for k,v in encodings['argument'].items()}

            kps = {k:v.to(device) for k,v in encodings['kp'].items()}

            labels = encodings['label']
            labels = labels.to(device)

            optimizer.zero_grad()
            outp = model(args, kps) # Perform feed-forward pass

            # Compute loss and performs a backward pass
            loss = loss_function(outp.float(), labels.float())
            loss.backward()
            
            # Compute start and end index of the slice to save in results
            start_idx = batch_idx*train_loader.batch_size;
            if batch_idx < (len(train_loader)-1):
                end_idx = (batch_idx+1)*(len(outp))
            else:
                end_idx = len(train_loader.dataset)
               
            # Store loss
            epoch_results['loss'][batch_idx] = loss
            
            """
            Stores predictions and targets in respective position
            of the epoch_results arrays, this has been done because the
            data is shuffled at each iteration
            """
            j = 0
            for i in range(0, len(encodings['id'])):
                index = encodings['id'][i]
                epoch_results['predicted'][index] = outp[j]
                epoch_results['labels'][index] = labels[j]
                j+=1
                
            # Clip norm of gradients
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Perform optimizer step
            optimizer.step()

            # Update the learning rate
            scheduler.step()
            
            # Print current information
            if verbose:
                if batch_idx % 10 == 0:
                    print(f'Train Epoch:', epoch+1, 'batch:',
                        batch_idx, 'loss:',
                        loss.mean())
        
        # Save average loss over the whole epoch
        results['loss'][epoch] = torch.mean(epoch_results['loss'], 0)
        
        # Save predictions for each epoch
        results['predicted'][epoch] = epoch_results['predicted']
        
    # Save labels
    results['labels'] = epoch_results['labels']
                                 
    return results
            
                
def test(model, device, test_loader, loss_function):
    """ Tests a Siamese Network model
    Parameters
    ----------
    model: SiameseNetwork object
        Siamese Network
    device: torch device
        Selected device on which to perform the grid search 
        (usually a GPU)
    test_loader: DataLoader object
        Data to perform test/evaluation on
    loss_function: function
        Loss function to be computed at each step
    Returns
    -------
    results: dict
        Contains, the loss for each mini-batch 
        (or sample if batch size is 1),
        the predictions and labels
    """
    
    # Set model in "evaluation/test mode"
    model.eval()
    
    loss_function.to(device)
    
    # Create structure to save results
    results = {'labels': torch.zeros([len(test_loader),1]),
               'predicted':torch.zeros([len(test_loader),1]),
               'loss':torch.zeros([len(test_loader),1])
              }
    
    results = {k:v.to(device) for k,v in results.items()}
    
    
    with torch.no_grad():
        for batch_idx, (encodings) in enumerate(test_loader):

            # Extract arguments, key_points and labels all from the same batch
            args = {k:v.to(device) for k,v in encodings['argument'].items()}

            kps = {k:v.to(device) for k,v in encodings['kp'].items()}

            labels = encodings['label']
            labels = labels.to(device)

            # Perform a feed-forward pass
            outp = model(args, kps)

            # Compute loss
            loss = loss_function(outp.float(), labels.float())
            
            # Store results for each mini-batch
            results['labels'][batch_idx] = labels
            results['predicted'][batch_idx] = outp
            results['loss'][batch_idx] = loss
            
    return results