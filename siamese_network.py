import torch
from torch import nn
from transformers import BertModel
from sentence_transformers import util
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import accuracy_score

class SiameseNetwork(nn.Module):
    """
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
    """
    def __init__(self, bert_type=None):
        super(SiameseNetwork, self).__init__()

        if bert_type is None:
            self.model = BertModel.from_pretrained("bert-base-uncased",
                                          num_labels = 2)
        else:
            self.model = bert_type

        self.output_fun = torch.nn.CosineSimilarity()
        
        # add linear layers to compare
        '''self.fc = nn.Sequential(
             nn.Linear(self.model.fc.in_features * 2, 256),
             nn.ReLU(inplace=True),
             nn.Linear(256, 1),
        )

        self.sigmoid = nn.Sigmoid()'''

        # initialize the weights
        #self.resnet.apply(self.init_weights)
        # self.fc.apply(self.init_weights)
        
    def init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform(m.weight)
            m.bias.data.fill_(0.01)
    
    def forward_once(self, input_ids, attention_masks):
        
        outputs = self.model(input_ids,
                            token_type_ids = None,
                            attention_mask = attention_masks)

        last_hidden_states = outputs.last_hidden_state

        return last_hidden_states

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1['input_ids'], input1['attention_masks'])
        output2 = self.forward_once(input2['input_ids'], input2['attention_masks'])
        
        # AVG of every token
        output1 = torch.mean(output1, 1)
        output2 = torch.mean(output2, 1)

        out = self.output_fun(output1, output2)

        return out

    def get_smart_batching_collate(self):
        return self.model.smart_batching_collate

def train(model, device, train_loader, loss_function, optimizer, epochs, scheduler, metrics, verbose=False):
    model.train()
    
    loss_function.to(device)
    results = {'loss': torch.zeros([epochs,1]),
              'predicted': torch.zeros([epochs,len(train_loader.dataset)]),
              'labels': torch.zeros([epochs,len(train_loader.dataset)])
              }
    
    results = {k:v.to(device) for k,v in results.items()}
    
    for epoch in range(0, epochs):
        
        epoch_results = {'loss': torch.zeros([len(train_loader),1]),
              'predicted': torch.zeros([len(train_loader.dataset)]),
              'labels': torch.zeros([len(train_loader.dataset)])
        }
        
        epoch_results = {k:v.to(device) for k,v in epoch_results.items()}
        
        #TODO compute metrics for each epoch and return the mean of each metric
        
        for batch_idx, (encodings) in enumerate(train_loader):

            # Extract arguments, key_points and labels all from the same batch
            args = {k:v.to(device) for k,v in encodings['argument'].items()}

            kps = {k:v.to(device) for k,v in encodings['kp'].items()}

            labels = encodings['label']
            labels = labels.to(device)

            optimizer.zero_grad()
            outp = model(args, kps)

            loss = loss_function(outp.float(), labels.float())
            loss.backward()
            
            # Compute start and end index of the slice to assign
            start_idx = batch_idx*train_loader.batch_size;
            if batch_idx < (len(train_loader)-1):
                end_idx = (batch_idx+1)*(len(outp))
            else:
                end_idx = len(train_loader.dataset)
                
            epoch_results['loss'][batch_idx] = loss
            epoch_results['predicted'][start_idx:end_idx] = outp
            epoch_results['labels'][start_idx:end_idx] = labels

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()
            
            if verbose:
                if batch_idx % 10 == 0:
                    print(f'Train Epoch:', epoch, 'batch:',
                        batch_idx, 'loss:',
                        loss.mean())

        
        results['metrics'] = compute_metrics(epoch_results['predicted'], epoch_results['labels'], metrics)
        
        results['loss'][epoch] = torch.mean(epoch_results['loss'], 0)
        results['predicted'][epoch] = epoch_results['predicted']
        results['labels'][epoch] = epoch_results['labels']
                                 
    return results
            
                
def test(model, device, test_loader, loss_function, metrics):
    model.train()
    
    loss_function.to(device)
    
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

            outp = model(args, kps)

            loss = loss_function(outp.float(), labels.float())
            
            results['labels'][batch_idx] = labels
            results['predicted'][batch_idx] = outp
            results['loss'][batch_idx] = loss
    
    results['metrics'] = compute_metrics(results['predicted'], results['labels'], metrics)
            
    return results
    
    
def compute_metrics(predicted, expected, metrics):
    
    #TODO add challenge metrics
    if torch.is_tensor(predicted):
        predicted = predicted.cpu().data.numpy()
        
    if torch.is_tensor(expected):
        expected = expected.cpu().data.numpy()
    
    metric_results = {}
    
    if "accuracy" in metrics:
        
        # Threshold
        predicted[predicted >= 0.5] = 1
        predicted[predicted < 0.5] = 0
    
        metric_results['accuracy'] = accuracy_score(expected, predicted)
    #if "map" in metrics:
    
    return metric_results
        