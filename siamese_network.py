import torch
from torch import nn
from transformers import BertModel, AutoModel
from sentence_transformers import util
from torch.optim.lr_scheduler import StepLR
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from challenge_metrics import get_predictions, evaluate_predictions
from sklearn.preprocessing import MinMaxScaler

class SiameseNetwork(nn.Module):
    """
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
    """
    def __init__(self, model_type=None):
        super(SiameseNetwork, self).__init__()

        if model_type is None:
            self.model = AutoModel.from_pretrained("bert-base-uncased",
                                          num_labels = 2)
        else:
            self.model = model_type

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
                            attention_mask = attention_masks)

        last_hidden_states = outputs.last_hidden_state

        return last_hidden_states

    def forward(self, input1, input2):
        # get two images' features
        output1 = self.forward_once(input1['input_ids'], input1['attention_masks'])
        output2 = self.forward_once(input2['input_ids'], input2['attention_masks'])
        
        # AVG of every token
        output1 = torch.mean(output1[:, 1:, :], 1)
        output2 = torch.mean(output2[:, 1:, :], 1)

        out = self.output_fun(output1, output2)

        return out

    def get_smart_batching_collate(self):
        return self.model.smart_batching_collate

def train(model, device, train_loader, loss_function, optimizer, epochs, scheduler, verbose=False):
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
            
            # Saves predictions and targets in respective position
            # of the epoch_results arrays, this has been done because the
            # data shuffles at each iteration
            j = 0
            for i in range(0, len(encodings['id'])):
                index = encodings['id'][i]
                epoch_results['predicted'][index] = outp[j]
                epoch_results['labels'][index] = labels[j]
                j+=1
                
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
                    print(f'Train Epoch:', epoch+1, 'batch:',
                        batch_idx, 'loss:',
                        loss.mean())
        
        results['loss'][epoch] = torch.mean(epoch_results['loss'], 0)
        
        results['predicted'][epoch] = epoch_results['predicted']
        
    results['labels'] = epoch_results['labels']
                                 
    return results
            
                
def test(model, device, test_loader, loss_function):
    model.eval()
    
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
    
    #results['metrics'] = compute_metrics(results['predicted'], results['labels'], metrics)
            
    return results
    
    
def compute_metrics(predicted, expected, metrics):
    
    if torch.is_tensor(predicted):
        predicted = predicted.cpu().data.numpy()
        pred = predicted.copy().T
        
    if torch.is_tensor(expected):
        expected = expected.cpu().data.numpy()
        expected = expected.T
    
    metric_results = {}
    
    # Threshold
    pred[pred >= 0.5] = 1
    pred[pred < 0.5] = 0
        
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
    
    np_predicted = predictions.cpu().data.numpy()
    np_predicted = np_predicted.T
    
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
    evaluate_predictions(merged_df)
    