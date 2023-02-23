import torch
from torch import nn
from transformers import AutoModel, PegasusForConditionalGeneration

class PegasusModel(nn.Module):
    """
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
    """
    def __init__(self, model_type=None):
        super(PegasusModel, self).__init__()
        
        if model_type is None:
            self.model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum",
                                          num_labels = 2)
        else:
            self.model = model_type

    def forward(self, input_arg):
       
        self.model(input_arg['input_ids'],
                    attention_mask = input_arg['attention_mask'])

        last_hidden_states = outputs.last_hidden_state

        return last_hidden_states
    
    def generate(self, input_args):
        
        out_gen = self.model.generate(**input_args
                           #, length_penalty=0.8, num_beams=8, max_length=128)
        
        return out_gen
        

        
def train(model, device, train_loader, loss_function, optimizer, epochs, scheduler, verbose=False):
    
    model.train()
    
    # loss_function.to(device)
    results = {}
    
    '''results = {'loss': torch.zeros([epochs,1]),
              'predicted': torch.zeros([epochs,len(train_loader.dataset)]),
              'labels': torch.zeros([epochs,len(train_loader.dataset)])
              }
    
    results = {k:v.to(device) for k,v in results.items()}'''
    
    for epoch in range(0, epochs):
        
        '''epoch_results = {'loss': torch.zeros([len(train_loader),1]),
              'predicted': torch.zeros([len(train_loader.dataset)]),
              'labels': torch.zeros([len(train_loader.dataset)])
        }
        
        epoch_results = {k:v.to(device) for k,v in epoch_results.items()}'''
        
        for batch_idx, (encodings) in enumerate(train_loader):

            # Extract arguments, key_points and labels all from the same batch
            '''args = {k:v.to(device) for k,v in encodings['argument'].items()}

            kps = {k:v.to(device) for k,v in encodings['kp'].items()}'''
            
            args = encodings['argument']
            kps = encodings['kp']

            labels = encodings['label']
            #labels = labels.to(device)

            optimizer.zero_grad()
            outp = model(args)

            loss = loss_function(outp.float(), labels.float())
            loss.backward()
            
            # Compute start and end index of the slice to assign
            '''start_idx = batch_idx*train_loader.batch_size;
            if batch_idx < (len(train_loader)-1):
                end_idx = (batch_idx+1)*(len(outp))
            else:
                end_idx = len(train_loader.dataset)
                
            epoch_results['loss'][batch_idx] = loss'''
            
            # Saves predictions and targets in respective position
            # of the epoch_results arrays, this has been done because the
            # data shuffles at each iteration
            '''j = 0
            for i in range(0, len(encodings['id'])):
                index = encodings['id'][i]
                epoch_results['predicted'][index] = outp[j]
                epoch_results['labels'][index] = labels[j]
                j+=1'''
                
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

        #TODO check this row
        #results['metrics'] = compute_metrics(epoch_results['predicted'], epoch_results['labels'], metrics)
        
        '''results['loss'][epoch] = torch.mean(epoch_results['loss'], 0)
        
        results['predicted'][epoch] = epoch_results['predicted']
        
    results['labels'] = epoch_results['labels']'''
                                 
    return results