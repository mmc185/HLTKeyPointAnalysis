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
            #"google/pegasus-xsum"
            self.model = PegasusForConditionalGeneration.from_pretrained("google/pegasus-xsum", num_labels = 2)
        else:
            self.model = model_type

    def forward(self, input_ids, attention_mask, decoder_input_ids, decoder_attention_mask, labels):
       
        outputs = self.model(input_ids, attention_mask = attention_mask,
                  decoder_input_ids = decoder_input_ids, decoder_attention_mask = decoder_attention_mask,
                            labels = labels)

        #last_hidden_states = outputs.last_hidden_state
        #loss = outputs.loss
        return outputs #last_hidden_states
    
    def generate(self, input_args, attention_masks):
        
        out_gen = self.model.generate(input_ids = input_args, attention_mask = attention_masks)
                           #, length_penalty=0.8, num_beams=8, max_length=128)
        
        return out_gen
        

        
def train(model, device, train_loader, optimizer, epochs, loss_function, scheduler, max_length, verbose=False):
    model.train()
    
    
    results = {'loss': torch.zeros([epochs,1]),
              'predicted': torch.zeros([epochs,len(train_loader.dataset), max_length]),
              'labels': torch.zeros([epochs,len(train_loader.dataset), max_length])
              }
    
    #results = {k:v.to(device) for k,v in results.items()}
    
    for epoch in range(0, epochs):
        
        epoch_results = {'loss': torch.zeros([len(train_loader),1]),
              'predicted': torch.zeros([len(train_loader.dataset), max_length]),
              'labels': torch.zeros([len(train_loader.dataset), max_length])
        }
        
        #epoch_results = {k:v.to(device) for k,v in epoch_results.items()}
        
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
            outs = model(input_ids, attention_mask, 
                         decoder_input_ids, decoder_attention_mask, 
                         labels)

            if loss_function is None:
                loss = outs.loss
            epoch_results['loss'][batch_idx] = loss.cpu()
            
            loss.backward()
                
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
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
            
            if verbose:
                if batch_idx % 10 == 0:
                    print(f'Train Epoch:', epoch, 'batch:',
                        batch_idx, 'loss:',
                        loss.mean())

                    
        results['loss'][epoch] = torch.mean(epoch_results['loss'], 0)
        results['predicted'][epoch] = epoch_results['predicted']        
        results['labels'][epoch] = epoch_results['labels']
                                 
    return results

def test(model, device, test_loader, max_length=100):
    
    model.eval()
    
    results = {
            'predicted': torch.zeros([1, len(test_loader.dataset), max_length]),
            'labels': torch.zeros([1, len(test_loader.dataset), max_length])
          }
    
    results = {k:v.to(device) for k,v in results.items()}
    
    
    with torch.no_grad():
        for batch_idx, (encodings) in enumerate(test_loader):

            # Extract arguments, key_points and labels all from the same batch
            input_ids = encodings['input_ids'].to(device)
            attention_mask = encodings['attention_mask'].to(device)
            
            labels = encodings['labels'].to(device)

            outp = model.generate(input_args = input_ids, attention_masks = attention_mask)
            print(labels.shape)
            print(batch_idx)
            results['labels'][batch_idx] = labels
            results['predicted'][batch_idx] = outp
            
    return results