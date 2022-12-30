import torch
from torch import nn
from transformers import BertModel
from sentence_transformers import util
from torch.optim.lr_scheduler import StepLR

class SiameseNetwork(nn.Module):
    """
        The network is composed of two identical networks, one for each input.
        The output of each network is concatenated and passed to a linear layer. 
        The output of the linear layer passed through a sigmoid function.
    """
    def __init__(self, bert_type=None, output_type=None):
        super(SiameseNetwork, self).__init__()

        if bert_type is None:
            self.model = BertModel.from_pretrained("bert-base-uncased",
                                          num_labels = 2)
        else:
            self.model = bert_type

        if output_type is None:
            self.output_type = util.cos_sim
        else:
            self.output_type = output_type
        
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
    
    def forward_once(self, input_ids, attention_masks, label):
        
        outputs = self.model(input_ids,
                            token_type_ids = None,
                            attention_mask = attention_masks)

        last_hidden_states = outputs.last_hidden_state

        return last_hidden_states

    def forward(self, input1, input2, label):
        # get two images' features
        output1 = self.forward_once(input1['input_ids'], input1['attention_masks'], label)
        output2 = self.forward_once(input2['input_ids'], input2['attention_masks'], label)

        # concatenate both images' features
        #output = torch.cat((output1, output2), 1)
        #output = []
        #for i in range(output1.shape[0]):
         # output.append(self.output_type(output1[i], output2[i]))


        # pass the concatenation to the linear layers
        #output = self.fc(output)

        # pass the out of the linear layers to sigmoid layer
        #output = self.sigmoid(output)
        
        return output1, output2

    def get_smart_batching_collate(self):
        return self.model.smart_batching_collate

def train(model, device, train_loader, loss_function, optimizer, epoch, scheduler):
    model.train()
    
    for batch_idx, (encodings) in enumerate(train_loader):
      #images_1, images_2, targets = images_1.to(device), images_2.to(device), targets.to(device)
      
      # Extract arguments, key_points and labels all from the same batch
        args = encodings['arg']
        kps = encodings['kp']
        labels = encodings['label']
        
        optimizer.zero_grad()
        output1, output2 = model(args, kps, labels)
        
        #print(output)
        '''new_shape = (output1.shape[0], output1.shape[1]* output1.shape[2])
        labels = [labels] #.shape[0],))
        print(labels.shape)
        output1 = output1.reshape(new_shape)
        output2 = output2.reshape(new_shape)'''
        loss = loss_function(output1, output2, labels)

      
        #loss = loss_function(tf.convert_to_tensor(labels.numpy()), tf.convert_to_tensor(outputs.numpy()))
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
      
        if batch_idx % 10 == 0:
            print(f'Train Epoch:', epoch, 'batch:',
                batch_idx, '/', len(train_loader.dataset), 'loss:',
                loss.item())