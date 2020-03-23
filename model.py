import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        # Embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        # LSTM unit with input -> embedded vector, output -> hidden state
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # Fully connected output layer
        self.fc = nn.Linear(hidden_size, vocab_size)
        
    
    def forward(self, features, captions):
        
        embed_captions = self.embedding(captions[:,:-1])
        lstm_input = torch.cat((features.unsqueeze(dim=1), embed_captions), dim=1)
        lstm_output,_ = self.lstm(lstm_input)
        output = self.fc(lstm_output)
        
        return output

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        
        output = []
        i = 0
        while i < max_len:
            lstm_out, states = self.lstm(inputs, states)
            out = self.fc(lstm_out)
            #print(f'out {out.shape})

            _, word = out.max(2)
            output.append(word.item())
            
            # input for next iteration
            inputs = self.embedding(word)
            
            i += 1
        
        return output