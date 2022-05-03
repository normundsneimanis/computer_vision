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
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.2):
        super(DecoderRNN, self).__init__()
        # Features: [batch, embed_size]
        # Captions have length [batch, cap_len]
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)
    
    def forward(self, features, captions):
        embeds = self.embed(captions[:,:-1])
        inputs = torch.cat((features.unsqueeze(1), embeds), 1)
        lstm_out, hidden = self.lstm(inputs, None)
        out = self.linear(lstm_out)
        return out

    def sample(self, embeddings, hidden_state=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sentence = []
        for _ in range(max_len): 
            out1, hidden_state = self.lstm(embeddings, hidden_state)
            out2 = self.linear(out1)
            out3 = out2.squeeze(1)
            word = out3.max(1)[1]
            sentence.append(word.item())
            embeddings = self.embed(word).unsqueeze(1)
        return sentence
