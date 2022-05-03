## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self, dropout_p=0.2, n_classes=136):
        super(Net, self).__init__()
        
        self.dropout_p = dropout_p
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        # Conv, Activation, Maxpool, Dropout
        self.conv1 = nn.Conv2d(1, 32, 5)
        I.xavier_uniform_(self.conv1.weight)
        self.bn1 = nn.BatchNorm2d(32)
        self.drop1 = nn.Dropout(p=dropout_p)
        
        self.conv2 = nn.Conv2d(32, 64, 4)
        I.xavier_uniform_(self.conv2.weight)
        self.bn2 = nn.BatchNorm2d(64)
        self.drop2 = nn.Dropout(p=dropout_p)
        
        self.conv3 = nn.Conv2d(64, 128, 3)
        I.xavier_uniform_(self.conv3.weight)
        self.bn3 = nn.BatchNorm2d(128)
        self.drop3 = nn.Dropout(p=dropout_p)
        
        self.conv4 = nn.Conv2d(128, 256, 2)
        I.xavier_uniform_(self.conv4.weight)
        self.bn4 = nn.BatchNorm2d(256)
        self.drop4 = nn.Dropout(p=dropout_p)
        
        self.pool = nn.MaxPool2d(2, 2)

        self.fc1 = nn.Linear(36864, 10000)
        I.xavier_uniform_(self.fc1.weight)
        self.drop5 = nn.Dropout(p=dropout_p)
        self.fc2 = nn.Linear(10000, 1000)
        I.xavier_uniform_(self.fc2.weight)
        self.drop6 = nn.Dropout(p=dropout_p)
        self.fc3 = nn.Linear(1000, n_classes)
        I.xavier_uniform_(self.fc2.weight)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        x = self.conv1(x)
        #print(x.shape)
        x = self.drop1(self.bn1(self.pool(F.relu(x))))
        #print(x.shape)
        x = self.drop2(self.bn2(self.pool(F.relu(self.conv2(x)))))
        #print(x.shape)
        x = self.drop3(self.bn3(self.pool(F.relu(self.conv3(x)))))
        #print(x.shape)
        x = self.drop4(self.bn4(self.pool(F.relu(self.conv4(x)))))
        #print(x.shape)
        
        # Flatten
        x = x.view(x.size(0), -1)
        #print(x.shape)
        x = F.relu(self.fc1(x))
        x = self.drop5(x)
        x = F.relu(self.fc2(x))
        x = self.drop6(x)
        x = self.fc3(x)
        
        return x
