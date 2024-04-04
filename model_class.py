import torch
import torchvision
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torchvision.io import read_image
from torchvision.transforms import ToTensor, Lambda
import torchvision.models as models
from torch import nn
import torch.nn.functional as F
import torch.optim as opt

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import os

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()

        # convolution layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 5)

        # convolution layer with 20 5x5 filters
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 5)

        # drop layer, rate = 50%
        self.conv2_drop = nn.Dropout2d(p = 0.50)

        # fully connected layer with 50 nodes
        self.fc1 = nn.Linear(320, 50)

        # fully connected layer with 10 nodes
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):

        # max pooling with reLU, 2x2
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # max pooling with reLU 2x2
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # flattening operation
        x = x.view(-1, 320)

        # reLU on output of fully connected layer
        x = F.relu(self.fc1(x))

        # final fully connected later
        x = self.fc2(x)

        # apply log softmax
        return F.log_softmax(x, dim = 1)

    
class NeuralNetwork_2(nn.Module):
    def __init__(self):
        super(NeuralNetwork_2, self).__init__()

        # convolution layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5, padding = 2)

        # convolution layer with 10 5x5 filters 
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5, padding = 2)  

        # convolution layer with 10 5x5 filters 
        self.conv3 = nn.Conv2d(20, 40, kernel_size=5, padding = 2)  

        # drop layer, rate = 50%
        self.conv2_drop = nn.Dropout2d(p = 0.50)

        # drop layer, rate = 50%
        self.conv3_drop = nn.Dropout2d(p = 0.50)  

        # fully connected layer with 50 nodes
        self.fc1 = nn.Linear(40*7*7, 50)  

        # fully connected layer with 10 nodes
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):

        # max pooling with reLU, 2x2
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # max pooling with reLU, 2x2
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # reLU for layer 3
        x = F.relu(self.conv3_drop(self.conv3(x)))
        
        # flattening operation
        x = x.view(-1, 40*7*7) 

        # reLU on output of fully connected layer
        x = F.relu(self.fc1(x))

        # final fully connected later
        x = self.fc2(x)

        # apply log softmax
        return F.log_softmax(x, dim=1)

class NeuralNetwork_3(nn.Module):
    def __init__(self):
        super(NeuralNetwork_3, self).__init__()

        # convolution layer with 10 5x5 filters
        self.conv1 = nn.Conv2d(1, 10, kernel_size = 3, padding = 1)

        # convolution layer with 10 5x5 filters 
        self.conv2 = nn.Conv2d(10, 20, kernel_size = 3, padding = 1)  

        # convolution layer with 10 5x5 filters 
        self.conv3 = nn.Conv2d(20, 40, kernel_size = 3, padding = 1)  

        # drop layer, rate = 50%
        self.conv2_drop = nn.Dropout2d(p = 0.50)

        # drop layer, rate = 50%
        self.conv3_drop = nn.Dropout2d(p = 0.50)  

        # fully connected layer with 50 nodes
        self.fc1 = nn.Linear(40*7*7, 50)  

        # fully connected layer with 10 nodes
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):

        # max pooling with reLU, 2x2
        x = F.relu(F.max_pool2d(self.conv1(x), 2))

        # max pooling with reLU, 2x2
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))

        # reLU for layer 3
        x = F.relu(self.conv3_drop(self.conv3(x)))
        
        # flattening operation
        x = x.view(-1, 40*7*7) 

        # reLU on output of fully connected layer
        x = F.relu(self.fc1(x))

        # final fully connected later
        x = self.fc2(x)

        # apply log softmax
        return F.log_softmax(x, dim=1)

    

