import time
import numpy as np
import torch
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Constants
IMAGE_WIDTH = 32
IMAGE_HEIGHT = 32
COLOR_CHANNELS = 3
EPOCHS = 300
LEARNING_RATES = [.00001, 0.0001, 0.001, 0.01, 0.1]
KEEP_RATES = [.5, .65, .8]
MOMENTUM_RATES = [.25, .5, .75]
WEIGHT_DECAY_RATES = [.0005, .005, .05]
BATCH_SIZE = 32
BATCH_IMAGE_COUNT = 10000
TRAIN_BATCHS = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4"]
TEST_BATCHES = ["data_batch_5"]
CLASSES = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
N_CLASSES = len(CLASSES)
PLOT = False



class MLP(torch.nn.Module):
    def __init__(self, n_hidden_nodes=100, n_hidden_layers=2, activation="relu", keep_rate=0):
        super(MLP, self).__init__()
        self.n_hidden_nodes = n_hidden_nodes
        self.n_hidden_layers = n_hidden_layers
        self.activation = activation
        if not keep_rate:
            keep_rate = 0.5
        self.keep_rate = keep_rate
        # Set up perceptron layers and add dropout
        self.fc1 = torch.nn.Linear(IMAGE_WIDTH * IMAGE_WIDTH * COLOR_CHANNELS,
                                   n_hidden_nodes)
        self.fc1_drop = torch.nn.Dropout(1 - keep_rate)
        if n_hidden_layers == 2:
            self.fc2 = torch.nn.Linear(n_hidden_nodes,
                                       n_hidden_nodes)
            self.fc2_drop = torch.nn.Dropout(1 - keep_rate)

        self.out = torch.nn.Linear(n_hidden_nodes, N_CLASSES)

    def forward(self, x):
        x = x.view(-1, IMAGE_WIDTH * IMAGE_WIDTH * COLOR_CHANNELS)
        if self.activation == "sigmoid":
            sigmoid = torch.nn.Sigmoid()
            x = sigmoid(self.fc1(x))
        elif self.activation == "relu":
            x = torch.nn.functional.relu(self.fc1(x))
        x = self.fc1_drop(x)
        if self.n_hidden_layers == 2:
            if self.activation == "sigmoid":
                x = sigmoid(self.fc2(x))
            elif self.activation == "relu":
                x = torch.nn.functional.relu(self.fc2(x))
            x = self.fc2_drop(x)
        return torch.nn.functional.log_softmax(self.out(x))
