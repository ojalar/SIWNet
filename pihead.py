import torch
import torch.nn as nn
from torch import Tensor

class PIHead(nn.Module):
    # implementation of the prediction interval head
    def __init__(self, input_size):
        super().__init__()
        # create the prediction interval head layers
        self.net = nn.Sequential(
                nn.Linear(input_size, 128),
                nn.ReLU(),
                nn.Linear(128, 512),
                nn.ReLU(),
                nn.Linear(512, 1),
                nn.Sigmoid()
                )
    
    def forward(self, x):
        return self.net(x)

