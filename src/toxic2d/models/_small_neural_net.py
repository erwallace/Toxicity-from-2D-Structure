import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallNeuralNet(nn.Module):
    def __init__(self, input_size=125):
        super(SmallNeuralNet, self).__init__()
        self.fc1 = nn.Linear(input_size, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        return x
