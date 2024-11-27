import torch.nn as nn
import torch

class PoorPerformingCNN(nn.Module):
    def __init__(self):
        super(PoorPerformingCNN, self).__init__()
        ##############################
        ###     CHANGE THIS CODE   ###
        ##############################  
        self.conv1 = nn.Conv2d(3, 4, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.conv2 = nn.Conv2d(4, 8, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        # After two max pooling layers, the size will be (8, 8, 8)
        # So the flattened size is 8 * 8 * 8 = 512
        # Also CIFAR-10 only needs 10 outputs
        self.fc1 = nn.Linear(8 * 8 * 8, 10)

    def forward(self, x):
        x = self.pool(self.relu1(self.conv1(x)))
        x = self.pool(self.relu2(self.conv2(x)))
        x = x.view(-1, 8 * 8 * 8)
        x = self.fc1(x)
        return x