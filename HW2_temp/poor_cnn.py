import torch.nn as nn
import torch

class PoorPerformingCNN(nn.Module):
    def __init__(self):
        super(PoorPerformingCNN, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.relu2 = nn.ReLU()

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)

        self.fc1 = nn.Linear(8 * 8 * 128, 256)
        self.fc2 = nn.Linear(256, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        # First conv + pooling layer
        x = self.relu1(self.conv1(x))

        # Second conv + pooling layer
        x = self.pool(self.relu2(self.conv2(x)))

        x = self.pool(self.relu2(self.conv3(x)))

        # Flatten layer
        x = x.view(-1, 8 * 8 * 128)

        # Linears
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        return x