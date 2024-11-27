import torch
from torch import nn
from torch.nn import functional as F    


class CustomCNN(nn.Module):
    def __init__(self):
        super(CustomCNN, self).__init__()
        ##############################
        self.features = nn.Sequential(
            nn.Conv2d(3, 20, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(20),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(20, 35, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(35),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            nn.Conv2d(35, 60, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(60),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Dropout(0.20)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(60 * 4 * 4, 512),
            nn.ReLU(),
            nn.Dropout(0.45),
            nn.Linear(512, 10)
        )
        ##############################  
        # Define the convolutional layers
        
        pass

    def forward(self, x):
        ##############################
        x = self.features(x)
        x = x.view(-1, 60 * 4 * 4)
        x = self.classifier(x)
        ##############################  
        # Define the forward pass
        return x