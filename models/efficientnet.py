import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet

class EfficientNetModel(nn.Module):
    def __init__(self, num_classes):
        super(EfficientNetModel, self).__init__()
        # Load the pretrained EfficientNet model
        self.model = EfficientNet.from_pretrained('efficientnet-b0')
        # Replace the final fully connected layer with a new one for our specific task
        self.fc = nn.Linear(self.model._fc.in_features, num_classes)
        self.model._fc = self.fc

    def forward(self, x):
        return self.model(x)
