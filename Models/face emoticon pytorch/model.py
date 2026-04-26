import torch.nn as nn
from torchvision import models

EMOTIONS = ['angry', 'disgust', 'fear', 'happy', 'sad', 'surprise', 'neutral']

def build_model():
    model = models.resnet18(pretrained=True)

    # Replace final layer for 7-class output
    model.fc = nn.Linear(model.fc.in_features, len(EMOTIONS))
    return model