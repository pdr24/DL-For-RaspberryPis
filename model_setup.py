import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchvision.models as models

# sets up the model specified by the user
def setup_model(model_name):
    if model_name == "mobilenet":
        return setup_mobilenet_model()
    elif model_name == "inception":
        return setup_inception_model()
    else:
        print("Error: model not detected")
        sys.exit(1)

# sets up mobilenetv2 model for use with cifar-10 dataset 
def setup_mobilenet_model():
    # load the pretrained mobilenetv2 model
    model = models.mobilenet_v2(pretrained=True)

    # modify the model to match cifar-10's 10 output classes
    model.classifier[1] = nn.Linear(model.last_channel, 10)

    return model

# sets up inceptionv3 model for use with cifar-10 dataset
def setup_inception_model():
    model = models.inception_v3(pretrained=True, aux_logits=True)
    
    # modify the model to match cifar-10's 10 output classes
    model.fc = nn.Linear(model.fc.in_features, 10)

    return model
