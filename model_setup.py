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

def load_saved_model(model_name, model_path, device):
    if model_name == "mobilenet":
        return load_saved_mobilenet_model(model_path, device).to(device)
    elif model_name == "inception":
        return load_saved_inception_model(model_path, device).to(device)
    else:
        print(f"Error: invalid model name {model_name} received by load_saved_model() in model_setup.py")

def load_saved_mobilenet_model(model_path="mobilenetv2_cifar10.pth", device="cpu"):
    # initialize the model architecture
    model = models.mobilenet_v2(weights=None)  
    model.classifier[1] = nn.Linear(model.last_channel, 10)  # adjust for cifar-10

    # load the saved state dictionary, ensuring it is mapped to the correct device
    state_dict = torch.load(model_path, map_location=torch.device(device), weights_only=True)
    model.load_state_dict(state_dict)

    # set the model to evaluation mode
    model.eval()
    
    return model

def load_saved_inception_model(model_path="inception_cifar10.pth", device="cpu"):
    # initialize model architecture 
    model = models.inception_v3(weights=None, aux_logits=True)
    model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for CIFAR-10 classes
    
    # load the saved state directory 
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)

    # set the model to evaluation mode
    model.eval()
    
    return model