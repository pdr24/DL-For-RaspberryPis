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
    elif model_name == "resnet18":
        return setup_resnet18_model()
    elif model_name == "alexnet":
        return setup_alexnet_model()
    elif model_name == "vgg16":
        return setup_vgg16_model()
    elif model_name == "squeezenet":
        return setup_squeezenet_model()
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

# sets up resnet18 model for use with cifar-10 dataset 
def setup_resnet18_model():
    model = models.resnet18(pretrained=True)

    # modify to match cifar-10's 10 output classes 
    model.fc = nn.Linear(model.fc.in_features, 10)  

    return model

# sets up alexnet model for use with cifar-10 dataset 
def setup_alexnet_model():
    model = models.alexnet(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)  # Adjust for CIFAR-10 classes
    return model

# sets up vgg16 model for use with cifar10 dataset
def setup_vgg16_model():
    model = models.vgg16(pretrained=True)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)  # Adjust for CIFAR-10 classes
    return model

# sets up squeezenet model for use with cifar10 dataset
def setup_squeezenet_model():
    model = models.squeezenet1_1(pretrained=True)
    model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1))  # Adjust for CIFAR-10 classes
    model.num_classes = 10
    return model

def load_saved_model(model_name, model_path, device):
    if model_name == "mobilenet":
        return load_saved_mobilenet_model(model_path, device).to(device)
    elif model_name == "inception":
        return load_saved_inception_model(model_path, device).to(device)
    elif model_name == "resnet18":
        return load_saved_resnet18_model(model_path, device).to(device)
    elif model_name == "alexnet":
        return load_saved_alexnet_model(model_path, device).to(device)
    elif model_name == "vgg16":
        return load_saved_vgg16_model(model_path, device).to(device)
    elif model_name == "squeezenet"
        return load_saved_squeezenet_model(model_path, device).to(device)
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

def load_saved_resnet18_model(model_path="resnet18_cifar10.pth", device="cpu"):
    # initialize model architecture
    model = models.resnet18(weights=None)  
    model.fc = nn.Linear(model.fc.in_features, 10)  # Adjust for CIFAR-10 classes
    
    # load the saved state directory 
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    
    model.eval()  
    
    return model

def load_saved_alexnet_model(model_path="alexnet_cifar10.pth", device="cpu"):
    model = models.alexnet(weights=None)  # Initialize AlexNet without pretrained weights
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)  # Adjust for CIFAR-10 classes
    
    # Load model weights with correct device mapping
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    
    model.to(device)  # Move model to the specified device
    model.eval()  # Set model to evaluation mode
    
    return model

def load_saved_vgg16_model(model_path="vgg16_cifar10.pth", device="cpu"):
    model = models.vgg16(weights=None)  # Initialize VGG16 without pretrained weights
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, 10)  # Adjust for CIFAR-10 classes
    
    # Load model weights with correct device mapping
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    
    model.eval()  # Set model to evaluation mode
    
    return model

# loads squeezenet model based on saved trained model
def load_saved_squeezenet_model(model_path="squeezenet_cifar10.pth", device="cpu"):
    model = models.squeezenet1_1(weights=None)  # Initialize SqueezeNet without pretrained weights
    model.classifier[1] = nn.Conv2d(512, 10, kernel_size=(1, 1))  # Adjust for CIFAR-10 classes
    model.num_classes = 10
    
    # Load model weights with correct device mapping
    state_dict = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(state_dict)
    
    model.eval() 
    
    return model

