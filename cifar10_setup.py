import model_setup

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import torchvision.models as models

def setup_data(model_name):
    preprocess = return_preprocess(model_name)
    return setup_train_data_loader(preprocess), setup_test_data_loader(preprocess)

def setup_train_data_loader(model_name):
    preprocess = return_preprocess(model_name)

    # load cifar-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=preprocess)
    train_loader = DataLoader(trainset, batch_size=16, shuffle=True)

    return train_loader

def setup_test_data_loader(model_name):
    preprocess = return_preprocess(model_name)

    # load cifar-10 dataset
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=preprocess)
    test_loader = DataLoader(testset, batch_size=16, shuffle=False) 

    return test_loader


# determines and returns preprocess steps based on model selected by the user 
def return_preprocess(model_name):
    if model_name == "mobilenet":
        preprocess = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization (mobilenetv2 was pretrained on ImageNet)
            ])
    elif model_name == "inception":
        preprocess = transforms.Compose([
            transforms.Resize((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
    return preprocess