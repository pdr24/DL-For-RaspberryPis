import collect_data
import model_setup
import cifar10_setup

import sys 
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pandas as pd
import time 

def train(model_name, model, criterion, optimizer, epochs, train_loader, path):
    # Set the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    print(f"Device: {device}\n")

    # Train model on CIFAR-10
    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()

            start_time = time.time()

            # Handle auxiliary classifier in Inception
            if model_name == "inception":
                outputs, aux_outputs = model(images)  # Inception returns a tuple
                loss = criterion(outputs, labels) + 0.4 * criterion(aux_outputs, labels)  # Use both losses
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)

            end_time = time.time()

            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}")

    torch.save(model.state_dict(), path)
    print(f"Model saved as '{path}'")


if __name__ == "__main__":

    # check if command line arguments were valid 
    if len(sys.argv) != 3:
        print("Usage: python3 train.py [mobilenet|inception|resnet18|alexnet|vgg16] [path_to_save_model]")
        sys.exit(1)
    elif sys.argv[1].lower() not in ["mobilenet", "inception", "resnet18", "alexnet", "vgg16"]:
        print("Error: model name must be either 'mobilenet', 'inception', 'resnet18', 'alexnet', 'vgg16'")
        print("Usage: python3 train.py [mobilenet|inception|resnet18|alexnet|vgg16] [path_to_save_model]")
        sys.exit(1)

    # determine model to run training on 
    model_name = sys.argv[1]
    print(f"Model: {model_name}")

    # file to save model weights to 
    path = sys.argv[2]

    # setup model specified by the user 
    model = model_setup.setup_model(model_name)

    # setup data loaders (cifar-10) 
    train_loader, test_loader = cifar10_setup.setup_data(model_name)

    # model parameters
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10

    # create df to store data 
    system_metrics = collect_data.setup_df()
    pd.set_option('display.max_columns', None)  # Show all columns

    # train the model 
    train(model_name, model, criterion, optimizer, epochs, train_loader, path)

    