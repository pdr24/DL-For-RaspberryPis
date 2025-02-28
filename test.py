import collect_data
import model_setup
import cifar10_setup

import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import pandas as pd
import numpy as np
import time 
import os

def test(model, test_loader, device, system_metrics):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            start_time = time.time()
            outputs = model(images)
            end_time = time.time()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            inference_time = end_time - start_time
            cpu_percent, load_1min, load_5min, load_15min, cpu_freq_current = collect_data.collect_cpu_statistics()
            available_memory, used_memory, percent_used, active_memory, inactive_memory, buffers, cached, shared, swap_used, swap_free, swap_percent = collect_data.collect_memory_statistics()

            system_metrics = collect_data.add_metrics_to_df(system_metrics, inference_time, cpu_percent, load_1min, load_5min, load_15min, cpu_freq_current, available_memory, used_memory, percent_used, active_memory, inactive_memory, buffers, cached, shared, swap_used, swap_free, swap_percent)
            

    # Compute accuracy
    accuracy = 100.0 * correct / total
    print(f'Accuracy: {accuracy:.2f}%')

    return system_metrics, accuracy

if __name__ == "__main__":
    # check if command line arguments were valid 
    if len(sys.argv) != 5:
        print("Usage: python3 test.py [mobilenet|inception|resnet18|alexnet|vgg16] [path_to_saved_model] [path_to_save_results] [number_iterations]")
        sys.exit(1)
    elif sys.argv[1].lower() not in ["mobilenet", "inception", "resnet18", "alexnet", "vgg16"]:
        print("Error: model name must be either 'mobilenet', 'inception', 'resnet18', 'alexnet', 'vgg16'")
        print("Usage: python3 test.py [mobilenet|inception|resnet18|alexnet|vgg16] [path_to_saved_model] [path_to_save_results] [number_iterations]")
        sys.exit(1)

    # determine model to run training on 
    model_name = sys.argv[1]
    print(f"Model: {model_name}")

    # path the saved model weights are located at  
    path = sys.argv[2]

    # path to file to save results to
    results_path = sys.argv[3]

    # number of iterations to test 
    number_iterations = int(sys.argv[4])

    # set the device (GPU or CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n")

    # load model specified by the user 
    model = model_setup.load_saved_model(model_name, path, device)

    # setup data loaders (cifar-10) 
    number_test_samples = 500 # number of samples to test on 
    train_loader, test_loader = cifar10_setup.setup_data(model_name, number_test_samples)

    # create df to store data 
    system_metrics = collect_data.setup_df()
    pd.set_option('display.max_columns', None)  # Show all columns
    
    if number_iterations == 1:
        # test
        system_metrics, accuracy = test(model, test_loader, device, system_metrics)

        # save summary of collected system metrics 
        system_metrics.describe(include='all').to_csv(results_path)
    else:
        for i in range(number_iterations):
            # test
            system_metrics = collect_data.setup_df()
            system_metrics, accuracy = test(model, test_loader, device, system_metrics)

            # file name for specific iteration
            results_filename = os.path.basename(results_path)
            new_results_filename = str(i + 1) + "_" + results_filename
            directory_path = os.path.dirname(results_path)
            new_path = os.path.join(directory_path, new_results_filename)

             # save summary of collected system metrics 
            system_metrics.describe(include='all').to_csv(new_path)
