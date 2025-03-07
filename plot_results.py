import pandas as pd
import matplotlib.pyplot as plt
import glob
import sys

# Get a list of all CSV files 
csv_files = glob.glob("results/alexnet_results/*_results_*.csv")  
print(csv_files)

df = pd.read_csv(csv_files[0], index_col=0)  # Read CSV with first column as index

for column in df.columns:
    if column != '':
        avgs = []
        for file in sorted(csv_files):  # Sorting ensures correct order
            df = pd.read_csv(file, index_col=0)  # Read CSV with first column as index
            avg = df.loc["mean", column] 
            avgs.append(avg)

        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(avgs) + 1), avgs, marker='o', linestyle='-')
        plt.xlabel("Iteration")
        plt.ylabel(column)
        plt.title(f"{column} Across Iterations")
        plt.grid(True)

        if len(avgs) > 0:
            aggregate_avg = sum(avgs) / len(avgs)
        else:
            aggregate_avg = "Missing"
        print(f"Average {column} across iterations: {aggregate_avg:.3f}")

'''
# commenting out for now as all accuracies were the same for resnet18, inception
with open("mobilenetv2/results/copy_accuracies.txt", "r") as file:
    accuracies = []
    for line in file:
        accuracies.append(float(line.strip()))

    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(accuracies) + 1), accuracies, marker='o', linestyle='-')
    plt.xlabel("Iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Across Iterations")
    plt.grid(True)

    if len(avgs) > 0:
        aggregate_avg = sum(accuracies) / len(accuracies)
    else:
        aggregate_avg = "Missing"
    print(f"Average Accuracy across iterations: {aggregate_avg:.3f}")  
'''

plt.show()
