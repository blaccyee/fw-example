from time import time

import matplotlib.pyplot as plt
import torch
from tqdm import tqdm


def plot_accuracy(epsilons, accuracies, avg_time, attack_name):
    plt.figure(figsize=(15, 5))
    plt.plot(epsilons, accuracies, "o-")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title(f"{attack_name} Attack: Accuracy vs Epsilon")
    plt.text(0.0, -0.15, f"AVG Time: {avg_time}", ha='center', va='center', transform=plt.gca().transAxes)
    plt.xscale('log')
    plt.show()


def test_attack(model, attack, test_loader, epsilon, params=None):
    correct = 0
    total = 0
    start_time = time()

    for images, labels in test_loader:
        if params:  # FW manual implementations -- NOT USED
            _, pertrubed, _ = attack(model, images, labels, epsilon, params=params)
        else:  # foolbox implementations
            _, pertrubed, _ = attack(model, images, labels, epsilons=epsilon)
        outputs = model(pertrubed)
        _, predicted = torch.max(outputs.data, 1)

        original_outputs = model(images)
        _, predicted_orig = torch.max(original_outputs.data, 1)

        correct += (predicted == predicted_orig).sum().item()
        total += predicted_orig.size(0)

    end_time = time()
    accuracy = correct / total
    return accuracy, end_time - start_time


def attack_and_collect_stats(epsilons, model, attack, test_loader, attack_name, params=None, plot=True):
    accuracies = []
    times = []

    for epsilon in tqdm(epsilons):
        accuracy, elapsed_time = test_attack(model, attack, test_loader, epsilon, params)
        accuracies.append(accuracy)
        times.append(elapsed_time)

    avg_time = round(sum(times) / len(times), 4)
    if plot:
        plot_accuracy(epsilons, accuracies, avg_time, attack_name)

    return accuracies, avg_time
