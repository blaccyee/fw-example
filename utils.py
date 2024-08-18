from time import time
from IPython.display import clear_output

import matplotlib.pyplot as plt
import torch


def plot_accuracy(epsilons, accuracies, avg_time, attack_name):
    plt.figure(figsize=(15, 5))
    plt.plot(epsilons, accuracies, "o-")
    plt.xlabel("Epsilon")
    plt.ylabel("Accuracy")
    plt.title(f"{attack_name} Attack: Accuracy vs Epsilon")
    plt.text(0.0, -0.15, f"AVG Time: {avg_time}", ha='center', va='center', transform=plt.gca().transAxes)
    plt.xscale('log')
    plt.show()


def calculate_avg_time(times):
    return round(sum(times) / len(times), 4)


def test_attack(model, attack, test_loader, epsilon, params=None):
    correct = 0
    total = 0
    start_time = time()

    for images, labels in test_loader:
        if params:  # FW manual implementations
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


def attack_and_collect_stats(epsilons, model, attack, test_loader, attack_name, params=None):
    accuracies = []
    times = []

    for i, epsilon in enumerate(epsilons):
        print("Progress:", i + 1, "/", len(epsilons))
        accuracy, elapsed_time = test_attack(model, attack, test_loader, epsilon, params)
        accuracies.append(accuracy)
        times.append(elapsed_time)

    clear_output(wait=False)

    avg_time = calculate_avg_time(times)
    plot_accuracy(epsilons, accuracies, avg_time, attack_name)

    return accuracies, avg_time
