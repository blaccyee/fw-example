import random

import torch
import torchvision
import torch.nn as nn
import torchvision.transforms as transforms

from src.params import RANDOM_SEED, USE_MANUAL_INDICES_PERMUTATION, PRESELECTED_INDICES_5


# Load pre-trained model for CIFAR10 dataset from: https://github.com/chenyaofo/pytorch-cifar-models
# The model is a ResNet20 model trained on CIFAR10 dataset
class ResNet20():
    def __init__(self):
        self.model = torch.hub.load(
            "chenyaofo/pytorch-cifar-models", "cifar10_resnet20", pretrained=True, verbose=False
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        self.model.eval()

        self.classes = [
            "airplane",
            "automobile",
            "bird",
            "cat",
            "deer",
            "dog",
            "frog",
            "horse",
            "ship",
            "truck",
        ]
        self.num_classes = len(self.classes)

        self.loss = nn.CrossEntropyLoss()

        self.transform = transforms.Compose([transforms.ToTensor()])

    def get_model(self):
        return self.model

    def make_test_loader_subset(self, dataset_size=5, seed=RANDOM_SEED, use_manual_indices=True):
        random.seed(seed)
        torch.manual_seed(seed)  # doesn't seem to work on mac

        # Load the full CIFAR10 test dataset
        full_test_dataset = torchvision.datasets.CIFAR10(
            root="./data", train=False, download=True, transform=self.transform
        )

        if USE_MANUAL_INDICES_PERMUTATION and use_manual_indices and dataset_size == 5:
            # Use manually selected indices for subset
            indices = PRESELECTED_INDICES_5
        else:
            # Randomly select indices for subset
            indices = torch.randperm(len(full_test_dataset)).tolist()[:dataset_size]

        # Create a subset of the test dataset using the selected indices
        test_dataset = torch.utils.data.Subset(full_test_dataset, indices)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

        return test_loader
