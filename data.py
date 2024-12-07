import os
import numpy as np

import torch
from torch.utils.data import  Dataset, Subset
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, Food101, StanfordCars, Flowers102

class Cifar100(CIFAR100):
    def __init__(self, root, train=False, augment=False, download=False) -> None:
        if augment:
            transform = transforms.Compose([
                    transforms.RandomResizedCrop(size=(224, 224)),
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
            ]) 
        else:
            transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.CenterCrop((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5071, 0.4865, 0.4409], std=[0.2673, 0.2564, 0.2762]),
            ])
        super().__init__(root, train, transform, None, download)
        self.data = np.array(self.data)
        self.targets = np.array(self.targets)
        self.num_classes = 100