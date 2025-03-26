import medmnist
from medmnist import BloodMNIST, INFO, Evaluator
import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms


data_flag = 'pathmnist'
download = True

NUM_EPOCHS = 3
BATCH_SIZE = 128
lr = 0.001

info = INFO[data_flag]
task = info['task']
n_channels = info['n_channels']
n_classes = len(info['label'])

DataClass = getattr(medmnist, info['python_class'])

"""## First, we read the MedMNIST data, preprocess them and encapsulate them into dataloader form."""

# preprocessing
data_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[.5], std=[.5])
])

# load the data
train_dataset = DataClass(split='train', transform=data_transform, download=download)
test_dataset = DataClass(split='test', transform=data_transform, download=download)

