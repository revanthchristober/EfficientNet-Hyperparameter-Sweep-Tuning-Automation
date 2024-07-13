import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import yaml

# Load configuration
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def load_data(batch_size):
    # Define transformation steps for data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # Load training and validation datasets
    train_dataset = datasets.CIFAR10(root=config['data']['processed'], train=True, download=True, transform=transform)
    val_dataset = datasets.CIFAR10(root=config['data']['processed'], train=False, download=True, transform=transform)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader
