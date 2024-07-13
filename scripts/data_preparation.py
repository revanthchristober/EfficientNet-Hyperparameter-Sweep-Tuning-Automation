import os
import shutil
from torchvision import datasets, transforms
import yaml

# Load configuration
with open('../config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def prepare_data():
    # Define transformation steps for data preprocessing
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    data_dir = config['data']['raw']
    processed_data_dir = config['data']['processed']

    # Download and transform the CIFAR10 dataset
    train_dataset = datasets.CIFAR10(root=data_dir, train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=transform)

    if not os.path.exists(processed_data_dir):
        os.makedirs(processed_data_dir)

    # Move the raw data to the processed directory
    shutil.move(os.path.join(data_dir, 'cifar-10-batches-py'), os.path.join(processed_data_dir, 'cifar-10-batches-py'))

if __name__ == '__main__':
    prepare_data()
