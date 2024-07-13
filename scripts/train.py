import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.efficientnet import EfficientNetModel
from utils.training_utils import train, evaluate
import yaml
import json
import mlflow
import mlflow.pytorch

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def main():
    # Configurations
    num_classes = config['hyperparameters']['num_classes']
    learning_rate = config['hyperparameters']['learning_rate']
    batch_size = config['hyperparameters']['batch_size']
    num_epochs = config['hyperparameters']['num_epochs']

    # Data loading
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

    # Initialize the model, loss function, and optimizer
    model = EfficientNetModel(num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Prepare directories for results
    os.makedirs(config['results']['logs'], exist_ok=True)
    os.makedirs(config['results']['models'], exist_ok=True)
    os.makedirs(config['results']['metrics'], exist_ok=True)

    with mlflow.start_run():
        mlflow.log_params({
            'learning_rate': learning_rate,
            'batch_size': batch_size,
            'num_epochs': num_epochs
        })

        # Training loop
        metrics = {'train_loss': [], 'val_loss': []}
        for epoch in range(num_epochs):
            train_loss = train(model, train_loader, criterion, optimizer)
            val_loss = evaluate(model, val_loader, criterion)
            print(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}')
            
            # Log metrics to MLflow
            mlflow.log_metric('train_loss', train_loss, step=epoch)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            
            # Save losses to metrics dictionary
            metrics['train_loss'].append(train_loss)
            metrics['val_loss'].append(val_loss)

            # Save logs
            with open(os.path.join(config['results']['logs'], f'epoch_{epoch+1}.log'), 'w') as f:
                f.write(f'Epoch {epoch+1}, Train Loss: {train_loss}, Val Loss: {val_loss}\n')

        # Save the final model
        model_path = os.path.join(config['results']['models'], 'efficientnet.pth')
        torch.save(model.state_dict(), model_path)
        mlflow.pytorch.log_model(model, 'model')
        
        # Save metrics
        with open(os.path.join(config['results']['metrics'], 'metrics.json'), 'w') as f:
            json.dump(metrics, f)

if __name__ == '__main__':
    main()
