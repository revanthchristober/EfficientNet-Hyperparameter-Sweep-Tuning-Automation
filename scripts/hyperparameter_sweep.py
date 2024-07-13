import optuna
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from models.efficientnet import EfficientNetModel
from utils.data_loader import load_data
from utils.training_utils import train, evaluate
import yaml
import mlflow
import mlflow.pytorch

# Load configuration
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def objective(trial):
    # Hyperparameters to tune
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
    batch_size = trial.suggest_int('batch_size', 32, 128)
    num_classes = config['hyperparameters']['num_classes']

    # Initialize the model, loss function, and optimizer
    model = EfficientNetModel(num_classes)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Load training and validation datasets
    train_loader, val_loader = load_data(batch_size)

    with mlflow.start_run():
        mlflow.log_params({'learning_rate': learning_rate, 'batch_size': batch_size})
        
        for epoch in range(10):  # Dummy epochs
            train(model, train_loader, criterion, optimizer)
            val_loss = evaluate(model, val_loader, criterion)
            mlflow.log_metric('val_loss', val_loss, step=epoch)
            trial.report(val_loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()
        
        # Log the model to MLflow
        mlflow.pytorch.log_model(model, 'model')

    return val_loss

if __name__ == '__main__':
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)
    print(study.best_params)
