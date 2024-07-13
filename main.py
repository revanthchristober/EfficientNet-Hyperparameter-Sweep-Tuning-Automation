import yaml
from tqdm import tqdm
from scripts.data_preparation import prepare_data
from scripts.train import main as train_main
from scripts.hyperparameter_sweep import objective
import optuna
import os
import json

# Load configuration from config.yaml file
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

def run_data_preparation():
    """
    Function to prepare the data for training.
    This function calls the prepare_data function from the data_preparation script.
    """
    print("Preparing data...")
    prepare_data()
    print("Data preparation done.")

def run_training():
    """
    Function to train the model.
    This function calls the main function from the train script.
    """
    print("Training model...")
    train_main()
    print("Model training done.")

def run_hyperparameter_sweep():
    """
    Function to perform hyperparameter sweep using Optuna.
    This function creates a study, defines the objective function with progress tracking,
    and optimizes the hyperparameters.
    """
    print("Starting hyperparameter sweep...")
    
    def objective_with_tqdm(trial):
        """
        Wrapper for the objective function to add progress tracking using tqdm.
        """
        with tqdm(total=100, desc="Hyperparameter Sweep") as pbar:
            for i in range(100):
                result = objective(trial)
                pbar.update(1)
                if trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
            return result

    # Create a study and optimize the hyperparameters
    study = optuna.create_study(direction='minimize')
    study.optimize(objective_with_tqdm, n_trials=100)

    # Save the best hyperparameters found during the study to a JSON file
    os.makedirs(config['results']['logs'], exist_ok=True)
    with open(os.path.join(config['results']['logs'], 'study_results.json'), 'w') as f:
        json.dump(study.best_params, f)
        
    print("Best hyperparameters found: ", study.best_params)

if __name__ == '__main__':
    """
    Main function to run the full pipeline: data preparation, model training, and hyperparameter sweep.
    Progress is tracked using tqdm.
    """
    print("Running the full pipeline...")
    with tqdm(total=3, desc="Pipeline") as pbar:
        run_data_preparation()  # Step 1: Data preparation
        pbar.update(1)
        
        run_training()  # Step 2: Model training
        pbar.update(1)
        
        run_hyperparameter_sweep()  # Step 3: Hyperparameter sweep
        pbar.update(1)
    print("Pipeline completed.")
