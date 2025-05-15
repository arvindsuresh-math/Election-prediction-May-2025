import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import optuna
import os
import joblib
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, List, Tuple
from data_handling import DataHandler # for typing

# --- Device selection ---
if torch.backends.mps.is_available() and torch.backends.mps.is_built():
    DEVICE = torch.device("mps")
    print("Using MPS device (Apple Silicon GPU)")
elif torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using CUDA device (NVIDIA GPU)")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU device")

############################################

def weighted_cross_entropy_loss(outputs: torch.Tensor,
                                targets: torch.Tensor,
                                weights: torch.Tensor):
    """
    Calculates a custom weighted cross-entropy loss.
    Handles both standard batch inputs (2D tensors) and fold-batched inputs (3D tensors).
    
    For each sample (or sample within a fold):
    1. Scales the model outputs by the sum of the target probabilities for that sample (P(18plus)).
       This is because the targets are soft labels representing a subset of classes.
    2. Computes the cross-entropy: Sample_CE_Loss = - sum_k ( target_k * log(scaled_output_k) ).
    3. Computes the batch loss as the weighted average of these sample CE losses.
    The batch loss is the weighted average of these sample CE losses.
    If inputs are fold-batched (3D), it computes the sum of the losses per fold.
    and then returns the mean of these per-fold losses.

    Args:
        outputs (torch.Tensor): Model predictions (probabilities).
                                Shape: [batch_size, num_classes] or [num_folds, batch_size_per_fold, num_classes].
        targets (torch.Tensor): Ground truth probabilities.
                                Shape: [batch_size, num_classes] or [num_folds, batch_size_per_fold, num_classes].
        weights (torch.Tensor): Sample weights ('P(C)').
                                Shape: [batch_size, 1] or [num_folds, batch_size_per_fold, 1].

    Returns:
        torch.Tensor: Scalar tensor representing the final loss.
    """
    # Ensure 3D for unified processing
    if outputs.ndim == 2:
        outputs = outputs.unsqueeze(0)
        targets = targets.unsqueeze(0)
        weights = weights.unsqueeze(0)

    # Scale outputs by P(18plus) and clamp to avoid log(0)
    tots = targets.sum(dim=-1, keepdim=True) # Shape: (K, B, 1)
    outputs = outputs * tots
    outputs = torch.clamp(outputs, 1e-10, 1. - 1e-10) 

    # Tensors of shape (K, B, 1)
    sample_ce_loss = -torch.sum(targets * torch.log(outputs), dim=-1, keepdim=True)
    weights_reshaped = weights.view_as(sample_ce_loss) 
    weighted_sample_ce_losses = sample_ce_loss * weights_reshaped

    # Tensors of shape (K, 1, 1)
    sum_weighted_losses_fold = weighted_sample_ce_losses.sum(dim=1, keepdim=True)
    sum_weights_fold = weights_reshaped.sum(dim=1, keepdim=True)
    loss_per_fold = sum_weighted_losses_fold / sum_weights_fold
    
    return loss_per_fold.sum()

class BatchedLinear(nn.Module):
    """
    Batched version of `nn.Linear`; handles `K`-fold batched inputs. It performs `K` independent linear transformations (one per fold) using batched matrix multiplication (`torch.bmm`). 

    Weights shape: `(K, out_features, in_features)`. 
    Biases shape: `(K, out_features)`.
    """
    def __init__(self, K: int, in_features: int, out_features: int, bias: bool = True):
        """Initialize K parallel linear layers with shared parameters structure."""
        super().__init__()
        self.K = K
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(K, out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(K, out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize (fold-by-fold) the weights using Kaiming uniform and the biases within calculated bounds."""
        for k in range(self.K): 
            init.kaiming_uniform_(self.weight[k], mode='fan_in', nonlinearity='relu')
            if self.bias is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight[k])
                bound = 1 / (fan_in**0.5) if fan_in > 0 else 0
                init.uniform_(self.bias[k], -bound, bound)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply `K` parallel transformations to `K` batches. 
        Input shape: `(K, batch_size, in_features)`. 
        Output shape: `(K, batch_size, out_features)`.
        """
        output = torch.bmm(x, self.weight.transpose(1, 2))
        if self.bias is not None:
            output = output + self.bias.unsqueeze(1)
        return output

    def extra_repr(self) -> str:
        """Return string representation of layer parameters."""
        return f'K={self.K}, in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}'

def build_network(input_dim: int, depth: int, hparams: Dict[str, Any], batched=False):
    """Returns an MLP (nn.Sequential) with the specified input dimension, depth, and hyperparameters. If batched=True, uses BatchedLinear layers; otherwise, uses standard nn.Linear layers."""
    layers = []
    current_dim = input_dim

    for i in range(1, depth + 1):
        n_hidden = hparams[f"n_hidden_{i}"]
        if batched:
            layers.append(BatchedLinear(3, current_dim, n_hidden))
        else:
            layers.append(nn.Linear(current_dim, n_hidden))
        layers.append(nn.ReLU()) 

        dropout_rate = hparams.get(f"dropout_rate_{i}", 0.0)
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        current_dim = n_hidden

    layers.append(nn.Linear(current_dim, 4))
    layers.append(nn.Softmax(dim=-1))
    return nn.Sequential(*layers)

def objective_mlp(trial: optuna.trial.Trial, 
                  input_dim: int,
                  depth: int, 
                  cv_dataloaders: List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
                  max_epochs: int = 100):
    """
    Optuna objective function for BatchedMLP hyperparameter optimization.
    
    Args:
        trial: Current Optuna trial
        input_dim: Set this as `dh.input_dim`
        depth: Number of hidden layers
        dataloaders: List of `(train_loader, val_loader)` tuples. Set this as `dh.get_mlp_data('cv', batch_size)`
        max_epochs: Maximum training epochs
        
    Returns:
        Best validation loss achieved
    """
    
    params = {}
    params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    for i in range(1, depth + 1):
        params[f'n_hidden_{i}'] = trial.suggest_int(f'n_hidden_{i}', 8, 128, step=8)
        params[f'dropout_rate_{i}'] = trial.suggest_float(f'dropout_rate_{i}', 0.0, 0.5, log=False)

    model = build_network(input_dim, depth, params, batched=True).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(),
                             lr=params['learning_rate'],
                             weight_decay=params['weight_decay'])

    train_loader_list = [dl_pair[0] for dl_pair in cv_dataloaders]
    val_loader_list = [dl_pair[1] for dl_pair in cv_dataloaders]
    num_batches = len(val_loader_list[0])

    best_epoch_loss = float('inf')      
    for epoch in range(max_epochs):
        model.train()
        for batched_train_data in zip(*train_loader_list):
            stacked_features = torch.stack([data_fold[0] for data_fold in batched_train_data], dim=0).to(DEVICE)
            stacked_targets  = torch.stack([data_fold[1] for data_fold in batched_train_data], dim=0).to(DEVICE)
            stacked_weights  = torch.stack([data_fold[2] for data_fold in batched_train_data], dim=0).to(DEVICE)
            
            optimizer.zero_grad()
            outputs = model(stacked_features)
            loss = weighted_cross_entropy_loss(outputs, stacked_targets, stacked_weights)
            loss.backward()
            optimizer.step()
            
        model.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for batched_val_data in zip(*val_loader_list):
                stacked_features = torch.stack([data_fold[0] for data_fold in batched_val_data], dim=0).to(DEVICE)
                stacked_targets  = torch.stack([data_fold[1] for data_fold in batched_val_data], dim=0).to(DEVICE)
                stacked_weights  = torch.stack([data_fold[2] for data_fold in batched_val_data], dim=0).to(DEVICE)
                
                outputs = model(stacked_features)
                loss = weighted_cross_entropy_loss(outputs, stacked_targets, stacked_weights)
                epoch_loss += loss.item()
        
        epoch_loss = epoch_loss / (3 * num_batches)
        best_epoch_loss = min(best_epoch_loss, epoch_loss)
        
        trial.report(epoch_loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_epoch_loss

class MLPModel:
    """
    Handler for training, tuning, and using Multi-Layer Perceptron (MLP) models.

    Args:
        model_name (str): Unique name for this model (used for saving files).
        depth (int): Number of hidden layers in the MLP.

    After initialization, use:
    - run_optuna_study(...) to tune hyperparameters
    - train_final_model(...) to train the model
    - make_final_predictions(...) to get predictions on the test set

    Typical usage in a Jupyter notebook:
    ```python
    mlp = MLPModel(model_name="mlp1", depth=1)
    mlp.run_optuna_study(dh, batch_size=64, min_resource=5, reduction_factor=2, n_trials=30, timeout=15, max_epochs=50)
    mlp.train_final_model(dh, batch_size=64)
    preds = mlp.make_final_predictions(dh, save=True)
    ```
    """

    def __init__(self, model_name: str, depth: int):
        self.model_name = model_name
        self.depth = depth

        # Set after running/loading an optuna study
        self.best_params = None
        self.best_epoch = None

        # Set after final training, or loading state dict
        self.model = None
        self.final_train_history = []

    def run_optuna_study(self,
                         dh: DataHandler,
                         batch_size: int,
                         min_resource: int,
                         reduction_factor: int,
                         n_trials: int,
                         timeout: int, #in minutes
                         max_epochs: int,
                         save: bool = True):
        """Runs an optuna study for hyperparameter tuning using the ASHA pruning algorithm. Updates the best hyperparameters and best epoch after running the study. If `save` is `True`, then it saves the study to file and updates the study filepath."""

        # Set up the ASHA pruner
        asha_pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=min_resource,
            reduction_factor=reduction_factor,
            min_early_stopping_rate=0
            )
        
        study_name = f"{self.model_name}_study"
        study = optuna.create_study(
            study_name=study_name,
            direction='minimize',
            pruner=asha_pruner
            )
        
        print(f"Running Optuna study for {self.model_name} with depth {self.depth}.")
        print(f'Min resource: {min_resource}, reduction factor: {reduction_factor}, n_trials: {n_trials}, timeout: {timeout} seconds')
        print(f"Batch size: {batch_size}, max epochs: {max_epochs}")
        print('-' * 20)

        study.optimize(
            lambda trial: objective_mlp(
                trial,
                input_dim=dh.input_dim,
                depth=self.depth,
                dataloaders=dh.get_mlp_data('cv', batch_size),
                max_epochs=max_epochs
                ),
            n_trials=n_trials,
            timeout=timeout * 60,
            n_jobs=-1
            )
        
        best_trial = study.best_trial
        intermediate_vals = list(best_trial.intermediate_values.values())

        self.best_params = best_trial.params
        self.best_epoch = int(np.argmin(intermediate_vals))+1

        print('-' * 20)
        print('Study concluded. Results:')
        print(f"Best trial: {best_trial.number}")
        print(f"Best loss: {best_trial.value}")
        print(f"Best params: {self.best_params}")
        print(f"Best epoch: {self.best_epoch}")

        if save:
            study_path = os.path.join(dh.optuna_dir, f'{study_name}.pkl')
            joblib.dump(study, study_path)
            print(f"Study saved to {study_path}")  

    def load_optuna_study(self, study_path: str):
        """Loads an Optuna study from a specified path. Updates the best hyperparameters and best epoch. Returns the study object."""
        study = joblib.load(study_path)
        best_trial = study.best_trial
        intermediate_vals = list(best_trial.intermediate_values.values())

        self.best_params = best_trial.params
        self.best_epoch = int(np.argmin(intermediate_vals)) + 1
        
        print(f"Best params: {self.best_params}")
        print(f"Best epoch: {self.best_epoch}")
        return study

    def train_final_model(self,
                          dh: DataHandler,
                          batch_size: int,
                          max_epochs: int = None,
                          patience: int = 30,
                          save: bool = True):
        """Trains the final model using the best hyperparameters from the Optuna study. If `save` is `True`, then it saves the model to file and updates the model filepath."""
        if self.best_params is None:
            raise ValueError("No best parameters found. Please run or load an Optuna study first.") 
        if max_epochs is None:
            max_epochs = self.best_epoch

        model = build_network(dh.input_dim, self.depth, self.best_params).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(),
                                 lr=self.best_params['learning_rate'],
                                 weight_decay=self.best_params['weight_decay'])

        train_loader = dh.get_mlp_data('train', batch_size)
        model.train()

        print(f'Training {self.model_name} with depth {self.depth} for {max_epochs} epochs.')
        print(f'Using best params: {self.best_params} and batch size: {batch_size}.')

        patience_used = 0
        best_epoch_loss = float('inf')
        start_time = time.time()
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            for features, targets, weights in train_loader:
                features, targets, weights = features.to(DEVICE), targets.to(DEVICE), weights.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(features)
                loss = weighted_cross_entropy_loss(outputs, targets, weights)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(train_loader)
            self.final_train_history.append(np.mean(epoch_loss))
            print(f"Epoch {str(epoch).rjust(4)}, Loss: {epoch_loss:.6f}")

            # Early stopping
            if epoch_loss < best_epoch_loss:
                best_epoch_loss = epoch_loss
                patience_used = 0
            else:
                patience_used += 1
                if patience_used >= patience:
                    print(f"Early stopping at epoch {epoch}.")
                    break

        time_taken = time.time() - start_time
        print(f"Training completed in {time_taken:.2f} seconds.")

        if save:
            model_save_path = os.path.join(dh.models_dir, f"{self.model_name}.pth")
            torch.save(self.model.state_dict(), model_save_path)
            print(f"Model saved to {model_save_path}")

        model.eval()
        self.model = model

    def load_model(self, model_path: str):
        """Loads a pre-trained model from a specified path into self.model."""
        self.model = build_network(self.input_dim, self.depth, self.best_params).to(DEVICE)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()
        print(f"Model loaded from {model_path}.")

    def make_final_predictions(self,
                               dh: DataHandler):
        """
        Makes predictions on the test set using the trained model. Saves the predictions to a CSV file, and returns the predictions as a numpy array.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")
        
        X_test, y_test = dh.get_mlp_data('test', 1)
        y_tots = y_test.sum(axis = 1, keepdims=True)

        X_test = X_test.to(DEVICE)
        y_test = y_test.to(DEVICE)
        y_tots = y_tots.to(DEVICE)
        self.model.to(DEVICE)
        self.model.eval()

        with torch.no_grad():
            outputs = self.model(X_test)
            y_pred = outputs * y_tots

        y_pred = y_pred.cpu().numpy()
        pred_df = pd.DataFrame(y_pred, columns=dh.targets)

        pred_save_path = os.path.join(dh.preds_dir, f"{self.model_name}_preds.csv")
        pred_df.to_csv(pred_save_path, index=False)
        print(f"{self.model_name} predictions saved to: {pred_save_path}.")

        return y_pred







