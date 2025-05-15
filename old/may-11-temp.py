import torch
import torch.nn as nn
import torch.optim as optim
import optuna
import os
import json
import copy
import joblib
import pandas as pd
import numpy as np
import time
from typing import Dict, Any, Optional, Union, List, Tuple, Type

from constants import DEVICE
from data_handling import DataHandler
from constants import RESULTS_DIR, MODELS_DIR, PREDS_DIR

############################################################
# --- OLD CODE ---
############################################################

def weighted_cross_entropy_loss(outputs: torch.Tensor,
                                targets: torch.Tensor,
                                weights: torch.Tensor):
    """
    Calculates a custom weighted cross-entropy loss for a batch.
    Scales the outputs by the sum of the target probabilities (P(18plus)) for each sample. This is because the targets are soft labels comprising 4 out of 5 classes; the fifth class is 1 - P(18plus), where P(18plus) is the sum of the soft labels.
    Sample Loss = - sum_k ( target_k * log(output_k) )
    Batch Loss = Expected sample loss = sum_C ( P(C) * Loss(C) ) / sum_C ( P(C) ), where P(C) is the sample weight.

    Args:
        outputs (torch.Tensor): Model predictions (probabilities), shape [batch_size, num_classes].
        targets (torch.Tensor): Ground truth probabilities, shape [batch_size, num_classes].
        weights (torch.Tensor): Sample weights ('P(C)'), shape [batch_size, 1].

    Returns:
        torch.Tensor: Scalar tensor representing the weighted average loss.
    """
    # scale outputs by P(18plus) and clamp to avoid log(0)
    tots = targets.sum(dim=1, keepdim=True) # P(18plus) per sample
    outputs = outputs * tots
    outputs = torch.clamp(outputs, 1e-10, 1. - 1e-9)
    # Sample cross-entropy loss
    sample_loss = -torch.sum(targets * torch.log(outputs), dim=1, keepdim=True)
    weights_reshaped = weights.view_as(sample_loss)
    weighted_sample_losses = sample_loss * (weights.view_as(sample_loss))
    batch_loss = weighted_sample_losses.sum() / weights_reshaped.sum()
    return batch_loss

def build_network(input_dim: int, depth: int, hparams: Dict[str, Any]):
    """Returns an MLP (nn.Sequential) with the specified input dimension, depth, and hyperparameters."""
    layers = []
    current_dim = input_dim

    for i in range(1, depth + 1):
        n_hidden = hparams[f"n_hidden_{i}"]
        layers.append(nn.Linear(current_dim, n_hidden))
        layers.append(nn.ReLU()) 

        dropout_rate = hparams.get(f"dropout_rate_{i}", 0.0)
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        current_dim = n_hidden

    layers.append(nn.Linear(current_dim, 4))
    layers.append(nn.Softmax(dim=1))
    return nn.Sequential(*layers)

def objective_mlp(trial: optuna.trial.Trial, 
                  input_dim: int,
                  depth: int, 
                  dataloaders: List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
                  max_epochs: int = 100):
    """
    Objective function for MLP hyperparam optimization using Optuna. Returns the val loss for the trial. 
    """
    # Get hyperparams for the current trial
    params = {}
    params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    for i in range(1, depth + 1):
        params[f'n_hidden_{i}'] = trial.suggest_int(f'n_hidden_{i}', 8, 128, step=8)
        params[f'dropout_rate_{i}'] = trial.suggest_float(f'dropout_rate_{i}', 0.0, 0.5, log=False)

    # Make dict with (model, optimizer) for each fold
    model = build_network(input_dim, depth, params)
    models = {}
    for i in range(3):
        model_i = copy.deepcopy(model).to(DEVICE)
        optimizer_i = optim.AdamW(model_i.parameters(),
                                 lr=params['learning_rate'],
                                 weight_decay=params['weight_decay'])
        models[i] = (model_i, optimizer_i)

    best_epoch_loss = float('inf')      
    for epoch in range(max_epochs):
        epoch_fold_losses = [] 

        # Train/validate one epoch on each fold
        for fold_idx, (train_loader, val_loader) in enumerate(dataloaders):
            model, optimizer = models[fold_idx]

            model.train()
            for features, targets, weights in train_loader:
                features = features.to(DEVICE)
                targets = targets.to(DEVICE)
                weights = weights.to(DEVICE)
                optimizer.zero_grad()
                outputs = model(features)
                loss = weighted_cross_entropy_loss(outputs, targets, weights)
                loss.backward()
                optimizer.step()
            
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for features, targets, weights in val_loader:
                    features = features.to(DEVICE)
                    targets = targets.to(DEVICE)
                    weights = weights.to(DEVICE)
                    outputs = model(features)
                    loss = weighted_cross_entropy_loss(outputs, targets, weights)
                    val_loss += loss.item() 

            epoch_fold_losses.append(val_loss / len(val_loader))

        # Report + Pruning check
        epoch_loss = np.mean(epoch_fold_losses)
        best_epoch_loss = min(best_epoch_loss, epoch_loss)
        trial.report(epoch_loss, step=epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()
    
    return best_epoch_loss


############################################################
# --- NEW CODE ---
############################################################

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
    
    The overall loss is the weighted average of these sample CE losses.
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
    tots = targets.sum(dim=2, keepdim=True) # Shape: (K, B, 1)
    outputs = outputs * tots
    outputs = torch.clamp(outputs, 1e-10, 1. - 1e-10) 

    # Tensors of shape (K, B, 1)
    sample_ce_loss = -torch.sum(targets * torch.log(outputs), dim=2, keepdim=True)
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
            init.kaiming_uniform_(self.weight[k], a=0, mode='fan_in', nonlinearity='leaky_relu')
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

def build_network(input_dim: int, 
                 depth: int, 
                 hparams: Dict[str, Any], 
                 K: int = 3, 
                 num_classes: int = 4):    
    layers = []
    current_dim = input_dim

    for i in range(1, depth + 1):
        n_hidden = hparams[f"n_hidden_{i}"]
        layers.append(BatchedLinear(K, current_dim, n_hidden))
        layers.append(nn.ReLU())
        dropout_rate = hparams.get(f"dropout_rate_{i}", 0.0)
        if dropout_rate > 0.0:
            layers.append(nn.Dropout(dropout_rate))
        current_dim = n_hidden
    
    layers.append(BatchedLinear(K, current_dim, num_classes))
    layers.append(nn.Softmax(dim=2))
    
    return nn.Sequential(*layers)
    
def objective_mlp(trial: optuna.trial.Trial, 
                  input_dim: int,
                  depth: int, 
                  dataloaders: List[Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]],
                  max_epochs: int = 100):
    """
    Optuna objective function for BatchedMLP hyperparameter optimization.
    
    Args:
        trial: Current Optuna trial
        input_dim: Set this as `dh.input_dim`
        depth: Number of hidden layers
        dataloaders: List of `(train_loader, val_loader)` tuples. Set this as `dh.get_nn_data('cv', batch_size)`
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

    model = build_network(input_dim, depth, params).to(DEVICE)
    optimizer = optim.AdamW(model.parameters(),
                             lr=params['learning_rate'],
                             weight_decay=params['weight_decay'])

    train_loader_list = [dl_pair[0] for dl_pair in dataloaders]
    val_loader_list = [dl_pair[1] for dl_pair in dataloaders]
    num_batches = len(train_loader_list[0])

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


############################################################
# --- CODE TO RUN OPTUNA STUDY IN NOTEBOOK ---
############################################################

dh = DataHandler()

asha_pruner = optuna.pruners.SuccessiveHalvingPruner(
    min_resource=30,        # Minimum number of steps before pruning
    reduction_factor=2,    # Reduction factor for successive halving
    min_early_stopping_rate=0
)

study_mlp3 = optuna.create_study(direction="minimize", pruner=asha_pruner)

study_mlp3.optimize(
    lambda trial: objective_mlp(trial,
                            input_dim=dh.input_dim,
                            depth=3,
                            dataloaders=dh.get_nn_data('cv', batch_size=256),
                            max_epochs=128),
    n_trials=64,  # Number of trials to run
    timeout=3600,   # Timeout in 1 hour
    n_jobs=-1,     # Use all available cores
)

