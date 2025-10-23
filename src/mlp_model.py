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

def weighted_cross_entropy_loss(y_pred: torch.Tensor,
                                y_true: torch.Tensor,
                                wts: torch.Tensor):
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
        y_pred (torch.Tensor): Model predictions (probabilities).
                                Shape: [batch_size, num_classes] or [num_folds, batch_size_per_fold, num_classes].
        y_true (torch.Tensor): Ground truth probabilities.
                                Shape: [batch_size, num_classes] or [num_folds, batch_size_per_fold, num_classes].
        wts (torch.Tensor): Sample weights ('P(C)').
                                Shape: [batch_size, 1] or [num_folds, batch_size_per_fold, 1].

    Returns:
        torch.Tensor: Scalar tensor representing the final loss.
    """
    # Ensure 3D for unified processing
    if y_pred.ndim == 2:
        y_pred = y_pred.unsqueeze(0)
        y_true = y_true.unsqueeze(0)
        wts = wts.unsqueeze(0)

    # Scale y_pred by P(18plus) and clamp to avoid log(0)
    y_pred = torch.clamp(y_pred, 1e-10, 1. - 1e-10) 

    # Tensors of shape (K, B, 1)
    sample_ce_loss = -torch.sum(y_true * torch.log(y_pred), dim=-1, keepdim=True)
    wts_reshaped = wts.view_as(sample_ce_loss) 
    weighted_sample_ce_losses = sample_ce_loss * wts_reshaped

    # Tensors of shape (K, 1, 1)
    sum_weighted_losses_fold = weighted_sample_ce_losses.sum(dim=1, keepdim=True)
    sum_wts_fold = wts_reshaped.sum(dim=1, keepdim=True)
    loss_per_fold = sum_weighted_losses_fold / sum_wts_fold
    
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
                  dh: DataHandler,
                  depth: int, 
                  max_epochs: int,
                  use_pca: bool = False
                  ):
    """
    Optuna objective function for BatchedMLP hyperparameter optimization.
    
    Args:
        trial: Current Optuna trial
        dh: DataHandler instance
        depth: Number of hidden layers
        max_epochs: Maximum training epochs
        
    Returns:
        Best validation loss achieved
    """
    # Suggest hyperparameters
    params = {}
    params['learning_rate'] = trial.suggest_float('learning_rate', 1e-4, 1e-1, log=True)
    params['weight_decay'] = trial.suggest_float('weight_decay', 1e-6, 1e-2, log=True)
    params['batch_size'] = trial.suggest_int('batch_size', 64, 256, step=64)
    for i in range(1, depth + 1):
        params[f'n_hidden_{i}'] = trial.suggest_int(f'n_hidden_{i}', 8, 128, step=8)
        params[f'dropout_rate_{i}'] = trial.suggest_float(f'dropout_rate_{i}', 0.0, 0.5, log=False)
    if use_pca:
        params['n_components'] = trial.suggest_int('n_components', 10, (dh.n_features // 10)*10, step=10)
        input_dim = params['n_components']
        cv_dataloaders = dh.get_mlp_data('cv', params['batch_size'], input_dim)
    else:
        input_dim = dh.n_features
        cv_dataloaders = dh.get_mlp_data('cv', params['batch_size'])

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
            # Stack (X, y, wts) cross folds, move to DEVICE
            stacked_data = [(torch.stack([data_fold[i] for data_fold in batched_train_data], dim=0))
                            for i in range(3)]  # 0 = X, 1 = y, 2 = wts
            X, y_true, wts = [d.to(DEVICE) for d in stacked_data]
            
            optimizer.zero_grad()
            y_pred = model(X)
            loss = weighted_cross_entropy_loss(y_pred, y_true, wts)
            loss.backward()
            optimizer.step()
            
        model.eval()
        epoch_loss = 0.0
        with torch.no_grad():
            for batched_val_data in zip(*val_loader_list):
                stacked_data = [(torch.stack([data_fold[i] for data_fold in batched_val_data], dim=0))
                                for i in range(3)]
                X, y_true, wts = [d.to(DEVICE) for d in stacked_data]
                
                y_pred = model(X)
                loss = weighted_cross_entropy_loss(y_pred, y_true, wts)
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

    Typical usage in a Jupyter notebook:
    ```python
    dh = DataHandler(test_year=2020)
    mlp = MLPModel(dh, model_name="mlp2", depth=2)
    mlp.run_optuna_study(min_resource=5, reduction_factor=2, n_trials=30, timeout=15, max_epochs=50, use_pca=True)
    mlp.train_final_model(batch_size=64)
    preds = mlp.make_final_predictions()
    ```
    """

    def __init__(self, dh: DataHandler, model_name: str, depth: int):
        self.dh = dh
        self.model_name = model_name
        self.depth = depth

        # will be set after tuning
        self.best_params = None
        self.best_epoch = None

        # will be set after training or loading
        self.model = None
        self.final_train_history: List[float] = []

        # precompute file paths
        self.study_path = os.path.join(dh.optuna_dir,   f"{model_name}_study.pkl")
        self.model_path = os.path.join(dh.models_dir,   f"{model_name}_model.pth")
        self.pred_path  = os.path.join(dh.preds_dir,    f"{model_name}_preds.csv")

        print(f'MLPModel initialized with model name {self.model_name} and depth {self.depth}.')
        print(f"Optuna study will be stored in  : {self.study_path}")
        print(f"Trained model will be stored in : {self.model_path}")
        print(f"Final preds will be stored in   : {self.pred_path}")

    def run_optuna_study(self,
                        min_resource: int,
                        reduction_factor: int,
                        n_trials: int,
                        timeout: int,
                        max_epochs: int,
                        use_pca: bool = False):
        """
        Runs hyperparameter optimization for the MLP model using Optuna with ASHA pruning.
        
        This method searches for optimal hyperparameters (batch size, layer sizes, dropout rates, learning rate, weight decay) and determines the best training epoch. The results are stored in 
        the model instance as `best_params` and `best_epoch`.
        
        Parameters
        ----------
        min_resource : int
            Minimum number of epochs for ASHA pruner before pruning can occur (typically 5-10)
        reduction_factor : int
            Reduction factor for ASHA pruner (typically 2-4)
        n_trials : int
            Maximum number of hyperparameter configurations to try
        timeout : int
            Maximum duration in minutes for the study to run
        max_epochs : int
            Maximum number of epochs for each trial
        use_pca : bool, default=False
            Whether to use PCA on features; if True, the number of components is also optimized
        """
        asha = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=min_resource,
            reduction_factor=reduction_factor,
            min_early_stopping_rate=0
        )
        study = optuna.create_study(
            study_name=f"{self.model_name}_study",
            direction="minimize",
            pruner=asha
        )
        study.optimize(
            lambda t: objective_mlp(t, 
                                    self.dh, 
                                    self.depth, 
                                    max_epochs, 
                                    use_pca),
            n_trials=n_trials,
            timeout=timeout*60,
            n_jobs=-1
        )
        
        best_trial = study.best_trial
        vals = list(best_trial.intermediate_values.values())

        self.best_params = study.best_params
        self.best_epoch = int(np.argmin(vals)) + 1

        joblib.dump(study, self.study_path)

        print('-' * 20)
        print(f'Study concluded and saved to {self.study_path}.')
        print(f"Best trial: {best_trial.number}")
        print(f"Best loss: {best_trial.value}")
        print(f"Best params: {self.best_params}")
        print(f"Best epoch: {self.best_epoch}")

    def load_optuna_study(self):
        """Loads the Optuna study."""
        if not os.path.exists(self.study_path):
            raise FileNotFoundError(f"No study at {self.study_path}")
        
        study = joblib.load(self.study_path)
        print(f"Loaded study from {self.study_path}")
        return study
    
    def update_params(self):
        """Updates the best hyperparameters and best epoch from the Optuna study."""
        study = self.load_optuna_study()
        best_trial = study.best_trial
        vals = list(best_trial.intermediate_values.values())
        self.best_params = study.best_params
        self.best_epoch  = int(np.argmin(vals)) + 1
        print(f"Best params: {self.best_params}")
        print(f"Best epoch: {self.best_epoch}")

    def train_final_model(self,
                          batch_size: int = None,
                          max_epochs: int = None,
                          patience: int = 30):
        """Trains the final model using the best hyperparameters from the Optuna study. """
        if self.best_params is None:
            self.update_params()

        if max_epochs is None:
            max_epochs = self.best_epoch
        if batch_size is None:
            batch_size = self.best_params["batch_size"]

        # get data (with or without PCA)
        n_components = self.best_params.get("n_components", None)
        input_dim = self.best_params.get("n_components", self.dh.n_features)
        loader = self.dh.get_mlp_data("train", batch_size, n_components)

        model = build_network(input_dim, self.depth, self.best_params).to(DEVICE)
        optimizer = optim.AdamW(model.parameters(),
                                lr=self.best_params["learning_rate"],
                                weight_decay=self.best_params["weight_decay"])

        patience_used = 0
        best_epoch_loss = float('inf')

        print(f'Training {self.model_name} with depth {self.depth} for {max_epochs} epochs with patience of {patience} epochs.')
        print(f'Using batch size: {batch_size} and best params: {self.best_params}.')

        model.train()
        start_time = time.time()
        for epoch in range(max_epochs):
            epoch_loss = 0.0
            for X, y_true, wts in loader:
                X, y_true, wts = X.to(DEVICE), y_true.to(DEVICE), wts.to(DEVICE)
                optimizer.zero_grad()
                y_pred = model(X)
                loss = weighted_cross_entropy_loss(y_pred, y_true, wts)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            epoch_loss /= len(loader)
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

        torch.save(model.state_dict(), self.model_path)
        print(f"Model saved to {self.model_path}")

        model.eval()
        self.model = model

    def load_model(self):
        """Loads the trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No model at {self.model_path}")
        if self.best_params is None:
            self.update_params()
        self.model = build_network(
            self.best_params.get("n_components", self.dh.n_features),
            self.depth,
            self.best_params
        ).to(DEVICE)
        self.model.load_state_dict(torch.load(self.model_path, weights_only=True))
        print(f"Model loaded from {self.model_path}")

    def make_final_predictions(self):
        """
        Makes predictions on the test set using the trained model. Saves the predictions to a CSV file, and returns the predictions as a numpy array.
        """
        if self.model is None:
            self.load_model()
        
        X_test, y_test = self.dh.get_mlp_data('test', 1, self.best_params.get('n_components', None)) 

        X_test = X_test.to(DEVICE)
        y_test = y_test.to(DEVICE)
        self.model.to(DEVICE)

        self.model.eval()
        with torch.no_grad():
            y_pred = self.model(X_test)

        y_pred = y_pred.cpu().numpy()
        pred_df = pd.DataFrame(y_pred, columns=self.dh.targets)
        pred_df.to_csv(self.pred_path, index=False)
        print(f"{self.model_name} predictions saved to: {self.pred_path}.")

        return y_pred







