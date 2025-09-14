import os
import pandas as pd
import numpy as np
import json
import torch
import xgboost as xgb
from torch.utils.data import TensorDataset, DataLoader
from typing import List, Tuple

# --- WeightedStandardScaler Class ---

class WeightedStandardScaler:
    """
    Scales features using weighted mean and variance. Has same methods and attributes as sklearn's StandardScaler.

    Example
    -------
    >>> scaler = WeightedStandardScaler()
    >>> X_scaled = scaler.fit_transform(X, weights)
    >>> X_orig = scaler.inverse_transform(X_scaled)
    """
    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X: np.ndarray, weights: np.ndarray):
        """Fits the scaler to the data X using sample weights."""
        w = weights.reshape(-1, 1)
        self.mean_ = (X * w).sum(axis=0) / w.sum(axis=0)
        var = (w * (X - self.mean_)**2).sum(axis=0) / w.sum(axis=0)
        self.scale_ = np.sqrt(var)
        return self

    def transform(self, X: np.ndarray):
        """Transforms the data X using the fitted scaler."""
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X: np.ndarray, weights: np.ndarray):
        """Fits and transforms the data X using sample weights."""
        return self.fit(X, weights).transform(X)

    def inverse_transform(self, X: np.ndarray):
        """Undo the scaling of X: X_original = X_scaled * scale_ + mean_."""
        return X * self.scale_ + self.mean_

# --- WeightedPCA Class ---

class WeightedPCA:
    """
    Performs Principal Component Analysis using a weighted covariance matrix.
    Assumes input data `X_scaled` is already standardized using WeightedStandardScaler.
    Fits all components; transformation selects the top n.
    """
    def __init__(self):
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.input_dim_ = None 

    def fit(self, X_scaled: np.ndarray, weights: np.ndarray):
        """
        Fits the Weighted PCA model to the standardized data X_scaled using sample weights. Computes and stores ALL principal components.
        """
        self.input_dim_ = X_scaled.shape[1]

        # Calculate Weighted Covariance Matrix 
        sqrt_weights = np.sqrt(weights)
        weighted_X_scaled = X_scaled * sqrt_weights 
        weighted_cov = (weighted_X_scaled.T @ weighted_X_scaled) / weights.sum() 

        # Eigendecomposition
        eigenvalues, eigenvectors = np.linalg.eigh(weighted_cov)

        # Sort eigenvalues and corresponding eigenvectors in descending order
        sorted_indices = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[sorted_indices]
        eigenvectors = eigenvectors[:, sorted_indices]

        # Store all components and explained variance
        total_variance = np.sum(eigenvalues)
        self.explained_variance_ = eigenvalues
        self.explained_variance_ratio_ = eigenvalues / total_variance
        self.components_ = eigenvectors.T # Store components as rows

        return self

    def transform(self, X_scaled: np.ndarray, n_components: int):
        """
        Applies the Weighted PCA transformation using the top n_components.
        Assumes X_scaled is already standardized using the SAME scaler used for fitting PCA.
        """
        # Select the top n components
        selected_components = self.components_[:n_components]

        # Project data onto the selected components
        X_transformed = X_scaled.dot(selected_components.T)

        return X_transformed

    def fit_transform(self, X_scaled: np.ndarray, weights: np.ndarray, n_components: int):
        """Fits PCA (all components) and transforms using the top n_components."""
        return self.fit(X_scaled, weights).transform(X_scaled, n_components)

    def get_explained_variance_ratio(self):
        """Returns the explained variance ratio for all components."""
        return self.explained_variance_ratio_

# --- DataHandler Class ---

class DataHandler:
    """
    Handles loading, preprocessing, and splitting of election prediction data.

    Parameters
    ----------
    test_year : int, optional
        Year to use as the test set (default: 2020).
    features_to_drop : list, optional
        List of feature names to exclude from model input (default: ['P(C)']).

    Attributes
    ----------
    raw_data : pd.DataFrame
        Raw DataFrame loaded from CSV.
    idx : list
        Index columns (['year', 'gisjoin', 'state', 'county']).
    targets : list
        Target columns (['P(democrat|C)', 'P(republican|C)', 'P(other|C)', 'P(non-voter|C)']).
    features : list
        List of features used for model input.
    input_dim : int
        Number of features used for model input.
    years : list
        List of years (e.g., [2008, 2012, 2016, 2020]).
    test_year : int
        Year used for test data.
    train_years : list
        Years used for training data (all but test_year).
    cv_year_pairs : list of tuples
        Each tuple is (train_years, val_year) for cross-validation splits.
    cv_scalers : list
        List of scalers fitted to the training years for each CV fold.
    final_scaler : WeightedStandardScaler
        Scaler fitted to the training years for final model training.
    data_dir : str
        Directory containing the dataset.
    results_dir : str
        Directory for storing results.
    models_dir : str
        Directory for storing models after final training.
    optuna_dir : str
        Directory for storing Optuna studies.
    preds_dir : str
        Directory for storing final predictions on the test set.

    Example
    -------
    >>> dh = DataHandler(test_year=2020)
    >>> for train_data, val_data in dh.get_mlp_data('cv', batch_size=32):
    >>>     # training/validation loop goes here
    """

    def __init__(self, test_year: int = 2020, features_to_drop=['P(C)']):
        """Initializes the DataHandler, creates directories to store results, and determines input/output dimensions."""
        # --- Create directories for results ---
        self._make_dirs()

        # --- Load full dataset ---
        self.raw_data = pd.read_csv(os.path.join(self.data_dir, 'final_dataset.csv'))

        # --- Feature and Target Definitions ---
        with open(os.path.join(self.data_dir, 'variables.json'), 'r') as f:
            vars = json.load(f)
        self.idx = vars['idx']
        self.targets = vars['targets']
        feature_keys = set(vars.keys()) - set(['targets', 'years', 'idx'])
        self.features = sorted([item for key in feature_keys for item in vars[key] if item not in features_to_drop])
        self.input_dim = len(self.features)

        # --- Years for CV, Training, Testing ---
        self.years = vars['years']
        self.test_year = test_year
        self.train_years = sorted(set(self.years) - {test_year})
        self.cv_year_pairs = [
                ([self.train_years[0], self.train_years[1]], self.train_years[2]),
                ([self.train_years[0], self.train_years[2]], self.train_years[1]),
                ([self.train_years[1], self.train_years[2]], self.train_years[0])
                    ]
        
        # --- Fitted scalers for CV and final training ---
        self.cv_scalers = [self._fit_scaler(fit_years=train_years) for (train_years,_) in self.cv_year_pairs]
        self.final_scaler = self._fit_scaler(fit_years=self.train_years)

        # --- Fitted PCA for CV and final training ---
        self.cv_pca = [self._fit_wpca(fit_years=train_years, scaler=self.cv_scalers[i]) for i, (train_years,_) in enumerate(self.cv_year_pairs)]
        self.final_pca = self._fit_wpca(fit_years=self.train_years, scaler=self.final_scaler)

        print(f"DataHandler initialized - Using {self.input_dim} features - Test year: {self.test_year}")

    def _make_dirs(self):
        """Creates a directory named f"{test_year}-results-{current date}" in the repo root, with subdirs for models, optuna, and predictions."""
        from datetime import datetime

        date_str = datetime.now().strftime("%Y%m%d")
        # Get the root directory of the repo (assumes this file is in src/)
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        results_dir = os.path.join(repo_root, f"{self.test_year}-results-{date_str}")
        subdirs = ["models", "optuna", "predictions"]

        for sub in subdirs:
            path = os.path.join(results_dir, sub)
            os.makedirs(path, exist_ok=True)

        # Store the paths as attributes
        self.data_dir = os.path.join(repo_root, "data")
        self.results_dir = results_dir
        self.models_dir = os.path.join(results_dir, "models")
        self.optuna_dir = os.path.join(results_dir, "optuna_studies")
        self.preds_dir = os.path.join(results_dir, "predictions")

    def _load_data(self,
                   train_years: List[int], 
                   test_year: int 
                   ):
        """Loads data for a specific train/validation split."""
        data = self.raw_data

        # Make datasets with fit years and transform years
        df_train = data[data['year'].isin(train_years)].reset_index(drop=True)
        df_test = data[data['year'] == test_year].reset_index(drop=True)

        # Shape (n_samples, 1)
        wts_train = df_train['P(C)'].values.reshape(-1, 1) 
        wts_test = df_test['P(C)'].values.reshape(-1, 1)

        # Shape (n_samples, 4)
        y_train = df_train[self.targets].values
        y_test = df_test[self.targets].values

        # Shape (n_samples, n_features)
        X_train = df_train[self.features].values
        X_test = df_test[self.features].values

        return (X_train, y_train, wts_train), (X_test, y_test, wts_test)

    def _fit_scaler(self, fit_years: List[int]):
        """Fits a WeightedStandardScaler to the given years."""
        df = self.raw_data[self.raw_data['year'].isin(fit_years)].reset_index(drop=True)
        wts = df['P(C)'].values
        X = df[self.features].values

        scaler = WeightedStandardScaler()
        scaler.fit(X, wts)
        return scaler
    
    def _fit_wpca(self, fit_years: List[int], scaler: WeightedStandardScaler):
        """Fits a WeightedPCA to the given years using the provided scaler."""
        df = self.raw_data[self.raw_data['year'].isin(fit_years)].reset_index(drop=True)
        wts = df['P(C)'].values
        X_raw = df[self.features].values
        X_scaled = scaler.transform(X_raw)

        pca = WeightedPCA()
        pca.fit(X_scaled, wts)
        return pca

    def _create_tensors(self, data: Tuple[np.ndarray, np.ndarray, np.ndarray]):
        """Converts NumPy arrays (X, y, wts) to Pytorch tensors."""
        X_np, y_np, wts_np = data
        X_tensor = torch.tensor(X_np, dtype=torch.float32)
        y_tensor = torch.tensor(y_np, dtype=torch.float32)
        wts_tensor = torch.tensor(wts_np, dtype=torch.float32).unsqueeze(1) # Ensure shape (n_samples, 1)
        return (X_tensor, y_tensor, wts_tensor)
        
    def _create_dataloader(self,
                            data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                            batch_size: int,
                            shuffle: bool = True
                            ) -> Tuple[DataLoader, DataLoader]:
        """Creates DataLoaders for training and validation sets."""
        X, y, wts = self._create_tensors(data)
        dataset = TensorDataset(X, y, wts)
        loader = DataLoader(dataset,
                            batch_size=batch_size,
                            shuffle=shuffle) 
        return loader

    def _create_dmatrix(self,
                         data: Tuple[np.ndarray, np.ndarray, np.ndarray],
                         only_X: bool = False
                        ) -> xgb.DMatrix:
        """
        Creates an xgb.DMatrix from input data. If `only_X` is True, only X is used, otherwise X, y, and wts are used.
        """
        X, y, wts = data
        if only_X:
            dmatrix = xgb.DMatrix(X)
        else:
            dmatrix = xgb.DMatrix(X, label=y, weight=wts)
        return dmatrix

    def get_ridge_data(self, task: str, n_components: int = None):
        """
        Returns the following depending on `task`:
        - 'cv': List of tuples (train_data, val_data) for each CV fold.
        - 'train': Tuple (X_train, y_train, wts_train).
        - 'test': Tuple (X_test, y_test, wts_test).
        If `n_components` (positive int) is provided, applies weighted PCA and returns only the top n components.
        """
        if task == 'cv':
            train_val_data_list = []
            for j, (train_years, val_year) in enumerate(self.cv_year_pairs):
                scaler = self.cv_scalers[j]
                (X_train, y_train, wts_train), (X_test, y_test, wts_test) = self._load_data(train_years, val_year)
                X_train = scaler.transform(X_train)
                X_test = scaler.transform(X_test)
                if n_components is not None:
                    X_train = self.cv_pca[j].transform(X_train, n_components)
                    X_test = self.cv_pca[j].transform(X_test, n_components)
                train_val_data_list.append(((X_train, y_train, wts_train), (X_test, y_test, wts_test)))
            return train_val_data_list
        
        elif task == 'train':
            (X_train, y_train, wts_train), _ = self._load_data(self.train_years, self.test_year)
            X_train = self.final_scaler.transform(X_train)
            if n_components is not None:
                X_train = self.final_pca.transform(X_train, n_components)
            return X_train, y_train, wts_train
        
        elif task == 'test':
            _, (X_test, y_test, wts_test) = self._load_data(self.train_years, self.test_year)
            X_test = self.final_scaler.transform(X_test)
            if n_components is not None:
                X_test = self.final_pca.transform(X_test, n_components)
            return X_test, y_test, wts_test
        else:
            raise ValueError("Invalid task specified. Use 'cv', 'train', or 'test'.")

    def get_mlp_data(self, task: str, batch_size: int, n_components: int = None):
        """
        Returns the data needed for Neural Networks depending on task:
        - 'cv': List of tuples (train_loader, val_loader) for each CV fold.
        - 'train': Train_loader for 3 training years.
        - 'test': Tuple of tensors (X_test, y_test).
        If `n_components` (positive int) is provided, applies weighted PCA and returns only the top n components.
        """
        if task == 'cv':
            loaders = []
            for train_data, val_data in self.get_ridge_data('cv', n_components):
                train_loader = self._create_dataloader(train_data, batch_size)
                val_loader = self._create_dataloader(val_data, batch_size, shuffle=False)
                loaders.append((train_loader, val_loader))
            return loaders
        
        elif task == 'train':
            train_data = self.get_ridge_data('train', n_components)
            return self._create_dataloader(train_data, batch_size)
        
        elif task == 'test':
            test_data = self.get_ridge_data('test', n_components)
            X_tensor, y_tensor, _ = self._create_tensors(test_data)
            return X_tensor, y_tensor

    def get_xgb_data(self, task: str, n_components: int = None):
        """Returns the data needed for XGBoost based on the task, depending on task:
        - 'cv': DMatrix with labels and weights with all train years combined; folds are determined internally by xgb.cv
        - 'train': Same as 'cv'.
        - 'test': DMatrix for test year (X only, no labels or weights).
        If `n_components` (positive int) is provided, applies weighted PCA and returns only the top n components.
        """
        if task == 'cv' or task == 'train':
            # Use all training years for XGBoost CV or final training
            train_data = self.get_ridge_data('train', n_components)
            return self._create_dmatrix(train_data, only_X=False)
        
        elif task == 'test':
            test_data = self.get_ridge_data('test', n_components)
            return self._create_dmatrix(test_data, only_X=True)
        
        else:
            raise ValueError("Invalid task specified. Use 'cv', 'train', or 'test'.")
        
