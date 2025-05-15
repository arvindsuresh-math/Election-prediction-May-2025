import os
import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from typing import List, Tuple
from data_handling import DataHandler # For type hinting

def objective_ridge(trial: optuna.trial.Trial,
                    cv_data: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
                    ):
    """Objective function for Ridge hyperparameter optimization using Optuna. Uses weighted MSE as the loss function."""
    alpha = trial.suggest_float('alpha', 1e-6, 10.0, log=True)
    n_c
    fold_val_losses = []

    for (X_train, y_train, wts_train), (X_val, y_val, wts_val) in cv_data:
        model_fold = Ridge(alpha=alpha)
        model_fold.fit(X_train, y_train, sample_weight=wts_train.ravel())
        y_pred = model_fold.predict(X_val)
        val_loss = mean_squared_error(y_val, y_pred, sample_weight=wts_val)
        fold_val_losses.append(val_loss)
    return np.mean(fold_val_losses)

# --- Main Ridge handler class ---

class RidgeModel:
    """Handles Ridge Regression CV, training, loading, and evaluation."""

    def __init__(self, model_name: str = 'ridge'):
        """Initializes the RidgeModel handler."""
        self.model_name = model_name

        # Set after running/loading optuna study
        self.best_alpha = None 

        # Set after final training, or loading saved model
        self.model = None
        
    def run_optuna_study(self,
                         dh: DataHandler,
                         n_trials: int,
                         save: bool = True
                        ):
        """Runs an Optuna study for Ridge hyperparameter tuning. Saves the study results."""
        cv_data = dh.get_ridge_data('cv')
        study_name = f'{self.model_name}_study'
        study = optuna.create_study(direction='minimize', study_name=study_name)
        study.optimize(
            lambda trial: objective_ridge(trial, cv_data), 
            n_trials=n_trials
            )

        self.best_alpha = study.best_params['alpha']
        print(f"Best alpha found: {self.best_alpha}")

        if save:
            study_path = os.path.join(dh.optuna_dir, f"{study_name}.pkl")
            joblib.dump(study, study_path)
            print(f"Optuna study saved to: {study_path}")

    def load_optuna_study(self, study_path: str):
        """Loads an Optuna study from a specified path. Updates the best alpha."""
        study = joblib.load(study_path)
        self.best_alpha = study.best_params['alpha']
        print(f"Best alpha: {self.best_alpha}")
        return study

    def train_final_model(self,
                          dh: DataHandler,
                          save: bool = True
                          ):
        """Trains the final Ridge model using the best alpha and n_components (PCA) from Optuna study. If `save` is `True`, saves the model to file."""
        if self.best_alpha is None:
            raise ValueError("No best alpha found. Please run or load an Optuna study first.") 
        print(f"Using best alpha: {self.best_alpha}")

        X_train, y_train, wts_train = dh.get_ridge_data('train')
        self.model = Ridge(alpha=self.best_alpha)       
        self.model.fit(X_train, y_train, sample_weight=wts_train.ravel())
        print(f'Final training for {self.model_name} complete.')

        if save:
            model_save_path = os.path.join(dh.models_dir, f"{dh.test_year}_{self.model_name}.pkl")
            joblib.dump(self.model, model_save_path)
            print(f"Model saved to {self.model_save_path}.")

    def load_model(self, model_path: str):
        """Loads a pre-trained model from a specified path into `self.model`."""
        self.model = joblib.load(model_path)
        print(f"Model loaded from {model_path}.")

    def make_final_predictions(self, dh: DataHandler):
        """
        Makes predictions on the test set using the trained model. Saves the predictions to a CSV file, and returns the predictions as a numpy array.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")

        X_test = dh.get_ridge_data('test')[0] 
        y_pred = self.model.predict(X_test)

        pred_save_path = os.path.join(dh.preds_dir, f"{dh.test_year}_{self.model_name}_preds.csv")
        pred_df = pd.DataFrame(y_pred, columns=dh.targets)
        pred_df.to_csv(pred_save_path, index=False)
        print(f"{self.model_name} predictions saved to: {self.pred_save_path}.")
        
        return y_pred
