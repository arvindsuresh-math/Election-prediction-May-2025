import os
import pandas as pd
import numpy as np
import optuna
import xgboost as xgb
import joblib
import time
from data_handling import DataHandler # For type hinting
from optuna.integration import XGBoostPruningCallback

############################################

def softmax(x: np.ndarray):
    """Numerically stable softmax function."""
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

def weighted_softprob_obj(preds: np.ndarray, dtrain: xgb.DMatrix):
    """Custom XGBoost objective for weighted cross-entropy with soft labels.
    """
    # Get true labels (soft probabilities) and weights
    n_samples = preds.shape[0]
    labels = dtrain.get_label().reshape((n_samples, 4))

    # Calculate predicted probabas
    tots = labels.sum(axis=1, keepdims=True) # P(18plus) per sample
    probs = softmax(preds)
    probs = probs * tots 

    # Calculate Gradient: (q - p)
    grad = probs - labels

    # Calculate Hessian (diagonal approximation): tots * q * (1 - q)
    hess = tots * probs * (1 - probs)
    hess = np.maximum(hess, 1e-12)

    return (grad,hess)

def weighted_cross_entropy_eval(preds: np.ndarray, dtrain: xgb.DMatrix):
    """Custom evaluation metric for weighted cross-entropy with soft labels."""
    # Get true probas and weights
    n_samples = preds.shape[0]
    labels = dtrain.get_label().reshape((n_samples, 4))
    weights = dtrain.get_weight().reshape((n_samples, 1))
    weights = weights / weights.sum() # Normalize weights

    # Calculate predicted probas
    tots = labels.sum(axis=1, keepdims=True) # Total votes per sample
    probs = softmax(preds)
    probs = probs * tots

    # Calculate weighted cross-entropy per sample
    epsilon = 1e-9
    probs = np.clip(probs, epsilon, 1. - epsilon)
    sample_surprisals = - labels * np.log(probs) #surprisal weighted by labels
    sample_loss = sample_surprisals.sum(axis=1) #sum across classes to get CE loss per sample

    # Calculate average weighted cross-entropy
    weighted_avg_ce = np.average(sample_loss, weights=weights.flatten())

    return 'weighted-CE', weighted_avg_ce

def objective_xgb(trial: optuna.trial.Trial, 
                  dtrain: xgb.DMatrix,
                  num_boost_rounds: int = 150,
                  early_stopping_rounds: int = 30):
    """
    Objective function for XGBoost hyperparameter optimization using Optuna. Uses weighted_softprob_obj and weighted_cross_entropy_eval as custom objective and evaluation metric. Returns the mean val loss for the trial.
    """
    # Get hyperparams for the current trial
    params = {
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0, log=False),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0, log=False),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.4, 1.0, log=False),
        'colsample_bylevel': trial.suggest_float('colsample_bylevel', 0.4, 1.0, log=False),
        'colsample_bynode': trial.suggest_float('colsample_bynode', 0.4, 1.0, log=False),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10.0, log=True),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10.0, log=True),
        'max_depth': trial.suggest_int('max_depth', 3, 12, step=1),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 20, step=1)
    }

    pruning_callback = XGBoostPruningCallback(trial, "test-weighted-CE")

    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_rounds,
        nfold=3,
        obj=weighted_softprob_obj, #custom objective
        custom_metric=weighted_cross_entropy_eval, #name='weighted-CE'
        maximize=False,
        early_stopping_rounds=early_stopping_rounds,
        callbacks=[pruning_callback],
        verbose_eval=False,
        shuffle=False # Ensures folds match what's used by other models
    )
    
    best_score = cv_results['test-weighted-CE-mean'].min()
    return best_score

# --- Main xgboost model handler class ---

class XGBoostModel:
    """
    Handler for training, tuning, and using XGBoost models.

    Attributes:
        model_name (str): Name of the model instance.
        best_params (dict or None): Best hyperparameters found by Optuna study.
        optimal_boost_rounds (int or None): Number of boosting rounds corresponding to best validation loss.
        model (xgb.Booster or None): Trained XGBoost model after final training.

    After initialization, use:
        - run_optuna_study(...) to tune hyperparameters
        - train_final_model(...) to train the model
        - make_final_predictions(...) to get predictions on the test set

    Typical usage in a Jupyter notebook:
    ```python
    xgbm = XGBoostModel(model_name="xgb1")
    xgbm.run_optuna_study(dh, min_resource=10, reduction_factor=2, n_trials=30, timeout=15)
    xgbm.train_final_model(dh)
    preds = xgbm.make_final_predictions(dh)
    ```
    """

    def __init__(self, model_name: str = 'xgboost'):
        self.model_name = model_name
        # Set after running/loading an optuna study
        self.best_params = None 
        self.optimal_boost_rounds = None 
        # Set after final training, or loading a saved model
        self.model = None

    def run_optuna_study(self,
                         dh: DataHandler,
                         min_resource: int,
                         reduction_factor: int,
                         n_trials: int,
                         timeout: int, #in minutes
                         num_boost_rounds: int = 150,
                         early_stopping_rounds: int = 30,
                         save: bool = True
                        ):
        """Runs an Optuna study for hyperparameter tuning of XGBoost using the ASHA pruning algorithm. Updates the best hyperparameters and optimal boosting rounds after running the study. If `save` is `True`, then it saves the study to file."""

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
        
        print(f"Running Optuna study for {self.model_name}.")
        print(f'Min resource: {min_resource}, Reduction factor: {reduction_factor}, n_trials: {n_trials}, Timeout: {timeout} mins')
        print('-' * 20)

        study.optimize(
            lambda trial: objective_xgb(
                trial,
                dtrain=dh.get_xgb_data('train'),
                num_boost_rounds=num_boost_rounds,
                early_stopping_rounds=early_stopping_rounds
            ),
            n_trials=n_trials,
            timeout=timeout * 60, # Convert to seconds
            n_jobs=-1
            )
        
        best_trial = study.best_trial
        intermediate_vals = list(study.best_trial.intermediate_values.values())

        self.best_params = best_trial.params
        self.optimal_boost_rounds = int(np.argmin(intermediate_vals)) + 1 

        print('-' * 20)
        print('Study concluded. Results:')
        print(f"Best trial: {best_trial.number}")
        print(f"Best loss: {best_trial.value}")
        print(f"Best params: {self.best_params}")
        print(f"Optimal boosting rounds: {self.optimal_boost_rounds}")

        if save:
            study_path = os.path.join(dh.optuna_dir, f'{study_name}.pkl')
            joblib.dump(study, study_path)
            print(f"Study saved to {study_path}")     

    def load_optuna_study(self, study_path: str):
        """Loads an Optuna study from a specified path. Updates the best hyperparameters and optimal boosting rounds. Returns the study object."""
        study = joblib.load(study_path)
        best_trial = study.best_trial
        intermediate_vals = list(best_trial.intermediate_values.values())

        self.best_params = best_trial.params
        self.optimal_boost_rounds = int(np.argmin(intermediate_vals)) + 1 

        print(f"Best params: {self.best_params}")
        print(f"Optimal boosting rounds: {self.optimal_boost_rounds}")

        return study

    def train_final_model(self, 
                          dh: DataHandler,
                          num_boost_round: int = None,
                          save: bool = True
                          ):
        """
        Trains the final XGBoost model using the best hyperparameters and
        the optimal number of boosting rounds from the Optuna study. If `save` is `True`, then it saves the model to file and updates the model filepath.
        """
        if self.best_params is None:
            raise ValueError("No best parameters found. Please run or load an Optuna study first.")
        if num_boost_round is None:
            num_boost_round = self.optimal_boost_round

        final_train_params = {
            **self.best_params,
            'n_jobs': -1, # Use all available cores
        }
        dtrain = dh.get_xgb_data('train') 

        print(f'Training {self.model_name} for {num_boost_round} boosting rounds...')
        print(f"Using best params: {self.best_params}")

        start_time = time.time()
        bst = xgb.train(
            params=final_train_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round, 
            custom_metric=weighted_cross_entropy_eval, 
            maximize=False,
            verbose_eval=10 # Print progress every 10 rounds if evals is used
            )
        time_taken = time.time() - start_time
        print(f"Training completed in {time_taken:.2f} seconds.")

        if save:
            model_save_path = os.path.join(dh.models_dir, f"{dh.test_year}_{self.model_name}.json")
            bst.save_model(model_save_path)
            print(f"Model saved to: {model_save_path}")

        self.model = bst

    def load_model(self, model_path: str = None):
        """Loads a trained XGBoost Booster model from a specified path."""
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        print(f"Model loaded from {model_path}")

    def make_final_predictions(self,
                               dh: DataHandler
                               ):
        """
        Makes predictions on the test set using the trained model. Saves the predictions to a CSV file, and returns the predictions as a numpy array.
        """
        if self.model is None:
            raise ValueError("Model not trained or loaded. Please train or load a model first.")

        dtest = dh.get_xgb_data('test') 
        X_test, y_test, _ = dh.get_ridge_data('test') 
        n_samples = X_test.shape[0]
        y_tots = y_test.sum(axis=1, keepdims=True) # P(18plus|C), shape [n_samples, 1]

        # Create final, scaled predictions
        y_pred = self.model.predict(dtest)
        y_pred = y_pred.reshape((n_samples, 4))
        y_pred = softmax(y_pred) 
        y_pred = y_pred * y_tots

        pred_save_path = os.path.join(dh.preds_dir, f"{dh.test_year}_{self.model_name}_preds.csv")
        pred_df = pd.DataFrame(y_pred, columns=dh.targets)
        pred_df.to_csv(pred_save_path, index=False)
        print(f"{self.model_name} predictions saved to: {pred_save_path}.")

        return y_pred


