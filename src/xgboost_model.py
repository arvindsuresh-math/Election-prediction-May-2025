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

# def weighted_softprob_obj(preds: np.ndarray, dtrain: xgb.DMatrix):
#     """Custom XGBoost objective for weighted cross-entropy with soft labels.
#     """
#     # Get true labels (soft probabilities) and weights
#     n_samples = preds.shape[0]
#     labels = dtrain.get_label().reshape((n_samples, 4))

#     # Calculate predicted probabas
#     tots = labels.sum(axis=1, keepdims=True) # P(18plus) per sample
#     probs = softmax(preds)
#     probs = probs * tots 

#     # Calculate Gradient: (q - p)
#     grad = probs - labels

#     # Calculate Hessian (diagonal approximation): tots * q * (1 - q)
#     hess = tots * probs * (1 - probs)
#     hess = np.maximum(hess, 1e-12)

#     return (grad,hess)

def weighted_softprob_obj(preds: np.ndarray, dtrain: xgb.DMatrix):
    """
    Static custom XGBoost objective for weighted cross-entropy with soft labels.
    """
    # 1. Get true labels (soft probabilities) and weights
    n_samples = preds.shape[0]
    labels = dtrain.get_label().reshape((n_samples, 4))
    weights = dtrain.get_weight().reshape((n_samples, 1))
    weights = weights / weights.sum() # Normalize weights

    # 3. Calculate predicted probabas
    tots = labels.sum(axis=1, keepdims=True) # P(18plus) per sample
    probs = softmax(preds)
    probs = probs * tots 

    # 4. Calculate Gradient: (q - p)
    grad = probs - labels

    # 5. Calculate Hessian (diagonal approximation): w * q * (1 - q)
    # hess = weights * probs * (tots - probs)
    # hess = np.maximum(hess, 1e-12) # Ensure non-negative hessian
    hess = probs * (1 - probs)
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
    probs = softmax(preds)

    # Calculate weighted cross-entropy per sample
    epsilon = 1e-9
    probs = np.clip(probs, epsilon, 1. - epsilon)
    sample_surprisals = - labels * np.log(probs) #surprisal weighted by labels
    sample_loss = sample_surprisals.sum(axis=1) #sum across classes to get CE loss per sample

    # Calculate average weighted cross-entropy
    weighted_avg_ce = np.average(sample_loss, weights=weights.flatten())

    return 'weighted-CE', weighted_avg_ce

def objective_xgb(trial: optuna.trial.Trial, 
                  dh: DataHandler,
                  num_boost_rounds: int,
                  early_stopping_rounds: int,
                  use_pca: bool = False
                  ):
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

    if use_pca:
        params['n_components'] = trial.suggest_int('n_components', 10, (dh.n_features // 10)*10, step=10)
        dtrain = dh.get_xgb_data('train', params['n_components'])
    else:
        dtrain = dh.get_xgb_data('train')

    pruning_callback = XGBoostPruningCallback(trial, "test-weighted-CE")

    cv_results = xgb.cv(
        params={**params, "verbosity": 0},
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
    Handler for training, tuning, and using XGBoost models.     After initialization, use:
        - run_optuna_study(...) to tune hyperparameters
        - train_final_model(...) to train the model
        - make_final_predictions(...) to get predictions on the test set

    Typical usage in a Jupyter notebook:
    ```python
    xgbm = XGBoostModel(model_name="xgb")
    xgbm.run_optuna_study(dh, min_resource=10, reduction_factor=2, n_trials=30, timeout=15)
    xgbm.train_final_model(dh)
    preds = xgbm.make_final_predictions(dh)
    ```
    """

    def __init__(self, dh: DataHandler, model_name: str = 'xgboost'):
        self.dh = dh
        self.model_name = model_name
        self.best_params = None 
        self.optimal_boost_rounds = None 
        self.model = None

        # precompute file paths
        self.study_path = os.path.join(dh.optuna_dir, f"{model_name}_study.pkl")
        self.model_path = os.path.join(dh.models_dir, f"{model_name}_model.json")
        self.pred_path  = os.path.join(dh.preds_dir,  f"{model_name}_preds.csv")

        print(f'XGBoostModel initialized with model name: {self.model_name}')
        print(f"Optuna study will be stored in  : {self.study_path}")
        print(f"Trained model will be stored in : {self.model_path}")
        print(f"Final preds will be stored in   : {self.pred_path}")

    def run_optuna_study(self,
                         min_resource: int,
                         reduction_factor: int,
                         n_trials: int,
                         timeout: int, #in minutes
                         num_boost_rounds: int = 150,
                         early_stopping_rounds: int = 30,
                         use_pca: bool = False
                        ):
        """
        Run an Optuna study to tune XGBoost hyperparameters using the ASHA pruning algorithm.

        Args:
            min_resource (int):
                Minimum number of boosting rounds before pruning can begin.
            reduction_factor (int):
                Factor by which the resource (boosting rounds) is reduced at each pruning step.
            n_trials (int):
                Maximum number of hyperparameter configurations to evaluate.
            timeout (int):
                Time limit for the entire study in minutes.
            num_boost_rounds (int, optional):
                Maximum number of boosting rounds to test in cross‐validation. Default is 150.
            early_stopping_rounds (int, optional):
                Number of rounds without improvement to stop a CV trial early. Default is 30.
            use_pca (bool, optional):
                If True, include PCA preprocessing and tune `n_components`. Default is False.

        Side Effects:
            - Updates `self.best_params` with the best hyperparameters.
            - Sets `self.optimal_boost_rounds` to the best number of rounds.
            - Always saves the completed Optuna study to `self.study_path`.
        """

        pruner = optuna.pruners.SuccessiveHalvingPruner(
            min_resource=min_resource,
            reduction_factor=reduction_factor,
            min_early_stopping_rate=0
        )
        study = optuna.create_study(
            study_name=f"{self.model_name}_study",
            direction='minimize',
            pruner=pruner
        )
        study.optimize(
            lambda t: objective_xgb(t, 
                                    self.dh,  
                                    num_boost_rounds, 
                                    early_stopping_rounds, 
                                    use_pca),
            n_trials=n_trials,
            timeout=timeout * 60, # Convert to seconds
            n_jobs=-1
            )
        
        best_trial = study.best_trial
        vals = list(best_trial.intermediate_values.values())

        self.best_params = best_trial.params
        self.optimal_boost_rounds = int(np.argmin(vals)) + 1 

        print('-' * 20)
        print('Study concluded. Results:')
        print(f"Best trial: {best_trial.number}")
        print(f"Best loss: {best_trial.value}")
        print(f"Best params: {self.best_params}")
        print(f"Optimal boosting rounds: {self.optimal_boost_rounds}")

        joblib.dump(study, self.study_path)
        print(f"Study saved to {self.study_path}")    

    def load_optuna_study(self):
        """Loads an Optuna study. Returns the study object."""
        if not os.path.exists(self.study_path):
            raise FileNotFoundError(f"No study at {self.study_path}")
        study = joblib.load(self.study_path)
        print(f"Study loaded from {self.study_path}")
        return study
    
    def update_params(self):
        """Updates the best hyperparameters and optimal boosting rounds from the study."""
        study = self.load_optuna_study()
        best_trial = study.best_trial
        vals = list(best_trial.intermediate_values.values())

        self.best_params = best_trial.params
        self.optimal_boost_rounds = int(np.argmin(vals)) + 1 

        print(f"Best params: {self.best_params}")
        print(f"Optimal boosting rounds: {self.optimal_boost_rounds}")

        return study

    def train_final_model(self, num_boost_round: int = None):
        """
        Trains the final XGBoost model using the best hyperparameters and
        the optimal number of boosting rounds from the Optuna study. Saves the trained model.
        """
        if self.best_params is None or self.optimal_boost_rounds is None:
            self.update_params()
        if num_boost_round is None:
            num_boost_round = self.optimal_boost_rounds

        dtrain = self.dh.get_xgb_data('train', self.best_params.get('n_components', None))
        final_train_params = {
            **self.best_params,
            'n_jobs': -1, # Use all available cores
            'verbosity': 0
        }

        print(f'Training {self.model_name} for {num_boost_round} boosting rounds...')
        print(f"Using best params: {self.best_params}")

        start_time = time.time()
        bst = xgb.train(
            params=final_train_params,
            dtrain=dtrain,
            num_boost_round=num_boost_round, 
            obj=weighted_softprob_obj,
            custom_metric=weighted_cross_entropy_eval, 
            maximize=False
            )
        time_taken = time.time() - start_time
        print(f"Training completed in {time_taken:.2f} seconds.")

        bst.save_model(self.model_path)
        print(f"Model saved to {self.model_path}")

        self.model = bst

    def load_model(self):
        """Loads a trained XGBoost Booster model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No model at {self.model_path}")
        self.model = xgb.Booster()
        self.model.load_model(self.model_path)
        print(f"Model loaded from {self.model_path}")

    def make_final_predictions(self):
        """
        Makes predictions on the test set using the trained model. Saves the predictions to a CSV file, and returns the predictions as a numpy array.
        """
        if self.model is None:
            self.load_model()

        dtest = self.dh.get_xgb_data('test', self.best_params.get('n_components', None)) 
        X_test, y_test, _ = self.dh.get_ridge_data('test', self.best_params.get('n_components', None)) 
        n_samples = X_test.shape[0]

        y_pred = self.model.predict(dtest).reshape((n_samples, 4))
        y_pred = softmax(y_pred)

        pred_df = pd.DataFrame(y_pred, columns=self.dh.targets)
        pred_df.to_csv(self.pred_path, index=False)
        print(f"{self.model_name} predictions saved to: {self.pred_path}.")

        return y_pred


