import os
import pandas as pd
import numpy as np
import joblib
import optuna
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from data_handling import DataHandler # For type hinting

def objective_ridge(trial: optuna.trial.Trial,
                    dh: DataHandler,
                    use_pca: bool = False
                    ):
    """Objective function for Ridge hyperparameter optimization using Optuna. Uses weighted MSE as the loss function."""
    params = {}
    params['alpha'] = trial.suggest_float('alpha', 1e-6, 10.0, log=True)
    if use_pca:
        params['n_components'] = trial.suggest_int('n_components', 10, (dh.n_features // 10)*10, step=10)
        cv_data = dh.get_ridge_data('cv', params['n_components'])
    else:
        cv_data = dh.get_ridge_data('cv')
    fold_val_losses = []

    for (X_train, y_train, wts_train), (X_val, y_val, wts_val) in cv_data:
        model_fold = Ridge(alpha=params['alpha'])
        model_fold.fit(X_train, y_train, sample_weight=wts_train.ravel())
        y_pred = model_fold.predict(X_val)
        val_loss = mean_squared_error(y_val, y_pred, sample_weight=wts_val.ravel())
        fold_val_losses.append(val_loss)
    return np.mean(fold_val_losses)

# --- Main Ridge handler class ---

class RidgeModel:
    """Handles Ridge Regression CV, training, loading, and evaluation."""

    def __init__(self, dh: DataHandler, model_name: str = 'ridge'):
        """Initializes the RidgeModel handler and pre-computes all paths."""
        self.dh = dh
        self.model_name = model_name
        self.best_params = None
        self.model = None

        # precompute file paths
        self.study_path = os.path.join(dh.optuna_dir, f"{model_name}_study.pkl")
        self.model_path = os.path.join(dh.models_dir, f"{model_name}_model.pkl")
        self.pred_path  = os.path.join(dh.preds_dir,  f"{model_name}_preds.csv")

        print(f'RidgeModel initialized with model name: {self.model_name}')
        print(f"Optuna study will be stored in  : {self.study_path}")
        print(f"Trained model will be stored in : {self.model_path}")
        print(f"Final preds will be stored in   : {self.pred_path}")

    def run_optuna_study(self,
                        n_trials: int,
                        use_pca: bool = False):
        """Runs an Optuna study for Ridge hyperparameter tuning. Saves the study object."""
        study = optuna.create_study(direction='minimize',
                                    study_name=f"{self.model_name}_study")
        study.optimize(lambda t: objective_ridge(t, self.dh, use_pca),
                       n_trials=n_trials)

        self.best_params = study.best_params
        print(f"Best alpha: {self.best_params['alpha']}")
        if use_pca:
            print(f"Best n_components: {self.best_params['n_components']}")

        joblib.dump(study, self.study_path)
        print(f"Optuna study saved to: {self.study_path}")

    def load_optuna_study(self):
        """Loads an Optuna study."""
        if not os.path.exists(self.study_path):
            raise FileNotFoundError(f"No study at {self.study_path}. Run run_optuna_study first.")
        return joblib.load(self.study_path)

    def update_params(self):
        """Updates the best parameters from the Optuna study."""
        study = self.load_optuna_study()
        self.best_params = study.best_params
        print(f"Updated best_params: {self.best_params}")

    def train_final_model(self):
        """Trains the final Ridge model using the best parameters from the Optuna study. Saves the model."""
        self.update_params()
        # fetch train data (with or without PCA)
        X, y, wts = self.dh.get_ridge_data('train', self.best_params.get('n_components', None))

        self.model = Ridge(alpha=self.best_params['alpha'])
        self.model.fit(X, y, sample_weight=wts.ravel())

        joblib.dump(self.model, self.model_path)
        print(f"Model saved to: {self.model_path}")

    def load_model(self):
        """Loads a pre-trained model."""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"No model at {self.model_path}. Run train_final_model first.")
        self.model = joblib.load(self.model_path)
        print(f"Model loaded from: {self.model_path}")

    def make_final_predictions(self):
        if self.model is None:
            self.load_model()

        X_test = self.dh.get_ridge_data('test', self.best_params.get('n_components', None))[0]
        y_pred = self.model.predict(X_test)

        pred_df = pd.DataFrame(y_pred, columns=self.dh.targets)
        pred_df.to_csv(self.pred_path, index=False)
        print(f"Predictions saved to: {self.pred_path}")

        return y_pred