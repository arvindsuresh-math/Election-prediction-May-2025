"""
Custom metrics, loss functions, and objective functions for model training and evaluation.

Includes:
- weighted_cross_entropy_loss: For PyTorch models with soft labels and sample weights.
- softmax: Numerically stable softmax implementation.
- weighted_softprob_obj: Custom XGBoost objective for weighted cross-entropy.
- weighted_cross_entropy_eval: Custom XGBoost evaluation metric for weighted cross-entropy.
"""
import numpy as np
import torch
import xgboost as xgb # Primarily for type hinting DMatrix in XGBoost functions
import pandas as pd

from data_handling import DataHandler # For type hinting
from constants import RESULTS_DIR 
from typing import Dict





# --- Evaluation Function for Model Predictions ---

def evaluate_predictions(pred_dict: Dict[str, np.ndarray],
                         dh: DataHandler,
                         save: bool = False):
    """
    Evaluates and compares model predictions. `pred_dict` should be a dictionary
    where keys are model names (str) and values are numpy arrays containing the predictions. If `save` is True, the resulting evaluation DataFrame is saved to a CSV file.

    Returns:
        pd.DataFrame: A DataFrame summarizing the evaluation. 
        Columns:
        - 'P(democrat)', 'P(other)', 'P(republican)', 'P(non_voter)': Aggregate shares.
        - 'P(underage)': Calculated as 1 minus the sum of the four target shares.
        - 'Cross-entropy': Cross-entropy against true distribution.
        - 'KL Div': Kullback-Leibler divergence against true distribution.
        - 'KL Div%': Percentage of KL divergence relative to true self-entropy.
    """
    print("\n--- Evaluating Model Predictions ---")

    _, y_true, wts = dh.get_ridge_data('test') 
    pred_dict['true'] = y_true

    for key, y in pred_dict.items():
        y = np.clip(y, 1e-9, 1.0) # For RidgeModel
        y = (wts * y).sum(axis=0) 
        y = np.append(y,1.0 - np.sum(y)) # Append P(underage)
        pred_dict[key] = list(y)

    y_true = pred_dict['true']
    ce_true = -np.sum(y_true * np.log(y))

    for key, y in pred_dict.items():
        ce = -np.sum(y_true * np.log(y))
        kl_div = ce - ce_true
        kl_div_percent = (kl_div / ce_true) * 100
        pred_dict[key] = list(y) + [ce, kl_div, kl_div_percent] 

    target_cols = [t.replace('|C)', ')') for t in dh.targets] + ['P(underage)', 'Cross-entropy', 'KL Div', 'KL Div%']
    eval_df = pd.DataFrame.from_dict(pred_dict, orient='index', columns=target_cols)
    eval_df = eval_df.sort_values(by='KL Div', ascending=True)

    if save:
        eval_save_path = os.path.join(RESULTS_DIR, f"Final_evaluation_{dh.test_year}.csv")
        eval_df.to_csv(eval_save_path, index=True, index_label='model')
        print(f"\nEvaluation summary saved to: {eval_save_path}")

    return eval_df