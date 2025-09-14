def softmax(x: np.ndarray):
    """Numerically stable softmax function."""
    # Subtract max for numerical stability before exp
    e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return e_x / np.sum(e_x, axis=1, keepdims=True)

# --- XGBoost Custom Objective Function ---

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

# --- XGBoost Custom Evaluation Metric ---

def weighted_cross_entropy_eval(preds: np.ndarray, dtrain: xgb.DMatrix):
    """
    Static custom evaluation metric for weighted cross-entropy with soft labels.
    """
    # 1. Get true labels (soft probabilities) and weights
    n_samples = preds.shape[0]
    labels = dtrain.get_label().reshape((n_samples, 4))
    weights = dtrain.get_weight().reshape((n_samples, 1))
    weights = weights / weights.sum() # Normalize weights

    # 3. Calculate predicted probabilities using softmax
    tots = labels.sum(axis=1, keepdims=True) # Total votes per sample
    probs = softmax(preds)
    probs = probs * tots

    # 4. Calculate weighted cross-entropy per sample
    epsilon = 1e-9
    probs = np.clip(probs, epsilon, 1. - epsilon)
    sample_surprisals = - labels * np.log(probs) #surprisal weighted by labels
    sample_loss = sample_surprisals.sum(axis=1) #sum across classes to get CE loss per sample

    # 5. Calculate average weighted cross-entropy
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
    
    # Create pruning callback
    pruning_callback = XGBoostPruningCallback(trial, "test-weighted-CE-mean")
    
    # Run cross-validation
    cv_results = xgb.cv(
        params=params,
        dtrain=dtrain,
        num_boost_round=num_boost_rounds,
        nfold=3,
        obj=weighted_softprob_obj, #custom objective
        custom_metric=weighted_cross_entropy_eval, #custom eval, name='weighted-CE'
        maximize=False,
        early_stopping_rounds=early_stopping_rounds,
        callbacks=[pruning_callback],
        verbose_eval=False,
        shuffle=False # Ensures folds match what's used by other models
    )
    
    # Return best validation score
    best_score = cv_results['test-weighted-CE-mean'].min()
    return best_score