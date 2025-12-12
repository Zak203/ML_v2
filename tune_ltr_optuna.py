import optuna
import pickle
import numpy as np
from catboost import CatBoostRanker, Pool
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

def map_at_k(y_true, y_score, k=10):
    """Calculates MAP@k for a single user."""
    # Sort by score descending
    order = np.argsort(y_score)[::-1]
    y_true_sorted = y_true[order]
    
    score = 0.0
    num_hits = 0.0
    
    for i, rel in enumerate(y_true_sorted[:k]):
        if rel > 0:
            num_hits += 1.0
            score += num_hits / (i + 1.0)
            
    return score / min(len(y_true), k) if len(y_true) > 0 else 0.0

def evaluate_fold(model_type, params, X, y, groups):
    """Evaluates a model configuration using 5-fold CV."""
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    
    # Reconstruct group boundaries for splitting
    # groups is a list of group sizes [g1, g2, ...]
    # We need to split by groups, not samples
    
    # Create group indices
    group_indices = np.arange(len(groups))
    
    # Cumulative sum for slicing X and y
    group_ptr = np.cumsum(np.concatenate(([0], groups)))
    
    for train_idx_g, val_idx_g in kf.split(group_indices):
        # Map group indices to sample indices
        train_idx = []
        for g_idx in train_idx_g:
            train_idx.extend(range(group_ptr[g_idx], group_ptr[g_idx+1]))
            
        val_idx = []
        for g_idx in val_idx_g:
            val_idx.extend(range(group_ptr[g_idx], group_ptr[g_idx+1]))
            
        X_train, y_train = X[train_idx], y[train_idx]
        X_val, y_val = X[val_idx], y[val_idx]
        groups_train = groups[train_idx_g]
        groups_val = groups[val_idx_g]
        
        # Train
        if model_type == 'catboost':
            train_pool = Pool(data=X_train, label=y_train, group_id=np.repeat(np.arange(len(groups_train)), groups_train))
            val_pool = Pool(data=X_val, label=y_val, group_id=np.repeat(np.arange(len(groups_val)), groups_val))
            
            model = CatBoostRanker(**params, verbose=0)
            model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
            preds = model.predict(X_val)
            
        elif model_type == 'xgboost':
            model = xgb.XGBRanker(**params)
            model.fit(X_train, y_train, group=groups_train, 
                      eval_set=[(X_val, y_val)], eval_group=[groups_val], 
                      verbose=False, early_stopping_rounds=50)
            preds = model.predict(X_val)
            
        elif model_type == 'lightgbm':
            model = lgb.LGBMRanker(**params)
            model.fit(X_train, y_train, group=groups_train,
                      eval_set=[(X_val, y_val)], eval_group=[groups_val],
                      eval_at=[10], early_stopping_rounds=50, verbose=False)
            preds = model.predict(X_val)
            
        # Evaluate MAP@10
        cursor = 0
        fold_map = []
        for g in groups_val:
            p = preds[cursor : cursor+g]
            t = y_val[cursor : cursor+g]
            fold_map.append(map_at_k(t, p, k=10))
            cursor += g
        scores.append(np.mean(fold_map))
        
    return np.mean(scores)

def objective_catboost(trial):
    params = {
        'loss_function': 'YetiRank',
        'iterations': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'depth': trial.suggest_int('depth', 4, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1e-3, 10.0, log=True),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'random_seed': 42,
        'task_type': 'CPU'
    }
    return evaluate_fold('catboost', params, X, y, groups)

def objective_xgboost(trial):
    params = {
        'objective': 'rank:ndcg',
        'eval_metric': 'ndcg@10',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'gamma': trial.suggest_float('gamma', 0.0, 5.0),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }
    return evaluate_fold('xgboost', params, X, y, groups)

def objective_lightgbm(trial):
    params = {
        'objective': 'lambdarank',
        'metric': 'ndcg',
        'n_estimators': 1000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'num_leaves': trial.suggest_int('num_leaves', 15, 63),
        'max_depth': trial.suggest_int('max_depth', 3, 12),
        'feature_fraction': trial.suggest_float('feature_fraction', 0.5, 1.0),
        'bagging_fraction': trial.suggest_float('bagging_fraction', 0.5, 1.0),
        'bagging_freq': 1,
        'lambda_l1': trial.suggest_float('lambda_l1', 1e-8, 10.0, log=True),
        'lambda_l2': trial.suggest_float('lambda_l2', 1e-8, 10.0, log=True),
        'random_state': 42,
        'n_jobs': -1
    }
    return evaluate_fold('lightgbm', params, X, y, groups)

if __name__ == "__main__":
    print("Loading LTR Data...")
    with open('ltr_data.pkl', 'rb') as f:
        X, y, groups = pickle.load(f)
    print(f"Loaded X: {X.shape}, y: {y.shape}, groups: {len(groups)}")
    
    print("\n--- Tuning CatBoost ---")
    study_cb = optuna.create_study(direction='maximize')
    study_cb.optimize(objective_catboost, n_trials=20)
    print("Best CatBoost Params:", study_cb.best_params)
    
    print("\n--- Tuning XGBoost ---")
    study_xgb = optuna.create_study(direction='maximize')
    study_xgb.optimize(objective_xgboost, n_trials=20)
    print("Best XGBoost Params:", study_xgb.best_params)
    
    print("\n--- Tuning LightGBM ---")
    study_lgb = optuna.create_study(direction='maximize')
    study_lgb.optimize(objective_lightgbm, n_trials=20)
    print("Best LightGBM Params:", study_lgb.best_params)
    
    # Save best params
    best_params = {
        'catboost': study_cb.best_params,
        'xgboost': study_xgb.best_params,
        'lightgbm': study_lgb.best_params
    }
    with open('best_ltr_params.pkl', 'wb') as f:
        pickle.dump(best_params, f)
    print("\nSaved best_ltr_params.pkl")
