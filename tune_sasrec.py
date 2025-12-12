import optuna
import pandas as pd
import numpy as np
from recommender import DataLoader, map_at_k
from sasrec_recommender import SASRecRecommender

def tune():
    print("Loading Data...")
    loader = DataLoader('interactions_train.csv', 'items.csv')
    loader.preprocess()
    train_df, val_df = loader.get_train_val_split(val_ratio=0.2, strategy='user_time')
    
    val_ground_truth = val_df.groupby('u')['i'].apply(list).to_dict()
    train_users_set = set(train_df['u'].unique())
    val_users = [u for u in val_ground_truth.keys() if u in train_users_set]
    val_user_codes = [loader.reverse_user_map.get(u, -1) for u in val_users]
    
    actual = [val_ground_truth[u] for u in val_users]
    
    num_users = len(loader.user_map)
    num_items = len(loader.item_map)

    def objective(trial):
        # Hyperparameters
        hidden_units = trial.suggest_categorical('hidden_units', [32, 64, 128])
        num_blocks = trial.suggest_int('num_blocks', 1, 3)
        num_heads = trial.suggest_categorical('num_heads', [1, 2, 4])
        dropout_rate = trial.suggest_float('dropout_rate', 0.0, 0.5)
        maxlen = trial.suggest_categorical('maxlen', [20, 50, 100])
        lr = trial.suggest_loguniform('lr', 1e-4, 1e-2)
        batch_size = trial.suggest_categorical('batch_size', [64, 128, 256])
        
        # Constraints
        if hidden_units % num_heads != 0:
            raise optuna.TrialPruned()
            
        print(f"Trial params: {trial.params}")
        
        model = SASRecRecommender(
            hidden_units=hidden_units,
            num_blocks=num_blocks,
            num_heads=num_heads,
            dropout_rate=dropout_rate,
            maxlen=maxlen,
            batch_size=batch_size,
            lr=lr,
            epochs=15, # Reduced epochs for speed
            device='cpu' # Force CPU if needed, or let it detect
        )
        
        try:
            model.fit(train_df, num_users, num_items)
            ids, scores = model.recommend(val_users, val_user_codes, k=10, filter_already_liked_items=False)
            recs = [[loader.item_map.get(i) for i in r] for r in ids]
            
            score = map_at_k(actual, recs, k=10)
            return score
        except Exception as e:
            print(f"Trial failed: {e}")
            return 0.0

    print("Starting Optimization...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)
    
    print("Best params:", study.best_params)
    print("Best score:", study.best_value)

if __name__ == "__main__":
    tune()
