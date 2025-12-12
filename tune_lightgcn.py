import optuna
import pandas as pd
import numpy as np
from recommender import DataLoader, map_at_k
from lightgcn_recommender import LightGCNRecommender

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
        embedding_dim = trial.suggest_categorical('embedding_dim', [64, 128, 256])
        n_layers = trial.suggest_int('n_layers', 2, 4) # Force deeper layers
        lr = trial.suggest_float('lr', 1e-4, 1e-2, log=True)
        batch_size = trial.suggest_categorical('batch_size', [1024, 2048])
        epochs = trial.suggest_int('epochs', 20, 50, step=10)
        
        print(f"Trial params: {trial.params}")
        
        model = LightGCNRecommender(
            embedding_dim=embedding_dim,
            n_layers=n_layers,
            lr=lr,
            epochs=epochs,
            batch_size=batch_size
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
    study.optimize(objective, n_trials=20)
    
    print("Best params:", study.best_params)
    print("Best score:", study.best_value)

if __name__ == "__main__":
    tune()
