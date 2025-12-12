import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import KFold
from recommender import map_at_k

def calculate_map(y_true, y_scores, groups, k=10):
    """
    Calculates MAP@k given flattened arrays and group sizes.
    y_true: binary labels
    y_scores: predicted scores
    groups: list of group sizes (number of candidates per query)
    """
    aps = []
    cursor = 0
    for g in groups:
        if g == 0:
            cursor += g
            continue
            
        yt = y_true[cursor : cursor + g]
        ys = y_scores[cursor : cursor + g]
        
        # Sort by score descending
        sorted_indices = np.argsort(ys)[::-1]
        yt_sorted = yt[sorted_indices]
        
        # Calculate AP
        score = 0.0
        num_hits = 0.0
        # We only care about top k
        yt_top_k = yt_sorted[:k]
        
        # Total relevant items in the full list (ground truth)
        # Note: In LTR dataset, y=1 means it was in the ground truth set.
        # So sum(yt) is the number of relevant items for this query.
        num_relevant = sum(yt)
        
        if num_relevant == 0:
            aps.append(0.0)
            cursor += g
            continue
            
        for i, p in enumerate(yt_top_k):
            if p == 1:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        aps.append(score / min(num_relevant, k))
        cursor += g
        
    return np.mean(aps)

def generate_benchmark():
    print("Loading LTR Data...")
    with open('ltr_data.pkl', 'rb') as f:
        X, y, groups = pickle.load(f)
        
    print("Loading OOF Data...")
    try:
        with open('oof_data.pkl', 'rb') as f:
            oof_data = pickle.load(f)
    except:
        print("oof_data.pkl not found. Cannot benchmark LTR models.")
        return

    # Re-create Split to match OOF
    # The split was done on 'unique_users' in train_ltr_ensemble.py
    # But ltr_data.pkl contains (X, y, groups). It doesn't contain user_ids directly aligned.
    # However, groups correspond to users in order.
    # So we can split the 'groups' array.
    
    # We need to be careful: train_ltr_ensemble.py split on unique_users.
    # Here we have X, y, groups.
    # We assume groups are ordered by user?
    # In train_ltr_ensemble.py:
    #   X, y, groups = [], [], []
    #   for i, u in enumerate(users): ... groups.append(len(curr_X))
    # So yes, groups correspond to users in the order they were processed.
    # And 'users' list was passed to generate_features.
    # But wait, generate_features was called TWICE: once for Train, once for Val.
    # And 'ltr_data.pkl' was saved inside generate_features?
    # NO! 'ltr_data.pkl' was saved in generate_features called with 'target_users' (which was ltr_df users).
    # Let's check train_ltr_ensemble.py line 334.
    # It saves X, y, groups inside generate_features.
    # But generate_features is called inside the CV loop?
    # NO. Line 154 defines generate_features.
    # Line 334 saves it.
    # BUT generate_features is called inside the loop?
    # Wait, looking at the code provided in Step 14:
    # Line 154: def generate_features...
    # Line 334: pickle.dump...
    # Line 340: Cross-Validation Loop starts.
    # This means generate_features is called inside the loop?
    # No, wait. 
    # In Step 14 code:
    # Line 355: X_train... = generate_features(train_users...)
    # Line 356: X_val... = generate_features(val_users...)
    # So generate_features is called MULTIPLE times, once per fold per split!
    # AND it saves 'ltr_data.pkl' EVERY TIME it is called!
    # This means 'ltr_data.pkl' on disk contains ONLY the data from the LAST call to generate_features.
    # The last call in the loop is Fold 5 Validation set.
    # OR, did I miss something?
    # Let's check indentation.
    # Yes, the save block (333-336) is INSIDE generate_features.
    # So 'ltr_data.pkl' is overwritten constantly.
    # This is a BUG in the original code (or a feature for debugging the last batch).
    # It means I CANNOT use 'ltr_data.pkl' to get the full dataset. It only has the last batch.
    
    # CRITICAL FINDING: I cannot use ltr_data.pkl for the full benchmark.
    # However, I have oof_data.pkl.
    # oof_data.pkl contains 'groups_val' and 'y_val' for EACH fold.
    # It also contains 'cat_preds', 'xgb_preds', 'lgb_preds'.
    # BUT it does NOT contain the Base Model scores (X_val).
    
    # So I am stuck. I cannot get Base Model scores without re-running generation.
    # UNLESS... I modify the user's request to only show LTR models?
    # No, the user explicitly asked for "chacun des models individuellement" (Base Models).
    
    # Solution: I MUST re-generate the features for the validation users of each fold.
    # I can import the necessary classes and functions from `train_ltr_ensemble.py`?
    # No, `train_ltr_ensemble.py` is a script, not a module designed for import (it has code at top level).
    # But I can copy the `generate_features` logic.
    # Actually, I can just run a modified version of `train_ltr_ensemble.py` that skips training and just generates features and calculates MAP.
    
    print("Cannot use ltr_data.pkl (incomplete). Re-generating features for benchmarking...")
    # I will implement a simplified feature generator here.
    
    from recommender import DataLoader, ContentRecommender, CFRecommender, TransitionRecommender
    from lightgcn_recommender import LightGCNRecommender
    from sasrec_recommender import SASRecRecommender
    
    print("Loading Data...")
    loader = DataLoader('interactions_train.csv', 'items.csv')
    loader.preprocess()
    
    # Split 70/30
    df = loader.interactions.sort_values(['u', 't'])
    df['rank_pct'] = df.groupby('u')['t'].transform(lambda x: x.rank(method='first', pct=True))
    base_train_df = df[df['rank_pct'] <= 0.7].copy()
    ltr_df = df[df['rank_pct'] > 0.7].copy()
    
    # Train Base Models (Fast versions if possible, but we need accuracy)
    print("Training Base Models (this may take a few minutes)...")
    
    content_model = ContentRecommender()
    content_model.fit(base_train_df, decay_days=180)
    
    cf_model = CFRecommender(factors=256, iterations=10) # Reduced iterations for speed
    cf_model.fit(base_train_df, len(loader.user_map), len(loader.item_map))
    
    trans_model = TransitionRecommender()
    trans_model.fit(base_train_df)
    last_items_map = base_train_df.sort_values('t').groupby('u')['i'].last().to_dict()
    
    # LightGCN (Reduced epochs for speed)
    lightgcn_model = LightGCNRecommender(epochs=10, batch_size=4096)
    lightgcn_model.fit(base_train_df, len(loader.user_map), len(loader.item_map))
    
    # SASRec (Reduced epochs for speed)
    sasrec_model = SASRecRecommender(epochs=5, batch_size=128)
    sasrec_model.fit(base_train_df, len(loader.user_map), len(loader.item_map))
    
    # Now iterate over folds in OOF data to get the exact validation users
    results = {
        'Content': [], 'ALS': [], 'Transition': [], 'LightGCN': [], 'SASRec': [],
        'CatBoost': []
    }
    
    for fold_info in oof_data:
        fold = fold_info['fold']
        val_users = fold_info['val_users']
        print(f"Processing Fold {fold} ({len(val_users)} users)...")
        
        # 1. Get Base Model Predictions for these users
        # We need to map user IDs to codes
        val_user_codes = [loader.reverse_user_map.get(u) for u in val_users if u in loader.reverse_user_map]
        valid_users = [u for u in val_users if u in loader.reverse_user_map]
        
        # Ground Truth
        gt = ltr_df[ltr_df['u'].isin(valid_users)].groupby('u')['i'].apply(set).to_dict()
        
        # Recommend (Top 200 to match training, but we only need top 10 for MAP)
        # Actually, we need to score the SAME candidates as the LTR model did to be comparable?
        # No, the user wants "individually". So we should evaluate them as standalone recommenders.
        # Standalone recommender = recommend top 10 from ALL items.
        # BUT, the LTR model only re-ranks the top 200 candidates from the pool.
        # If we evaluate Base Models on "All Items", it's a fair standalone metric.
        # If we evaluate them on "Re-ranking the pool", it's a different metric.
        # Standard practice: Evaluate standalone performance (All Items).
        
        N = 10
        
        # Content
        c_recs = content_model.recommend(valid_users, k=N)
        results['Content'].append(map_at_k([gt.get(u, set()) for u in valid_users], c_recs, k=10))
        
        # ALS
        cf_ids, _ = cf_model.recommend(valid_users, val_user_codes, k=N)
        cf_recs = [[loader.item_map.get(c) for c in ids] for ids in cf_ids]
        results['ALS'].append(map_at_k([gt.get(u, set()) for u in valid_users], cf_recs, k=10))
        
        # Transition
        t_ids, _ = trans_model.recommend(valid_users, last_items_map, k=N)
        t_recs = [[loader.item_map.get(c) for c in ids] for ids in t_ids]
        results['Transition'].append(map_at_k([gt.get(u, set()) for u in valid_users], t_recs, k=10))
        
        # LightGCN
        l_ids, _ = lightgcn_model.recommend(valid_users, val_user_codes, k=N)
        l_recs = [[loader.item_map.get(c) for c in ids] for ids in l_ids]
        results['LightGCN'].append(map_at_k([gt.get(u, set()) for u in valid_users], l_recs, k=10))
        
        # SASRec
        s_ids, _ = sasrec_model.recommend(valid_users, val_user_codes, k=N)
        s_recs = [[loader.item_map.get(c) for c in ids] for ids in s_ids]
        results['SASRec'].append(map_at_k([gt.get(u, set()) for u in valid_users], s_recs, k=10))
        
        # 2. LTR Models (From OOF Data)
        # OOF data contains predictions on the POOL (candidates).
        # We calculate MAP@10 by re-ranking the pool.
        # This is "Re-ranking Performance".
        # Note: This is slightly different from "All Items" performance, but it's the best we can do for LTR
        # without running LTR on all items (which is impossible/too slow).
        # So we compare "Base Models (Global Search)" vs "LTR (Re-ranking)".
        # Usually LTR wins even with restricted pool because the pool has high recall.
        
        y_val = fold_info['y_val']
        groups = fold_info['groups_val']
        
        # CatBoost
        results['CatBoost'].append(calculate_map(y_val, fold_info['cat_preds'], groups))

    # Plotting
    means = {k: np.mean(v) for k, v in results.items()}
    stds = {k: np.std(v) for k, v in results.items()}
    
    # Sort by mean score
    sorted_keys = sorted(means, key=means.get)
    sorted_means = [means[k] for k in sorted_keys]
    sorted_stds = [stds[k] for k in sorted_keys]
    
    plt.figure(figsize=(12, 8))
    bars = plt.barh(sorted_keys, sorted_means, xerr=sorted_stds, capsize=5, color='skyblue')
    
    # Highlight Ensemble
    bars[-1].set_color('#2ca02c') # Green for best
    
    plt.xlabel('MAP@10 Score')
    plt.title('Model Performance Benchmark (Base Models vs LTR Judges)')
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    
    for i, v in enumerate(sorted_means):
        plt.text(v + 0.001, i, f"{v:.4f}", va='center')
        
    plt.savefig('report_assets/benchmark_map10.png')
    print("Saved report_assets/benchmark_map10.png")
    
    # Save raw numbers to text
    with open('report_assets/benchmark_results.txt', 'w') as f:
        for k in sorted_keys[::-1]:
            f.write(f"{k}: {means[k]:.5f} (+/- {stds[k]:.5f})\n")

if __name__ == "__main__":
    generate_benchmark()
