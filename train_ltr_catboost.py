import pandas as pd
import numpy as np
# Imports moved to local scope to avoid OpenMP conflicts with PyTorch
HAS_XGB = True
HAS_LGB = True

from catboost import CatBoostRanker, Pool
from recommender import DataLoader, ContentRecommender, CFRecommender, TransitionRecommender, map_at_k
from lightgcn_recommender import LightGCNRecommender
from sasrec_recommender import SASRecRecommender
import pickle
import os
from sklearn.model_selection import KFold

def train_ltr_ensemble():
    print("Loading data...")
    loader = DataLoader('interactions_train.csv', 'items.csv')
    loader.preprocess()
    
    # Load enriched book data for metadata features
    print("Loading enriched book data...")
    books_df = pd.read_csv('books_enriched_FINAL.csv')
    # Create mappings for features
    item_to_author = books_df.set_index('i')['Author'].to_dict()
    item_to_year = books_df.set_index('i')['published_year'].to_dict()
    item_to_pages = books_df.set_index('i')['page_count'].to_dict()
    item_to_publisher = books_df.set_index('i')['Publisher'].to_dict()
    
    # Process Year (handle NaNs and convert to int)
    def clean_year(y):
        try:
            return int(float(str(y)[:4]))
        except:
            return -1
            
    item_to_year = {k: clean_year(v) for k, v in item_to_year.items()}
    
    # Process Subjects (split by ;)
    print("Processing subjects...")
    item_to_subjects = books_df.set_index('i')['Subjects'].to_dict()
    def clean_subjects(s):
        if pd.isna(s): return []
        return [x.strip() for x in str(s).split(';')]
    item_to_subjects = {k: clean_subjects(v) for k, v in item_to_subjects.items()}
    
    # Load Embeddings
    print("Loading embeddings...")
    with open('item_embeddings.pkl', 'rb') as f:
        emb_data = pickle.load(f)
    
    item_embeddings = dict(zip(emb_data['item_ids'], emb_data['embeddings']))
    
    # 1. Split Data: Base Train (70%) / LTR Data (30%)
    df = loader.interactions.sort_values(['u', 't'])
    df['rank_pct'] = df.groupby('u')['t'].transform(
        lambda x: x.rank(method='first', pct=True)
    )
    
    base_train_df = df[df['rank_pct'] <= 0.7].copy()
    ltr_df = df[df['rank_pct'] > 0.7].copy()
    
    print(f"Base Train: {len(base_train_df)} interactions")
    print(f"LTR Data: {len(ltr_df)} interactions")
    
    # 2. Train Base Models (On Base Train ONLY to avoid leakage)
    print("Training Base Models (On 70% Split)...")
    
    # Content
    print("Training Content...")
    content_model = ContentRecommender()
    content_model.fit(base_train_df, decay_days=180)
    
    # CF
    print("Training CF...")
    cf_model = CFRecommender(factors=256, regularization=0.01, iterations=20)
    num_users = len(loader.user_map)
    num_items = len(loader.item_map)
    cf_model.fit(base_train_df, num_users, num_items, alpha=80)
    
    # Transition
    print("Training Transition...")
    trans_model = TransitionRecommender()
    trans_model.fit(base_train_df)
    last_items_map = base_train_df.sort_values('t').groupby('u')['i'].last().to_dict()
    
    # LightGCN
    print("Training LightGCN...")
    lightgcn_model = LightGCNRecommender(
        embedding_dim=256,
        n_layers=3,
        reg_weight=1e-4,
        lr=0.0032,
        epochs=50,
        batch_size=2048
    )
    lightgcn_model.fit(base_train_df, num_users, num_items)
    
    # SASRec
    print("Training SASRec...")
    sasrec_model = SASRecRecommender(
        hidden_units=128,
        num_blocks=2,
        num_heads=4,
        dropout_rate=0.43,
        maxlen=100,
        batch_size=64,
        lr=0.0005,
        epochs=20
    )
    sasrec_model.fit(base_train_df, num_users, num_items)
    
    # Pre-compute global stats (from base_train_df)
    item_pop = base_train_df['i'].value_counts().to_dict()
    user_act = base_train_df['u'].value_counts().to_dict()
    
    # Author Popularity
    author_pop = {}
    for i, count in item_pop.items():
        auth = item_to_author.get(i, "Unknown")
        author_pop[auth] = author_pop.get(auth, 0) + count
        
    # User History for Affinity (using full interactions)
    print("Building user history for affinity features...")
    # Store (item, timestamp) tuples
    # CRITICAL FIX: Use base_train_df ONLY to avoid leakage!
    user_history_items = base_train_df.groupby('u')[['i', 't']].apply(lambda x: list(zip(x['i'], x['t']))).to_dict()
    
    user_author_counts = {}
    user_subject_counts = {}
    user_publisher_counts = {}
    user_centroids = {}
    
    # Pre-calculate centroids with Time Decay
    print("Calculating Time-Weighted User Centroids...")
    DECAY_DAYS = 180
    
    for u, history in user_history_items.items():
        u_auths = {}
        u_subjs = {}
        u_pubs = {}
        u_embs = []
        u_weights = []
        
        if not history:
            user_centroids[u] = np.zeros(384)
            continue
            
        # Find max time for this user to calculate relative decay
        max_t = max([t for _, t in history])
        
        for i, t in history:
            # Author
            auth = item_to_author.get(i)
            if auth: u_auths[auth] = u_auths.get(auth, 0) + 1
            # Subjects
            subjs = item_to_subjects.get(i, [])
            for s in subjs: u_subjs[s] = u_subjs.get(s, 0) + 1
            # Publisher
            pub = item_to_publisher.get(i)
            if pub: u_pubs[pub] = u_pubs.get(pub, 0) + 1
            
            # Embedding & Weight
            if i in item_embeddings: 
                u_embs.append(item_embeddings[i])
                # Time Decay Weight: exp(-ln(2) * days / half_life)
                days_diff = (max_t - t) / (24*3600*1000) if t > 1e10 else (max_t - t) # Handle ms vs s if needed, assuming days or similar unit
                # Assuming 't' is integer days or similar from dataset
                # Let's use the same as ContentRecommender: decay_days=180
                weight = np.exp(-(max_t - t) / DECAY_DAYS)
                u_weights.append(weight)
                
        user_author_counts[u] = u_auths
        user_subject_counts[u] = u_subjs
        user_publisher_counts[u] = u_pubs
        
        if u_embs: 
            # Weighted Average
            user_centroids[u] = np.average(u_embs, axis=0, weights=u_weights)
        else: 
            user_centroids[u] = np.zeros(384)

    # 3. Generate Features (Function)
    def generate_features(target_users, name=""):
        print(f"Generating features for {name}...")
        users = [u for u in target_users if u in loader.reverse_user_map]
        user_codes = [loader.reverse_user_map[u] for u in users]
        
        # Ground Truth (from ltr_df for these users)
        ground_truth = ltr_df[ltr_df['u'].isin(users)].groupby('u')['i'].apply(set).to_dict()
        
        N = 200 # Increased Candidate Size
        print(f"  Getting top {N} candidates...")
        c_recs, c_scores = content_model.recommend(users, k=N, return_scores=True)
        cf_ids, cf_scores = cf_model.recommend(users, user_codes, k=N, filter_already_liked_items=False)
        t_ids, t_scores = trans_model.recommend(users, last_items_map, k=N)
        l_ids, l_scores = lightgcn_model.recommend(users, user_codes, k=N, filter_already_liked_items=False)
        s_ids, s_scores = sasrec_model.recommend(users, user_codes, k=N, filter_already_liked_items=False)
        
        X = []
        y = []
        groups = []
        all_items = [] 
        
        print("  Constructing dataset...")
        for i, u in enumerate(users):
            gt = ground_truth.get(u, set())
            item_feats = {}
            
            # Temporary structures for aggregation
            # We have 5 models
            item_ranks = {} # item -> [999, 999, 999, 999, 999]
            item_norm_scores = {} # item -> [0.0, 0.0, 0.0, 0.0, 0.0]
            
            u_auth_prefs = user_author_counts.get(u, {})
            u_subj_prefs = user_subject_counts.get(u, {})
            u_pub_prefs = user_publisher_counts.get(u, {})
            u_centroid = user_centroids.get(u, np.zeros(384))
            
            last_item = last_items_map.get(u)
            last_item_year = item_to_year.get(last_item, -1)
            last_item_emb = item_embeddings.get(last_item, np.zeros(384)) if last_item else np.zeros(384)
            
            def add_feats(items, scores, model_idx):
                # Normalize scores for this model batch (MinMax to 0-1)
                if len(scores) > 0:
                    min_s = np.min(scores)
                    max_s = np.max(scores)
                    if max_s > min_s:
                        norm_scores = (scores - min_s) / (max_s - min_s)
                    else:
                        norm_scores = np.ones_like(scores) if max_s > 0 else np.zeros_like(scores)
                else:
                    norm_scores = []

                for rank, (item, score, n_score) in enumerate(zip(items, scores, norm_scores)):
                    if item is None: continue
                    
                    if item not in item_feats: 
                        # Initialize feature vector (30 features)
                        item_feats[item] = [0.0] * 30
                        item_ranks[item] = [999] * 5
                        item_norm_scores[item] = [0.0] * 5
                        
                        # Static Features
                        item_feats[item][5] = item_pop.get(item, 0)
                        item_feats[item][6] = user_act.get(u, 0)
                        
                        auth = item_to_author.get(item, "Unknown")
                        item_feats[item][7] = author_pop.get(auth, 0)
                        
                        item_year = item_to_year.get(item, -1)
                        item_feats[item][8] = item_year
                        item_feats[item][9] = item_to_pages.get(item, -1)
                        
                        # Affinity Features
                        auth_count = u_auth_prefs.get(auth, 0)
                        item_feats[item][10] = auth_count
                        
                        subjs = item_to_subjects.get(item, [])
                        subj_score = sum([u_subj_prefs.get(s, 0) for s in subjs])
                        item_feats[item][11] = subj_score
                        
                        pub = item_to_publisher.get(item)
                        pub_count = u_pub_prefs.get(pub, 0)
                        item_feats[item][12] = pub_count
                        
                        # Semantic Features
                        item_emb = item_embeddings.get(item, np.zeros(384))
                        
                        def cosine(a, b):
                            norm_a = np.linalg.norm(a)
                            norm_b = np.linalg.norm(b)
                            if norm_a == 0 or norm_b == 0: return 0.0
                            return np.dot(a, b) / (norm_a * norm_b)
                            
                        item_feats[item][13] = cosine(u_centroid, item_emb)
                        item_feats[item][14] = cosine(last_item_emb, item_emb)
                        
                        # --- Advanced Features ---
                        
                        # 15. Ratio of Author Reads
                        # history is now list of tuples, so len(history)
                        total_reads = len(user_history_items.get(u, []))
                        if total_reads > 0:
                            item_feats[item][15] = auth_count / total_reads
                            item_feats[item][16] = pub_count / total_reads
                        else:
                            item_feats[item][15] = 0.0
                            item_feats[item][16] = 0.0
                            
                        # 17. Subject Match Ratio
                        if subjs:
                            match_count = sum([1 for s in subjs if s in u_subj_prefs])
                            item_feats[item][17] = match_count / len(subjs)
                        else:
                            item_feats[item][17] = 0.0
                            
                        # --- Phase 7 New Features ---
                        
                        # 18. Time since last read (Year Diff)
                        if last_item_year != -1 and item_year != -1:
                            item_feats[item][18] = abs(item_year - last_item_year)
                        else:
                            item_feats[item][18] = -1
                            
                        # 19. Co-occurrence (Same Author as Last Read)
                        auth_last = item_to_author.get(last_item)
                        if auth_last and auth == auth_last:
                            item_feats[item][19] = 1.0
                        else:
                            item_feats[item][19] = 0.0
                        
                    # Store Raw Score
                    item_feats[item][model_idx] = score
                    
                    # Store Rank and Norm Score for aggregation
                    item_ranks[item][model_idx] = rank + 1 # 1-based rank
                    item_norm_scores[item][model_idx] = n_score
            
            add_feats(c_recs[i], c_scores[i], 0)
            add_feats([loader.item_map.get(c) for c in cf_ids[i]], cf_scores[i], 1)
            add_feats([loader.item_map.get(c) for c in t_ids[i]], t_scores[i], 2)
            add_feats([loader.item_map.get(c) for c in l_ids[i]], l_scores[i], 3)
            add_feats([loader.item_map.get(c) for c in s_ids[i]], s_scores[i], 4)
            
            if not item_feats: 
                groups.append(0)
                continue
            
            curr_X = []
            curr_y = []
            curr_items = []
            
            for item, feats in item_feats.items():
                # --- Phase 8: Rank & Aggregation Features ---
                ranks = item_ranks[item]
                n_scores = item_norm_scores[item]
                
                # 20-24: Ranks
                for m_idx in range(5):
                    feats[20 + m_idx] = ranks[m_idx]
                    
                # 25: Vote Count (how many models ranked it < 999)
                vote_count = sum(1 for r in ranks if r < 999)
                feats[25] = vote_count
                
                # 26-29: Score Aggregates
                feats[26] = np.mean(n_scores)
                feats[27] = np.std(n_scores)
                feats[28] = np.min(n_scores)
                feats[29] = np.max(n_scores)
                
                curr_X.append(feats)
                curr_y.append(1 if item in gt else 0)
                curr_items.append(item)
                
            X.extend(curr_X)
            y.extend(curr_y)
            groups.append(len(curr_X))
            all_items.append(curr_items)
            
        # Save LTR Data for Optuna Tuning
        print("Saving LTR Data to ltr_data.pkl...")
        with open('ltr_data.pkl', 'wb') as f:
            pickle.dump((np.array(X), np.array(y), np.array(groups)), f)
        print("Saved ltr_data.pkl")
            
        return np.array(X), np.array(y), np.array(groups), users, ground_truth, all_items

    # 4. Cross-Validation Loop on LTR Data
    print("Starting 5-Fold Cross-Validation on LTR Data...")
    unique_users = ltr_df['u'].unique()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_scores = []
    
    oof_data = []
    
    for fold, (train_idx, val_idx) in enumerate(kf.split(unique_users)):
        print(f"--- Fold {fold+1}/5 ---")
        train_users = unique_users[train_idx]
        val_users = unique_users[val_idx]
        
        # We can use all train_users now since ltr_df is smaller (30% of total)
        X_train, y_train, groups_train, _, _, _ = generate_features(train_users, f"Fold {fold+1} Train")
        X_val, y_val, groups_val, val_users_list, val_gt, val_items_list = generate_features(val_users, f"Fold {fold+1} Val")
        
        # Train Models for this Fold
        
        # --- CatBoost ---
        print(f"Training CatBoost (Fold {fold+1})...")
        train_pool = Pool(data=X_train, label=y_train, group_id=np.repeat(range(len(groups_train)), groups_train))
        val_pool = Pool(data=X_val, label=y_val, group_id=np.repeat(range(len(groups_val)), groups_val))
        
        cat_model = CatBoostRanker(
            iterations=1000,
            learning_rate=0.1,
            depth=6,
            loss_function='YetiRank',
            eval_metric='MAP:top=10',
            verbose=False,
            random_seed=42
        )
        cat_model.fit(train_pool, eval_set=val_pool, early_stopping_rounds=50)
        cat_model.save_model(f"ltr_catboost_fold{fold}.cbm")
        
        # Evaluate Fold
        print(f"Evaluating Fold {fold+1}...")
        
        cat_preds = cat_model.predict(X_val)
        
        # Save OOF Data (CatBoost Only)
        oof_data.append({
            'fold': fold,
            'cat_preds': cat_preds,
            'y_val': y_val,
            'groups_val': groups_val,
            'val_users': val_users_list
        })
        
        final_recs = []
        cursor = 0
        for i, g in enumerate(groups_val):
            if g == 0:
                final_recs.append([])
                continue
            p = cat_preds[cursor : cursor + g]
            items = val_items_list[i]
            sorted_indices = np.argsort(p)[::-1]
            top_items = [items[idx] for idx in sorted_indices[:10]]
            final_recs.append(top_items)
            cursor += g
            
        actual = [val_gt[u] for u in val_users_list]
        score = map_at_k(actual, final_recs, k=10)
        print(f"Fold {fold+1} MAP@10: {score:.5f}")
        fold_scores.append(score)
        
    print(f"Average CV MAP@10: {np.mean(fold_scores):.5f}")
    
    print("Saving OOF data...")
    with open('oof_data.pkl', 'wb') as f:
        pickle.dump(oof_data, f)
    print("Saved oof_data.pkl")

if __name__ == "__main__":
    train_ltr_ensemble()
