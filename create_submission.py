import pandas as pd
import numpy as np
from recommender import DataLoader, ContentRecommender, CFRecommender, TransitionRecommender
from lightgcn_recommender import LightGCNRecommender
from sasrec_recommender import SASRecRecommender
from catboost import CatBoostRanker, CatBoostClassifier
import pickle
import os
import joblib

# Imports moved to local scope to avoid OpenMP conflicts with PyTorch
HAS_XGB = True
HAS_LGB = True

def create_submission():
    print("Loading data...")
    loader = DataLoader('interactions_train.csv', 'items.csv')
    loader.preprocess()
    
    # Train on full dataset
    print("Training Content Model on full dataset...")
    content_model = ContentRecommender()
    content_model.fit(loader.interactions, decay_days=180)
    
    print("Training CF Model on full dataset...")
    cf_model = CFRecommender(factors=256, regularization=0.01, iterations=20)
    num_users = len(loader.user_map)
    num_items = len(loader.item_map)
    cf_model.fit(loader.interactions, num_users, num_items, alpha=80)
    
    print("Training Transition Model on full dataset...")
    trans_model = TransitionRecommender()
    trans_model.fit(loader.interactions)
    
    print("Training LightGCN Model on full dataset...")
    lightgcn_model = LightGCNRecommender(
        embedding_dim=256,
        n_layers=2,
        lr=0.0032,
        epochs=50,
        batch_size=2048
    )
    lightgcn_model.fit(loader.interactions, num_users, num_items)
    
    print("Training SASRec Model (Optimized) on full dataset...")
    sasrec_model = SASRecRecommender(
        hidden_units=32,
        num_blocks=1,
        num_heads=4,
        dropout_rate=0.43,
        maxlen=100,
        batch_size=64,
        lr=0.0005,
        epochs=20
    )
    sasrec_model.fit(loader.interactions, num_users, num_items)
    
    # Load enriched book data for metadata features
    print("Loading enriched book data...")
    books_df = pd.read_csv('books_enriched_FINAL.csv')
    item_to_author = books_df.set_index('i')['Author'].to_dict()
    item_to_year = books_df.set_index('i')['published_year'].to_dict()
    item_to_pages = books_df.set_index('i')['page_count'].to_dict()
    item_to_publisher = books_df.set_index('i')['Publisher'].to_dict()
    
    # Process Year
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
    
    # Load CatBoost Models (5 Folds)
    print("Loading CatBoost models (5 Folds)...")
    
    cat_models = []
    n_folds = 5
    
    # CatBoost
    for i in range(n_folds):
        try:
            m = CatBoostRanker()
            m.load_model(f"ltr_catboost_fold{i}.cbm")
            cat_models.append(m)
            print(f"Loaded ltr_catboost_fold{i}.cbm")
        except:
            print(f"Could not load ltr_catboost_fold{i}.cbm")
            
    # Load submission users
    submission_df = pd.read_csv('sample_submission.csv')
    target_users = submission_df['user_id'].unique()
    target_user_codes = [loader.reverse_user_map.get(u, -1) for u in target_users]
    
    # We need last items for Transition
    last_items_map = loader.interactions.sort_values('t').groupby('u')['i'].last().to_dict()
    
    # Candidates
    N = 500 # Increased Candidate Size
    print(f"  Getting top {N} candidates...")
    c_recs, c_scores = content_model.recommend(target_users, k=N, return_scores=True)
    cf_ids, cf_scores = cf_model.recommend(target_users, target_user_codes, k=N, filter_already_liked_items=False)
    t_ids, t_scores = trans_model.recommend(target_users, last_items_map, k=N)
    l_ids, l_scores = lightgcn_model.recommend(target_users, target_user_codes, k=N, filter_already_liked_items=False)
    s_ids, s_scores = sasrec_model.recommend(target_users, target_user_codes, k=N, filter_already_liked_items=False)
    
    # Pre-compute stats
    item_pop = loader.interactions['i'].value_counts().to_dict()
    user_act = loader.interactions['u'].value_counts().to_dict()
    
    author_pop = {}
    for i, count in item_pop.items():
        auth = item_to_author.get(i, "Unknown")
        author_pop[auth] = author_pop.get(auth, 0) + count
        
    # User History for Affinity (using full interactions)
    print("Building user history for affinity features...")
    # Store (item, timestamp) tuples
    user_history_items = loader.interactions.groupby('u')[['i', 't']].apply(lambda x: list(zip(x['i'], x['t']))).to_dict()
    
    user_author_counts = {}
    user_subject_counts = {}
    user_publisher_counts = {}
    user_centroids = {}
    
    # Pre-calculate centroids with Time Decay
    print("Calculating Time-Weighted User Centroids...")
    DECAY_DAYS = 180
    
    # Only need to compute for target users
    for u in target_users:
        items = user_history_items.get(u, [])
        u_auths = {}
        u_subjs = {}
        u_pubs = {}
        u_embs = []
        u_weights = []
        
        if not items:
            user_centroids[u] = np.zeros(384)
            user_author_counts[u] = {}
            user_subject_counts[u] = {}
            user_publisher_counts[u] = {}
            continue
            
        # Find max time for this user to calculate relative decay
        max_t = max([t for _, t in items])
        
        for i, t in items:
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
                # Assuming 't' is integer days or similar from dataset
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
    
    final_recs = []
    
    print("  Re-ranking...")
    
    def standardize(x):
        if len(x) == 0: return x
        return (x - np.mean(x)) / (np.std(x) + 1e-8)
        
    for i, u in enumerate(target_users):
        item_feats = {}
        
        # Temporary structures for aggregation
        # We have 5 models
        item_ranks = {} # item -> [999, 999, 999, 999, 999]
        item_norm_scores = {} # item -> [0.0, 0.0, 0.0, 0.0, 0.0]
        
        # Get user preferences
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
            final_recs.append([])
            continue
        
        curr_X = []
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
            curr_items.append(item)
            
        
        # Weighted Averaging
        # Load best weights if available
        # Force CatBoost Only (Best Single Model)
        print("Using CatBoost Only (w=1.0)...")
        # Predict with CatBoost Ensemble (Average of 5 folds)
        # Using CatBoost Only
        cat_preds = np.zeros(len(curr_X))
        if cat_models:
            for model in cat_models:
                cat_preds += model.predict(curr_X)
            cat_preds /= len(cat_models)
        
        final_scores = cat_preds
        
        # Sort
        sorted_indices = np.argsort(final_scores)[::-1]
        top_items = [curr_items[idx] for idx in sorted_indices[:10]]
        final_recs.append(top_items)
        
    # Create Submission File
    print("Creating submission file...")
    with open('submission.csv', 'w') as f:
        f.write("user_id,recommendation\n")
        for u, recs in zip(target_users, final_recs):
            # Convert back to original item IDs (already done if item_map used correctly)
            # recs are already original item IDs
            items_str = " ".join([str(x) for x in recs])
            f.write(f"{u},{items_str}\n")
            
    print("Saved submission.csv")

if __name__ == "__main__":
    create_submission()
