import pandas as pd
import numpy as np
import scipy.sparse as sparse
import implicit
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics.pairwise import cosine_similarity

class DataLoader:
    def __init__(self, interactions_path, items_path):
        self.interactions = pd.read_csv(interactions_path)
        self.items = pd.read_csv(items_path)
        self.user_map = None
        self.item_map = None
        self.reverse_user_map = None
        self.reverse_item_map = None

    def preprocess(self):
        # Create mappings
        self.interactions['u'] = self.interactions['u'].astype("category")
        self.interactions['i'] = self.interactions['i'].astype("category")
        
        self.user_map = dict(enumerate(self.interactions['u'].cat.categories))
        self.item_map = dict(enumerate(self.interactions['i'].cat.categories))
        
        self.reverse_user_map = {v: k for k, v in self.user_map.items()}
        self.reverse_item_map = {v: k for k, v in self.item_map.items()}
        
        self.interactions['user_id_code'] = self.interactions['u'].cat.codes
        self.interactions['item_id_code'] = self.interactions['i'].cat.codes


    def get_train_val_split(self, val_ratio=0.2, strategy='user_time'):
        print(f"Splitting data with strategy='{strategy}' and ratio={val_ratio}...")
        
        if strategy == 'user_time':
            # Cas 2 : Split par utilisateur (Time-based per user)
            # 1. On trie d'abord par utilisateur ET par temps
            df_sorted = self.interactions.sort_values(['u', 't'])

            # 2. On calcule le rang de chaque interaction pour chaque utilisateur
            df_sorted['rank_pct'] = df_sorted.groupby('u')['t'].transform(
                lambda x: x.rank(method='first', pct=True)
            )

            # 3. On coupe
            train_df = df_sorted[df_sorted['rank_pct'] <= (1 - val_ratio)]
            val_df = df_sorted[df_sorted['rank_pct'] > (1 - val_ratio)]

            train_df = train_df.drop(columns=['rank_pct'])
            val_df = val_df.drop(columns=['rank_pct'])
            
        elif strategy == 'global_time':
            # Split global par temps
            df_sorted = self.interactions.sort_values('t')
            split_idx = int(len(df_sorted) * (1 - val_ratio))
            
            train_df = df_sorted.iloc[:split_idx]
            val_df = df_sorted.iloc[split_idx:]
            
        elif strategy == 'random':
            # Split random global
            train_df, val_df = train_test_split(self.interactions, test_size=val_ratio, random_state=42)
            
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        print(f"Split terminé.")
        print(f"Train size: {len(train_df)}")
        print(f"Val size: {len(val_df)}")
        print(f"Users in Train: {train_df['u'].nunique()}")
        print(f"Users in Val: {val_df['u'].nunique()}")

        return train_df, val_df

class PopularityRecommender:
    def __init__(self):
        self.popular_items = []

    def fit(self, train_df):
        self.popular_items = train_df['i'].value_counts().index.tolist()

    def recommend(self, user_ids, k=10):
        recs = []
        top_k = self.popular_items[:k]
        for _ in user_ids:
            recs.append(top_k)
        return recs

#Ancien CFRecommender a reprendre au caou le nouveau plante
# class CFRecommender:
#     def __init__(self, factors=256, regularization=0.01, iterations=20):
#         self.model = implicit.als.AlternatingLeastSquares(
#             factors=factors, regularization=regularization, iterations=iterations, random_state=42
#         )
#         self.user_item_matrix = None

#     def fit(self, train_df, num_users, num_items, alpha=80):
#         # implicit 0.7.x expects (user, item) matrix for fit
        
#         row = train_df['user_id_code'].values
#         col = train_df['item_id_code'].values
#         data = np.ones(len(train_df))
        
#         self.user_item_matrix = sparse.csr_matrix((data, (row, col)), shape=(num_users, num_items))
        
#         # Alpha scaling
#         data_scaled = (self.user_item_matrix * alpha).astype('double')
        
#         self.model.fit(data_scaled)

#     def recommend(self, user_ids, user_id_codes, k=10, filter_already_liked_items=False):
#         # user_id_codes: list of internal codes corresponding to user_ids
#         print(f"DEBUG: recommend called with {len(user_id_codes)} users")
#         print(f"DEBUG: matrix shape: {self.user_item_matrix.shape}")
#         if len(user_id_codes) > 0:
#             print(f"DEBUG: user_id_codes min: {min(user_id_codes)}, max: {max(user_id_codes)}")
        
#         # Implicit 0.7.x batch recommend might require user_items to match the batch size
#         # Slice the matrix to get rows corresponding to user_id_codes
#         user_items_batch = self.user_item_matrix[user_id_codes]
        
#         ids, scores = self.model.recommend(
#             user_id_codes, user_items_batch, N=k, filter_already_liked_items=filter_already_liked_items
#         )
#         return ids, scores

class CFRecommender:
    def __init__(self, factors=256, regularization=0.01, iterations=20):
        self.model = implicit.als.AlternatingLeastSquares(
            factors=factors, regularization=regularization, iterations=iterations, random_state=42
        )
        self.user_item_matrix = None

    def fit(self, train_df, num_users, num_items, alpha=80):
        # implicit 0.7.x expects (user, item) matrix for fit
        
        row = train_df['user_id_code'].values
        col = train_df['item_id_code'].values
        data = np.ones(len(train_df))
        
        self.user_item_matrix = sparse.csr_matrix((data, (row, col)), shape=(num_users, num_items))
        
        # Alpha scaling
        data_scaled = (self.user_item_matrix * alpha).astype('double')
        
        self.model.fit(data_scaled)

    def recommend(self, user_ids, user_id_codes, k=10, filter_already_liked_items=False):
        # user_id_codes: list/array de codes internes
        user_id_codes = np.asarray(user_id_codes, dtype=int)

        print(f"DEBUG: recommend called with {len(user_id_codes)} users")
        print(f"DEBUG: matrix shape: {self.user_item_matrix.shape}")
        if len(user_id_codes) > 0:
            print(f"DEBUG: user_id_codes min: {user_id_codes.min()}, max: {user_id_codes.max()}")

        n_users_cf, _ = self.user_item_matrix.shape

        # Si aucun user_id_code → renvoyer des arrays vides propres
        if len(user_id_codes) == 0:
            empty_ids = np.zeros((len(user_ids), 0), dtype=int)
            empty_scores = np.zeros((len(user_ids), 0), dtype=float)
            return empty_ids, empty_scores

        max_code = user_id_codes.max()
        min_code = user_id_codes.min()

        # ⚠️ Si le code utilisateur sort des bornes de la matrice CF → on ignore CF
        if max_code >= n_users_cf or min_code < 0:
            print(f"⚠️ CF ignoré pour ces users (codes {user_id_codes} hors [0, {n_users_cf-1}])")
            empty_ids = np.zeros((len(user_ids), 0), dtype=int)
            empty_scores = np.zeros((len(user_ids), 0), dtype=float)
            return empty_ids, empty_scores

        # ✅ Ici, tous les codes sont valides
        user_items_batch = self.user_item_matrix[user_id_codes]
        
        ids, scores = self.model.recommend(
            user_id_codes,
            user_items_batch,
            N=k,
            filter_already_liked_items=filter_already_liked_items
        )
        return ids, scores

class ContentRecommender:
    def __init__(self, embeddings_path='item_embeddings.pkl'):
        with open(embeddings_path, 'rb') as f:
            data = pickle.load(f)
        self.item_ids = data['item_ids']
        self.embeddings = data['embeddings']
        self.item_id_to_idx = {iid: idx for idx, iid in enumerate(self.item_ids)}
        self.user_profiles = {}
        self.user_history = {}

    def fit(self, train_df, decay_days=180):
        # Store history for filtering
        self.user_history = train_df.groupby('u')['i'].apply(set).to_dict()
        
        print(f"Fitting ContentRecommender with decay_days={decay_days}")
        
        # Compute user profiles: weighted mean of embeddings based on time
        # Group by user
        # We need timestamps for decay
        
        # Calculate max time for relative decay or use absolute?
        # Let's use decay relative to the user's last interaction or global max?
        # Global max is safer for "current" state.
        max_time = train_df['t'].max()
        
        # Group by user and get list of (item, time)
        # To do this efficiently, let's join or iterate
        
        # Let's iterate over groups
        user_groups = train_df.groupby('u')[['i', 't']].apply(lambda x: list(zip(x['i'], x['t'])))
        
        if decay_days:
            decay_seconds = decay_days * 24 * 3600
        
        for user, interactions in user_groups.items():
            valid_interactions = [(i, t) for i, t in interactions if i in self.item_id_to_idx]
            if not valid_interactions:
                continue
            
            indices = [self.item_id_to_idx[i] for i, t in valid_interactions]
            times = np.array([t for i, t in valid_interactions])
            
            # Weight = exp(-(max_time - t) / decay_seconds)
            # If decay_days is None or very large, weights are ~1
            if decay_days:
                time_diff = max_time - times
                weights = np.exp(-time_diff / decay_seconds)
                # Normalize weights? Not strictly necessary for weighted average but good for stability
                weights = weights / weights.sum()
            else:
                weights = np.ones(len(indices)) / len(indices)
                
            item_embs = self.embeddings[indices]
            
            # Weighted average
            user_profile = np.average(item_embs, axis=0, weights=weights)
            self.user_profiles[user] = user_profile
            
        print(f"Generated profiles for {len(self.user_profiles)} users")

    def recommend(self, user_ids, k=10, filter_history=True, return_scores=False):
        recs = []
        rec_scores = []
        
        # Filter users who have profiles
        users_with_profiles = [u for u in user_ids if u in self.user_profiles]
        if not users_with_profiles:
            if return_scores:
                return [[] for _ in user_ids], [[] for _ in user_ids]
            return [[] for _ in user_ids]
            
        # Create matrix of user profiles
        user_matrix = np.array([self.user_profiles[u] for u in users_with_profiles])
        
        # Compute similarity: (n_users, n_items)
        sim_scores = cosine_similarity(user_matrix, self.embeddings)
        
        # Get top k
        fetch_k = 50 if filter_history else k
        fetch_k = min(fetch_k, self.embeddings.shape[0])
        
        top_k_indices = np.argpartition(sim_scores, -fetch_k, axis=1)[:, -fetch_k:]
        
        results_map = {}
        scores_map = {}
        
        for i, u in enumerate(users_with_profiles):
            indices = top_k_indices[i]
            scores = sim_scores[i, indices]
            
            # Sort by score descending
            sorted_args = np.argsort(scores)[::-1]
            sorted_indices = indices[sorted_args]
            sorted_scores = scores[sorted_args]
            
            # Convert to item ids
            candidates = [self.item_ids[idx] for idx in sorted_indices]
            
            # if filter_history and u in self.user_history:
            #     history = self.user_history[u]
            #     candidates = [c for c in candidates if c not in history]
            
            results_map[u] = candidates[:k]
            scores_map[u] = sorted_scores[:k]
            
        # Align with input user_ids
        for u in user_ids:
            recs.append(results_map.get(u, []))
            rec_scores.append(scores_map.get(u, []))
            
        if return_scores:
            return recs, rec_scores
        return recs

class TransitionRecommender:
    def __init__(self):
        self.transitions = {} # item_id -> {next_item_id: count}
        self.item_map = {}
        
    def fit(self, train_df):
        print("Fitting Transition Model...")
        # Sort by user and time
        df = train_df.sort_values(['u', 't'])
        
        # Group by user and get sequence of items
        # We need original item IDs, not codes, or codes?
        # Let's use codes 'i'
        sequences = df.groupby('u')['i'].apply(list)
        
        for seq in sequences:
            if len(seq) < 2: continue
            for i in range(len(seq) - 1):
                curr_item = seq[i]
                next_item = seq[i+1]
                
                if curr_item not in self.transitions:
                    self.transitions[curr_item] = {}
                
                if next_item not in self.transitions[curr_item]:
                    self.transitions[curr_item][next_item] = 0
                
                self.transitions[curr_item][next_item] += 1
                
        # Normalize
        for item in self.transitions:
            total = sum(self.transitions[item].values())
            for next_item in self.transitions[item]:
                self.transitions[item][next_item] /= total
                
    def recommend(self, user_ids, last_items, k=10):
        # last_items: dict {user_id: last_item_code}
        recs = []
        scores = []
        
        for u in user_ids:
            last_i = last_items.get(u)
            if last_i is not None and last_i in self.transitions:
                # Get top next items
                next_probs = self.transitions[last_i]
                sorted_items = sorted(next_probs.items(), key=lambda x: x[1], reverse=True)
                
                top_k = sorted_items[:k]
                recs.append([x[0] for x in top_k])
                scores.append([x[1] for x in top_k])
            else:
                recs.append([])
                scores.append([])
                
        return recs, scores

class HybridRecommender:
    def __init__(self, content_model, cf_model, trans_model, lightgcn_model, sasrec_model, item_map, 
                 content_weight=0.4, trans_weight=0.05, lightgcn_weight=0.1, sasrec_weight=0.1):
        self.content_model = content_model
        self.cf_model = cf_model
        self.transition_model = trans_model
        self.lightgcn_model = lightgcn_model
        self.sasrec_model = sasrec_model
        self.item_map = item_map
        self.content_weight = content_weight
        self.trans_weight = trans_weight
        self.lightgcn_weight = lightgcn_weight
        self.sasrec_weight = sasrec_weight

    def recommend(self, user_ids, user_id_codes, last_items_map, k=10, use_rrf=False):
        # Get more candidates
        N = 50
        
        # Content recommendations
        content_recs, content_scores = self.content_model.recommend(user_ids, k=N, return_scores=True)
        
        # CF recommendations
        cf_ids_codes, cf_scores = self.cf_model.recommend(user_ids, user_id_codes, k=N, filter_already_liked_items=False)
        
        # Transition recommendations
        trans_recs, trans_scores = self.transition_model.recommend(user_ids, last_items_map, k=N)
        
        # LightGCN recommendations
        lgcn_ids_codes, lgcn_scores = self.lightgcn_model.recommend(user_ids, user_id_codes, k=N, filter_already_liked_items=False)
        
        # SASRec recommendations
        sas_ids_codes, sas_scores = self.sasrec_model.recommend(user_ids, user_id_codes, k=N, filter_already_liked_items=False)
        
        final_recs = []
        
        for i, user in enumerate(user_ids):
            # Content
            c_items = content_recs[i]
            c_scores = content_scores[i]
            
            # CF
            cf_codes = cf_ids_codes[i]
            cf_vals = cf_scores[i]
            cf_items = [self.item_map.get(c, None) for c in cf_codes]
            
            # Transition
            t_codes = trans_recs[i]
            t_vals = trans_scores[i]
            t_items = [self.item_map.get(c, None) for c in t_codes]
            
            # LightGCN
            l_codes = lgcn_ids_codes[i]
            l_vals = lgcn_scores[i]
            l_items = [self.item_map.get(c, None) for c in l_codes]
            
            # SASRec
            s_codes = sas_ids_codes[i]
            s_vals = sas_scores[i]
            s_items = [self.item_map.get(c, None) for c in s_codes]
            
            item_scores = {}
            
            if use_rrf:
                # Reciprocal Rank Fusion
                k_rrf = 60
                
                def add_rrf(items, weight):
                    for rank, item in enumerate(items):
                        if item is None: continue
                        rrf_score = weight * (1.0 / (k_rrf + rank + 1))
                        item_scores[item] = item_scores.get(item, 0) + rrf_score
                        
                add_rrf(c_items, self.content_weight)
                add_rrf(cf_items, 1.0 - self.content_weight - self.trans_weight - self.lightgcn_weight - self.sasrec_weight)
                add_rrf(t_items, self.trans_weight)
                add_rrf(l_items, self.lightgcn_weight)
                add_rrf(s_items, self.sasrec_weight)
                
            else:
                # Weighted Score Averaging
                # Normalize Content
                if len(c_scores) > 0:
                    min_c = min(c_scores)
                    max_c = max(c_scores)
                    range_c = max_c - min_c if max_c != min_c else 1.0
                    for item, score in zip(c_items, c_scores):
                        norm_score = (score - min_c) / range_c
                        item_scores[item] = item_scores.get(item, 0) + self.content_weight * norm_score
                
                # Normalize CF
                cf_weight = 1.0 - self.content_weight - self.trans_weight - self.lightgcn_weight - self.sasrec_weight
                if len(cf_vals) > 0:
                    min_cf = min(cf_vals)
                    max_cf = max(cf_vals)
                    range_cf = max_cf - min_cf if max_cf != min_cf else 1.0
                    for item, score in zip(cf_items, cf_vals):
                        if item is None: continue
                        norm_score = (score - min_cf) / range_cf
                        item_scores[item] = item_scores.get(item, 0) + cf_weight * norm_score
                        
                # Normalize Transition
                if len(t_vals) > 0:
                    min_t = min(t_vals)
                    max_t = max(t_vals)
                    range_t = max_t - min_t if max_t != min_t else 1.0
                    for item, score in zip(t_items, t_vals):
                        if item is None: continue
                        norm_score = (score - min_t) / range_t
                        item_scores[item] = item_scores.get(item, 0) + self.trans_weight * norm_score
                        
                # Normalize LightGCN
                if len(l_vals) > 0:
                    min_l = min(l_vals)
                    max_l = max(l_vals)
                    range_l = max_l - min_l if max_l != min_l else 1.0
                    for item, score in zip(l_items, l_vals):
                        if item is None: continue
                        norm_score = (score - min_l) / range_l
                        item_scores[item] = item_scores.get(item, 0) + self.lightgcn_weight * norm_score

                # Normalize SASRec
                if len(s_vals) > 0:
                    min_s = min(s_vals)
                    max_s = max(s_vals)
                    range_s = max_s - min_s if max_s != min_s else 1.0
                    for item, score in zip(s_items, s_vals):
                        if item is None: continue
                        norm_score = (score - min_s) / range_s
                        item_scores[item] = item_scores.get(item, 0) + self.sasrec_weight * norm_score
            
            # Sort by combined score
            sorted_items = sorted(item_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Take top k
            final_recs.append([item for item, score in sorted_items[:k]])
            
        return final_recs

def map_at_k(actual, predicted, k=10):
    # actual: list of lists of ground truth items
    # predicted: list of lists of predicted items
    
    scores = []
    for act, pred in zip(actual, predicted):
        if not act:
            scores.append(0.0)
            continue
            
        act_set = set(act)
        pred = pred[:k]
        
        score = 0.0
        num_hits = 0.0
        
        for i, p in enumerate(pred):
            if p in act_set:
                num_hits += 1.0
                score += num_hits / (i + 1.0)
        
        scores.append(score / min(len(act), k))
        
    return np.mean(scores)
