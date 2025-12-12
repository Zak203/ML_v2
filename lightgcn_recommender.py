import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from tqdm import tqdm
import scipy.sparse as sp

class LightGCN(nn.Module):
    def __init__(self, num_users, num_items, embedding_dim=64, n_layers=3):
        super(LightGCN, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Initialize embeddings
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.item_embedding = nn.Embedding(num_items, embedding_dim)
        
        # Normal initialization
        nn.init.normal_(self.user_embedding.weight, std=0.1)
        nn.init.normal_(self.item_embedding.weight, std=0.1)
        
    def forward(self, adj_matrix):
        # adj_matrix is a sparse tensor
        
        # Initial embeddings
        ego_embeddings = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        all_embeddings = [ego_embeddings]
        
        for k in range(self.n_layers):
            # Graph convolution: E^(k+1) = D^-0.5 A D^-0.5 E^k
            # We assume adj_matrix is already normalized
            ego_embeddings = torch.sparse.mm(adj_matrix, ego_embeddings)
            all_embeddings.append(ego_embeddings)
            
        # Average layers
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = torch.mean(all_embeddings, dim=1)
        
        user_embeddings, item_embeddings = torch.split(all_embeddings, [self.num_users, self.num_items])
        return user_embeddings, item_embeddings

class LightGCNRecommender:
    def __init__(self, embedding_dim=64, n_layers=3, reg_weight=1e-4, lr=0.001, epochs=20, batch_size=1024, device=None):
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        self.reg_weight = reg_weight
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None
        self.user_map = None
        self.item_map = None
        self.reverse_item_map = None
        
    def _build_adj_matrix(self, df, num_users, num_items):
        # Create user-item adjacency matrix
        # R = [0, A]
        #     [A.T, 0]
        
        user_idx = df['user_id_code'].values
        item_idx = df['item_id_code'].values
        
        # Interaction matrix A
        # We treat all interactions as 1
        ratings = np.ones(len(user_idx))
        
        # Create sparse matrix
        A = sp.coo_matrix((ratings, (user_idx, item_idx)), shape=(num_users, num_items))
        
        # Build symmetric adjacency matrix
        # shape: (num_users + num_items, num_users + num_items)
        rows = np.concatenate([user_idx, item_idx + num_users])
        cols = np.concatenate([item_idx + num_users, user_idx])
        data = np.ones(len(rows))
        
        adj = sp.coo_matrix((data, (rows, cols)), shape=(num_users + num_items, num_users + num_items))
        
        # Normalize: D^-0.5 A D^-0.5
        rowsum = np.array(adj.sum(1))
        d_inv_sqrt = np.power(rowsum, -0.5).flatten()
        d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
        d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
        
        norm_adj = d_mat_inv_sqrt.dot(adj).dot(d_mat_inv_sqrt)
        
        # Convert to sparse tensor
        norm_adj = norm_adj.tocoo()
        indices = torch.from_numpy(np.vstack((norm_adj.row, norm_adj.col)).astype(np.int64))
        values = torch.from_numpy(norm_adj.data.astype(np.float32))
        shape = torch.Size(norm_adj.shape)
        
        return torch.sparse_coo_tensor(indices, values, shape).to(self.device)

    def fit(self, train_df, num_users, num_items):
        print(f"Training LightGCN on {self.device}...")
        self.model = LightGCN(num_users, num_items, self.embedding_dim, self.n_layers).to(self.device)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        
        # Build Graph
        adj_matrix = self._build_adj_matrix(train_df, num_users, num_items)
        self.adj_matrix = adj_matrix # Store for inference
        
        # Training Loop (BPR Loss)
        # ... (rest of training loop)
        
        user_ids = torch.LongTensor(train_df['user_id_code'].values).to(self.device)
        item_ids = torch.LongTensor(train_df['item_id_code'].values).to(self.device)
        
        # Create a set of all items for negative sampling
        all_items = set(range(num_items))
        # Group items by user for fast negative sampling
        self.user_positives = train_df.groupby('user_id_code')['item_id_code'].apply(set).to_dict()
        
        for epoch in range(self.epochs):
            self.model.train()
            total_loss = 0
            
            # Shuffle data
            perm = torch.randperm(len(user_ids))
            user_ids = user_ids[perm]
            item_ids = item_ids[perm]
            
            num_batches = (len(user_ids) + self.batch_size - 1) // self.batch_size
            
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.epochs}")
            
            for i in pbar:
                start_idx = i * self.batch_size
                end_idx = min((i + 1) * self.batch_size, len(user_ids))
                
                batch_users = user_ids[start_idx:end_idx]
                batch_pos_items = item_ids[start_idx:end_idx]
                
                # Sample negatives
                batch_neg_items = []
                for u in batch_users.cpu().numpy():
                    pos = self.user_positives.get(u, set())
                    while True:
                        neg = np.random.randint(0, num_items)
                        if neg not in pos:
                            batch_neg_items.append(neg)
                            break
                batch_neg_items = torch.LongTensor(batch_neg_items).to(self.device)
                
                # Forward
                user_emb, item_emb = self.model(self.adj_matrix)
                
                u_e = user_emb[batch_users]
                pos_e = item_emb[batch_pos_items]
                neg_e = item_emb[batch_neg_items]
                
                # BPR Loss
                pos_scores = torch.sum(u_e * pos_e, dim=1)
                neg_scores = torch.sum(u_e * neg_e, dim=1)
                
                loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
                
                # Regularization
                reg_loss = (1/2) * (u_e.norm(2).pow(2) + pos_e.norm(2).pow(2) + neg_e.norm(2).pow(2)) / float(len(batch_users))
                loss = loss + self.reg_weight * reg_loss
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (i + 1)})
                
    # Ancienne fontion a reprendre si ca plante
    # def recommend(self, user_ids, user_id_codes, k=10, filter_already_liked_items=False):
    #     self.model.eval()
    #     with torch.no_grad():
    #         user_emb, item_emb = self.model(self.adj_matrix)
            
    #         recs = []
    #         rec_scores = []
            
    #         # Process in batches to avoid OOM
    #         batch_size = 100
    #         for i in range(0, len(user_id_codes), batch_size):
    #             batch_codes = user_id_codes[i:i+batch_size]
                
    #             # Filter out -1 (unknown users)
    #             valid_indices = [idx for idx, c in enumerate(batch_codes) if c != -1]
    #             valid_codes = [c for c in batch_codes if c != -1]
                
    #             if not valid_codes:
    #                 for _ in batch_codes:
    #                     recs.append([])
    #                     rec_scores.append([])
    #                 continue
                
    #             batch_u_emb = user_emb[torch.LongTensor(valid_codes).to(self.device)]
                
    #             # Scores: (batch_size, num_items)
    #             scores = torch.matmul(batch_u_emb, item_emb.t())
                
    #             # Filter history
    #             if filter_already_liked_items:
    #                 for idx, u_code in enumerate(valid_codes):
    #                     pos_items = self.user_positives.get(u_code, set())
    #                     if pos_items:
    #                         scores[idx, list(pos_items)] = -float('inf')
                
    #             # Top K
    #             top_scores, top_indices = torch.topk(scores, k)
                
    #             top_indices = top_indices.cpu().numpy()
    #             top_scores = top_scores.cpu().numpy()
                
    #             # Map back to full batch
    #             current_batch_recs = [[] for _ in batch_codes]
    #             current_batch_scores = [[] for _ in batch_codes]
                
    #             for idx, valid_idx in enumerate(valid_indices):
    #                 current_batch_recs[valid_idx] = top_indices[idx].tolist()
    #                 current_batch_scores[valid_idx] = top_scores[idx].tolist()
                    
    #             recs.extend(current_batch_recs)
    #             rec_scores.extend(current_batch_scores)
                
    #         return recs, rec_scores

    def recommend(self, user_ids, user_id_codes, k=10, filter_already_liked_items=False):
        """
        Version originale (batch + -1 pour unknown) avec un petit patch :
        - on ignore aussi les codes en dehors de [0, n_users_lgcn-1]
        pour éviter les IndexError sur les nouveaux users.
        """
        self.model.eval()
        with torch.no_grad():
            user_emb, item_emb = self.model(self.adj_matrix)
            n_users_lgcn = user_emb.size(0)

            recs = []
            rec_scores = []
            
            # Process in batches to avoid OOM
            batch_size = 100
            for i in range(0, len(user_id_codes), batch_size):
                batch_codes = user_id_codes[i:i+batch_size]

                # On convertit en array numpy pour être sûr
                batch_codes_arr = np.asarray(batch_codes, dtype=int)

                # 🔑 VALID CODES :
                # - on garde ceux qui ne sont pas -1
                # - ET qui sont dans [0, n_users_lgcn-1]
                valid_indices = [
                    idx for idx, c in enumerate(batch_codes_arr)
                    if c != -1 and 0 <= c < n_users_lgcn
                ]
                valid_codes = [int(batch_codes_arr[idx]) for idx in valid_indices]

                if len(batch_codes_arr) > 0:
                    print(
                        f"DEBUG LGCN batch: codes min={batch_codes_arr.min()}, "
                        f"max={batch_codes_arr.max()}, n_users_lgcn={n_users_lgcn}"
                    )
                    print(f"DEBUG LGCN valid_codes={valid_codes}")

                # Si aucun user valable dans ce batch → que des listes vides
                if not valid_codes:
                    for _ in batch_codes_arr:
                        recs.append([])
                        rec_scores.append([])
                    continue

                # Embeddings des users valides
                batch_u_emb = user_emb[torch.LongTensor(valid_codes).to(self.device)]
                
                # Scores: (len(valid_codes), num_items)
                scores = torch.matmul(batch_u_emb, item_emb.t())
                
                # Filter history
                if filter_already_liked_items:
                    for local_idx, u_code in enumerate(valid_codes):
                        pos_items = self.user_positives.get(u_code, set())
                        if pos_items:
                            scores[local_idx, list(pos_items)] = -float('inf')
                
                # Top K
                top_scores, top_indices = torch.topk(scores, k)
                
                top_indices = top_indices.cpu().numpy()
                top_scores = top_scores.cpu().numpy()
                
                # Reconstruction alignée avec la taille du batch
                current_batch_recs = [[] for _ in batch_codes_arr]
                current_batch_scores = [[] for _ in batch_codes_arr]
                
                for local_pos, batch_pos in enumerate(valid_indices):
                    current_batch_recs[batch_pos] = top_indices[local_pos].tolist()
                    current_batch_scores[batch_pos] = top_scores[local_pos].tolist()
                    
                recs.extend(current_batch_recs)
                rec_scores.extend(current_batch_scores)
            
            return recs, rec_scores