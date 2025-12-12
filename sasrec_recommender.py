import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class SASRec(nn.Module):
    def __init__(self, num_users, num_items, args):
        super(SASRec, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.dev = args.device
        
        # Embeddings
        self.item_emb = nn.Embedding(self.num_items + 1, args.hidden_units, padding_idx=0)
        self.pos_emb = nn.Embedding(args.maxlen, args.hidden_units)
        self.emb_dropout = nn.Dropout(p=args.dropout_rate)
        
        # Transformer Blocks
        self.attention_layernorms = nn.ModuleList()
        self.attention_layers = nn.ModuleList()
        self.forward_layernorms = nn.ModuleList()
        self.forward_layers = nn.ModuleList()
        
        self.last_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
        
        for _ in range(args.num_blocks):
            new_attn_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.attention_layernorms.append(new_attn_layernorm)
            
            new_attn_layer = nn.MultiheadAttention(args.hidden_units, args.num_heads, args.dropout_rate)
            self.attention_layers.append(new_attn_layer)
            
            new_fwd_layernorm = nn.LayerNorm(args.hidden_units, eps=1e-8)
            self.forward_layernorms.append(new_fwd_layernorm)
            
            new_fwd_layer = nn.Sequential(
                nn.Linear(args.hidden_units, args.hidden_units * 4),
                nn.GELU(),
                nn.Dropout(p=args.dropout_rate),
                nn.Linear(args.hidden_units * 4, args.hidden_units),
                nn.Dropout(p=args.dropout_rate)
            )
            self.forward_layers.append(new_fwd_layer)
            
    def log2feats(self, log_seqs):
        seqs = self.item_emb(torch.LongTensor(log_seqs).to(self.dev))
        seqs *= self.item_emb.embedding_dim ** 0.5
        positions = np.tile(np.array(range(log_seqs.shape[1])), [log_seqs.shape[0], 1])
        seqs += self.pos_emb(torch.LongTensor(positions).to(self.dev))
        seqs = self.emb_dropout(seqs)
        
        timeline_mask = torch.BoolTensor(log_seqs == 0).to(self.dev)
        seqs *= ~timeline_mask.unsqueeze(-1) # broadcast in last dim
        
        tl = seqs.shape[1]
        attention_mask = ~torch.tril(torch.ones((tl, tl), dtype=torch.bool, device=self.dev))
        
        for i in range(len(self.attention_layers)):
            seqs = torch.transpose(seqs, 0, 1)
            Q = self.attention_layernorms[i](seqs)
            mha_outputs, _ = self.attention_layers[i](Q, seqs, seqs, attn_mask=attention_mask)
            seqs = Q + mha_outputs
            seqs = torch.transpose(seqs, 0, 1)
            
            seqs = self.forward_layernorms[i](seqs)
            seqs = self.forward_layers[i](seqs) + seqs
            seqs *= ~timeline_mask.unsqueeze(-1)
            
        log_feats = self.last_layernorm(seqs)
        return log_feats

    def forward(self, user_ids, log_seqs, pos_seqs, neg_seqs):
        log_feats = self.log2feats(log_seqs) # (batch, maxlen, hidden)
        
        pos_embs = self.item_emb(torch.LongTensor(pos_seqs).to(self.dev))
        neg_embs = self.item_emb(torch.LongTensor(neg_seqs).to(self.dev))
        
        pos_logits = (log_feats * pos_embs).sum(dim=-1)
        neg_logits = (log_feats * neg_embs).sum(dim=-1)
        
        return pos_logits, neg_logits
    
    def predict(self, user_ids, log_seqs, item_indices):
        log_feats = self.log2feats(log_seqs) # (batch, maxlen, hidden)
        final_feat = log_feats[:, -1, :] # (batch, hidden)
        
        item_embs = self.item_emb(torch.LongTensor(item_indices).to(self.dev)) # (num_items, hidden)
        
        logits = item_embs.matmul(final_feat.unsqueeze(-1)).squeeze(-1)
        return logits

class SASRecRecommender:
    def __init__(self, hidden_units=128, num_blocks=2, num_heads=4, dropout_rate=0.2, maxlen=100, batch_size=128, lr=0.001, epochs=20, device=None):
        self.args = type('Args', (), {})()
        self.args.hidden_units = hidden_units
        self.args.num_blocks = num_blocks
        self.args.num_heads = num_heads
        self.args.dropout_rate = dropout_rate
        self.args.maxlen = maxlen
        self.args.batch_size = batch_size
        self.args.lr = lr
        self.args.epochs = epochs
        self.args.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.model = None
        self.user_train = {}
        self.user_valid = {}
        self.user_test = {}
        self.item_num = 0
        self.user_num = 0
        
    def fit(self, train_df, num_users, num_items):
        print(f"Training SASRec on {self.args.device}...")
        self.user_num = num_users
        self.item_num = num_items
        
        # Group by user and get sequences
        # We need to ensure items are 1-indexed for padding (0)
        # Our codes are 0-indexed. So we shift by +1.
        
        # Create sequences
        print("Creating sequences...")
        # Use codes!
        df_sorted = train_df.sort_values(['user_id_code', 't'])
        user_group = df_sorted.groupby('user_id_code')
        
        self.user_train = {}
        for u, group in user_group:
            # Shift item ids by +1 (0 is padding)
            items = (group['item_id_code'].values + 1).tolist()
            self.user_train[u] = items
            
        max_item_in_train = df_sorted['item_id_code'].max() + 1
        print(f"Max item index in train: {max_item_in_train}")
        print(f"Num items (from map): {self.item_num}")
        
        if max_item_in_train > self.item_num:
            print(f"WARNING: Train data has more items than item_map. Adjusting item_num to {max_item_in_train}")
            self.item_num = max_item_in_train
            
        print(f"Embedding size: {self.item_num + 1}")
            
        self.model = SASRec(self.user_num, self.item_num, self.args).to(self.args.device)
        
        # Training Loop
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.lr, betas=(0.9, 0.98))
        bce_criterion = torch.nn.BCEWithLogitsLoss()
        
        for epoch in range(self.args.epochs):
            self.model.train()
            
            # Sampler
            # We need to sample (user, seq, pos, neg)
            # This is slow in python loop.
            
            # Let's create a list of users to sample from
            users = list(self.user_train.keys())
            num_batches = (len(users) + self.args.batch_size - 1) // self.args.batch_size
            
            total_loss = 0
            pbar = tqdm(range(num_batches), desc=f"Epoch {epoch+1}/{self.args.epochs}")
            
            # Shuffle users
            np.random.shuffle(users)
            
            for i in pbar:
                batch_users = users[i*self.args.batch_size : (i+1)*self.args.batch_size]
                
                # Prepare sequences
                seqs = np.zeros([len(batch_users), self.args.maxlen], dtype=np.int32)
                pos = np.zeros([len(batch_users), self.args.maxlen], dtype=np.int32)
                
                for idx, u in enumerate(batch_users):
                    ts = self.user_train[u]
                    
                    # Input: seq[:-1], Target: seq[1:]
                    input_seq = ts[:-1]
                    target_seq = ts[1:]
                    
                    pad_len = self.args.maxlen - len(input_seq)
                    if pad_len > 0:
                        seqs[idx, pad_len:] = input_seq
                        pos[idx, pad_len:] = target_seq
                    else:
                        seqs[idx] = input_seq[-self.args.maxlen:]
                        pos[idx] = target_seq[-self.args.maxlen:]
                        
                # Convert to tensors
                seqs_tensor = torch.LongTensor(seqs).to(self.args.device)
                pos_tensor = torch.LongTensor(pos).to(self.args.device)
                
                # Forward Pass
                # log_feats: (batch, maxlen, hidden)
                log_feats = self.model.log2feats(seqs)
                
                # We only care about the last step for InfoNCE usually, OR all steps?
                # Standard SASRec with BCE uses all steps.
                # For InfoNCE, we can also use all steps.
                
                # Let's use the standard SASRec approach but with Cross Entropy over items.
                # However, full softmax is too expensive (15k items).
                # We can use Batch Negatives (other items in the batch as negatives).
                
                # Target embeddings: (batch, maxlen, hidden)
                pos_embs = self.model.item_emb(pos_tensor)
                
                # Mask padding (where pos == 0)
                mask = (pos_tensor != 0).float() # (batch, maxlen)
                
                # Compute logits: (batch, maxlen, batch, maxlen) - too big?
                # No, we want for each (user, time) step, to classify the correct item.
                # But typically InfoNCE is done on the final embedding for the next item prediction.
                # If we want to train on all steps, it's complex to do batch negatives efficiently for every step.
                
                # SIMPLIFICATION: Train only on the LAST item prediction for InfoNCE?
                # Or use the standard BCE implementation but with more negatives?
                # The user approved InfoNCE.
                
                # Let's implement a simplified InfoNCE:
                # 1. Take the final state of the sequence: log_feats[:, -1, :] -> (batch, hidden)
                # 2. Positive item: pos[:, -1] -> (batch,)
                # 3. Negatives: All other items in the batch (from other users' positives).
                
                final_feats = log_feats[:, -1, :] # (batch, hidden)
                final_pos = pos_tensor[:, -1] # (batch,)
                
                # Positive embeddings
                final_pos_embs = self.model.item_emb(final_pos) # (batch, hidden)
                
                # Logits: (batch, batch)
                # Row i: scores for user i against all items in the batch (positives of user 0..N)
                logits = torch.matmul(final_feats, final_pos_embs.t()) # (batch, batch)
                
                # Temperature
                temperature = 0.07
                logits /= temperature
                
                # Labels: diagonal is the positive
                labels = torch.arange(len(batch_users)).to(self.args.device)
                
                # Cross Entropy Loss
                loss = torch.nn.functional.cross_entropy(logits, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                pbar.set_postfix({'loss': total_loss / (i + 1)})

    def recommend(self, user_ids, user_id_codes, k=10, filter_already_liked_items=False):
        self.model.eval()
        
        # Prepare sequences for inference
        # We need the full sequence for each user
        
        recs = []
        rec_scores = []
        
        # Batch inference
        batch_size = 100
        all_items = np.arange(1, self.item_num + 1)
        
        with torch.no_grad():
            for i in range(0, len(user_ids), batch_size):
                batch_u_ids = user_ids[i:i+batch_size]
                batch_u_codes = user_id_codes[i:i+batch_size]
                
                seqs = np.zeros([len(batch_u_codes), self.args.maxlen], dtype=np.int32)
                
                valid_indices = []
                
                for idx, u_code in enumerate(batch_u_codes):
                    if u_code == -1: continue
                    if u_code not in self.user_train: continue
                    
                    valid_indices.append(idx)
                    ts = self.user_train[u_code]
                    
                    # Use last maxlen items
                    if len(ts) > self.args.maxlen:
                        seqs[idx] = ts[-self.args.maxlen:]
                    else:
                        seqs[idx, -len(ts):] = ts
                        
                if not valid_indices:
                    for _ in batch_u_codes:
                        recs.append([])
                        rec_scores.append([])
                    continue
                
                # Predict
                # We want scores for ALL items for the last step
                # predict returns (batch, num_items)
                
                # We pass all items to predict
                # Actually predict takes item_indices. We can pass all items.
                
                # To save memory, we might need to batch items too?
                # But item_num ~15k is small enough.
                
                # log_feats: (batch, maxlen, hidden)
                log_feats = self.model.log2feats(seqs)
                final_feat = log_feats[:, -1, :] # (batch, hidden)
                
                # Item embeddings
                item_embs = self.model.item_emb.weight[1:] # (num_items, hidden) - skip padding
                
                # Scores: (batch, num_items)
                scores = torch.matmul(final_feat, item_embs.t())
                
                # Filter history
                if filter_already_liked_items:
                    for idx, u_code in enumerate(batch_u_codes):
                        if u_code == -1 or u_code not in self.user_train: continue
                        ts = self.user_train[u_code]
                        # ts are 1-indexed, scores are 0-indexed (item 1 is at index 0)
                        ts_indices = [t-1 for t in ts]
                        scores[idx, ts_indices] = -float('inf')
                
                # Top K
                top_scores, top_indices = torch.topk(scores, k)
                
                top_indices = top_indices.cpu().numpy() # 0-indexed indices
                top_scores = top_scores.cpu().numpy()
                
                # Map back to item codes (index + 1) -> actually we return item codes (0-indexed) for compatibility?
                # My recommender system uses 0-indexed codes internally.
                # SASRec used 1-indexed internally.
                # So top_indices (0-indexed relative to item_embs which starts at 1)
                # item_embs[0] is item 1.
                # So index 0 corresponds to item 1.
                # Item 1 in SASRec is Item 0 in global map.
                # So top_indices are exactly the global item codes!
                
                current_batch_recs = [[] for _ in batch_u_codes]
                current_batch_scores = [[] for _ in batch_u_codes]
                
                for idx, valid_idx in enumerate(range(len(batch_u_codes))):
                    current_batch_recs[idx] = top_indices[idx].tolist()
                    current_batch_scores[idx] = top_scores[idx].tolist()
                    
                # Debug predictions
                if i == 0:
                    print(f"Pred Batch 0:")
                    print(f"Top Indices: {top_indices[0]}")
                    print(f"Top Scores: {top_scores[0]}")
                    
                recs.extend(current_batch_recs)
                rec_scores.extend(current_batch_scores)
                
        return recs, rec_scores
