import pandas as pd
import numpy as np
from recommender import DataLoader, ContentRecommender, CFRecommender, TransitionRecommender
from lightgcn_recommender import LightGCNRecommender
from sasrec_recommender import SASRecRecommender
from catboost import CatBoostRanker
import pickle
import sys
import os
import csv

def generate_for_user(user_id, new_book_ids=None):
    """
    Génère les recommandations pour un seul utilisateur en chargeant les modèles pré-entraînés.
    Si les modèles n'existent pas, entraîne-les d'abord.
    
    Args:
        user_id: ID de l'utilisateur
        new_book_ids: Liste des IDs de livres que l'utilisateur aime (nouvelles interactions)
    """
    print(f"🎯 Génération des recommandations pour l'utilisateur {user_id}")
    
    if new_book_ids:
        print(f"📚 Ajout de {len(new_book_ids)} nouvelles interactions au profil")
    
    print("Loading data...")
    loader = DataLoader('interactions_train.csv', 'items.csv')
    loader.preprocess()
    
    # === AJOUTER LES NOUVELLES INTERACTIONS ===
    if new_book_ids and len(new_book_ids) > 0:
        print(f"Adding {len(new_book_ids)} new interactions for user {user_id}...")
        
        # Obtenir le timestamp max actuel
        max_timestamp = loader.interactions['t'].max() if not loader.interactions.empty else 0
        
        # Créer de nouvelles interactions
        new_interactions = []
        for idx, book_id in enumerate(new_book_ids):
            new_interactions.append({
                'u': user_id,
                'i': book_id,
                't': max_timestamp + idx + 1  # Timestamps croissants
            })
        
        new_df = pd.DataFrame(new_interactions)
        
        # Ajouter au DataFrame d'interactions
        loader.interactions = pd.concat([loader.interactions, new_df], ignore_index=True)
        print(f"✅ {len(new_book_ids)} nouvelles interactions ajoutées")
    
    num_users = len(loader.user_map)
    num_items = len(loader.item_map)
    
    # Vérifier que l'utilisateur existe (ou a été créé avec les nouvelles interactions)
    if user_id not in loader.reverse_user_map and user_id not in loader.interactions['u'].values:
        print(f"❌ Erreur: L'utilisateur {user_id} n'existe pas dans les données")
        return None
    
    # === CHARGER LES MODÈLES PRÉ-ENTRAÎNÉS (RAPIDE!) ===
    print("📦 Chargement des modèles pré-entraînés...")
    
    # Vérifier que les modèles existent
    required_models = ['content_model.pkl', 'cf_model.pkl', 'trans_model.pkl', 'lightgcn_model.pkl']
    missing_models = [m for m in required_models if not os.path.exists(m)]
    
    if missing_models:
        print(f"❌ Modèles manquants: {missing_models}")
        print("⚠️ Veuillez d'abord exécuter create_submission.py pour entraîner les modèles")
        return None
    
    # Charger les modèles
    try:
        with open('content_model.pkl', 'rb') as f:
            content_model = pickle.load(f)
        print("✅ Content Model chargé")
    except Exception as e:
        print(f"❌ Erreur chargement Content Model: {e}")
        return None
    
    try:
        with open('cf_model.pkl', 'rb') as f:
            cf_model = pickle.load(f)
        print("✅ CF Model chargé")
    except Exception as e:
        print(f"❌ Erreur chargement CF Model: {e}")
        return None
    
    try:
        with open('trans_model.pkl', 'rb') as f:
            trans_model = pickle.load(f)
        print("✅ Transition Model chargé")
    except Exception as e:
        print(f"❌ Erreur chargement Transition Model: {e}")
        return None
    
    try:
        with open('lightgcn_model.pkl', 'rb') as f:
            lightgcn_model = pickle.load(f)
        print("✅ LightGCN Model chargé")
    except Exception as e:
        print(f"❌ Erreur chargement LightGCN Model: {e}")
        return None
    
    # Pour SASRec, on va créer un modèle vide et générer des scores neutres
    # car il ne peut pas être sérialisé facilement
    print("⚠️ SASRec Model: utilisation de scores neutres (modèle non sérialisable)")
    sasrec_model = None  # On gérera ça différemment
    
    # Charger les données enrichies
    print("Loading enriched book data...")
    books_df = pd.read_csv('books_enriched_FINAL.csv')
    item_to_author = books_df.set_index('i')['Author'].to_dict()
    item_to_year = books_df.set_index('i')['published_year'].to_dict()
    item_to_pages = books_df.set_index('i')['page_count'].to_dict()
    item_to_publisher = books_df.set_index('i')['Publisher'].to_dict()
    
    def clean_year(y):
        try:
            return int(float(str(y)[:4]))
        except:
            return -1
    item_to_year = {k: clean_year(v) for k, v in item_to_year.items()}
    
    print("Processing subjects...")
    item_to_subjects = books_df.set_index('i')['Subjects'].to_dict()
    def clean_subjects(s):
        if pd.isna(s): return []
        return [x.strip() for x in str(s).split(';')]
    item_to_subjects = {k: clean_subjects(v) for k, v in item_to_subjects.items()}
    
    print("Loading embeddings...")
    with open('item_embeddings.pkl', 'rb') as f:
        emb_data = pickle.load(f)
    item_embeddings = dict(zip(emb_data['item_ids'], emb_data['embeddings']))
    
    # Charger les modèles CatBoost
    print("Loading CatBoost models...")
    cat_models = []
    n_folds = 5
    for i in range(n_folds):
        try:
            m = CatBoostRanker()
            m.load_model(f"ltr_catboost_fold{i}.cbm")
            cat_models.append(m)
        except:
            print(f"⚠️ Could not load ltr_catboost_fold{i}.cbm")
    
    # Générer les recommandations pour cet utilisateur
    target_users = [user_id]
    user_code = loader.reverse_user_map.get(user_id, -1)
    target_user_codes = [user_code]
    
    last_items_map = loader.interactions.sort_values('t').groupby('u')['i'].last().to_dict()
    
    N = 500
    print(f"Getting top {N} candidates...")
    c_recs, c_scores = content_model.recommend(target_users, k=N, return_scores=True)
    cf_ids, cf_scores = cf_model.recommend(target_users, target_user_codes, k=N, filter_already_liked_items=False)
    t_ids, t_scores = trans_model.recommend(target_users, last_items_map, k=N)
    l_ids, l_scores = lightgcn_model.recommend(target_users, target_user_codes, k=N, filter_already_liked_items=False)
    # Pour SASRec, générer des scores neutres (le modèle ne peut pas être sérialisé)
    # On utilise des scores uniformes pour ne pas biaiser le ranking final
    s_ids = [np.array([0] * N) for _ in target_users]  # IDs neutres
    s_scores = [np.array([0.5] * N) for _ in target_users]  # Scores neutres
    print("⚠️ SASRec: scores neutres utilisés (modèle non chargé)")
    
    # Stats
    item_pop = loader.interactions['i'].value_counts().to_dict()
    user_act = loader.interactions['u'].value_counts().to_dict()
    
    author_pop = {}
    for i, count in item_pop.items():
        auth = item_to_author.get(i, "Unknown")
        author_pop[auth] = author_pop.get(auth, 0) + count
    
    print("Building user history...")
    user_history_items = loader.interactions.groupby('u')[['i', 't']].apply(lambda x: list(zip(x['i'], x['t']))).to_dict()
    
    DECAY_DAYS = 180
    
    # Calculer pour cet utilisateur
    u = user_id
    items = user_history_items.get(u, [])
    u_auths = {}
    u_subjs = {}
    u_pubs = {}
    u_embs = []
    u_weights = []
    
    if items:
        max_t = max([t for _, t in items])
        
        for i, t in items:
            auth = item_to_author.get(i)
            if auth: u_auths[auth] = u_auths.get(auth, 0) + 1
            subjs = item_to_subjects.get(i, [])
            for s in subjs: u_subjs[s] = u_subjs.get(s, 0) + 1
            pub = item_to_publisher.get(i)
            if pub: u_pubs[pub] = u_pubs.get(pub, 0) + 1
            
            if i in item_embeddings:
                u_embs.append(item_embeddings[i])
                weight = np.exp(-(max_t - t) / DECAY_DAYS)
                u_weights.append(weight)
    
    u_centroid = np.average(u_embs, axis=0, weights=u_weights) if u_embs else np.zeros(384)
    
    last_item = last_items_map.get(u)
    last_item_year = item_to_year.get(last_item, -1)
    last_item_emb = item_embeddings.get(last_item, np.zeros(384)) if last_item else np.zeros(384)
    
    print("Re-ranking...")
    item_feats = {}
    item_ranks = {}
    item_norm_scores = {}
    
    def add_feats(items, scores, model_idx):
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
                item_feats[item] = [0.0] * 30
                item_ranks[item] = [999] * 5
                item_norm_scores[item] = [0.0] * 5
                
                item_feats[item][5] = item_pop.get(item, 0)
                item_feats[item][6] = user_act.get(u, 0)
                
                auth = item_to_author.get(item, "Unknown")
                item_feats[item][7] = author_pop.get(auth, 0)
                
                item_year = item_to_year.get(item, -1)
                item_feats[item][8] = item_year
                item_feats[item][9] = item_to_pages.get(item, -1)
                
                auth_count = u_auths.get(auth, 0)
                item_feats[item][10] = auth_count
                
                subjs = item_to_subjects.get(item, [])
                subj_score = sum([u_subjs.get(s, 0) for s in subjs])
                item_feats[item][11] = subj_score
                
                pub = item_to_publisher.get(item)
                pub_count = u_pubs.get(pub, 0)
                item_feats[item][12] = pub_count
                
                item_emb = item_embeddings.get(item, np.zeros(384))
                
                def cosine(a, b):
                    norm_a = np.linalg.norm(a)
                    norm_b = np.linalg.norm(b)
                    if norm_a == 0 or norm_b == 0: return 0.0
                    return np.dot(a, b) / (norm_a * norm_b)
                
                item_feats[item][13] = cosine(u_centroid, item_emb)
                item_feats[item][14] = cosine(last_item_emb, item_emb)
                
                total_reads = len(items)
                if total_reads > 0:
                    item_feats[item][15] = auth_count / total_reads
                    item_feats[item][16] = pub_count / total_reads
                else:
                    item_feats[item][15] = 0.0
                    item_feats[item][16] = 0.0
                
                if subjs:
                    match_count = sum([1 for s in subjs if s in u_subjs])
                    item_feats[item][17] = match_count / len(subjs)
                else:
                    item_feats[item][17] = 0.0
                
                if last_item_year != -1 and item_year != -1:
                    item_feats[item][18] = abs(item_year - last_item_year)
                else:
                    item_feats[item][18] = -1
                
                auth_last = item_to_author.get(last_item)
                if auth_last and auth == auth_last:
                    item_feats[item][19] = 1.0
                else:
                    item_feats[item][19] = 0.0
            
            item_feats[item][model_idx] = score
            item_ranks[item][model_idx] = rank + 1
            item_norm_scores[item][model_idx] = n_score
    
    add_feats(c_recs[0], c_scores[0], 0)
    add_feats([loader.item_map.get(c) for c in cf_ids[0]], cf_scores[0], 1)
    add_feats([loader.item_map.get(c) for c in t_ids[0]], t_scores[0], 2)
    add_feats([loader.item_map.get(c) for c in l_ids[0]], l_scores[0], 3)
    add_feats([loader.item_map.get(c) for c in s_ids[0]], s_scores[0], 4)
    
    curr_X = []
    curr_items = []
    
    for item, feats in item_feats.items():
        ranks = item_ranks[item]
        n_scores = item_norm_scores[item]
        
        for m_idx in range(5):
            feats[20 + m_idx] = ranks[m_idx]
        
        vote_count = sum(1 for r in ranks if r < 999)
        feats[25] = vote_count
        
        feats[26] = np.mean(n_scores)
        feats[27] = np.std(n_scores)
        feats[28] = np.min(n_scores)
        feats[29] = np.max(n_scores)
        
        curr_X.append(feats)
        curr_items.append(item)
    
    # Prédiction avec CatBoost
    cat_preds = np.zeros(len(curr_X))
    if cat_models:
        for model in cat_models:
            cat_preds += model.predict(curr_X)
        cat_preds /= len(cat_models)
    
    final_scores = cat_preds
    
    sorted_indices = np.argsort(final_scores)[::-1]
    top_items = [curr_items[idx] for idx in sorted_indices[:10]]
    
    print(f"✅ Top 10 recommandations générées pour l'utilisateur {user_id}")
    print(f"Items: {top_items}")
    
    return top_items

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python generate_single_user_reco.py <user_id> [book_id1 book_id2 ...]")
        sys.exit(1)
    
    user_id = int(sys.argv[1])
    
    # Récupérer les IDs de livres si fournis
    new_book_ids = []
    if len(sys.argv) > 2:
        new_book_ids = [int(book_id) for book_id in sys.argv[2:]]
        print(f"📚 Livres sélectionnés: {new_book_ids}")
    
    recommendations = generate_for_user(user_id, new_book_ids)
    
    if recommendations:
        # Mettre à jour submission.csv
        # print("Updating submission.csv...")
        # try:
        #     submission_df = pd.read_csv('submission.csv')
        # except:
        #     submission_df = pd.DataFrame(columns=['user_id', 'item_id'])
        
        # items_str = " ".join([str(x) for x in recommendations])
        
        # if user_id in submission_df['user_id'].values:
        #     submission_df.loc[submission_df['user_id'] == user_id, 'item_id'] = items_str
        # else:
        #     new_row = pd.DataFrame([{'user_id': user_id, 'item_id': items_str}])
        #     submission_df = pd.concat([submission_df, new_row], ignore_index=True)
        
        # submission_df.to_csv('submission.csv', index=False)
        # print("Saved submission.csv")
    

        print("Updating submission.csv...")
        try:
            submission_df = pd.read_csv("submission.csv")
        except:
            # Fichier inexistant → on crée proprement
            submission_df = pd.DataFrame(columns=["user_id", "recommendation"])

        # Harmoniser les colonnes si jamais il reste un vieux 'item_id'
        if "recommendation" not in submission_df.columns:
            if "item_id" in submission_df.columns:
                submission_df = submission_df.rename(columns={"item_id": "recommendation"})
            else:
                submission_df["recommendation"] = ""

        # On garde uniquement ces deux colonnes
        submission_df = submission_df[["user_id", "recommendation"]]

        items_str = " ".join(str(x) for x in recommendations)

        if user_id in submission_df["user_id"].values:
            # 👉 on écrase les anciennes recos pour cet user
            submission_df.loc[submission_df["user_id"] == user_id, "recommendation"] = items_str
        else:
            # 👉 nouvel user → 2 colonnes, pas plus
            new_row = pd.DataFrame([{"user_id": user_id, "recommendation": items_str}])
            submission_df = pd.concat([submission_df, new_row], ignore_index=True)

        # Sauvegarde sans index ; les guillemets ici sont juste du CSV normal
        submission_df.to_csv(
            "submission.csv",
            index=False,
            quoting=csv.QUOTE_MINIMAL  # comportement standard
        )
        print("Saved submission.csv")


