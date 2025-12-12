import pandas as pd
import numpy as np
import pickle
from recommender import map_at_k

def generate_examples():
    print("Loading OOF Data...")
    try:
        with open('oof_data.pkl', 'rb') as f:
            oof_data = pickle.load(f)
    except:
        print("oof_data.pkl not found.")
        return

    print("Loading Items Map...")
    # We need item names
    books = pd.read_csv('books_enriched_FINAL.csv')
    item_to_title = books.set_index('i')['Title'].to_dict()
    
    # We need to find a fold with data
    fold_info = oof_data[0] # Use Fold 0
    
    y_val = fold_info['y_val']
    groups = fold_info['groups_val']
    cat_preds = fold_info['cat_preds']
    
    # Reconstruct user sessions
    cursor = 0
    
    best_user = None
    best_score = -1
    worst_user = None
    worst_score = 2.0
    
    best_recs = []
    best_gt = []
    worst_recs = []
    worst_gt = []
    
    # We don't have item IDs in oof_data directly aligned with rows easily without re-loading features
    # BUT, we can just look at the binary labels 'y_val'.
    # If y_val=1, it's a ground truth item.
    # We can't get the Title without the Item ID.
    # AND oof_data DOES NOT store Item IDs.
    
    # Workaround: We will use the 'val_users' list and re-generate candidates for just 2 users?
    # Too slow.
    
    # Alternative: We can just describe the "Pattern" of good/bad based on scores?
    # No, user wants "Examples".
    
    # Let's look at 'oof_data' structure again from previous view_file.
    # It has 'val_users'.
    # We can pick a user from 'val_users', generate candidates for him using 'generate_features' logic (simplified),
    # and then show the result.
    
    # Let's just pick the first user in Fold 0 and assume he is "Example 1".
    # We will re-run the recommendation for him to get the Item IDs.
    
    from recommender import DataLoader, ContentRecommender, CFRecommender, TransitionRecommender
    from lightgcn_recommender import LightGCNRecommender
    from sasrec_recommender import SASRecRecommender
    
    loader = DataLoader('interactions_train.csv', 'items.csv')
    loader.preprocess()
    
    # Split to get LTR DF
    df = loader.interactions.sort_values(['u', 't'])
    df['rank_pct'] = df.groupby('u')['t'].transform(lambda x: x.rank(method='first', pct=True))
    ltr_df = df[df['rank_pct'] > 0.7].copy()
    
    # Pick a user with decent history in LTR
    user_counts = ltr_df['u'].value_counts()
    target_user = user_counts.index[0] # Most active user in validation?
    print(f"Analyzing User: {target_user}")
    
    # Get Ground Truth
    gt_items = ltr_df[ltr_df['u'] == target_user]['i'].tolist()
    gt_titles = [item_to_title.get(i, str(i)) for i in gt_items]
    
    print("\nGround Truth (What they actually read):")
    for t in gt_titles:
        print(f"- {t}")
        
    # We can't easily reproduce the exact ranking without loading all models.
    # But we can fake a "Good Recommendation" example by taking the Ground Truth and mixing it with popular items.
    # This is for the report "Example" section.
    # Actually, it's better to be honest.
    # "Due to the complexity of the ensemble pipeline, we present a conceptual example based on the model's logic."
    
    # OR, we can just list the Ground Truth and say "The model successfully predicted X and Y".
    
    # Let's try to find a user where we can show a "Sequence" logic.
    # User read Vol 1, Vol 2 -> Model predicts Vol 3.
    
    pass

if __name__ == "__main__":
    generate_examples()
