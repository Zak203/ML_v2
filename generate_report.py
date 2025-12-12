import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import os
from recommender import map_at_k

# Create assets directory
if not os.path.exists('report_assets'):
    os.makedirs('report_assets')

def generate_report():
    print("Loading data for report generation...")
    
    # 1. Load Data
    interactions = pd.read_csv('interactions_train.csv')
    books = pd.read_csv('books_enriched_FINAL.csv')
    
    try:
        with open('oof_data.pkl', 'rb') as f:
            oof_data = pickle.load(f)
    except FileNotFoundError:
        print("oof_data.pkl not found! Cannot generate performance metrics.")
        oof_data = None

    # --- EDA ---
    print("Generating EDA...")
    
    n_users = interactions['u'].nunique()
    n_items = interactions['i'].nunique()
    n_interactions = len(interactions)
    sparsity = 1 - (n_interactions / (n_users * n_items))
    
    # User Activity
    user_counts = interactions['u'].value_counts()
    plt.figure(figsize=(10, 6))
    sns.histplot(user_counts, bins=50, log_scale=(True, True))
    plt.title('User Activity Distribution (Log-Log)')
    plt.xlabel('Number of Interactions')
    plt.ylabel('Count of Users')
    plt.savefig('report_assets/user_activity.png')
    plt.close()
    
    # Item Popularity
    item_counts = interactions['i'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.plot(item_counts.values)
    plt.title('Item Popularity (Long Tail)')
    plt.xlabel('Item Rank')
    plt.ylabel('Number of Interactions')
    plt.yscale('log')
    plt.savefig('report_assets/item_popularity.png')
    plt.close()
    
    # Content Analysis
    top_authors = books['Author'].value_counts().head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(x=top_authors.values, y=top_authors.index, palette='viridis')
    plt.title('Top 10 Authors by Book Count')
    plt.xlabel('Number of Books')
    plt.savefig('report_assets/top_authors.png')
    plt.close()

    # --- Performance Analysis ---
    print("Analyzing Performance...")
    
    metrics = []
    
    if oof_data:
        fold_scores = {'CatBoost': [], 'XGBoost': [], 'LightGBM': [], 'Ensemble': []}
        
        for fold_info in oof_data:
            fold = fold_info['fold']
            val_users = fold_info['val_users']
            groups = fold_info['groups_val']
            y_val = fold_info['y_val']
            
            # Reconstruct Ground Truth
            # The oof_data structure in train_ltr_ensemble.py saves 'val_users' list
            # But we need the actual ground truth items for these users.
            # We can get it from interactions dataframe
            
            # Filter interactions for validation users
            val_interactions = interactions[interactions['u'].isin(val_users)]
            # We need to be careful: oof_data was generated on the LTR split (30% of data)
            # But let's assume we want to evaluate on the ground truth available in interactions
            # Actually, train_ltr_ensemble.py calculates MAP@10 during training using 'val_gt'.
            # Let's re-calculate it here to be sure, or use what we have.
            # The oof_data doesn't save 'val_gt' explicitly in the dictionary I saw in view_file, 
            # wait, let me check the view_file output again.
            # It saves: 'cat_preds', 'xgb_preds', 'lgb_preds', 'y_val', 'groups_val', 'val_users'
            # It DOES NOT save the candidate items corresponding to the predictions!
            # Without the candidate items (the 'i' column for the X_val rows), we cannot map predictions back to items to calculate MAP!
            
            # CRITICAL FIX: We need to know which item corresponds to which prediction score.
            # Since we can't easily reconstruct that without re-running the feature generation,
            # I will check if 'train_ltr_ensemble.py' prints the scores and if I can extract them? No.
            
            # Wait, if I can't calculate MAP, I can at least show the distribution of scores.
            # OR, I can rely on the logs if I had them.
            # BUT, I can generate a "Mock" performance chart based on the values I saw in the logs previously 
            # (Fold 1 MAP@10: 0.048... etc) if I had them.
            
            # Actually, let's look at 'oof_data.pkl' content structure again.
            # The user wants "examples, all precision or error calcul etc".
            # If I cannot calculate exact MAP because I miss item IDs, I will note that.
            
            # HOWEVER, I can calculate the ROC AUC or LogLoss on the binary targets 'y_val' provided in oof_data!
            # It's a ranking task, so AUC is a decent proxy for "ranking quality" in a binary sense.
            # Let's do AUC.
            
            from sklearn.metrics import roc_auc_score
            
            try:
                auc_cat = roc_auc_score(y_val, fold_info['cat_preds'])
                auc_xgb = roc_auc_score(y_val, fold_info['xgb_preds'])
                auc_lgb = roc_auc_score(y_val, fold_info['lgb_preds'])
                
                # Ensemble
                def standardize(x):
                    return (x - np.mean(x)) / (np.std(x) + 1e-8)
                ens_preds = (standardize(fold_info['cat_preds']) + 
                             standardize(fold_info['xgb_preds']) + 
                             standardize(fold_info['lgb_preds'])) / 3.0
                auc_ens = roc_auc_score(y_val, ens_preds)
                
                fold_scores['CatBoost'].append(auc_cat)
                fold_scores['XGBoost'].append(auc_xgb)
                fold_scores['LightGBM'].append(auc_lgb)
                fold_scores['Ensemble'].append(auc_ens)
            except:
                pass

        # Plot Performance
        if fold_scores['CatBoost']:
            means = [np.mean(fold_scores[m]) for m in fold_scores]
            stds = [np.std(fold_scores[m]) for m in fold_scores]
            
            plt.figure(figsize=(10, 6))
            plt.bar(fold_scores.keys(), means, yerr=stds, capsize=5, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            plt.title('Model Performance Comparison (ROC AUC on LTR Data)')
            plt.ylabel('ROC AUC Score')
            plt.ylim(0.5, 1.0)
            for i, v in enumerate(means):
                plt.text(i, v + 0.01, f"{v:.4f}", ha='center')
            plt.savefig('report_assets/model_comparison.png')
            plt.close()
            
            metrics = means

    # --- Generate Markdown Report ---
    print("Generating REPORT.md...")
    
    with open('REPORT.md', 'w') as f:
        f.write("# AI Librarian Recommender System - Final Report\n\n")
        
        f.write("## 1. Executive Summary\n")
        f.write(f"This report summarizes the performance and characteristics of the Hybrid Recommender System.\n")
        f.write(f"The system utilizes a 2-stage architecture (Candidate Generation + Learning to Rank) to personalize book recommendations.\n\n")
        
        f.write("## 2. Dataset Overview (EDA)\n")
        f.write(f"- **Total Users**: {n_users}\n")
        f.write(f"- **Total Items**: {n_items}\n")
        f.write(f"- **Total Interactions**: {n_interactions}\n")
        f.write(f"- **Sparsity**: {sparsity:.4%}\n\n")
        
        f.write("### User Activity\n")
        f.write("The user activity follows a power-law distribution, typical of recommender datasets.\n")
        f.write("![User Activity](report_assets/user_activity.png)\n\n")
        
        f.write("### Item Popularity\n")
        f.write("A small number of 'head' items account for a large portion of interactions (Long Tail).\n")
        f.write("![Item Popularity](report_assets/item_popularity.png)\n\n")
        
        f.write("### Content Insights\n")
        f.write("Top authors by book count:\n")
        f.write("![Top Authors](report_assets/top_authors.png)\n\n")
        
        f.write("## 3. Model Performance\n")
        if metrics:
            f.write("Performance evaluated using 5-Fold Cross-Validation on the LTR dataset (30% split).\n")
            f.write("Metric used: **ROC AUC** (Proxy for Ranking Quality on binary targets).\n\n")
            f.write("| Model | Mean AUC |\n")
            f.write("|-------|----------|\n")
            f.write(f"| CatBoost | {metrics[0]:.4f} |\n")
            f.write(f"| XGBoost | {metrics[1]:.4f} |\n")
            f.write(f"| LightGBM | {metrics[2]:.4f} |\n")
            f.write(f"| **Ensemble** | **{metrics[3]:.4f}** |\n\n")
            f.write("![Model Comparison](report_assets/model_comparison.png)\n\n")
        else:
            f.write("Performance metrics could not be calculated (missing OOF data).\n\n")
            
        f.write("## 4. System Architecture\n")
        f.write("### Candidate Generators (Base Models)\n")
        f.write("- **Content-Based**: Uses SentenceTransformer embeddings (`paraphrase-multilingual-MiniLM-L12-v2`).\n")
        f.write("- **Collaborative Filtering**: ALS Matrix Factorization.\n")
        f.write("- **Sequential**: Transition Matrix (Markov) & SASRec (Transformer).\n")
        f.write("- **Graph**: LightGCN.\n\n")
        
        f.write("### Learning to Rank (Ensemble)\n")
        f.write("- **Stacking**: Combines predictions from CatBoost, XGBoost, and LightGBM.\n")
        f.write("- **Features**: 30 features per candidate (Scores, Ranks, Popularity, Affinity, Semantic Similarity).\n")

    print("Report generation complete!")

if __name__ == "__main__":
    generate_report()
