import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from catboost import CatBoostRanker
import pickle

def analyze_catboost():
    print("Loading CatBoost Model (Fold 0)...")
    try:
        model = CatBoostRanker()
        model.load_model("ltr_catboost_fold0.cbm")
    except:
        print("Could not load model.")
        return

    # Feature Names (Must match generate_features order)
    feature_names = [
        # 0-4: Base Model Scores
        "Content_Score", "ALS_Score", "Transition_Score", "LightGCN_Score", "SASRec_Score",
        # 5-9: Static Item Features
        "Item_Popularity", "User_Activity", "Author_Popularity", "Year", "Page_Count",
        # 10-12: Affinity Counts
        "Author_Count", "Subject_Score", "Publisher_Count",
        # 13-14: Semantic Similarity
        "Sim_User_Centroid", "Sim_Last_Item",
        # 15-17: Ratios
        "Author_Ratio", "Publisher_Ratio", "Subject_Ratio",
        # 18-19: New Features (Phase 7)
        "Time_Since_Last", "Same_Author_Last",
        # 20-24: Ranks
        "Content_Rank", "ALS_Rank", "Transition_Rank", "LightGCN_Rank", "SASRec_Rank",
        # 25: Vote Count
        "Vote_Count",
        # 26-29: Aggregates
        "Mean_Score", "Std_Score", "Min_Score", "Max_Score"
    ]
    
    # Get Feature Importance
    importance = model.get_feature_importance(type='PredictionValuesChange')
    
    # Create DataFrame
    df_imp = pd.DataFrame({'feature': feature_names, 'importance': importance})
    df_imp = df_imp.sort_values('importance', ascending=False)
    
    print("\nTop 10 Features:")
    print(df_imp.head(10))
    
    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(df_imp['feature'][:20][::-1], df_imp['importance'][:20][::-1], color='skyblue')
    plt.xlabel('Importance (PredictionValuesChange)')
    plt.title('CatBoost Feature Importance')
    plt.tight_layout()
    plt.savefig('report_assets/feature_importance.png')
    print("\nSaved report_assets/feature_importance.png")

if __name__ == "__main__":
    analyze_catboost()
