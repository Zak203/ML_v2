import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import pickle

def generate_embeddings_enrichi():
    print("Loading items...")
    # Load enriched data
    enriched_items = pd.read_csv('books_enriched_FINAL.csv')
    
    # Load base items to ensure coverage (safety check)
    base_items = pd.read_csv('items.csv')
    
    print(f"Enriched items: {len(enriched_items)}")
    print(f"Base items: {len(base_items)}")
    
    # Merge to ensure we have all items, even if enriched data is missing some
    items = pd.merge(base_items[['i']], enriched_items, on='i', how='left')
    
    # Handle NaNs
    items['Title'] = items['Title'].fillna('')
    items['Author'] = items['Author'].fillna('')
    items['Subjects'] = items['Subjects'].fillna('')
    items['summary'] = items['summary'].fillna('')
    items['Publisher'] = items['Publisher'].fillna('')
    items['published_year'] = items['published_year'].fillna('').astype(str).str.replace(r'\.0$', '', regex=True)
    items['language'] = items['language'].fillna('')

    print("Constructing rich text for embedding...")
    # Construct text with prefixes for better semantic understanding
    items['text'] = (
        "Title: " + items['Title'] + " " + 
        "Author: " + items['Author'] + " " + 
        "Subjects: " + items['Subjects'] + " " + 
        "Summary: " + items['summary'] + " " + 
        "Publisher: " + items['Publisher'] + " " + 
        "Year: " + items['published_year'] + " " + 
        "Language: " + items['language']
    )
    
    print("Loading model (paraphrase-multilingual-MiniLM-L12-v2)...")
    # Using the better multilingual model as requested
    model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
    
    print("Encoding items...")
    embeddings = model.encode(items['text'].tolist(), show_progress_bar=True)
    
    print(f"Embeddings shape: {embeddings.shape}")
    
    # Save embeddings and item ids
    with open('item_embeddings.pkl', 'wb') as f:
        pickle.dump({'item_ids': items['i'].values, 'embeddings': embeddings}, f)
        
    print("Saved embeddings to item_embeddings.pkl")

if __name__ == "__main__":
    generate_embeddings_enrichi()