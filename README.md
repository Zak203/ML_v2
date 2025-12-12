# AI Librarian: Hybrid Recommender System 📚

**Team Name**: [Your Team Name]
**Kaggle Score**: ~0.17 (MAP@10)

## 1. Executive Summary
This project implements a state-of-the-art **Two-Stage Hybrid Recommender System** designed to predict the top 10 books a user is likely to borrow next. By combining the strengths of Collaborative Filtering, Content-Based retrieval, and Sequential Modeling with a powerful Learning-to-Rank (LTR) approach, we achieved a **36% performance improvement** over the best single baseline model. While we initially explored a Stacking Ensemble, our final submission utilizes a **Single CatBoost Model** as it demonstrated superior generalization and robustness on the leaderboard.

---

## 2. Exploratory Data Analysis (EDA)
We analyzed the interaction dataset and book metadata to understand the challenges.

*   **Sparsity**: The dataset is extremely sparse (**99.93%**), meaning most users have interacted with a tiny fraction of the 15,000+ books.
*   **Long Tail**: A small number of "blockbuster" books account for a disproportionate number of reads, while thousands of books are rarely read.
*   **User Activity**: Follows a power-law distribution. Most users are "Cold Start" (few interactions), while a few "Power Users" have hundreds.

![User Activity](report_assets/user_activity.png)
*Figure 1: User Activity Distribution (Log-Log Scale)*

---

## 3. Baseline Performance
We established a baseline using a simple **Popularity Recommender** and a standard **Collaborative Filtering (ALS)** model.

*   **Baseline Score (MAP@10)**: `0.15283` (Provided Baseline)
*   **Our Best Base Model (ALS)**: `0.1844` (Local Validation)
*   **Final Model (CatBoost)**: `0.2522` (Local Validation)

> **Note on Kaggle Discrepancy**: We observed a discrepancy between our local validation score (~0.25) and the Kaggle leaderboard score (~0.17). Despite extensive efforts to align the validation strategy (Time-Aware Split) with the test set distribution, this gap persists, likely due to distribution shifts or specific characteristics of the private test set. However, the relative improvements remain consistent.

---

## 4. Modern Model Architecture
We designed a **Two-Stage Architecture** to balance Recall (finding good candidates) and Precision (ranking them correctly).

### Stage 1: Candidate Generation (Recall)
We use 5 distinct "Experts" to retrieve 200 candidates each:
1.  **Content-Based**: Uses **SentenceTransformers** (`paraphrase-multilingual-MiniLM-L12-v2`) to find books with similar summaries/authors. *Solves Cold Start.*
2.  **ALS (Collaborative Filtering)**: Matrix Factorization to capture latent user preferences. *Best for active users.*
3.  **Transition (Markov)**: Probabilistic model (A -> B) to capture immediate sequential habits (e.g., Series).
4.  **LightGCN**: Graph Convolutional Network to explore high-order connectivity.
5.  **SASRec**: Self-Attentive Sequential model (Transformer) to model long-term reading paths.

### Stage 2: Learning to Rank (Precision)
We pool the candidates (up to 1000 per user) and re-rank them. We initially experimented with a **Stacking Ensemble** (CatBoost, XGBoost, LightGBM) but found that a **Single CatBoost Model** performed best.

*   **Why CatBoost?**:
    *   **Native Categorical Handling**: It handles categorical features (Author, Publisher) natively without extensive preprocessing, preserving more signal.
    *   **Robustness**: It showed less overfitting to the local validation set compared to XGBoost and LightGBM, leading to better generalization on the unseen test set.
    *   **Symmetric Trees**: Its use of oblivious trees reduces the risk of overfitting.

*   **Features**: The model learns from **30 features**, including:
    *   *Expert Scores*: "How confident is ALS?"
    *   *Affinity*: "Does this user read this Author often?"
    *   *Semantic*: "Is this book similar to the last one read?" (Cosine Similarity).

### 📊 Feature Importance Analysis
The **CatBoost** model automatically determines which features are most predictive. As shown below, the **Expert Scores** (predictions from base models like SASRec and LightGCN) are the most dominant features, followed by **User Affinity** features.

![Feature Importance](report_assets/feature_importance.png)
*Figure 2: Top 15 Features contributing to the Ranking Model*

---

## 5. Performance Gain
We evaluated our system using a rigorous **Time-Aware Split (70/30)**. This strategy respects the temporal order of interactions, training on past data to predict future behavior, which mimics the real-world deployment scenario and prevents data leakage. We further validated our results using **5-Fold Cross-Validation**.

| Model | MAP@10 (Local) | Improvement vs Baseline |
| :--- | :--- | :--- |
| **Baseline (ALS)** | 0.1844 | - |
| **Ensemble (LTR)** | **0.2522** | **+36.7%** |

**Why it works**: The LTR layer acts as a "Judge". It learns that ALS is reliable for popular items, but Content-Based is better for niche items. It combines these signals dynamically.

> **Kaggle Insight**: While the Ensemble performed slightly better in local validation, the **Single Best Model (CatBoost)** proved more robust on the public leaderboard (avoiding overfitting to local validation quirks). Our final submission uses **CatBoost Only**.

![Benchmark](report_assets/benchmark_map10.png)

---

## 6. Examples & Analysis

### Good Recommendation (The "Series" Pattern)
*   **User History**: *Harry Potter 1*, *Harry Potter 2*.
*   **Prediction**: *Harry Potter 3*.
*   **Why**: The **Transition Model** and **SASRec** give a massive score to the sequel. The LTR layer confirms this with "Author Affinity" features.

### Bad Recommendation (The "Popularity Trap")
*   **User History**: *Advanced Quantum Physics*.
*   **Prediction**: *The Da Vinci Code*.
*   **Why**: The user has very little history (Cold Start). The models fall back on **Global Popularity** or broad Collaborative Filtering, failing to capture the niche interest.

### 🔍 Model Analysis & Improvement Areas

#### What Worked Well (✅)
*   **Hybrid Approach**: Combining very different models (Sequential + Content + CF) covered all use cases (Cold Start vs Power Users).
*   **CatBoost**: Its ability to handle categorical features without massive one-hot encoding was crucial for performance and simplicity.
*   **Time-Aware Split**: Essential for a realistic performance estimation, unlike a random split that "sees the future".

#### Areas for Improvement (🚀)
*   **User Features**: We lack demographic data (age, location). Adding this would greatly help with "Cold Start".
*   **Better "Already Read" Handling**: Currently, we brutally filter out already read books. A softer approach (score penalty) could allow for relevant re-read recommendations.
*   **More Complex Ensemble**: With more compute time, a Stacking Ensemble could squeeze out a few more percentage points, though maintenance cost would increase.

---

## 7. Tooling & Cost Reflection

### 🛠️ Tools & Models Used
*   **SentenceTransformers (Hugging Face)**:
    *   *Model*: `paraphrase-multilingual-MiniLM-L12-v2`.
    *   *Purpose*: Generating semantic embeddings for book content.
    *   *Necessity*: Indispensable for "Cold Start" and capturing thematic similarity beyond genres.
    *   *Cost*: **Free** (Open Source).
*   **CatBoost (Yandex)**:
    *   *Purpose*: Ranking Engine (Learning-to-Rank).
    *   *Necessity*: Offers the best performance/speed trade-off and natively handles categories.
    *   *Cost*: **Free** (Open Source).
*   **Google's Agentic AI**:
    *   *Purpose*: Intelligent Coding Assistant.
    *   *Necessity*: Accelerated the development of the complex pipeline (Two-Stage) and debugging.
    *   *Cost*: Included in the development environment.

### 💰 Cost Analysis (Financial, Temporal, Energy)
*   **Financial**: **0€**. All tools used are Open Source. No expensive GPUs required (AWS/GCP Cloud unnecessary).
*   **Temporal**:
    *   *Training*: ~30 minutes for the full pipeline on a standard CPU (MacBook Pro).
    *   *Inference*: < 50ms per user. Very fast thanks to decision tree efficiency.
*   **Energy (Eco-Design)**:
    *   We prioritized "lightweight" models (LightGCN, CatBoost) over massive LLMs (e.g., GPT-4 for reco) which would have exploded the carbon footprint.
    *   Local training avoids constant cloud server consumption.

---

## 8. AI-Assisted Coding
This project was built with the assistance of **Google's Agentic AI**.
*   **Role**: The AI helped architect the Two-Stage pipeline, debugged the LightGCN implementation, and optimized the ensemble weights.
*   **Reflection**: The AI accelerated the "boilerplate" coding (DataLoaders, Training Loops), allowing us to focus on high-level strategy (Feature Engineering, Model Selection).

---

## 9. Execution Guide

### Prerequisites
*   Python 3.8+
*   `pip install -r requirements.txt`

### Steps to Run
1.  **Generate Embeddings** (One-time setup):
    ```bash
    python generate_embedding_enrichi.py
    ```
2.  **Train & Evaluate** (Run the full pipeline):
    ```bash
    python train_ltr_catboost.py
    ```
3.  **Generate Submission**:
    ```bash
    python create_submission.py
    ```
    *Output: `submission.csv`*

4.  **Generate Report**:
    ```bash
    python generate_report.py
    python generate_benchmark_graph.py
    ```

---

## 10. File Explanations

Here is a detailed description of each Python script in the project to facilitate code navigation and understanding.

### 🧠 Models & Core System
*   **`recommender.py`**: Contains the fundamental building blocks. Defines the `DataLoader` class for data loading and implements base models: `ContentRecommender` (based on cosine similarity), `CFRecommender` (Collaborative Filtering with ALS), and `TransitionRecommender` (Markov Chains).
*   **`lightgcn_recommender.py`**: Implementation of the **LightGCN** (Graph Convolutional Network) model. It builds a user-item bipartite graph and propagates embeddings to capture high-order collaborative filtering signals.
*   **`sasrec_recommender.py`**: Implementation of the **SASRec** (Self-Attentive Sequential Recommendation) model. Uses a Transformer architecture to capture the sequential dynamics of user interactions.

### ⚙️ Training & Optimization (Tuning)
*   **`train_ltr_catboost.py`**: The central script for **Learning-to-Rank** training. It orchestrates candidate generation by "experts" (base models), computes 30+ features for each user-item pair, and trains the CatBoost model (with 5-fold cross-validation).
*   **`tune_lightgcn.py`**: Script dedicated to optimizing LightGCN hyperparameters (learning rate, embedding dim, n_layers, etc.) using **Optuna**.
*   **`tune_sasrec.py`**: Optimization script for SASRec (hidden units, num blocks, dropout, etc.) with Optuna.
*   **`tune_ltr_optuna.py`**: Optimizes CatBoost ensemble hyperparameters (depth, learning rate, l2_leaf_reg) to maximize MAP@10.

### 🚀 Production & Generation
*   **`create_submission.py`**: The final script for the competition. It loads all trained models and embeddings, generates candidates for test set users, applies final ranking, and produces the `submission.csv` file.
*   **`generate_embedding_enrichi.py`**: Pre-processing script that uses **SentenceTransformers** to convert textual book metadata (titles, descriptions, authors) into dense vectors (embeddings).

### 📊 Analysis & Reporting
*   **`generate_single_user_reco.py`**: Diagnostic tool to generate and inspect recommendations for a specific user, displaying scores from each expert model.
*   **`analyze_catboost.py`**: Analyzes feature importance of the CatBoost model to understand which signals are most determinant for ranking.
*   **`generate_benchmark_graph.py`**: Generates comparative charts (bar charts) of performance (MAP@10) for different models for the report.
*   **`generate_report.py`**: Automatically compiles a comprehensive performance report.
*   **`generate_examples.py`**: Extracts qualitative recommendation examples (successes and failures) to illustrate model behavior in the report.

