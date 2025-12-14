# AI Librarian: Hybrid Recommender System 📚

**Team Name**: Ouchy
**Kaggle Score**: 0.17689 (MAP@10)

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
This project was built with the assistance of **Google's Agentic AI**. We leveraged AI as an "agentic" support system primarily when facing blockers, specifically to:
*   Identify relevant sources, sites, and endpoints (e.g., public book APIs).
*   Generate inspiration and ideas (candidate features, overall strategies).
*   Accelerate debugging and boilerplate code (data loading, pipelines, scripts).
*   Challenge specific decisions (e.g., "Is data enrichment worth the cost?").
*   We emphasize that while AI served as an accelerator, the final choices—regarding architecture, features, evaluation, and trade-offs—were driven by the team.
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

---

## 11. Data Enrichment via Public Book APIs (Google Books + OpenLibrary)

### 🎯 Objective

At the very beginning of the project, we decided to enrich the book database (items) with missing metadata in order to:
* Better handle the **cold start** problem (items with little to no user history).
* Strengthen **content-based retrieval** and semantic features.
* Standardize fields for future feature engineering (language, pages, year, summary, etc.).

Concretely, we added the following columns to our items file:
* `summary` (description/abstract)
* `published_year` (publication date)
* `page_count` (number of pages)
* `language` (language code)

We hypothesized that richer descriptions combined with language, year, and page counts would capture item similarity better and significantly assist in "cold-start" scenarios.

### 📚 Key Challenge: Multiple ISBNs per Book

In our dataset, a single book entity often possesses multiple ISBNs (ISBN-10/ISBN-13, different editions, variations, etc.). Therefore, enriching "one book" required querying multiple ISBNs and intelligently merging the results.

**Process:**
1.  **Parse:** Split ISBN cells (often separated by `;` or `,`) to get a clean list.
2.  **Structure:** Created an `isbn_list` (list of ISBNs per row) and identified a `first_isbn` (primary).
3.  **Deduplicate:** Built a `unique_isbns` set to avoid redundant API calls.

### 🔌 APIs Used & Fallback Strategy (Pipeline)

We implemented a 3-step strategy using two public sources:

**1. Google Books API (Primary)**
* **Endpoint:** `https://www.googleapis.com/books/v1/volumes?q=isbn:{isbn}`
* **Pros:** Generally strong on descriptions, page counts, language, and dates.
* **Data retrieved:** `description`, `publishedDate`, `pageCount`, `language`.

**2. OpenLibrary API via ISBN (Fallback)**
* **Endpoint:** `https://openlibrary.org/api/books?bibkeys=ISBN:{isbn}&jscmd=details&format=json`
* **Pros:** Sometimes covers books missing from Google; provides alternative fields.
* **Implementation details:**
    * If `description` is empty, we attempt to fallback on the `notes` field.
    * For language, we parse OpenLibrary specific keys (e.g., `/languages/fre`).

**3. OpenLibrary via Title + Author Search (Final Fallback)**
* **Logic:** If the summary is still missing after checking ISBNs, we perform a search.
* **Endpoint:** `https://openlibrary.org/search.json?title=...&author=...&limit=1`
* **Note:** We then fetch the specific `/works/...` endpoint to get the description.
* **Constraint:** This step is slower and "noisier," so it is only executed if the summary is missing to limit costs and errors.

### 🧠 Intelligent Merging Logic

We implemented a robust fusion strategy:
* Initialize an `empty_record` with target fields set to `None`.
* For each API call (Google followed by OpenLibrary), we **only fill fields that are currently missing**.
* If one ISBN doesn't provide all data, we iterate to the next ISBN for the same book.

**Why?**
The dataset contains cases where:
* *ISBN A* → Has the language but no summary.
* *ISBN B* → Has the summary but no page count.
➡️ We "complete" the book record by accumulating the best available information.

### ⚡ Performance: Parallel Calls (ThreadPool)

To handle the large volume of ISBNs, we parallelized the API requests:
* `ThreadPoolExecutor(max_workers=20)`
* `timeout=5s` per request
* Global mapping `isbn_to_data[isbn] = meta`

Finally, we reconstruct the final metadata at the dataset row level by iterating through the `isbn_list` until "core" fields are filled, exporting the result to an enriched Excel file (e.g., `books_enriched_....xlsx`).

### ✅ Real-World Result: High Effort, Zero Gain

Although data enrichment seemed logically essential at the start, we found that it **did not improve model performance** (neither on Kaggle nor in local validation), despite:
* Adding summaries (`summary`).
* Creating derived features (language/year/pages).
* Hoping for a significant boost on cold-start items.

**Retrospective:**
* The strongest signal came primarily from **interactions** (Collaborative Filtering + Sequential) and expert scores in LTR.
* The quality of retrieved summaries was inconsistent (often empty, noisy, or in the wrong language).
* The impact on `MAP@10` was insignificant or null.

👉 **Conclusion:** It was a good "product" intuition, but in the context of this challenge, it mostly resulted in lost time during the early phase. We subsequently refocused our efforts on the **Two-Stage + LTR** approach, which actually drove performance.

---

## 12. Collaboration & Support (Cross-Group Exchanges & Assistants)

We also benefited from discussions with:
*   Another student group ("Geneva"),
*   And the two assistants: Donia Gasmii and Stergios Konstantinidis.
  
These interactions helped us to:
*   Challenge our hypotheses (time-aware validation, data leakage, etc.),
*   Discuss candidate strategies (long-tail vs. popularity),
*   Clarify specific aspects of the pipeline and evaluation,
*   Resolve modeling decisions (LTR vs. stacking, robustness, etc.).

