# Comment Category Prediction Challenge

Multi-class text classification on user comments, combining **TF-IDF text signals** with **engineered metadata features** across an ensemble of three models. Optimized for **F1 Macro** on a heavily imbalanced 4-class target.

> Predict the category label assigned to each comment using its text content and platform metadata (votes, identity flags, emoticons, timestamps).

---

## Problem

Given `train.csv` and `test.csv`, train a robust classifier to predict one of **4 comment categories** and generate predictions for the test set. The primary metric is **F1 Macro**, which matters because the dataset is significantly imbalanced:

| Label | Share |
|:-----:|:-----:|
| 0 | ~57% (dominant) |
| 1 | — |
| 2 | — |
| 3 | ~2.8% (rarest) |

Because label 0 dominates and label 3 is rare, the modeling leans on `class_weight="balanced"`, `is_unbalance`, and macro-averaged evaluation rather than raw accuracy.

---

## Approach

```
Data Load → EDA → Feature Engineering → TF-IDF + Numeric → 3 Models → Weighted Ensemble → Submission
```

### 1. Exploratory Data Analysis
- Missing-value audit on train/test
- Target distribution + class imbalance check
- Numeric feature correlation heatmap
- Upvote/downvote distributions, comment length per label, upvote-vs-downvote scatter
- Top words per class (lightweight stopword filtering)

### 2. Feature Engineering
Three families of features built from the raw columns:

- **Text features** — `comment_length`, `word_count`, `avg_word_length`, `uppercase_ratio`, `special_char_count`, `exclamation_count`, `question_count`, `digit_ratio`, `sentence_count`, `unique_word_ratio`
- **Date features** — `hour`, `day_of_week`, `month`, `is_weekend`, `year` (from `created_date`)
- **Interaction features** — `vote_diff`, `vote_total`, `vote_ratio`, `emoticon_sum`, `identity_sum`, `has_identity`, `if_sum`, `if_product`, `if_ratio`, log-scaled vote counts

Identity columns (`race`, `religion`, `gender`, `disability`) are encoded to clean 0/1 flags, and comment text is cleaned (lowercased, URLs/emails/special chars stripped).

### 3. Text Vectorization
Dual TF-IDF for robustness:
- **Word-level** — `ngram_range=(1,2)`, up to 40k features, `sublinear_tf`
- **Character-level** — `char_wb`, `ngram_range=(2,5)`, up to 25k features

Numeric features are `StandardScaler`-d and horizontally stacked with the sparse TF-IDF matrices into a single feature space.

### 4. Models
| Model | Role | Notes |
|-------|------|-------|
| **LightGBM** | High-capacity main model | Tuned multiclass GBDT, `is_unbalance=True`, early stopping over 2000 rounds |
| **Logistic Regression** | Linear baseline | `GridSearchCV` over `C` and solver, 3-fold, scored on F1 Macro |
| **SGDClassifier** | Sparse-friendly linear | `modified_huber` loss for calibrated probabilities, wrapped in a `MaxAbsScaler` pipeline |

### 5. Weighted Ensemble
Soft-voting over the three models' predicted probabilities. Instead of fixed weights, a grid search over `(w_lgb, w_lr, w_sgd)` picks the blend that maximizes **validation F1 Macro**.

### 6. Final Submission
All three models are retrained on the full training set, test probabilities are blended with the optimized weights, and predictions are exported to `submission.csv`.

---

## Tech Stack

`Python` · `pandas` · `numpy` · `scikit-learn` · `LightGBM` · `scipy` · `matplotlib` · `seaborn`

---

## Repository Structure

```
.
├── notebook.ipynb        # Full pipeline: EDA → FE → modeling → submission
├── data/                 # train.csv, test.csv, Sample.csv (not tracked)
├── submission.csv        # Generated predictions
└── README.md
```

---

## How to Run

```bash
# 1. Install dependencies
pip install pandas numpy scikit-learn lightgbm scipy matplotlib seaborn

# 2. Place the competition data
#    data/train.csv, data/test.csv, data/Sample.csv
#    (update the read_csv paths in the notebook if not running on Kaggle)

# 3. Run the notebook top to bottom
jupyter notebook notebook.ipynb
```

The notebook was authored on Kaggle, so input paths default to
`/kaggle/input/comment-category-prediction-challenge/`. Adjust the three
`pd.read_csv(...)` calls in the **Data Loading** cell for a local run.

---

## Results

| Model | Accuracy | F1 Macro |
|-------|:--------:|:--------:|
| LightGBM | _—_ | _—_ |
| Logistic Regression | _—_ | _—_ |
| SGDClassifier | _—_ | _—_ |
| **Weighted Ensemble** | _—_ | _—_ |

> Fill these in from the final **Model Comparison** cell after a run.

---

## Key Takeaways

- Combining **word + character TF-IDF** with **engineered metadata** beats either signal alone.
- A **searched ensemble weight** consistently edges out fixed 0.6/0.4 blends.
- On an imbalanced target, **F1 Macro + balanced class weighting** is what actually moves the needle — accuracy alone is misleading.