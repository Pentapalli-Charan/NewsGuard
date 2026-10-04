import os
import json
import pandas as pd
import numpy as np
import scipy.sparse
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from src.features.tfidf_features import TfidfFeatureEngineer

def analyze_discriminative_terms(X_train, y_train, feature_names):
    """Calculate mean TF-IDF scores for REAL and FAKE classes."""
    # Ensure y_train is a numpy array
    y_train = y_train.values if isinstance(y_train, pd.Series) else y_train
    
    # Masks for classes
    is_real = (y_train == 'REAL')
    is_fake = (y_train == 'FAKE')
    
    # Calculate means
    # X_train is sparse CSR. We can mean over axis 0.
    # Note: np.asarray(X.mean(axis=0)).flatten() returns a 1D array
    mean_real = np.asarray(X_train[is_real].mean(axis=0)).flatten()
    mean_fake = np.asarray(X_train[is_fake].mean(axis=0)).flatten()
    
    diff = mean_real - mean_fake
    abs_diff = np.abs(diff)
    
    df = pd.DataFrame({
        'feature': feature_names,
        'mean_tfidf_real': mean_real,
        'mean_tfidf_fake': mean_fake,
        'difference': diff,
        'absolute_difference': abs_diff
    })
    
    df = df.sort_values(by='absolute_difference', ascending=False)
    return df

def plot_discriminative_terms(df_terms, top_n=20):
    os.makedirs("reports/figures/day3", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # 1. Top discriminative terms overall
    top_overall = df_terms.head(top_n)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='absolute_difference', y='feature', data=top_overall, color='purple')
    plt.title(f'Top {top_n} Discriminative Terms (Absolute Difference in Mean TF-IDF)')
    plt.tight_layout()
    plt.savefig("reports/figures/day3/top_discriminative_terms.png")
    plt.close()
    
    # 2. Top REAL-associated terms
    # Sort by difference (positive means REAL > FAKE)
    top_real = df_terms.sort_values(by='difference', ascending=False).head(top_n)
    plt.figure(figsize=(10, 8))
    sns.barplot(x='difference', y='feature', data=top_real, color='blue')
    plt.title(f'Top {top_n} Terms Associated with REAL')
    plt.tight_layout()
    plt.savefig("reports/figures/day3/top_real_terms.png")
    plt.close()
    
    # 3. Top FAKE-associated terms
    # Sort by difference ascending (negative means FAKE > REAL)
    top_fake = df_terms.sort_values(by='difference', ascending=True).head(top_n)
    # Plot absolute value of difference for clarity
    top_fake['fake_score'] = top_fake['difference'].abs()
    plt.figure(figsize=(10, 8))
    sns.barplot(x='fake_score', y='feature', data=top_fake, color='red')
    plt.title(f'Top {top_n} Terms Associated with FAKE')
    plt.tight_layout()
    plt.savefig("reports/figures/day3/top_fake_terms.png")
    plt.close()

def main():
    print("Loading preprocessed datasets...")
    train = pd.read_csv("data/processed/preprocessed_train.csv")
    val = pd.read_csv("data/processed/preprocessed_validation.csv")
    test = pd.read_csv("data/processed/preprocessed_test.csv")
    
    print("Initializing TF-IDF Engineer...")
    engineer = TfidfFeatureEngineer(ngram_range=(1, 2), max_features=10000)
    
    print("Fitting TF-IDF on training data...")
    X_train = engineer.fit_transform_train(train['cleaned_text'])
    
    print("Transforming validation and test data...")
    X_val = engineer.transform(val['cleaned_text'])
    X_test = engineer.transform(test['cleaned_text'])
    
    print("Saving sparse matrices...")
    scipy.sparse.save_npz("data/processed/tfidf_train.npz", X_train)
    scipy.sparse.save_npz("data/processed/tfidf_validation.npz", X_val)
    scipy.sparse.save_npz("data/processed/tfidf_test.npz", X_test)
    
    print("Saving vectorizer and feature names...")
    os.makedirs("models", exist_ok=True)
    engineer.save_vectorizer("models/tfidf_vectorizer.joblib")
    engineer.save_feature_names("data/processed/tfidf_feature_names.json")
    
    feature_names = engineer.get_feature_names()
    vocab_size = len(feature_names)
    unigrams = sum(1 for f in feature_names if ' ' not in f)
    bigrams = vocab_size - unigrams
    
    print("Analyzing discriminative terms (Training data only)...")
    df_terms = analyze_discriminative_terms(X_train, train['binary_label'], feature_names)
    os.makedirs("reports/metrics", exist_ok=True)
    df_terms.head(100).to_csv("reports/metrics/day3_discriminative_terms.csv", index=False)
    
    print("Generating visualizations...")
    plot_discriminative_terms(df_terms)
    
    # Metadata
    metadata = {
        "vectorizer_configuration": engineer.config,
        "ngram_range": engineer.config['ngram_range'],
        "max_features": engineer.config['max_features'],
        "min_df": engineer.config['min_df'],
        "max_df": engineer.config['max_df'],
        "vocabulary_size": vocab_size,
        "unigram_count": unigrams,
        "bigram_count": bigrams,
        "train_rows": X_train.shape[0],
        "validation_rows": X_val.shape[0],
        "test_rows": X_test.shape[0],
        "train_feature_shape": X_train.shape,
        "validation_feature_shape": X_val.shape,
        "test_feature_shape": X_test.shape,
        "sparsity_train": 1.0 - (X_train.nnz / (X_train.shape[0] * X_train.shape[1])),
        "fit_dataset": "preprocessed_train.csv",
        "preprocessing_source": "cleaned_text",
        "timestamp": datetime.now().isoformat(),
        "notes": "TF-IDF fitted on training split only."
    }
    
    with open("data/processed/tfidf_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Report markdown
    report_md = f"""# Day 3 TF-IDF Feature Engineering Report

## 1. Objective
Implement TF-IDF feature engineering using unigrams and bigrams based solely on the training dataset.

## 2. Configuration
- `ngram_range`: {engineer.config['ngram_range']}
- `max_features`: {engineer.config['max_features']} (Limits the vocabulary to the top features to manage sparse matrix dimensionality).
- `min_df`: {engineer.config['min_df']}
- `max_df`: {engineer.config['max_df']}

## 3. Vocabulary Size
- **Total Features**: {vocab_size}
- **Unigrams**: {unigrams} ({unigrams/vocab_size*100:.1f}%)
- **Bigrams**: {bigrams} ({bigrams/vocab_size*100:.1f}%)

## 4. Matrix Dimensions
- **Train**: {X_train.shape}
- **Validation**: {X_val.shape}
- **Test**: {X_test.shape}

## 5. Sparsity
- **Train Matrix Sparsity**: {metadata['sparsity_train']*100:.2f}% zero elements.

## 6. TRAINING-DERIVED INFORMATION: Top Discriminative Terms
*Note: This is exploratory feature analysis calculated using class-specific mean TF-IDF scores on the TRAINING data only. This is not model-based feature importance.*

**Top 5 REAL-associated terms:**
{', '.join(df_terms.sort_values(by='difference', ascending=False).head(5)['feature'].tolist())}

**Top 5 FAKE-associated terms:**
{', '.join(df_terms.sort_values(by='difference', ascending=True).head(5)['feature'].tolist())}

## 7. Leakage Prevention
**TF-IDF fitted on training split only.** Validation and test sets were transformed using the pre-fitted vectorizer, ensuring no vocabulary leakage.

## 8. Reproducibility
The output is completely deterministic for a given input dataset.
"""
    with open("reports/day3_tfidf_report.md", "w") as f:
        f.write(report_md)
        
    print("Day 3 execute completed successfully.")

if __name__ == "__main__":
    main()
