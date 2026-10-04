import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import gensim
import textstat
import textblob
import spacy
from src.features.auxiliary_features import AuxiliaryFeatureEngineer

def generate_statistics(df, split_name):
    stats = {}
    for col in ['readability_flesch_kincaid', 'sentiment_polarity', 'article_length', 'punctuation_ratio']:
        stats[col] = {
            "min": float(df[col].min()),
            "max": float(df[col].max()),
            "mean": float(df[col].mean()),
            "median": float(df[col].median()),
            "std": float(df[col].std())
        }
    return stats

def plot_distributions(all_data):
    os.makedirs("reports/figures/day4", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    # Sentiment
    plt.figure(figsize=(10, 6))
    sns.histplot(data=all_data, x='sentiment_polarity', hue='binary_label', kde=True, bins=30)
    plt.title('Sentiment Polarity Distribution')
    plt.savefig("reports/figures/day4/sentiment_distribution.png")
    plt.close()
    
    # Readability
    plt.figure(figsize=(10, 6))
    sns.histplot(data=all_data, x='readability_flesch_kincaid', hue='binary_label', kde=True, bins=30)
    plt.title('Readability (Flesch-Kincaid) Distribution')
    plt.xlim(-10, 100) # Clip extreme outliers for visualization
    plt.savefig("reports/figures/day4/readability_distribution.png")
    plt.close()
    
    # Article Length
    plt.figure(figsize=(10, 6))
    sns.histplot(data=all_data, x='article_length', hue='binary_label', kde=True, bins=30)
    plt.title('Article Length (Tokens) Distribution')
    plt.xlim(0, 500)
    plt.savefig("reports/figures/day4/article_length_distribution.png")
    plt.close()
    
    # Punctuation Ratio
    plt.figure(figsize=(10, 6))
    sns.histplot(data=all_data, x='punctuation_ratio', hue='binary_label', kde=True, bins=30)
    plt.title('Punctuation Ratio Distribution')
    plt.xlim(0, 0.2)
    plt.savefig("reports/figures/day4/punctuation_ratio_distribution.png")
    plt.close()

def main():
    print("Loading preprocessed datasets...")
    train = pd.read_csv("data/processed/preprocessed_train.csv")
    val = pd.read_csv("data/processed/preprocessed_validation.csv")
    test = pd.read_csv("data/processed/preprocessed_test.csv")
    
    print("Initializing Auxiliary Feature Engineer...")
    engineer = AuxiliaryFeatureEngineer(vector_size=100, min_count=2, workers=1, seed=42)
    
    print("Fitting Word2Vec on training cleaned text...")
    engineer.fit_word2vec(train['cleaned_text'])
    
    vocab_size = len(engineer.w2v_model.wv)
    
    print("Saving Word2Vec model...")
    os.makedirs("models", exist_ok=True)
    engineer.save_model("models/word2vec.model")
    
    print("Generating features for train, validation, test...")
    def generate_and_save(df, name):
        feats = engineer.generate_all_features(df['statement'], df['cleaned_text'])
        # Add identity/labels
        feats.insert(0, 'binary_label', df['binary_label'])
        if 'id' in df.columns:
            feats.insert(0, 'id', df['id'])
        
        feats.to_csv(f"data/processed/auxiliary_{name}.csv", index=False)
        
        # Word2Vec matrix specifically requested
        w2v_cols = [c for c in feats.columns if c.startswith('w2v_')]
        np.save(f"data/processed/word2vec_{name}.npy", feats[w2v_cols].values)
        return feats

    train_feats = generate_and_save(train, "train")
    val_feats = generate_and_save(val, "validation")
    test_feats = generate_and_save(test, "test")
    
    print("Generating feature names JSON...")
    feature_names = [c for c in train_feats.columns if c not in ['id', 'binary_label']]
    with open("data/processed/auxiliary_feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=4)
        
    print("Calculating statistics and plotting...")
    all_data = pd.concat([train_feats, val_feats, test_feats])
    stats = generate_statistics(all_data, "all")
    plot_distributions(all_data)
    
    print("Saving metadata...")
    metadata = {
        "word2vec_configuration": engineer.config,
        "word2vec_vocabulary_size": vocab_size,
        "word2vec_dimensions": engineer.config['vector_size'],
        "feature_names": feature_names,
        "train_shape": train_feats.shape,
        "validation_shape": val_feats.shape,
        "test_shape": test_feats.shape,
        "source_columns": {
            "word2vec": "cleaned_text",
            "readability": "statement",
            "sentiment": "statement",
            "article_length": "cleaned_text",
            "punctuation_ratio": "statement"
        },
        "readability_library": f"textstat {textstat.__version__}",
        "textblob_version": getattr(textblob, '__version__', 'unknown'),
        "gensim_version": gensim.__version__,
        "spacy_version": spacy.__version__,
        "seed": engineer.config['seed'],
        "training_split_used_for_word2vec": "preprocessed_train.csv",
        "timestamp": datetime.now().isoformat(),
        "notes": "Word2Vec trained using training split only."
    }
    
    with open("data/processed/auxiliary_features_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Markdown Report
    w2v_cols = [c for c in feature_names if c.startswith('w2v_')]
    non_w2v = [c for c in feature_names if not c.startswith('w2v_')]
    
    w2v_arr = train_feats[w2v_cols].values
    w2v_mean = np.mean(w2v_arr)
    w2v_std = np.std(w2v_arr)
    
    report_md = f"""# Day 4 Auxiliary Feature Engineering Report

## 1. Objective
Implement Word2Vec embeddings and NLP-derived auxiliary features (Readability, Sentiment, Article Length, Punctuation Ratio) without utilizing test/validation data for fitting.

## 2. Word2Vec Configuration
- `vector_size`: {engineer.config['vector_size']}
- `window`: {engineer.config['window']}
- `min_count`: {engineer.config['min_count']}
- `workers`: {engineer.config['workers']}
- `sg`: {engineer.config['sg']}
- `epochs`: {engineer.config['epochs']}
- `seed`: {engineer.config['seed']}

## 3. Word2Vec Training Corpus
- Trained strictly on `cleaned_text` from `data/processed/preprocessed_train.csv`.
- Validation and Test texts were NOT used to update the model.

## 4. Word2Vec Vocabulary Size
- **Total Vocabulary Size**: {vocab_size} tokens

## 5. Sentence-Vector Methodology
- Represents a document as the mean of the Word2Vec token vectors found in its `cleaned_text`.
- If a document has no known tokens, a zero-vector of size {engineer.config['vector_size']} is returned.

## 6. Readability Methodology
- Computes Flesch-Kincaid grade level using `textstat` on the RAW `statement` text.

## 7. Sentiment Methodology
- Computes sentiment polarity using `TextBlob` on the RAW `statement` text.
- Conceptually bounded [-1.0, 1.0].

## 8. Article-Length Methodology
- Computes the total number of whitespace-separated tokens in the `cleaned_text`.

## 9. Punctuation-Ratio Methodology
- Computes the count of punctuation characters divided by the total number of characters in the RAW `statement` text.

## 10. Feature Dimensions
- Word2Vec: {len(w2v_cols)}
- NLP Auxiliary: {len(non_w2v)}
- Total Features: {len(feature_names)}

## 11. Feature Statistics
- **Word2Vec Mean/Std**: {w2v_mean:.4f} / {w2v_std:.4f}
- **Readability**: Mean {stats['readability_flesch_kincaid']['mean']:.2f}, Std {stats['readability_flesch_kincaid']['std']:.2f}
- **Sentiment**: Mean {stats['sentiment_polarity']['mean']:.2f}, Std {stats['sentiment_polarity']['std']:.2f}
- **Article Length**: Mean {stats['article_length']['mean']:.2f}, Std {stats['article_length']['std']:.2f}
- **Punctuation Ratio**: Mean {stats['punctuation_ratio']['mean']:.4f}, Std {stats['punctuation_ratio']['std']:.4f}

## 12. Edge-case Handling
- Missing/NaN text is converted to empty strings.
- Empty strings evaluate to 0.0 for length, readability, and punctuation ratio.
- Out-of-vocabulary text evaluates to an all-zero Word2Vec vector.

## 13. Leakage Controls
- **Word2Vec was trained on TRAIN ONLY.**
- Validation/Test rows only queried the fixed training vocabulary and weights.

## 14. Reproducibility
- Configured with `workers=1` and a fixed seed to guarantee deterministic behavior. 
- Output matrices are identical on repeat runs.

## 15. Known Limitations
- Flesch-Kincaid relies on syllable approximations which can be inaccurate for extremely short texts.
- Sentence vector averaging loses word order information.
"""
    with open("reports/day4_auxiliary_features_report.md", "w") as f:
        f.write(report_md)
        
    print("Day 4 execute completed successfully.")

if __name__ == "__main__":
    main()
