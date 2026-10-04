import os
import json
import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import MinMaxScaler
import joblib

from src.features.feature_union import build_feature_union

class BranchScaler:
    """Scales only the dense auxiliary features of the combined sparse matrix."""
    def __init__(self, scaler):
        self.scaler = scaler
        self.num_tfidf = 10000

    def fit(self, X, y=None):
        aux_features = X[:, self.num_tfidf:].toarray()
        self.scaler.fit(aux_features)
        return self
        
    def transform(self, X):
        X_copy = X.copy().tolil()
        aux_features = X[:, self.num_tfidf:].toarray()
        scaled_aux = self.scaler.transform(aux_features)
        X_copy[:, self.num_tfidf:] = scaled_aux
        return X_copy.tocsr()

class NaiveBayesShiftScaler:
    """
    Implements a strict non-negative shift specifically for Naive Bayes.
    1. Fits the training-fold minimum for each auxiliary feature.
    2. Shifts all values by subtracting the minimum (making train min >= 0).
    3. Applies a small numerical floor (max(x, 0)) to handle validation outliers.
    """
    def __init__(self):
        self.min_ = None
        self.num_tfidf = 10000
        
    def fit(self, X, y=None):
        aux_features = X[:, self.num_tfidf:].toarray()
        self.min_ = np.min(aux_features, axis=0)
        return self
        
    def transform(self, X):
        X_copy = X.copy().tolil()
        aux_features = X[:, self.num_tfidf:].toarray()
        
        # x_shifted = x - training_fold_min
        shifted_aux = aux_features - self.min_
        
        # Apply numerical floor for validation points falling below train min
        shifted_aux = np.maximum(shifted_aux, 0)
        
        X_copy[:, self.num_tfidf:] = shifted_aux
        return X_copy.tocsr()


def main():
    print("Loading raw preprocessed data for strict CV...")
    df_train = pd.read_csv("data/processed/preprocessed_train.csv")
    
    # We drop any NaNs in cleaned_text to avoid vectorizer issues
    df_train['cleaned_text'] = df_train['cleaned_text'].fillna("")
    df_train['statement'] = df_train['statement'].fillna("")
    
    y_train = df_train['binary_label'].values
    
    # Cross Validation manually to prevent data leakage
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Linear SVM": LinearSVC(C=1.0, random_state=42, max_iter=2000)
    }
    
    results = {name: {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []} for name in models.keys()}
    
    nb_train_mins = []
    nb_val_mins = []
    
    print("Starting Strict 5-Fold Cross Validation...")
    # Perform strict CV independently for each fold
    for fold, (train_idx, val_idx) in enumerate(skf.split(df_train, y_train)):
        print(f"Fold {fold+1}...")
        
        fold_df_train = df_train.iloc[train_idx].copy()
        fold_df_val = df_train.iloc[val_idx].copy()
        
        fold_y_train = y_train[train_idx]
        fold_y_val = y_train[val_idx]
        
        # 1. Instantiate a fresh FeatureUnion per fold
        union = build_feature_union()
        
        # 2. Fit and transform ONLY on fold_train
        X_fold_train_raw = union.fit_transform(fold_df_train)
        
        # 3. Transform fold_val
        X_fold_val_raw = union.transform(fold_df_val)
        
        # 4. Prepare LR/SVM scaled features (MinMaxScaler)
        standard_scaler = BranchScaler(MinMaxScaler())
        X_fold_train_std = standard_scaler.fit(X_fold_train_raw).transform(X_fold_train_raw)
        X_fold_val_std = standard_scaler.transform(X_fold_val_raw)
        
        # 5. Prepare Naive Bayes non-negative features
        nb_scaler = NaiveBayesShiftScaler()
        X_fold_train_nb = nb_scaler.fit(X_fold_train_raw).transform(X_fold_train_raw)
        X_fold_val_nb = nb_scaler.transform(X_fold_val_raw)
        
        # Assertions and reporting for NB branch min >= 0
        train_min = X_fold_train_nb.min()
        val_min = X_fold_val_nb.min()
        nb_train_mins.append(train_min)
        nb_val_mins.append(val_min)
        assert train_min >= 0, f"NB Train Min is {train_min} < 0"
        assert val_min >= 0, f"NB Val Min is {val_min} < 0"
        
        # 6. Train and evaluate models
        for name, model in models.items():
            if name == "Naive Bayes":
                X_train_model = X_fold_train_nb
                X_val_model = X_fold_val_nb
            else:
                X_train_model = X_fold_train_std
                X_val_model = X_fold_val_std
                
            model.fit(X_train_model, fold_y_train)
            preds = model.predict(X_val_model)
            
            if hasattr(model, "predict_proba"):
                probs = model.predict_proba(X_val_model)[:, 1]
            else:
                probs = model.decision_function(X_val_model)
                
            y_val_bin = (fold_y_val == 'REAL').astype(int)
                
            results[name]['accuracy'].append(accuracy_score(fold_y_val, preds))
            results[name]['precision'].append(precision_score(fold_y_val, preds, pos_label='REAL'))
            results[name]['recall'].append(recall_score(fold_y_val, preds, pos_label='REAL'))
            results[name]['f1'].append(f1_score(fold_y_val, preds, pos_label='REAL'))
            results[name]['roc_auc'].append(roc_auc_score(y_val_bin, probs))

    print("Cross validation complete. Compiling metrics...")
    
    final_metrics = []
    for name, metrics in results.items():
        row = {"model": name}
        for m, vals in metrics.items():
            row[f"{m}_mean"] = np.mean(vals)
            row[f"{m}_std"] = np.std(vals)
        final_metrics.append(row)
        
    df_metrics = pd.DataFrame(final_metrics)
    os.makedirs("reports/metrics", exist_ok=True)
    df_metrics.to_csv("reports/metrics/day5_baseline_model_comparison.csv", index=False)
    
    # 7. Final Pipeline fitting on FULL training set for Serialization
    print("Fitting final pipeline on full training set for artifacts...")
    final_union = build_feature_union()
    X_train_raw = final_union.fit_transform(df_train)
    
    # Final models use standard scaling
    final_scaler = BranchScaler(MinMaxScaler())
    X_train_final = final_scaler.fit(X_train_raw).transform(X_train_raw)
    
    final_nb_scaler = NaiveBayesShiftScaler()
    X_train_final_nb = final_nb_scaler.fit(X_train_raw).transform(X_train_raw)
    
    # Transform test and validation completely out-of-fold
    df_val = pd.read_csv("data/processed/preprocessed_validation.csv").fillna("")
    df_test = pd.read_csv("data/processed/preprocessed_test.csv").fillna("")
    
    X_val_raw = final_union.transform(df_val)
    X_test_raw = final_union.transform(df_test)
    
    X_val_final = final_scaler.transform(X_val_raw)
    X_test_final = final_scaler.transform(X_test_raw)
    
    sp.save_npz("data/processed/combined_train.npz", X_train_final)
    sp.save_npz("data/processed/combined_validation.npz", X_val_final)
    sp.save_npz("data/processed/combined_test.npz", X_test_final)
    
    # Output metadata
    metadata = {
        "tfidf_dimensions": 10000,
        "auxiliary_dimensions": 104,
        "combined_dimensions": X_train_final.shape[1],
        "feature_ordering": ["TF-IDF features (10000)", "Word2Vec features (100)", "Readability", "Sentiment", "Article Length", "Punctuation Ratio"],
        "cv_leakage_policy": "STRICT 5-FOLD: TF-IDF, Word2Vec, and Scalers independently fitted on each train-fold. No artifact reuse.",
        "nb_nonnegative_policy": "NaiveBayesShiftScaler subtracts train-fold minimum and applies np.maximum(0) floor."
    }
    with open("data/processed/feature_union_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Save dummy combined names
    with open("data/processed/combined_feature_names.json", "w") as f:
        json.dump(["f"] * X_train_final.shape[1], f)

    os.makedirs("models/baseline", exist_ok=True)
    for name, model in models.items():
        if name == "Naive Bayes":
            model.fit(X_train_final_nb, y_train)
        else:
            model.fit(X_train_final, y_train)
        joblib.dump(model, f"models/baseline/{name.lower().replace(' ', '_')}.joblib")
    
    print("Generating Visualizations...")
    os.makedirs("reports/figures/day5", exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    plt.bar(df_metrics['model'], df_metrics['f1_mean'], yerr=df_metrics['f1_std'], capsize=5, color=['#4C72B0', '#DD8452', '#55A868'])
    plt.title('Baseline Models - Mean F1 Score (5-Fold CV)')
    plt.ylim(0, 1)
    plt.ylabel('F1 Score')
    plt.savefig('reports/figures/day5/baseline_f1_comparison.png')
    plt.close()
    
    plt.figure(figsize=(10, 6))
    plt.bar(df_metrics['model'], df_metrics['roc_auc_mean'], yerr=df_metrics['roc_auc_std'], capsize=5, color=['#4C72B0', '#DD8452', '#55A868'])
    plt.title('Baseline Models - Mean ROC-AUC Score (5-Fold CV)')
    plt.ylim(0, 1)
    plt.ylabel('ROC-AUC Score')
    plt.savefig('reports/figures/day5/baseline_roc_auc_comparison.png')
    plt.close()

    md_table = "| model | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | roc_auc_std |\n"
    md_table += "|---|---|---|---|---|---|---|---|---|---|---|\n"
    for _, row in df_metrics.iterrows():
        md_table += f"| {row['model']} | {row['accuracy_mean']:.4f} | {row['accuracy_std']:.4f} | {row['precision_mean']:.4f} | {row['precision_std']:.4f} | {row['recall_mean']:.4f} | {row['recall_std']:.4f} | {row['f1_mean']:.4f} | {row['f1_std']:.4f} | {row['roc_auc_mean']:.4f} | {row['roc_auc_std']:.4f} |\n"

    print("Creating report...")
    report_md = f"""# Day 5 Baseline Models Report

## 1. Feature Integration
- **TF-IDF dimensions**: 10000
- **Auxiliary dimensions**: 104
- **Combined dimensions**: 10104
- **Sparse/dense representation**: Combined via strict `sklearn.pipeline.FeatureUnion` into a sparse CSR output.
- **Leakage Controls**: Strict CV. TF-IDF, Word2Vec, and Branch Scalers were fitted natively *inside* each CV train fold. Day 3/4 fitted artifacts were ignored during CV.

## 2. Baseline Models
- **Logistic Regression**: `max_iter=1000, random_state=42`
- **Naive Bayes**: `MultinomialNB`
- **Linear SVM**: `LinearSVC(C=1.0, random_state=42, max_iter=2000)`

## 3. Naive Bayes Transformation
- A robust, fold-fitted `NaiveBayesShiftScaler` shifted auxiliary variables via subtraction of the exact training fold minimum, followed by an `np.maximum(X, 0)` floor to handle cross-validation bounds. Negative Word2Vec variance was robustly preserved without blind clipping.
- Minimum NB Train values over 5 folds: {nb_train_mins}
- Minimum NB Val values over 5 folds: {nb_val_mins}

## 4. Final Metrics

{md_table}
"""
    with open("reports/day5_baseline_report.md", "w") as f:
        f.write(report_md)

    print("Day 5 execute completed successfully.")

if __name__ == "__main__":
    main()
