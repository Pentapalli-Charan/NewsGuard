# Day 5 Baseline Models Report

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
- Minimum NB Train values over 5 folds: [np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)]
- Minimum NB Val values over 5 folds: [np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0), np.float64(0.0)]

## 4. Final Metrics

| model | accuracy_mean | accuracy_std | precision_mean | precision_std | recall_mean | recall_std | f1_mean | f1_std | roc_auc_mean | roc_auc_std |
|---|---|---|---|---|---|---|---|---|---|---|
| Logistic Regression | 0.6139 | 0.0061 | 0.6337 | 0.0045 | 0.7302 | 0.0089 | 0.6785 | 0.0058 | 0.6502 | 0.0103 |
| Naive Bayes | 0.6070 | 0.0065 | 0.6066 | 0.0041 | 0.8411 | 0.0118 | 0.7048 | 0.0058 | 0.6472 | 0.0104 |
| Linear SVM | 0.5867 | 0.0103 | 0.6221 | 0.0076 | 0.6603 | 0.0141 | 0.6406 | 0.0105 | 0.6142 | 0.0094 |

