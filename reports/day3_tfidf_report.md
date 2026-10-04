# Day 3 TF-IDF Feature Engineering Report

## 1. Objective
Implement TF-IDF feature engineering using unigrams and bigrams based solely on the training dataset.

## 2. Configuration
- `ngram_range`: (1, 2)
- `max_features`: 10000 (Limits the vocabulary to the top features to manage sparse matrix dimensionality).
- `min_df`: 2
- `max_df`: 0.95

## 3. Vocabulary Size
- **Total Features**: 10000
- **Unigrams**: 4186 (41.9%)
- **Bigrams**: 5814 (58.1%)

## 4. Matrix Dimensions
- **Train**: (10211, 10000)
- **Validation**: (1277, 10000)
- **Test**: (1277, 10000)

## 5. Sparsity
- **Train Matrix Sparsity**: 99.87% zero elements.

## 6. TRAINING-DERIVED INFORMATION: Top Discriminative Terms
*Note: This is exploratory feature analysis calculated using class-specific mean TF-IDF scores on the TRAINING data only. This is not model-based feature importance.*

**Top 5 REAL-associated terms:**
percent, 000, since, year, country

**Top 5 FAKE-associated terms:**
say, obama, president, obamacare, care

## 7. Leakage Prevention
**TF-IDF fitted on training split only.** Validation and test sets were transformed using the pre-fitted vectorizer, ensuring no vocabulary leakage.

## 8. Reproducibility
The output is completely deterministic for a given input dataset.
