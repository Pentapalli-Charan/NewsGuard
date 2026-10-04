# Day 4 Auxiliary Feature Engineering Report

## 1. Objective
Implement Word2Vec embeddings and NLP-derived auxiliary features (Readability, Sentiment, Article Length, Punctuation Ratio) without utilizing test/validation data for fitting.

## 2. Word2Vec Configuration
- `vector_size`: 100
- `window`: 5
- `min_count`: 2
- `workers`: 1
- `sg`: 1
- `epochs`: 10
- `seed`: 42

## 3. Word2Vec Training Corpus
- Trained strictly on `cleaned_text` from `data/processed/preprocessed_train.csv`.
- Validation and Test texts were NOT used to update the model.

## 4. Word2Vec Vocabulary Size
- **Total Vocabulary Size**: 5301 tokens

## 5. Sentence-Vector Methodology
- Represents a document as the mean of the Word2Vec token vectors found in its `cleaned_text`.
- If a document has no known tokens, a zero-vector of size 100 is returned.

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
- Word2Vec: 100
- NLP Auxiliary: 4
- Total Features: 104

## 11. Feature Statistics
- **Word2Vec Mean/Std**: -0.0152 / 0.1865
- **Readability**: Mean 9.55, Std 3.71
- **Sentiment**: Mean 0.05, Std 0.21
- **Article Length**: Mean 11.35, Std 7.55
- **Punctuation Ratio**: Mean 0.0270, Std 0.0176

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
