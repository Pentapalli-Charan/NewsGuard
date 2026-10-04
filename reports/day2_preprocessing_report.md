# Day 2 Preprocessing Report

## 1. Objective
Implement a deterministic NLP preprocessing pipeline to prepare text for TF-IDF and Word2Vec feature extraction.

## 2. Input Datasets
- `data/processed/train.csv`
- `data/processed/validation.csv`
- `data/processed/test.csv`

## 3. Text Column Used
- Original column: `statement`
- Output column: `cleaned_text`

## 4. Preprocessing Pipeline Order
1. Input validation
2. Lowercase normalization
3. URL removal
4. Noise/punctuation handling
5. SpaCy tokenization
6. Stopword removal
7. SpaCy lemmatization
8. Whitespace normalization

## 5. URL Removal Policy
URLs starting with `http://`, `https://`, or `www.` are completely removed via regex.

## 6. Noise/Punctuation Policy
All non-alphanumeric characters are removed (replaced by spaces). Numbers and alphabet letters are retained. Apostrophes are removed, treating subwords separately.

## 7. Tokenization Method
SpaCy tokenizer (`en_core_web_sm`).

## 8. Stopword Source
NLTK English stopwords.

## 9. Lemmatization Method
SpaCy lemmatizer (`en_core_web_sm`).

## 10. Case Normalization
All text is deterministically lowercased.

## 11. Empty-Input Handling
Empty or whitespace-only strings, or None values, are returned as empty strings `""`.

## 12. Processing Stats
- **Records Processed**: 12765
- **Empty Cleaned Outputs**: 0
- **Average Raw Length**: 107.3
- **Average Cleaned Length**: 73.8
- **Min Cleaned Length**: 6
- **Max Cleaned Length**: 2606
- **Records with URLs (Raw)**: 0 (0.00%)
- **Records with URLs (Cleaned)**: 0

## 13. Example (REAL class)
**Raw**: (George) LeMieux never requested a single earmark and pushed to ban them all.
**Cleaned**: george lemieux never request single earmark push ban

## 14. Known Limitations
- Removing punctuation completely also removes apostrophes, separating words like "don't" into "don" and "t", which lemmatizer then processes.
- The pipeline assumes English language solely.

## 15. Reproducibility
The output is completely deterministic. No fitting or learning is performed on any split.
