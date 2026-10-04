import os
import pandas as pd
import json
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from src.preprocessing.text_preprocessor import TextPreprocessor

def main():
    print("Loading datasets...")
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/validation.csv")
    test = pd.read_csv("data/processed/test.csv")
    
    print("Initializing TextPreprocessor...")
    preprocessor = TextPreprocessor()
    
    def process_df(df, name):
        print(f"Processing {name} ({len(df)} rows)...")
        # Apply deterministic preprocessing
        df['cleaned_text'] = df['statement'].apply(preprocessor.process)
        return df
        
    print("Applying preprocessing (batch)...")
    train_clean = process_df(train.copy(), "Train")
    val_clean = process_df(val.copy(), "Validation")
    test_clean = process_df(test.copy(), "Test")
    
    # Save datasets
    train_clean.to_csv("data/processed/preprocessed_train.csv", index=False)
    val_clean.to_csv("data/processed/preprocessed_validation.csv", index=False)
    test_clean.to_csv("data/processed/preprocessed_test.csv", index=False)
    
    # Quality metrics
    all_data = pd.concat([train_clean, val_clean, test_clean])
    total_processed = len(all_data)
    empty_outputs = len(all_data[all_data['cleaned_text'] == ""])
    
    all_data['raw_len'] = all_data['statement'].apply(lambda x: len(str(x)))
    all_data['clean_len'] = all_data['cleaned_text'].apply(lambda x: len(str(x)))
    
    avg_raw_len = all_data['raw_len'].mean()
    avg_clean_len = all_data['clean_len'].mean()
    min_clean = all_data['clean_len'].min()
    max_clean = all_data['clean_len'].max()
    
    # Check URLs
    import re
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    records_with_url = all_data['statement'].apply(lambda x: bool(url_pattern.search(str(x)))).sum()
    pct_url = (records_with_url / total_processed) * 100
    
    # Check that URLs were removed
    records_with_url_cleaned = all_data['cleaned_text'].apply(lambda x: bool(url_pattern.search(str(x)))).sum()
    
    metadata = {
        "preprocessing_version": "1.0",
        "python_version": "3.12.6",
        "spacy_version": spacy.__version__,
        "spacy_model": "en_core_web_sm",
        "nltk_version": nltk.__version__,
        "preprocessing_steps": [
            "Input validation",
            "Lowercase normalization",
            "URL removal",
            "Noise/punctuation handling",
            "SpaCy tokenization",
            "Stopword removal",
            "SpaCy lemmatization",
            "Whitespace normalization"
        ],
        "lowercase_policy": "All text is lowercased deterministically.",
        "url_policy": "Removed using regex (http://, https://, www.).",
        "punctuation_noise_policy": "Removed all non-alphanumeric characters except space.",
        "stopword_source": "NLTK English stopwords",
        "random_seed": 42,
        "records_processed": total_processed,
        "empty_outputs": empty_outputs,
        "timestamp": datetime.now().isoformat()
    }
    
    with open("data/processed/preprocessing_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)
        
    # Generate EDA figure
    os.makedirs("reports/figures/day2", exist_ok=True)
    sns.set_theme(style="whitegrid")
    
    plt.figure(figsize=(10, 6))
    sns.kdeplot(all_data['raw_len'], label='Raw Text Length', fill=True, color='blue')
    sns.kdeplot(all_data['clean_len'], label='Cleaned Text Length', fill=True, color='green')
    plt.title('Text Length Distribution: Raw vs Cleaned')
    plt.xlabel('Number of Characters')
    plt.xlim(0, 1000)
    plt.legend()
    plt.savefig("reports/figures/day2/text_length_comparison.png")
    plt.close()
    
    # Report markdown
    example_row = all_data[all_data['binary_label'] == 'REAL'].iloc[0]
    
    report_md = f"""# Day 2 Preprocessing Report

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
- **Records Processed**: {total_processed}
- **Empty Cleaned Outputs**: {empty_outputs}
- **Average Raw Length**: {avg_raw_len:.1f}
- **Average Cleaned Length**: {avg_clean_len:.1f}
- **Min Cleaned Length**: {min_clean}
- **Max Cleaned Length**: {max_clean}
- **Records with URLs (Raw)**: {records_with_url} ({pct_url:.2f}%)
- **Records with URLs (Cleaned)**: {records_with_url_cleaned}

## 13. Example (REAL class)
**Raw**: {example_row['statement']}
**Cleaned**: {example_row['cleaned_text']}

## 14. Known Limitations
- Removing punctuation completely also removes apostrophes, separating words like "don't" into "don" and "t", which lemmatizer then processes.
- The pipeline assumes English language solely.

## 15. Reproducibility
The output is completely deterministic. No fitting or learning is performed on any split.
"""
    with open("reports/day2_preprocessing_report.md", "w") as f:
        f.write(report_md)
        
    print("Day 2 execute completed successfully.")

if __name__ == "__main__":
    import spacy
    import nltk
    main()
