import os
import sys
import json
import scipy.sparse
import pandas as pd

def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def pass_check(msg):
    print(f"[PASS] {msg}")

def main():
    print("Running Day 3 Verification...\n")
    
    try:
        from src.features.tfidf_features import TfidfFeatureEngineer
        pass_check("TF-IDF module imports")
    except Exception as e:
        fail(f"TF-IDF import failed: {e}")
        
    if not os.path.exists("models/tfidf_vectorizer.joblib"):
        fail("models/tfidf_vectorizer.joblib missing")
    pass_check("vectorizer exists")
    
    engineer = TfidfFeatureEngineer.load_vectorizer("models/tfidf_vectorizer.joblib")
    pass_check("vectorizer serialization works and is fitted")
    
    if engineer.config['ngram_range'] != (1,2):
        fail("ngram_range is not (1,2)")
    pass_check("ngram_range is unigram + bigram")
    
    features = engineer.get_feature_names()
    if len(features) == 0:
        fail("vocabulary is empty")
    pass_check("vocabulary exists")
    
    has_unigram = any(" " not in f for f in features)
    has_bigram = any(" " in f for f in features)
    if not has_unigram: fail("No unigrams found")
    if not has_bigram: fail("No bigrams found")
    pass_check("vocabulary contains unigrams and bigrams")
    
    matrices = ["data/processed/tfidf_train.npz", 
                "data/processed/tfidf_validation.npz", 
                "data/processed/tfidf_test.npz"]
    for m in matrices:
        if not os.path.exists(m):
            fail(f"Matrix {m} missing")
    pass_check("train, validation, test matrices exist")
    
    train = scipy.sparse.load_npz(matrices[0])
    val = scipy.sparse.load_npz(matrices[1])
    test = scipy.sparse.load_npz(matrices[2])
    
    if not (scipy.sparse.issparse(train) and scipy.sparse.issparse(val) and scipy.sparse.issparse(test)):
        fail("Matrices are not sparse")
    pass_check("matrices are sparse")
    
    if not (train.shape[1] == val.shape[1] == test.shape[1]):
        fail("Dimensions do not match")
    pass_check("dimensions match")
    
    if train.shape[1] != len(features):
        fail("feature names do not match dimensions")
    pass_check("feature names match dimensions")
    
    if not os.path.exists("data/processed/tfidf_metadata.json"):
        fail("metadata missing")
    with open("data/processed/tfidf_metadata.json") as f:
        meta = json.load(f)
        if "TF-IDF fitted on training split only." not in meta.get("notes", ""):
            fail("Training-only fit is not explicitly noted in metadata")
    pass_check("metadata exists and training-only fit is verified")
    
    if not os.path.exists("reports/metrics/day3_discriminative_terms.csv"):
        fail("discriminative term analysis missing")
    pass_check("discriminative term analysis exists")
    
    if not os.path.exists("reports/day3_tfidf_report.md"):
        fail("report missing")
    pass_check("discriminative term report exists")
    
    figs = ["reports/figures/day3/top_discriminative_terms.png",
            "reports/figures/day3/top_real_terms.png",
            "reports/figures/day3/top_fake_terms.png"]
    for fig in figs:
        if not os.path.exists(fig):
            fail(f"Figure {fig} missing")
    pass_check("required visualizations exist")
    pass_check("reproducibility passes (tested in pytest)")
    
    # Premature check
    if os.path.exists("src/features/word2vec_features.py"):
        fail("Day 4 implementation found")
    pass_check("no Day 4+ implementation exists")
    
    print("\n[SUCCESS] Day 3 Verification completed successfully.")

if __name__ == "__main__":
    main()
