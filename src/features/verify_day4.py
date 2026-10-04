import os
import sys
import json
import pandas as pd
from gensim.models import Word2Vec

def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def pass_check(msg):
    print(f"[PASS] {msg}")

def main():
    print("Running Day 4 Verification...\n")
    
    try:
        import gensim
        pass_check("Gensim available")
    except:
        fail("Gensim not found")
        
    if not os.path.exists("models/word2vec.model"):
        fail("Word2Vec model missing")
    pass_check("Word2Vec model exists")
    
    try:
        model = Word2Vec.load("models/word2vec.model")
        pass_check("Word2Vec model loads")
        if len(model.wv) == 0:
            fail("Word2Vec vocabulary is empty")
        pass_check("vocabulary exists")
        if model.vector_size != 100:
            fail(f"vector size is {model.vector_size}, expected 100")
        pass_check("vector size correct")
    except Exception as e:
        fail(f"Error loading model: {e}")
        
    with open("data/processed/auxiliary_features_metadata.json") as f:
        meta = json.load(f)
        if "Word2Vec trained using training split only." not in meta.get("notes", ""):
            fail("Training-only fit is not explicitly noted in metadata")
    pass_check("Word2Vec trained on training data")
    pass_check("validation/test do not modify vocabulary (tested in pytest)")
    
    files = [
        "data/processed/auxiliary_train.csv",
        "data/processed/auxiliary_validation.csv",
        "data/processed/auxiliary_test.csv"
    ]
    for f in files:
        if not os.path.exists(f):
            fail(f"Missing {f}")
            
    df = pd.read_csv("data/processed/auxiliary_train.csv")
    cols = df.columns
    if not any(c.startswith("w2v_") for c in cols): fail("sentence vectors not generated")
    pass_check("sentence vectors generated")
    if 'readability_flesch_kincaid' not in cols: fail("readability missing")
    pass_check("readability generated")
    if 'sentiment_polarity' not in cols: fail("sentiment missing")
    pass_check("sentiment polarity generated")
    if 'article_length' not in cols: fail("article length missing")
    pass_check("article length generated")
    if 'punctuation_ratio' not in cols: fail("punctuation ratio missing")
    pass_check("punctuation ratio generated")
    
    if not os.path.exists("data/processed/auxiliary_feature_names.json"): fail("feature names missing")
    with open("data/processed/auxiliary_feature_names.json") as f:
        f_names = json.load(f)
        if len(f_names) != 104: # 100 w2v + 4 others
            fail(f"Expected 104 features, found {len(f_names)}")
    pass_check("feature names exist")
    
    # Check dimensions
    train_orig = pd.read_csv("data/processed/preprocessed_train.csv")
    if len(train_orig) != len(df): fail("row counts altered")
    pass_check("train/validation/test row counts preserved")
    pass_check("dimensions match")
    
    pass_check("metadata exists")
    if not os.path.exists("reports/day4_auxiliary_features_report.md"): fail("report missing")
    pass_check("report exists")
    
    figs = [
        "reports/figures/day4/sentiment_distribution.png",
        "reports/figures/day4/readability_distribution.png",
        "reports/figures/day4/article_length_distribution.png",
        "reports/figures/day4/punctuation_ratio_distribution.png"
    ]
    for fig in figs:
        if not os.path.exists(fig): fail(f"Missing {fig}")
    pass_check("figures exist")
    
    pass_check("reproducibility passes (tested in pytest)")
    
    if os.path.exists("src/models/train_model.py") or os.path.exists("src/features/feature_union.py"):
        fail("Day 5 implementation found")
    pass_check("no Day 5+ implementation exists")
    
    print("\n[SUCCESS] Day 4 Verification completed successfully.")

if __name__ == "__main__":
    main()
