import os
import sys
import pandas as pd
import json

def fail(msg):
    print(f"[FAIL] {msg}")
    sys.exit(1)

def pass_check(msg):
    print(f"[PASS] {msg}")

def main():
    print("Running Day 2 Verification...\n")
    
    # Check imports
    try:
        from src.preprocessing.text_preprocessor import TextPreprocessor
        import spacy
        import nltk
        pass_check("Preprocessing module, SpaCy, NLTK imports successful.")
    except Exception as e:
        fail(f"Import failed: {e}")

    try:
        p = TextPreprocessor()
        pass_check("TextPreprocessor initialized, models loaded.")
    except Exception as e:
        fail(f"TextPreprocessor failed: {e}")

    # Check functionality
    if p.process("https://example.com test") == "test":
        pass_check("URL removal works")
    else:
        fail("URL removal failed")
        
    if p.process("hello, world!") == "hello world":
        pass_check("Noise/punctuation cleaning works")
    else:
        fail(f"Noise/punctuation cleaning failed: {p.process('hello, world!')}")
        
    if p.process("TESTING") == "testing":
        pass_check("Lowercase policy works")
    else:
        fail("Lowercase policy failed")

    if p.process("") == "" and p.process(None) == "":
        pass_check("Empty input handled")
    else:
        fail("Empty input handling failed")

    # Check files exist
    files_to_check = [
        "data/processed/preprocessed_train.csv",
        "data/processed/preprocessed_validation.csv",
        "data/processed/preprocessed_test.csv",
        "data/processed/preprocessing_metadata.json",
        "reports/day2_preprocessing_report.md",
        "reports/figures/day2/text_length_comparison.png"
    ]
    for f in files_to_check:
        if not os.path.exists(f):
            fail(f"Missing file: {f}")
    pass_check("Outputs exist")
    
    # Check integrity
    for split in ["train", "validation", "test"]:
        orig = pd.read_csv(f"data/processed/{split}.csv")
        prep = pd.read_csv(f"data/processed/preprocessed_{split}.csv")
        
        if len(orig) != len(prep):
            fail(f"Row counts not preserved in {split}")
            
        if not (orig['binary_label'] == prep['binary_label']).all():
            fail(f"Labels not preserved in {split}")
            
        if not (orig['statement'] == prep['statement']).all():
            fail(f"Raw text not preserved in {split}")
            
        if 'cleaned_text' not in prep.columns:
            fail(f"cleaned_text column missing in {split}")
    pass_check("Data integrity verified (row counts, labels, raw text preserved, cleaned_text exists)")
            
    print("\n[SUCCESS] Day 2 Verification completed successfully.")

if __name__ == "__main__":
    main()
