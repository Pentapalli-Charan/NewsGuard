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
    print("Running Day 1 Verification...\n")

    if not os.path.exists("data/raw/train.tsv"):
        fail("Raw dataset train.tsv does not exist.")
    pass_check("Real dataset exists")

    if not os.path.exists("data/processed/dataset_metadata.json"):
        fail("Metadata file does not exist.")
    
    with open("data/processed/dataset_metadata.json", "r") as f:
        metadata = json.load(f)
    
    if metadata["total_usable_rows"] < 5000:
        fail(f"Usable rows < 5000. Found {metadata['total_usable_rows']}")
    pass_check(f">= 5,000 usable samples ({metadata['total_usable_rows']} rows)")

    train_path = "data/processed/train.csv"
    val_path = "data/processed/validation.csv"
    test_path = "data/processed/test.csv"

    if not all(os.path.exists(p) for p in [train_path, val_path, test_path]):
        fail("Train, validation, or test file missing.")
    pass_check("Train, validation, and test exist")

    train = pd.read_csv(train_path)
    val = pd.read_csv(val_path)
    test = pd.read_csv(test_path)

    valid_labels = {'FAKE', 'REAL'}
    for df, name in [(train, 'train'), (val, 'val'), (test, 'test')]:
        if not set(df['binary_label'].unique()).issubset(valid_labels):
            fail(f"Invalid labels found in {name}.")
    pass_check("Labels valid and binary mapping valid")

    total = len(train) + len(val) + len(test)
    if not (0.78 <= len(train)/total <= 0.82):
        fail("Train split ratio incorrect.")
    if not (0.08 <= len(val)/total <= 0.12):
        fail("Validation split ratio incorrect.")
    if not (0.08 <= len(test)/total <= 0.12):
        fail("Test split ratio incorrect.")
    pass_check("Split ratio correct (approx 80/10/10)")

    train_ids = set(train['statement'].tolist())
    val_ids = set(val['statement'].tolist())
    test_ids = set(test['statement'].tolist())

    if len(train_ids.intersection(val_ids)) > 0 or len(train_ids.intersection(test_ids)) > 0 or len(val_ids.intersection(test_ids)) > 0:
        fail("Split overlap detected.")
    pass_check("No split overlap")

    train_fake_pct = (train['binary_label'] == 'FAKE').mean()
    val_fake_pct = (val['binary_label'] == 'FAKE').mean()
    if abs(train_fake_pct - val_fake_pct) > 0.05:
        fail(f"Stratification failed. Train FAKE {train_fake_pct:.2f}, Val FAKE {val_fake_pct:.2f}")
    pass_check("Stratification verified")

    pass_check("Metadata exists")

    eda_files = [
        "overall_class_distribution.png",
        "split_class_distribution.png",
        "missing_values.png",
        "text_length_distribution.png",
        "text_length_by_class.png"
    ]
    for ef in eda_files:
        if not os.path.exists(f"reports/figures/day1/{ef}"):
            fail(f"EDA figure {ef} missing.")
    pass_check("EDA figures exist")

    if not os.path.exists("reports/day1_dataset_report.md"):
        fail("Day 1 report missing.")
    pass_check("Day 1 report exists")

    print("\n[SUCCESS] Day 1 Verification completed successfully.")

if __name__ == "__main__":
    main()
