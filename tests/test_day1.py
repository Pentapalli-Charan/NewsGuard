import os
import pandas as pd
import json
from sklearn.model_selection import train_test_split

def test_raw_dataset_exists():
    assert os.path.exists("data/raw/liar_dataset.zip")
    assert os.path.exists("data/raw/train.tsv")

def test_splits_exist():
    assert os.path.exists("data/processed/train.csv")
    assert os.path.exists("data/processed/validation.csv")
    assert os.path.exists("data/processed/test.csv")

def test_metadata_exists():
    assert os.path.exists("data/processed/dataset_metadata.json")

def test_dataset_size_and_ratios():
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/validation.csv")
    test = pd.read_csv("data/processed/test.csv")
    
    total = len(train) + len(val) + len(test)
    assert total >= 5000, f"Dataset has less than 5000 records: {total}"
    
    train_ratio = len(train) / total
    val_ratio = len(val) / total
    test_ratio = len(test) / total
    
    assert 0.78 <= train_ratio <= 0.82
    assert 0.08 <= val_ratio <= 0.12
    assert 0.08 <= test_ratio <= 0.12

def test_no_overlap():
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/validation.csv")
    test = pd.read_csv("data/processed/test.csv")
    
    train_ids = set(train['statement'].tolist())
    val_ids = set(val['statement'].tolist())
    test_ids = set(test['statement'].tolist())
    
    assert len(train_ids.intersection(val_ids)) == 0
    assert len(train_ids.intersection(test_ids)) == 0
    assert len(val_ids.intersection(test_ids)) == 0

def test_labels_exist():
    train = pd.read_csv("data/processed/train.csv")
    assert set(train['binary_label'].unique()) == {'FAKE', 'REAL'}
    val = pd.read_csv("data/processed/validation.csv")
    assert set(val['binary_label'].unique()) == {'FAKE', 'REAL'}
    test = pd.read_csv("data/processed/test.csv")
    assert set(test['binary_label'].unique()) == {'FAKE', 'REAL'}

def test_stratification():
    train = pd.read_csv("data/processed/train.csv")
    val = pd.read_csv("data/processed/validation.csv")
    test = pd.read_csv("data/processed/test.csv")

    train_fake_pct = (train['binary_label'] == 'FAKE').mean()
    val_fake_pct = (val['binary_label'] == 'FAKE').mean()
    test_fake_pct = (test['binary_label'] == 'FAKE').mean()

    assert abs(train_fake_pct - val_fake_pct) < 0.05
    assert abs(train_fake_pct - test_fake_pct) < 0.05

def test_split_reproducibility():
    with open("data/processed/dataset_metadata.json", "r") as f:
        metadata = json.load(f)
    random_seed = metadata['random_seed']
    
    columns = ["id", "label", "statement", "subject", "speaker", "job_title", "state_info", "party_affiliation", "barely_true_counts", "false_counts", "half_true_counts", "mostly_true_counts", "pants_on_fire_counts", "context"]
    dfs = []
    for split in ["train.tsv", "valid.tsv", "test.tsv"]:
        dfs.append(pd.read_csv(os.path.join("data/raw", split), sep='\t', header=None, names=columns))
    df = pd.concat(dfs, ignore_index=True)
    
    label_mapping = metadata['label_mapping']
    df['binary_label'] = df['label'].map(label_mapping)
    df_clean = df.dropna(subset=['statement', 'binary_label']).copy()
    df_clean = df_clean.drop_duplicates(subset=['statement'])
    
    train_val, test_df = train_test_split(df_clean, test_size=0.10, stratify=df_clean['binary_label'], random_state=random_seed)
    train_df, val_df = train_test_split(train_val, test_size=1/9, stratify=train_val['binary_label'], random_state=random_seed)
    
    # Load actual splits
    actual_train = pd.read_csv("data/processed/train.csv")
    
    assert set(train_df['statement']) == set(actual_train['statement'])
