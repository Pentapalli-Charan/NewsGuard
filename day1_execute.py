import os
import urllib.request
import zipfile
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
import json

# 1. Download and Extract LIAR Dataset
url = "https://www.cs.ucsb.edu/~william/data/liar_dataset.zip"
zip_path = "data/raw/liar_dataset.zip"
extract_dir = "data/raw/"

print("Downloading LIAR dataset...")
urllib.request.urlretrieve(url, zip_path)
print("Extracting...")
with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_dir)

# 2. Dataset Inspection
columns = [
    "id", "label", "statement", "subject", "speaker", "job_title", 
    "state_info", "party_affiliation", "barely_true_counts", 
    "false_counts", "half_true_counts", "mostly_true_counts", 
    "pants_on_fire_counts", "context"
]

dfs = []
for split in ["train.tsv", "valid.tsv", "test.tsv"]:
    df_part = pd.read_csv(os.path.join(extract_dir, split), sep='\t', header=None, names=columns)
    dfs.append(df_part)

df = pd.concat(dfs, ignore_index=True)
initial_row_count = len(df)
print(f"Total rows in raw data: {initial_row_count}")

# 3. Label mapping and cleaning
label_mapping = {
    'pants-fire': 'FAKE',
    'false': 'FAKE',
    'barely-true': 'FAKE',
    'half-true': 'REAL',
    'mostly-true': 'REAL',
    'true': 'REAL'
}

df['binary_label'] = df['label'].map(label_mapping)
df_clean = df.dropna(subset=['statement', 'binary_label']).copy()
df_clean = df_clean.drop_duplicates(subset=['statement'])
usable_rows = len(df_clean)

print(f"Usable rows after cleaning: {usable_rows}")
print("Original Label Distribution:")
print(df_clean['label'].value_counts())
print("Binary Label Distribution:")
print(df_clean['binary_label'].value_counts())

# 4. Stratified Split (80/10/10)
random_seed = 42
train_val, test_df = train_test_split(
    df_clean, test_size=0.10, stratify=df_clean['binary_label'], random_state=random_seed
)
train_df, val_df = train_test_split(
    train_val, test_size=1/9, stratify=train_val['binary_label'], random_state=random_seed
) # 10% of total is 1/9 of 90%

train_df.to_csv("data/processed/train.csv", index=False)
val_df.to_csv("data/processed/validation.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

# Metadata
metadata = {
    "total_raw_rows": initial_row_count,
    "total_usable_rows": usable_rows,
    "train_rows": len(train_df),
    "validation_rows": len(val_df),
    "test_rows": len(test_df),
    "class_counts": df_clean['binary_label'].value_counts().to_dict(),
    "random_seed": random_seed,
    "label_mapping": label_mapping,
    "dataset_source": url
}
with open("data/processed/dataset_metadata.json", "w") as f:
    json.dump(metadata, f, indent=4)

# 5. EDA
os.makedirs("reports/figures/day1", exist_ok=True)
sns.set_theme(style="whitegrid")

# Overall class distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='binary_label', data=df_clean, palette='Set2')
plt.title('Overall Binary Class Distribution')
plt.savefig("reports/figures/day1/overall_class_distribution.png")
plt.close()

# Train/Val/Test distribution
splits_info = pd.DataFrame({
    'Split': ['Train'] * len(train_df) + ['Validation'] * len(val_df) + ['Test'] * len(test_df),
    'Label': pd.concat([train_df['binary_label'], val_df['binary_label'], test_df['binary_label']])
})
plt.figure(figsize=(8, 5))
sns.countplot(x='Split', hue='Label', data=splits_info, palette='Set2')
plt.title('Class Distribution Across Splits')
plt.savefig("reports/figures/day1/split_class_distribution.png")
plt.close()

# Missing values
missing_counts = df_clean.isnull().sum()
plt.figure(figsize=(10, 6))
missing_counts[missing_counts > 0].plot(kind='bar', color='coral')
plt.title('Missing Values per Column (Usable Data)')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig("reports/figures/day1/missing_values.png")
plt.close()

# Text length distribution
df_clean['text_length'] = df_clean['statement'].apply(lambda x: len(str(x).split()))
plt.figure(figsize=(8, 5))
sns.histplot(df_clean['text_length'], bins=50, kde=True, color='skyblue')
plt.title('Text Length Distribution (Words)')
plt.xlabel('Number of Words')
plt.savefig("reports/figures/day1/text_length_distribution.png")
plt.close()

# Text length by class
plt.figure(figsize=(8, 5))
sns.boxplot(x='binary_label', y='text_length', data=df_clean, palette='Set2')
plt.title('Text Length by Class')
plt.xlabel('Binary Label')
plt.ylabel('Number of Words')
plt.savefig("reports/figures/day1/text_length_by_class.png")
plt.close()

# 6. Generate Report
report_content = f"""# Day 1 Dataset Report

## Dataset Acquisition
- **Source**: LIAR Dataset ({url})
- **Total Raw Rows**: {initial_row_count}
- **Usable Rows (after deduplication and removing missing text/labels)**: {usable_rows}

## Label Mapping
Original LIAR labels were mapped to binary classes (FAKE, REAL) as follows:
- `pants-fire`: FAKE
- `false`: FAKE
- `barely-true`: FAKE
- `half-true`: REAL
- `mostly-true`: REAL
- `true`: REAL

### Binary Class Distribution
- FAKE: {metadata['class_counts'].get('FAKE', 0)}
- REAL: {metadata['class_counts'].get('REAL', 0)}

## Stratified Split
- **Train**: {len(train_df)} rows ({len(train_df)/usable_rows*100:.1f}%)
- **Validation**: {len(val_df)} rows ({len(val_df)/usable_rows*100:.1f}%)
- **Test**: {len(test_df)} rows ({len(test_df)/usable_rows*100:.1f}%)
- **Random Seed**: {random_seed}

No overlap exists between splits.

## Exploratory Data Analysis (EDA)
EDA figures have been generated and saved to `reports/figures/day1/`:
1. `overall_class_distribution.png`: Overall binary class balance.
2. `split_class_distribution.png`: Class balance across train/validation/test sets.
3. `missing_values.png`: Analysis of missing fields in the dataset.
4. `text_length_distribution.png`: Overall distribution of text lengths (word counts).
5. `text_length_by_class.png`: Text length distribution categorized by binary label.
"""
with open("reports/day1_dataset_report.md", "w") as f:
    f.write(report_content)

print("Day 1 execution complete.")
