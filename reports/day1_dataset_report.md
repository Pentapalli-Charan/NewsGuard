# Day 1 Dataset Report

## Dataset Acquisition
- **Source**: LIAR Dataset (https://www.cs.ucsb.edu/~william/data/liar_dataset.zip)
- **Total Raw Rows**: 12791
- **Usable Rows (after deduplication and removing missing text/labels)**: 12765

## Label Mapping
Original LIAR labels were mapped to binary classes (FAKE, REAL) as follows:
- `pants-fire`: FAKE
- `false`: FAKE
- `barely-true`: FAKE
- `half-true`: REAL
- `mostly-true`: REAL
- `true`: REAL

### Binary Class Distribution
- FAKE: 5642
- REAL: 7123

## Stratified Split
- **Train**: 10211 rows (80.0%)
- **Validation**: 1277 rows (10.0%)
- **Test**: 1277 rows (10.0%)
- **Random Seed**: 42

No overlap exists between splits.

## Exploratory Data Analysis (EDA)
EDA figures have been generated and saved to `reports/figures/day1/`:
1. `overall_class_distribution.png`: Overall binary class balance.
2. `split_class_distribution.png`: Class balance across train/validation/test sets.
3. `missing_values.png`: Analysis of missing fields in the dataset.
4. `text_length_distribution.png`: Overall distribution of text lengths (word counts).
5. `text_length_by_class.png`: Text length distribution categorized by binary label.
