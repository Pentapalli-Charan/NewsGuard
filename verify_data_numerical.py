import pandas as pd
import json

train = pd.read_csv("data/processed/train.csv")
val = pd.read_csv("data/processed/validation.csv")
test = pd.read_csv("data/processed/test.csv")

t_real = len(train[train['binary_label'] == 'REAL'])
t_fake = len(train[train['binary_label'] == 'FAKE'])
v_real = len(val[val['binary_label'] == 'REAL'])
v_fake = len(val[val['binary_label'] == 'FAKE'])
te_real = len(test[test['binary_label'] == 'REAL'])
te_fake = len(test[test['binary_label'] == 'FAKE'])

total_real = t_real + v_real + te_real
total_fake = t_fake + v_fake + te_fake

print(f"train_rows={len(train)}")
print(f"val_rows={len(val)}")
print(f"test_rows={len(test)}")
print(f"total_rows={len(train) + len(val) + len(test)}")
print(f"total_REAL={total_real}")
print(f"total_FAKE={total_fake}")
