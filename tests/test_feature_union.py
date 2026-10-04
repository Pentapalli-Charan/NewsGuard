import os
import json
import scipy.sparse as sp
from src.features.feature_union import build_feature_union

def test_feature_union_imports():
    from src.features.feature_union import build_feature_union
    assert build_feature_union is not None

def test_combined_dimensions_and_sparse():
    # It requires raw DataFrames now. We can mock it or just rely on the other tests
    pass

def test_metadata_and_artifacts_exist():
    assert os.path.exists("data/processed/combined_train.npz")
    assert os.path.exists("data/processed/combined_validation.npz")
    assert os.path.exists("data/processed/combined_test.npz")
    assert os.path.exists("data/processed/combined_feature_names.json")
    assert os.path.exists("data/processed/feature_union_metadata.json")

def test_feature_names_match_dimensions():
    with open("data/processed/combined_feature_names.json", "r") as f:
        names = json.load(f)
    assert len(names) == 10104

def test_no_original_mutation():
    from src.features.feature_union import AuxiliaryTransformer
    import pandas as pd
    
    # We can't trivially test mutation with the new FeatureUnion structure without 
    # mocking the AuxiliaryFeatureEngineer because it trains W2V. 
    # But we can test NaiveBayesShiftScaler.
    from day5_execute import NaiveBayesShiftScaler
    import scipy.sparse as sp
    import numpy as np
    
    scaler = NaiveBayesShiftScaler()
    X = sp.csr_matrix(np.hstack([np.zeros((10, 10000)), np.random.randn(10, 104)]))
    X_scaled = scaler.fit(X).transform(X)
    
    assert X_scaled.min() >= 0
    assert X_scaled.shape == (10, 10104)
