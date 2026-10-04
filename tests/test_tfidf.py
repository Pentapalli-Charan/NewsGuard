import os
import pytest
import pandas as pd
import scipy.sparse
import json
from src.features.tfidf_features import TfidfFeatureEngineer

def test_vectorizer_loads_and_fits():
    engineer = TfidfFeatureEngineer(ngram_range=(1,2), max_features=100, min_df=1, max_df=1.0)
    train_text = pd.Series(["hello world", "this is a test"])
    X = engineer.fit_transform_train(train_text)
    assert X.shape[0] == 2
    assert engineer.config['ngram_range'] == (1,2)
    
def test_vocabulary_unigrams_bigrams():
    engineer = TfidfFeatureEngineer(ngram_range=(1,2), min_df=1, max_df=1.0)
    train_text = pd.Series(["hello world"])
    engineer.fit_transform_train(train_text)
    features = engineer.get_feature_names()
    assert len(features) > 0
    assert len(set(features)) == len(features)
    has_unigram = any(" " not in f for f in features)
    has_bigram = any(" " in f for f in features)
    assert has_unigram
    assert has_bigram

def test_matrices_exist_and_sparse():
    assert os.path.exists("data/processed/tfidf_train.npz")
    assert os.path.exists("data/processed/tfidf_validation.npz")
    assert os.path.exists("data/processed/tfidf_test.npz")
    
    train = scipy.sparse.load_npz("data/processed/tfidf_train.npz")
    val = scipy.sparse.load_npz("data/processed/tfidf_validation.npz")
    test = scipy.sparse.load_npz("data/processed/tfidf_test.npz")
    
    assert scipy.sparse.issparse(train)
    assert scipy.sparse.issparse(val)
    assert scipy.sparse.issparse(test)
    
    assert train.shape[1] == val.shape[1] == test.shape[1]
    
    with open("data/processed/tfidf_feature_names.json", "r") as f:
        features = json.load(f)
    assert train.shape[1] == len(features)

def test_serialization_works():
    assert os.path.exists("models/tfidf_vectorizer.joblib")
    engineer = TfidfFeatureEngineer.load_vectorizer("models/tfidf_vectorizer.joblib")
    assert engineer.config['ngram_range'] == (1,2)
    assert len(engineer.get_feature_names()) > 0

def test_leakage():
    # Prove TF-IDF vocabulary is based ONLY on training data
    engineer = TfidfFeatureEngineer(ngram_range=(1,1), min_df=1, max_df=1.0)
    train_text = pd.Series(["apple banana"])
    val_text = pd.Series(["apple validationonlytoken"])
    
    engineer.fit_transform_train(train_text)
    engineer.transform(val_text)
    
    features = engineer.get_feature_names()
    assert "apple" in features
    assert "banana" in features
    assert "validationonlytoken" not in features

def test_reproducibility():
    engineer1 = TfidfFeatureEngineer(ngram_range=(1,2), min_df=1, max_df=1.0)
    engineer2 = TfidfFeatureEngineer(ngram_range=(1,2), min_df=1, max_df=1.0)
    
    text = pd.Series(["test document one", "test document two"])
    X1 = engineer1.fit_transform_train(text)
    X2 = engineer2.fit_transform_train(text)
    
    assert engineer1.get_feature_names() == engineer2.get_feature_names()
    assert (X1 != X2).nnz == 0

def test_artifacts_exist():
    assert os.path.exists("data/processed/tfidf_metadata.json")
    assert os.path.exists("reports/day3_tfidf_report.md")
    assert os.path.exists("reports/figures/day3/top_discriminative_terms.png")
    assert os.path.exists("reports/metrics/day3_discriminative_terms.csv")
