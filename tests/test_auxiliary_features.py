import os
import pytest
import pandas as pd
import numpy as np
from src.features.auxiliary_features import AuxiliaryFeatureEngineer

def test_word2vec_loads_and_has_vocab():
    # Provide tiny dataset
    engineer = AuxiliaryFeatureEngineer(vector_size=10, min_count=1)
    train_text = pd.Series(["hello world", "test token"])
    engineer.fit_word2vec(train_text)
    
    assert engineer.w2v_model is not None
    assert len(engineer.w2v_model.wv) > 0
    assert "hello" in engineer.w2v_model.wv

def test_sentence_vector_dimensions_and_empty():
    engineer = AuxiliaryFeatureEngineer(vector_size=10, min_count=1)
    train_text = pd.Series(["hello world", "test token"])
    engineer.fit_word2vec(train_text)
    
    # Test valid text
    vectors = engineer.transform_word2vec(pd.Series(["hello world"]))
    assert vectors.shape == (1, 10)
    
    # Test unknown/empty text
    empty_vec = engineer.transform_word2vec(pd.Series([""]))
    assert empty_vec.shape == (1, 10)
    assert np.all(empty_vec == 0)
    
    unknown_vec = engineer.transform_word2vec(pd.Series(["unknownxyz"]))
    assert unknown_vec.shape == (1, 10)
    assert np.all(unknown_vec == 0)

def test_word2vec_leakage():
    engineer = AuxiliaryFeatureEngineer(vector_size=10, min_count=1)
    train_text = pd.Series(["hello world"])
    val_text = pd.Series(["validationonlytoken"])
    
    engineer.fit_word2vec(train_text)
    vocab_before = list(engineer.w2v_model.wv.key_to_index.keys())
    
    # Transform val text
    engineer.transform_word2vec(val_text)
    
    vocab_after = list(engineer.w2v_model.wv.key_to_index.keys())
    
    assert "validationonlytoken" not in vocab_after
    assert vocab_before == vocab_after

def test_auxiliary_nlp_features():
    engineer = AuxiliaryFeatureEngineer(min_count=1)
    raw = pd.Series(["This is a test sentence!"])
    clean = pd.Series(["test sentence"])
    
    engineer.fit_word2vec(clean)
    df = engineer.generate_all_features(raw, clean)
    assert 'readability_flesch_kincaid' in df.columns
    assert 'sentiment_polarity' in df.columns
    assert 'article_length' in df.columns
    assert 'punctuation_ratio' in df.columns
    
    # Range check
    assert -1.0 <= df['sentiment_polarity'].iloc[0] <= 1.0
    
    # Article length
    assert df['article_length'].iloc[0] == 2
    
    # Punctuation ratio = 1 punct / 24 total chars
    assert df['punctuation_ratio'].iloc[0] == 1/24

def test_auxiliary_nlp_features_empty():
    engineer = AuxiliaryFeatureEngineer(min_count=1)
    raw = pd.Series([""])
    clean = pd.Series([""])
    
    engineer.fit_word2vec(pd.Series(["dummy"]))
    df = engineer.generate_all_features(raw, clean)
    assert df['punctuation_ratio'].iloc[0] == 0.0
    assert df['article_length'].iloc[0] == 0.0

def test_reproducibility():
    engineer1 = AuxiliaryFeatureEngineer(vector_size=10, min_count=1, seed=42, workers=1)
    engineer2 = AuxiliaryFeatureEngineer(vector_size=10, min_count=1, seed=42, workers=1)
    
    train_text = pd.Series(["reproducibility is important"])
    engineer1.fit_word2vec(train_text)
    engineer2.fit_word2vec(train_text)
    
    v1 = engineer1.transform_word2vec(train_text)
    v2 = engineer2.transform_word2vec(train_text)
    
    np.testing.assert_array_almost_equal(v1, v2)

def test_artifacts_exist():
    assert os.path.exists("models/word2vec.model")
    assert os.path.exists("data/processed/auxiliary_train.csv")
    assert os.path.exists("data/processed/auxiliary_validation.csv")
    assert os.path.exists("data/processed/auxiliary_test.csv")
    assert os.path.exists("data/processed/auxiliary_features_metadata.json")
    assert os.path.exists("data/processed/auxiliary_feature_names.json")
    assert os.path.exists("reports/day4_auxiliary_features_report.md")
    assert os.path.exists("reports/figures/day4/sentiment_distribution.png")

def test_data_integrity():
    train = pd.read_csv("data/processed/preprocessed_train.csv")
    aux = pd.read_csv("data/processed/auxiliary_train.csv")
    
    assert len(train) == len(aux)
    assert (train['binary_label'] == aux['binary_label']).all()
