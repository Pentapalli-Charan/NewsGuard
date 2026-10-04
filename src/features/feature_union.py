import numpy as np
import scipy.sparse as sp
import pandas as pd
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_extraction.text import TfidfVectorizer

from src.features.auxiliary_features import AuxiliaryFeatureEngineer

class TextSelector(BaseEstimator, TransformerMixin):
    """Extracts a specified text column from a pandas DataFrame."""
    def __init__(self, key):
        self.key = key

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.key].fillna("")

class AuxiliaryTransformer(BaseEstimator, TransformerMixin):
    """Generates Word2Vec and NLP auxiliary features and applies MinMaxScaler."""
    def __init__(self, vector_size=100, min_count=2):
        self.vector_size = vector_size
        self.min_count = min_count
        self.engineer = None
        self.scaler = None

    def fit(self, X, y=None):
        self.engineer = AuxiliaryFeatureEngineer(vector_size=self.vector_size, min_count=self.min_count, workers=1, seed=42)
        self.scaler = MinMaxScaler()
        
        # Fit Word2Vec on the training cleaned_text
        self.engineer.fit_word2vec(X['cleaned_text'])
        
        # Generate features to fit the scaler
        feats = self.engineer.generate_all_features(X['statement'], X['cleaned_text'])
        
        # Fit scaler
        self.scaler.fit(feats.values)
        return self

    def transform(self, X):
        if self.engineer is None or self.scaler is None:
            raise ValueError("AuxiliaryTransformer has not been fitted.")
            
        feats = self.engineer.generate_all_features(X['statement'], X['cleaned_text'])
        scaled_feats = self.scaler.transform(feats.values)
        
        # Return sparse matrix to ensure FeatureUnion horizontally stacks as CSR
        return sp.csr_matrix(scaled_feats)

def build_feature_union(min_count=2):
    """
    Constructs and returns a true sklearn FeatureUnion combining TF-IDF and Auxiliary features.
    """
    union = FeatureUnion([
        ("tfidf_branch", Pipeline([
            ("selector", TextSelector(key="cleaned_text")),
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=min_count, max_df=0.95, max_features=10000))
        ])),
        ("aux_branch", AuxiliaryTransformer(vector_size=100, min_count=min_count))
    ])
    return union
