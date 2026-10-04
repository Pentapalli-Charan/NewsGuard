import json
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
import scipy.sparse
import pandas as pd
from typing import Tuple, List

class TfidfFeatureEngineer:
    def __init__(self, 
                 ngram_range: Tuple[int, int] = (1, 2), 
                 max_features: int = 10000, 
                 min_df: int = 2,
                 max_df: float = 0.95):
        """
        Initialize TF-IDF feature engineer with configurable parameters.
        Using max_features=10000 by default to keep sparse matrices manageable, 
        and filter extremely rare/common words using min_df/max_df.
        """
        self.config = {
            "ngram_range": ngram_range,
            "max_features": max_features,
            "min_df": min_df,
            "max_df": max_df,
            "lowercase": True  # Cleaned text is already lowercased, but scikit-learn defaults to True
        }
        self.vectorizer = TfidfVectorizer(**self.config)
        
    def fit_transform_train(self, train_text: pd.Series) -> scipy.sparse.csr_matrix:
        """Fit on training data and transform."""
        # Ensure we have strings and replace NaNs with empty string
        train_text = train_text.fillna("")
        return self.vectorizer.fit_transform(train_text)
        
    def transform(self, text: pd.Series) -> scipy.sparse.csr_matrix:
        """Transform validation or test data without fitting."""
        text = text.fillna("")
        return self.vectorizer.transform(text)
        
    def get_feature_names(self) -> List[str]:
        return self.vectorizer.get_feature_names_out().tolist()
        
    def save_vectorizer(self, filepath: str):
        joblib.dump(self.vectorizer, filepath)
        
    @classmethod
    def load_vectorizer(cls, filepath: str):
        obj = cls()
        obj.vectorizer = joblib.load(filepath)
        obj.config = obj.vectorizer.get_params()
        return obj

    def save_feature_names(self, filepath: str):
        with open(filepath, 'w') as f:
            json.dump(self.get_feature_names(), f)
