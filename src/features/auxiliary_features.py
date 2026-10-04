import os
import json
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
import textstat
from textblob import TextBlob
from typing import List, Optional

class AuxiliaryFeatureEngineer:
    def __init__(self, 
                 vector_size=100, 
                 window=5, 
                 min_count=2, 
                 workers=1, 
                 sg=1, 
                 epochs=10, 
                 seed=42):
        self.config = {
            "vector_size": vector_size,
            "window": window,
            "min_count": min_count,
            "workers": workers,
            "sg": sg,
            "epochs": epochs,
            "seed": seed
        }
        self.w2v_model = None

    def fit_word2vec(self, train_cleaned_text: pd.Series):
        """Fit Word2Vec model on training cleaned text only."""
        # Tokenize by splitting on whitespace (since it's already preprocessed)
        sentences = [str(x).split() for x in train_cleaned_text.fillna("")]
        
        self.w2v_model = Word2Vec(
            sentences=sentences,
            vector_size=self.config['vector_size'],
            window=self.config['window'],
            min_count=self.config['min_count'],
            workers=self.config['workers'],
            sg=self.config['sg'],
            epochs=self.config['epochs'],
            seed=self.config['seed']
        )
        
    def transform_word2vec(self, cleaned_text: pd.Series) -> np.ndarray:
        """Create document vectors by averaging Word2Vec token vectors."""
        if self.w2v_model is None:
            raise ValueError("Word2Vec model has not been fitted.")
            
        sentences = [str(x).split() for x in cleaned_text.fillna("")]
        vectors = []
        for tokens in sentences:
            valid_tokens = [t for t in tokens if t in self.w2v_model.wv]
            if valid_tokens:
                vec = np.mean([self.w2v_model.wv[t] for t in valid_tokens], axis=0)
            else:
                vec = np.zeros(self.config['vector_size'])
            vectors.append(vec)
        return np.array(vectors)

    def extract_readability(self, raw_text: pd.Series) -> np.ndarray:
        """Flesch-Kincaid readability on raw text."""
        def safe_flesch_kincaid(text):
            text = str(text) if pd.notnull(text) else ""
            if len(text.strip()) == 0:
                return 0.0
            try:
                # Textstat might raise issues with tiny or bizarre text
                return textstat.flesch_kincaid_grade(text)
            except Exception:
                return 0.0
        
        return np.array([safe_flesch_kincaid(x) for x in raw_text])

    def extract_sentiment(self, raw_text: pd.Series) -> np.ndarray:
        """TextBlob sentiment polarity on raw text."""
        def get_polarity(text):
            text = str(text) if pd.notnull(text) else ""
            return TextBlob(text).sentiment.polarity
        return np.array([get_polarity(x) for x in raw_text])
        
    def extract_article_length(self, cleaned_text: pd.Series) -> np.ndarray:
        """Token count of cleaned text."""
        def count_tokens(text):
            text = str(text) if pd.notnull(text) else ""
            return len(text.split())
        return np.array([count_tokens(x) for x in cleaned_text])
        
    def extract_punctuation_ratio(self, raw_text: pd.Series) -> np.ndarray:
        """(number of punctuation chars) / (total chars) on raw text."""
        import string
        punct_set = set(string.punctuation)
        
        def punct_ratio(text):
            text = str(text) if pd.notnull(text) else ""
            if len(text) == 0:
                return 0.0
            p_count = sum(1 for char in text if char in punct_set)
            return p_count / len(text)
            
        return np.array([punct_ratio(x) for x in raw_text])
        
    def generate_all_features(self, raw_text: pd.Series, cleaned_text: pd.Series) -> pd.DataFrame:
        """Generates all auxiliary features and returns them as a DataFrame."""
        w2v_feats = self.transform_word2vec(cleaned_text)
        readability = self.extract_readability(raw_text)
        sentiment = self.extract_sentiment(raw_text)
        length = self.extract_article_length(cleaned_text)
        punct = self.extract_punctuation_ratio(raw_text)
        
        feature_dict = {}
        for i in range(self.config['vector_size']):
            feature_dict[f"w2v_{i:03d}"] = w2v_feats[:, i]
            
        feature_dict['readability_flesch_kincaid'] = readability
        feature_dict['sentiment_polarity'] = sentiment
        feature_dict['article_length'] = length
        feature_dict['punctuation_ratio'] = punct
        
        return pd.DataFrame(feature_dict)

    def save_model(self, filepath: str):
        self.w2v_model.save(filepath)
        
    @classmethod
    def load_model(cls, filepath: str, config_path: str):
        with open(config_path, 'r') as f:
            meta = json.load(f)
        obj = cls(**meta['word2vec_configuration'])
        obj.w2v_model = Word2Vec.load(filepath)
        return obj
