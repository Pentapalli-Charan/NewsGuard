import os
import pandas as pd
from src.preprocessing.text_preprocessor import TextPreprocessor

def test_preprocessing_url_removal():
    preprocessor = TextPreprocessor()
    text = "Check out this link https://example.com and http://example.com/path or www.example.com for more info"
    cleaned = preprocessor.process(text)
    assert "http" not in cleaned
    assert "www" not in cleaned
    assert "example" not in cleaned
    assert "link" in cleaned
    assert "info" in cleaned

def test_preprocessing_punctuation_cleaning():
    preprocessor = TextPreprocessor()
    text = "Hello, world! This is a test... What's up?"
    cleaned = preprocessor.process(text)
    # punctuation removed, 's is removed, everything is lowercased
    # 'What's' becomes 'what s', lemmatized 'what' is 'what', 's' is removed if stopword, or kept if not
    assert "hello" in cleaned
    assert "world" in cleaned
    assert "!" not in cleaned
    assert "?" not in cleaned
    
def test_preprocessing_tokenization_and_lemmatization():
    preprocessor = TextPreprocessor()
    # 'running' -> 'run', 'dogs' -> 'dog', 'better' -> 'well' or 'good'
    text = "The running dogs are better"
    cleaned = preprocessor.process(text)
    assert "run" in cleaned
    assert "dog" in cleaned
    assert "running" not in cleaned
    assert "dogs" not in cleaned

def test_preprocessing_stopword_removal():
    preprocessor = TextPreprocessor()
    text = "This is a simple test of the stopword removal"
    cleaned = preprocessor.process(text)
    # 'this', 'is', 'a', 'of', 'the' should be removed
    assert "this" not in cleaned
    assert "is" not in cleaned
    assert "simple" in cleaned
    assert "test" in cleaned
    assert "stopword" in cleaned
    assert "removal" in cleaned

def test_preprocessing_lowercase():
    preprocessor = TextPreprocessor()
    text = "UPPERCASE TEXT"
    cleaned = preprocessor.process(text)
    assert cleaned == "uppercase text"

def test_preprocessing_empty_input():
    preprocessor = TextPreprocessor()
    assert preprocessor.process("") == ""
    assert preprocessor.process("   ") == ""
    assert preprocessor.process(None) == ""

def test_preprocessing_deterministic_output():
    preprocessor = TextPreprocessor()
    text = "Random sentence to test determinism!"
    cleaned1 = preprocessor.process(text)
    cleaned2 = preprocessor.process(text)
    assert cleaned1 == cleaned2

def test_preprocessed_files_exist():
    assert os.path.exists("data/processed/preprocessed_train.csv")
    assert os.path.exists("data/processed/preprocessed_validation.csv")
    assert os.path.exists("data/processed/preprocessed_test.csv")

def test_data_integrity():
    train = pd.read_csv("data/processed/train.csv")
    preprocessed_train = pd.read_csv("data/processed/preprocessed_train.csv")
    assert len(train) == len(preprocessed_train)
    assert list(train.columns) == [col for col in preprocessed_train.columns if col != 'cleaned_text']
    assert (train['binary_label'] == preprocessed_train['binary_label']).all()
    assert (train['statement'] == preprocessed_train['statement']).all()
    assert 'cleaned_text' in preprocessed_train.columns
