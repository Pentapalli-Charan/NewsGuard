import re
import spacy
import nltk
from nltk.corpus import stopwords
from typing import Optional

class TextPreprocessor:
    def __init__(self, spacy_model: str = "en_core_web_sm"):
        # Load SpaCy model once for efficiency
        # Disable ner and parser as they are not needed for tokenization and lemmatization
        self.nlp = spacy.load(spacy_model, disable=['ner', 'parser'])
        
        # Load NLTK stopwords
        try:
            self.stop_words = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            self.stop_words = set(stopwords.words('english'))

        # URL regex pattern: handles http://, https://, and www.
        self.url_pattern = re.compile(r'https?://\S+|www\.\S+')
        
        # Punctuation/noise pattern: keep alphanumeric and space
        self.noise_pattern = re.compile(r'[^a-zA-Z0-9\s]')
        
    def process(self, text: Optional[str]) -> str:
        """
        Processes a single string text through the pipeline.
        Pipeline Order:
        1. Input validation (handle None/empty)
        2. Lowercase normalization (deterministic case policy)
        3. URL removal
        4. Noise/punctuation removal
        5. SpaCy tokenization
        6. Stopword removal (via NLTK)
        7. SpaCy lemmatization
        8. Final whitespace normalization
        """
        # 1. Input validation
        if text is None:
            return ""
        if not isinstance(text, str):
            text = str(text)
        
        text = text.strip()
        if not text:
            return ""
            
        # 2. Lowercase normalization
        text = text.lower()
        
        # 3. URL removal
        text = self.url_pattern.sub('', text)
        
        # 4. Noise/punctuation handling (remove non-alphanumeric except spaces)
        text = self.noise_pattern.sub(' ', text)
        
        # 5, 7. SpaCy tokenization and lemmatization
        # Process the text using spacy pipeline
        doc = self.nlp(text)
        
        # 6, 8. Stopword removal, lemmatization and whitespace normalization
        cleaned_tokens = []
        for token in doc:
            # Tokenize: we iterate over doc.
            # Stopword removal: using NLTK stopwords
            # Note: SpaCy's tokenizer automatically splits by whitespace and punctuation.
            # We already removed punctuation, but we check token text against stopwords.
            if token.text.strip() and token.text not in self.stop_words:
                # Lemmatize
                cleaned_tokens.append(token.lemma_)
                
        # 9. Final whitespace normalization
        cleaned_text = " ".join(cleaned_tokens)
        return cleaned_text.strip()
