import pandas as pd
import numpy as np

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from gensim.sklearn_api import D2VTransformer

from sklearn.base import BaseEstimator, TransformerMixin


class Doc2VecExtractor(BaseEstimator, TransformerMixin):
    """Custom Doc2Vec Estimator Transformer class to use in sci-kit learn pipeline
    
    Johnson Kuan 2020
    
    """

    def __init__(self, dm = 1, min_count = 1, size = 20):
      
        self.dm = dm
        self.min_count = min_count
        self.size = size
        
        # https://radimrehurek.com/gensim/sklearn_api/d2vmodel.html
        self.d2v_model = D2VTransformer(dm = self.dm, min_count = self.min_count, size = self.size)   
        
    def tokenize_clean(self, text):
        """clean and tokenize text string"""

        tokens = word_tokenize(text)
        lemmatizer = WordNetLemmatizer()

        clean_tokens = []
        for tok in tokens:
            clean_tok = lemmatizer.lemmatize(tok).lower().strip()
            clean_tokens.append(clean_tok)

        return clean_tokens              
        
    def tokenize(self, x):
        return pd.Series(x).map(self.tokenize_clean).tolist() # prepare data for doc2vec algorithm
    
    def fit(self, x, y=None):
        
        text = self.tokenize(x)
        
        self.d2v_model.fit(text) # run doc2vec algorithm
        
        return self

    def transform(self, x):
        
        text = self.tokenize(x)

        return self.d2v_model.transform(text) # return document vectors