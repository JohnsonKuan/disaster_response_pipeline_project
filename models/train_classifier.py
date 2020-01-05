import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger', 'maxent_ne_chunker', 'words'])

import re
import numpy as np
import pandas as pd

from sqlalchemy import create_engine

from gensim.sklearn_api import D2VTransformer

from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.decomposition import TruncatedSVD

from utils import Doc2VecExtractor # I wrote a custom estimator transformer class

import sys
import pickle

from xgboost import XGBClassifier


def load_data(database_filepath):
    """ load data from table in sqlite database
    
    Parameters:
    database_filepath: path to database
    
    Returns:
    X: features dataset
    Y: labels dataset
    category_names: name of labels
    
    
    """  
  
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('message_category', engine) # read table
    X = df.message.values # text column
    Y = df.iloc[:,4:].values # 36 labels  
    
    df.drop(columns = ['child_alone'], inplace = True) # there's only 0 values for this indicator so we can drop for this project    
    
    category_names = df.columns[4:].tolist()    

    return X, Y, category_names


def tokenize(text):
    """clean and tokenize text string"""
    
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens  


def build_model():
    """build model pipeline"""
    
    pipeline = Pipeline([

            ('features', FeatureUnion([

                ('tsvd', Pipeline([
                    ('count_vect', CountVectorizer(tokenizer = tokenize))
                    ,('tfidf', TfidfTransformer())
                    # ,('tsvd', TruncatedSVD(n_components = 50))
                ]))

                ,('doc2vec', Doc2VecExtractor(size = 20)) # custom Estimator Transformer class using gensim Doc2Vec model
            ])),
            ('clf', MultiOutputClassifier(XGBClassifier(random_state = 1, gamma = 0.2, verbosity = 1)))
        ])
    
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
    """Report the f1 score, precision and recall for each output category of the dataset"""
  
    Y_pred = model.predict(X_test)
  
    for i, col in enumerate(category_names):
        print(f'Category: {col}')
        print(classification_report(Y_test[:,i], Y_pred[:,i]))
    
    pass


def save_model(model, model_filepath):
  
    with open(model_filepath, 'wb') as f:
        pickle.dump(model ,f)
    
    pass


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()