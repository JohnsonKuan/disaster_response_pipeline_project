import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath):
    """ load messages.csv and categories.csv datasets
    
    Parameters:
    messages_filepath: path to messages.csv file
    categories_filepath: path to categories.csv file
    
    Returns:
    df: merged messages and categories file on common id as dataframe
    
    
    """
    
    messages = pd.read_csv(messages_filepath) # read messages file
    categories = pd.read_csv(categories_filepath) # read categories file
    
    df = messages.merge(categories, on = 'id') # merge messages and categories file on common id
    
    return df


def clean_data(df):
    """ clean dataframe
    
    Parameters:
    df: merged messages and categories file on common id as dataframe
    
    Returns:
    df: cleansed dataframe
    
    """
  
    categories = df['categories'].str.split(';', expand = True) # split categories column on string ; and expand as individual columns
    
    category_colnames = categories.loc[0].map(lambda x: x[:-2]).tolist() # extract category column names
    
    categories.columns = category_colnames # change category column names
    
    # note: there are instances of 'related-2' in the categories field so need to apply additional logic to create a binary indicator    
    categories = categories.apply(lambda col: (pd.to_numeric(col.str[-1]) > 0).map(int)) # transform category string to binary indicator (0 or 1)
    
    df.drop(columns = ['categories'], inplace = True) # drop original categories column
    
    df = pd.concat([df, categories], axis = 1) # concatenate new categories dataframe with original dataframe
    
    df.drop_duplicates(inplace = True) # drop duplicates
    
    return df


def save_data(df, database_filename):
    """ save dataframe to a table in a sqlite database
    
    Parameters:
    df: dataframe to save
    
    Returns:
    database_filename: name of database to use
    
    """

    engine = create_engine('sqlite:///' + database_filename)
    df.to_sql('message_category', engine, index=False, if_exists = 'replace') # save to table in database
    
    pass


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()