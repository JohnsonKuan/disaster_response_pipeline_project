# Disaster Response Pipeline Project

## Description

* Analyze disaster data from Figure Eight
* Write code to run ETL process
* Build model that classifies disaster messages
* Deploy model as API for web app
* Run web app that can classify new messages

## Files

* data/process_data.py

  * ETL process on raw data

* models/train_classifier.py

  * ML pipeline on cleansed dataset

* app/run.py

  * Final code to run web app

* requirements.txt
  
## Data

Project uses disaster data from: https://www.figure-eight.com/

The dataset contains real messages from disasters and labels corresponding to the message category. There are 36 categories and each message can be tagged with multiple categories.

## Dependencies

I included a requirements.txt file with dependencies

```bash
pip install -r requirements.txt
```

## Instructions:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/
