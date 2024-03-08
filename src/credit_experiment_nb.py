# Weighs and balances is more enterprise standard https://wandb.ai/site
# sacred works locally for privacy
# an experiement file in sacred is special thing
# decoration are funcitons that wrap around other functions
# sqlobserver and sacred
from sacred import Experiment
from sacred.observers import SqlObserver

import pandas as pd


from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_validate, train_test_split
import os
# this is proper pipeline, not using Jupyter, 
from dotenv import load_dotenv

from logger import get_logger

# these are my ingredients
from credit_preproc_ingredient import preproc_ingredient, get_column_transformer
from credit_data_ingredient import data_ingredient, load_data
from credit_db_ingredient import db_ingredient, df_to_sql

# loaded here
load_dotenv()

# connection string
db_url = os.getenv('DB_URL')

# creating an experiement containing ingredients
# logger using our standard logger
_logs = get_logger(__name__)
ex  = Experiment("Credit Experiment",
                 ingredients=[data_ingredient, preproc_ingredient, db_ingredient])

# telling sacred this is the logger we want to use
ex.logger = _logs
# we have setup our experiment
ex.observers.append(SqlObserver(db_url))

# decorators that come from sacred, means we are using a configuration functions
# sacred will monitor this and make it availabel to any funcitons and make it available# 
@ex.config
def cfg():
    '''
    Config function: all variables here will be avialable in captured functions.
    '''
    preproc_pipe = "power"
    folds = 5
    scoring = ['neg_log_loss', 'roc_auc', 'f1', 'accuracy', 'precision', 'recall']

    
# sacred will be looking fir get-pipe in this functions
    # wants to keep config in my database, so create a file that allows me to store all my configurations details for the database
    #
@ex.capture
def get_pipe(preproc_pipe):
    '''
    Returns an NB pipeline.
    '''
    _logs.info(f'Getting Naive Bayes Pipeline')
    ct = get_column_transformer(preproc_pipe)
    pipe = Pipeline(
        steps  = [
            ('preproc', ct),
            ('clf', GaussianNB())
        ]
    )
    return pipe


# evaluating our pipeline
@ex.capture
def evaluate_model(pipe, X, Y, folds, scoring, _run):
    '''Evaluate model using corss validation.'''
    _logs.info(f'Evaluating model')
    res_dict = cross_validate(pipe, X, Y, cv = folds, scoring = scoring)
    res = (pd.DataFrame(res_dict)
           .reset_index()
           .rename(columns={'index': 'fold'})
           # givining it an id _run is special to sacred, idetifies every run you've done to your database
           # let you go back to every run you have done
           .assign(run_id = _run._id))
    return res

@ex.capture
def res_to_sql(res):
    '''Write results to db.'''
    _logs.info(f'Writing results to db')
    df_to_sql(res, "model_cv_fold_results")
    
    df_to_sql(res.groupby('run_id', group_keys=False).mean(), "model_cv_results")


# run is most important bit, not a caputre function, but automain,indicat this si the main function
    # to giv ea command line interface.
    # 
@ex.automain
def run():
    '''Main experiment run.'''
    _logs.info(f'Running experiment')
    X, Y  = load_data()
    pipe = get_pipe()
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 42)
    res = evaluate_model(pipe, X_train, Y_train)   
    res_to_sql(res)
   

# we want to track   
if __name__=="__main__":
    ex.run_commandline()