import numpy as np
import os
import sklearn
import gc
import pandas as pd
import pickle 
import argparse
import json 

from collections import defaultdict

#Auto sklearn
import autosklearn.classification
from sklearn.model_selection import GroupShuffleSplit

#Random forest
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification
from sklearn.pipeline import Pipeline

from lib.utils import seed_everything, validate_submission
from lib.helper import API_KEY

seed_everything()

parser = argparse.ArgumentParser() 
parser.add_argument('features', type = str) #The feature set to use in model
parser.add_argument('--submit', action='store_true') #Whether to submit to aicrowd
parser.add_argument('--parameterset', default = 'default')
parser.add_argument('--testrun', action = 'store_true')

#################################

def compute_metrics(y, pred):
    re = recall_score(y, pred, average = 'macro', labels = [0,1,2])
    pr = precision_score(y, pred, average = 'macro', labels = [0,1,2])
    f1 = f1_score(y, pred, labels = [0,1,2], average = 'macro')
    return [re, pr, f1]

def run_gridsearch_rf_xgb_cv(X_train, y_train, groups_train, X_val, y_val, refit = False, test_run = False):

    scorer = sklearn.metrics.make_scorer(f1_score, labels = [0,1,2], average = 'macro')

    gss = GroupShuffleSplit(n_splits = 5, test_size=7, random_state = 0)

    lr = Pipeline([('scaler', StandardScaler()), ('lr', sklearn.linear_model.LogisticRegression(max_iter = 10000))])
    rf = RandomForestClassifier()
    xgb = GradientBoostingClassifier()

    short_models = {#'logreg': [lr, [], {'lr__C': np.logspace(-2,-2,1)}, [], []], 
        'rf': [rf, [], {'n_estimators': [10], 'criterion': ['gini', 'entropy']}, [], []]}

    models = {'logreg': [lr, [], {'lr__C': np.logspace(-2,2,5)}, [], []], 
            'rf': [rf, [], {'n_estimators': [10, 30, 100], 'criterion': ['gini', 'entropy']}, [], []],
            'xgb': [xgb, [], {'loss': ['deviance', 'exponential']}, [], []]}

    if test_run:
        print("Test run: only doing a small grid search")
        models = short_models 

    for name in models:
        print('Estimating model ' + name)
        model = models[name][0]
        params = models[name][2]
        clf = GridSearchCV(model, params, cv = gss, verbose = 2, n_jobs = 10, scoring = scorer)
        clf.fit(X_train, y_train, groups_train)
        models[name][3] = clf.best_params_
        models[name][4] = clf
        preds = clf.predict(X_val)
        m = compute_metrics(y_val, preds)
        print('metrics:', m)
        models[name][1] = np.array(m)

    if refit:
        print('Refitting with all data')
        model.refit(X_train.copy(), y_train.copy())

    return models

#Model stacking
# Add HMM on output... to try to smooth estimates of behavior
# Once we have the output of the RF model (with enough estimators... 100?)

#TODO
# Implement stacking here. Currently building prototype in notebook

def run_gridsearch_stacking(X_train, y_train, groups_train, X_val, y_val, refit = False):

    gss = GroupShuffleSplit(n_splits = 5, test_size=7, random_state = 0)

    estimators = [
        ('rf', RandomForestClassifier()), 
        ('xgb', GradientBoostingClassifier()),
        ('lr', Pipeline([('scaler', StandardScaler()), ('lr', sklearn.linear_model.LogisticRegression(max_iter = 10000))]))
    ]
    
    clf = StackingClassifier(estimators=estimators, final_estimator=sklearn.linear_model.LogisticRegression(), cv = gss)

    models = [[], [], [], [], []]

    clf.fit(X_train, y_train, groups_train)
    models[name][3] = clf.best_params_
    models[name][4] = clf
    preds = clf.predict(X_val)
    metrics = compute_metrics(y_val, preds)
    print('metrics:', metrics)
    models[name][1] = np.array(metrics)

    if refit:
        print('Refitting with all data')
        model.refit(X_train.copy(), y_train.copy())

    return models

class Args(object):
    def __init__(self):
        self.features = 'features_distances_shifted'
        self.submit = False
        self.parameterset = 'default'
        self.testrun = True

#Load a set of default arguments, if running interactively. If __main__, then
#args are parsed from the command line, and these are not used
args = Args()

def main(args):

    if args.parameterset == 'default':
        params = None 
    else:
        #Load parameters from json file...
        #TODO:
        # Implement. For now, resort to default
        params = None

    run_model = run_gridsearch_rf_xgb_cv

    #Save training and validation predictions
    fn_out = f"results/train_validation_{args.features}_ml_gridsearch_cv_paramset_{args.parameterset}.pkl"
    json_out = "results/summary_gridsearch_cv.json"

    train_features = pd.read_csv(f'data/intermediate/train_{args.features}.csv')
    test_features = pd.read_csv(f'data/intermediate/test_{args.features}.csv')

    sample_submission = np.load('data/sample_submission.npy',allow_pickle=True).item()
    
    with open(f'data/intermediate/test_map_{args.features}.pkl', 'rb') as handle:
        test_map = pickle.load(handle)

    X = train_features.drop(columns = ['annotation', 'seq_id'])
    y = train_features['annotation']
    groups = train_features['seq_id']

    group_list = np.array(pd.unique(groups))

    #Split into train and validation by groups
    #70 total training videos
    #Select 7 at random to be validation videos...

    n_val_group = 7
    val_group = np.random.choice(group_list, n_val_group)

    X_train = X[~train_features['seq_id'].isin(val_group)]
    y_train = y[~train_features['seq_id'].isin(val_group)]
    groups_train = groups[~train_features['seq_id'].isin(val_group)]

    X_val = X[train_features['seq_id'].isin(val_group)]
    y_val = y[train_features['seq_id'].isin(val_group)]

    X_test = test_features.drop(columns = ['seq_id'])
    groups_test = test_features['seq_id']

    print("Running machine learning model")
    models = run_model(X_train, y_train, groups_train, X_val, y_val, test_run = args.testrun)

    #Find model with best validation f1 score
    best_f1 = 0
    for k in models:
        if models[k][1][2] > best_f1:
            model = models[k][4] 

    #Compute performance measures
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    #Also compute predicted probabilities
    save_data = {'pred_train': pred_train, 'pred_val': pred_val, 'X_train': X_train, 
                 'y_train': y_train, 'groups_train': groups_train, 'X_val': X_val, 
                 'y_val': y_val, 'val_group': val_group}

    with open(fn_out, 'wb') as handle:
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Performance on training data")
    print(classification_report(y_train, pred_train))

    print("Performance on validation data")
    print(classification_report(y_val, pred_val))

    #Compute f1 score, precision, recall, accuracy (only using the labels 0,1,2)
    performance = {}
    re, pr, f1 = compute_metrics(y_train, pred_train)
    performance['train'] = {'f1_macro': f1, 'recall': re, 'precision': pr}
    re, pr, f1 = compute_metrics(y_val, pred_val)
    performance['validation'] = {'f1_macro': f1, 'recall': re, 'precision': pr}
    performance['features'] = args.features
    performance['parameterset'] = args.parameterset

    #Save scores to summary.json
    with open(json_out, 'w') as outfile:
        json.dump(performance, outfile)

    print("Predicting fit model on test data")
    predictions = model.predict(X_test)

    print("Preparing submission")
    fn_out = f"results/submission_{args.features}_ml_gridsearch_cv_paramset_{args.parameterset}.npy"
    submission = defaultdict(list)
    for idx in range(len(predictions)):
        submission[test_map[groups_test[idx]]].append(predictions[idx])
    np.save(fn_out, submission)

    print("Validating submission")
    valid = validate_submission(submission, sample_submission)
    if not valid:
        print("Invalid submission format. Check submission.npy")
        return 
    print("Submission validated.")

    #results/submission_features_differences_ml_rf.npy

    if args.submit:
        login_cmd = f"aicrowd login --api-key {API_KEY}"
        os.system(login_cmd)
        sub_cmd = f"aicrowd submission create -c mabe-task-1-classical-classification -f {fn_out}"
        os.system(sub_cmd)
    else:
        print(f"Submission not made. Can do so manually as follows:\n\naicrowd submission create -c mabe-task-1-classical-classification -f {fn_out}\n")

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)