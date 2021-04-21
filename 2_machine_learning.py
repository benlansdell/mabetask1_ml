import numpy as np
import os
import sklearn
import gc
import pandas as pd
import pickle 
import argparse

from collections import defaultdict

#Auto sklearn
import autosklearn.classification
from sklearn.model_selection import GroupShuffleSplit

#Random forest
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, GridSearchCV, train_test_split, LeaveOneGroupOut
from sklearn.metrics import confusion_matrix, precision_score, recall_score, roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

from lib.utils import seed_everything, validate_submission
from lib.helper import API_KEY

seed_everything()

parser = argparse.ArgumentParser() 
parser.add_argument('features', type = str) #The feature set to use in model
parser.add_argument('model', type = str) #The model to use
parser.add_argument('--submit', action='store_true') #Whether to submit to aicrowd
parser.add_argument('--parameterset', default = 'default')
parser.add_argument('--test', action='store_true') #Whether to run on test data

def run_askl(X_train, y_train, groups, params = None):
    #TODO
    # How to get to run with n_jobs > 1?
    os.system('rm -r ./tmp_askl')

    askl = autosklearn.classification.AutoSklearnClassifier(
        resampling_strategy = GroupShuffleSplit,
        resampling_strategy_arguments = {'n_splits':5, 'test_size':7,
                                        'groups': groups},
        memory_limit = 100000,
        n_jobs = 1, 
        tmp_folder = './tmp_askl', 
        time_left_for_this_task = 36000, 
        seed = 5)

    print('Fitting a set of models')
    askl.fit(X_train, y_train)
    print('Refitting with all data')
    askl.refit(X_train.copy(), y_train.copy())
    return askl

def run_rf(X_train, y_train, groups, params = None, refit = False):

    #Setup default parameters
    if params is None:
        params = {}
        params['n_estimators'] = 10
        params['criterion'] = 'entropy'

    #Make random forest classifier, with group-level CV
    model = RandomForestClassifier(**params)

    print('Fitting random forest model')
    model.fit(X_train, y_train)

    if refit:
        print('Refitting with all data')
        model.refit(X_train.copy(), y_train.copy())
    return model

#def format_predictions(preds, groups):
#   sadf

# Add HMM on output... to try to smooth estimates of behavior
# Once we have the output of the RF model (with enough estimators... 100?)

class Args(object):
    def __init__(self):
        self.features = 'features_differences'
        self.model = 'rf'
        self.submit = False
        self.parameterset = None
        self.test = False

args = Args()

def main(args):

    supported_models = {'askl': run_askl, 
                          'rf': run_rf}

    if args.model not in supported_models:
        print("Model not found. Select one of", list(supported_models.keys()))
        return

    if args.parameterset == 'default':
        params = None 
    else:
        #Load parameters from json file...
        #TODO:
        # Implement. For now, resort to default
        params = None

    run_model = supported_models[args.model]

    train_features = pd.read_csv(f'data/intermediate/train_features_{args.features}.csv')
    if args.test:
        test_features = pd.read_csv(f'data/intermediate/test_features_{args.features}.csv')

    sample_submission = np.load('data/sample_submission.npy',allow_pickle=True).item()
    
    with open(f'data/intermediate/test_map_features_{args.features}.pkl', 'rb') as handle:
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

    if args.test:
        X_test = test_features.drop(columns = ['seq_id'])
        groups_test = test_features['seq_id']

    print("Running machine learning model")
    model = run_model(X_train, y_train, groups_train, params)

    #Compute performance measures
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    #Save training and validation predictions
    fn_out = f"results/train_validation_{args.features}_ml_{args.model}_paramset_{args.parameterset}.pkl"
    save_data = {'pred_train': pred_train, 'pred_val': pred_val, 'X_train': X_train, 
                 'y_train': y_train, 'groups_train': groups_train, 'X_val': X_val, 
                 'y_val': y_val, 'val_group': val_group}

    with open(fn_out, 'wb') as handle:
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("Performance on training data")
    print(classification_report(y_train, pred_train))

    print("Performance on validation data")
    print(classification_report(y_val, pred_val))

    if args.test:

        print("Predicting fit model on test data")
        predictions = model.predict(X_test)

        print("Preparing submission")
        fn_out = f"results/submission_{args.features}_ml_{args.model}_paramset_{args.parameterset}.npy"
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
    print(args.test)
    main(args)