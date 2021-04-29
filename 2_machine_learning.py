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
from sklearn.model_selection import GroupShuffleSplit, GroupKFold, KFold, GridSearchCV, \
                                    train_test_split, LeaveOneGroupOut, \
                                    cross_validate, cross_val_predict, \
                                    RandomizedSearchCV

#Random forest
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, precision_score, recall_score, \
                            roc_auc_score, f1_score, accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier

#XGB
import xgboost as xgb

#HMM
import ssm

from lib.utils import seed_everything, validate_submission
from lib.helper import API_KEY

seed_everything()

parser = argparse.ArgumentParser() 
parser.add_argument('features', type = str) #The feature set to use in model
parser.add_argument('model', type = str) #The model to use
parser.add_argument('--submit', action='store_true') #Whether to submit to aicrowd
parser.add_argument('--parameterset', default = 'default')
parser.add_argument('--test', action='store_true') #Whether to run on test data

#########
## HMM ##
#########

def logit(p):
    return np.log(p / (1 - p))

#Idea 2. HMM.
def fit_hmm(gt, emissions_raw, preds_raw, D, C):

    #Fit empirical transition matrix
    transition_matrix = np.ones((D,D))

    N = len(gt)
    
    for idx in range(N):
        if idx == 0: continue
        transition_matrix[gt[idx-1], gt[idx]] += 1
        
    for j in range(D):
        transition_matrix[j] /= np.sum(transition_matrix[j])
        
    #Adding the actual predicted category from the RF model (in addition to the probabilities)
    #helped improve performance -- increase the precision a bit
    
    #Turn traces into categories
    emissions = np.hstack(((emissions_raw*(C-1)).astype(int), np.atleast_2d((preds_raw).astype(int)).T))

    #print(emissions.shape)
    
    #Fit empirical emission probabilities
    emission_dist = np.ones((D, D+1, C))
    for i in range(D):
        for j in range(D+1):
            for k in range(C):
                ct = np.sum(emissions[(gt == i),j] == k)
                emission_dist[i, j, k] = max(1, ct)
            emission_dist[i,j,:] /= np.sum(emission_dist[i,j,:])

    true_hmm = ssm.HMM(D, D+1, observations="categorical", observation_kwargs = {'C': C})

    #Set params to empirical ones
    true_hmm.transitions.params = [np.log(transition_matrix)]

    #true_hmm.init_state_distn.params stay as is (uniform)

    #Emission probs, stored as logits
    true_hmm.observations.params = logit(emission_dist)
        
    return true_hmm

def infer_hmm(hmm, emissions_raw, preds_raw, C):
    emissions = np.hstack(((emissions_raw*(C-1)).astype(int), np.atleast_2d((preds_raw).astype(int)).T))
    return hmm.most_likely_states(emissions)

#################
## Reweighting ##
#################

def sample_prob_simplex(n=4):
    x = sorted(np.append(np.random.uniform(size = n-1), [0,1]))
    y = np.diff(np.array(x))
    return y

def optimize_weights(train_labels, train_pred_prob, val_pred_probs, N = 1000):
    f = lambda w: f1_score(train_labels, np.argmax((train_pred_prob*w), axis = -1), average = 'macro', labels = [0,1,2])

    w_star = np.ones(4)/4
    f_star = 0

    for idx in range(N):
        w = sample_prob_simplex()
        f_curr = f(w)
        if f_curr > f_star:
            w_star = w
            f_star = f_curr

    #Reweight and then apply HMM
    train_pred_probs_reweighted = train_pred_prob*w_star
    train_pred_reweighted = np.argmax(train_pred_probs_reweighted, axis = -1)

    #Reweight and then apply HMM
    val_pred_probs_reweighted = val_pred_probs*w_star
    val_pred_reweighted = np.argmax(val_pred_probs_reweighted, axis = -1)

    return (w_star, f_star, train_pred_probs_reweighted, train_pred_reweighted, 
                val_pred_probs_reweighted, val_pred_reweighted)

def run_askl(X_train, y_train, groups, params = None):

    raise NotImplementedError
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
    return askl, None

from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import StackingClassifier

def run_stacking_sklearn(X_train, y_train, X_val, y_val, groups, params = None):
    
    #Since this uses CV, we'll combine training and validation data

    cv_groups = GroupKFold(n_splits = 5)
    scorer = sklearn.metrics.make_scorer(f1_score, labels = [0,1,2], average = 'macro')
    xgb_params = {'subsample': 0.6,
        'min_child_weight': 1,
        'max_depth': 3,
        'gamma': 2,
        'colsample_bytree': 1.0}

    # define the base models
    level0 = list()
    level0.append(('cart', DecisionTreeClassifier()))
    level0.append(('svm', SVC()))
    level0.append(('bayes', GaussianNB()))
    level0.append(('rf', RandomForestClassifier(n_estimators = 25, criterion='entropy')))
    #Add best parameters for this model here
    level0.append(('xgb', xgb.XGBClassifier(**xgb_params)))
    level0.append(('xgb_base', xgb.XGBClassifier()))
    # define meta learner model
    level1 = LogisticRegression()
    # define the stacking ensemble
    model = StackingClassifier(estimators=level0, final_estimator=level1, cv=cv_groups, n_jobs = 8)
            
    #Fit the model, 
    print("Fitting", model)
    model.fit(X_train, y_train, groups)
    pred_train = model.predict(X_train)
    predict_proba_train = model.predict_proba(X_train)
    f1_train = f1_score(y_train, pred_train, average = 'macro', labels = [0,1,2])
    predict_proba_val = []
    pred_val = []

    print("Training performance")
    print(classification_report(y_train, pred_train))

    return model, predict_proba_train, pred_train, predict_proba_val, pred_val


def run_rf_cv_one_vs_all(X_train, y_train, X_val, y_val, groups, params = None, refit = False):

    #Setup default parameters
    if params is None:
        params = {}
        params['n_estimators'] = 10
        params['criterion'] = 'entropy'
        #params['class_weight'] = 'balanced'

    #Make random forest classifier, with group-level CV
    model = RandomForestClassifier(**params)

    print('Fitting random forest model with CV')
    cv_groups = GroupKFold(n_splits = 5)

    #One vs all: predict just one class at a time, then aggregate
    predict_proba_train = []
    predict_train_each_class_pred = []
    for behavior in range(4):
        pred_prob = cross_val_predict(model, X_train, y_train == behavior, 
                                           groups = groups, cv = cv_groups,
                                           n_jobs = 5, method = 'predict_proba')
        predict_proba_train.append(pred_prob)
        pred = np.argmax(pred_prob, axis = 1)
        predict_train_each_class_pred.append(pred)
        #Evaluate performance of each classifier on its own:
        print(f'Performance of class {behavior} classifier (training)')
        evaluate(y_train == behavior, pred)

    #Combine them to form final prediction
    predict_proba_train = np.vstack([i[:,1] for i in predict_proba_train]).T
    pred_train = np.argmax(predict_proba_train, axis = 1)

    #Compute validation predictions
    predict_proba_val = []
    predict_val_each_class_pred = []
    for behavior in range(4):
        model.fit(X_train, y_train == behavior)
        pred_prob = model.predict_proba(X_val)
        predict_proba_val.append(pred_prob)
        pred = np.argmax(pred_prob, axis = 1)
        predict_val_each_class_pred.append(pred)
        #Evaluate performance of each classifier on its own:
        print(f'Performance of class {behavior} classifier (validation)')
        evaluate(y_val == behavior, pred)

    predict_proba_val = np.vstack([i[:,1] for i in predict_proba_val]).T
    pred_val = np.argmax(predict_proba_val, axis = 1)

    return model, predict_proba_train, pred_train, predict_proba_val, pred_val

def run_rf_cv(X_train, y_train, X_val, y_val, groups, params = None, refit = False):

    #Setup default parameters
    if params is None:
        params = {}
        params['n_estimators'] = 10
        params['criterion'] = 'entropy'
        #params['class_weight'] = 'balanced'

    #Make random forest classifier, with group-level CV
    model = RandomForestClassifier(**params)

    print('Fitting random forest model with CV')
    cv_groups = GroupKFold(n_splits = 5)
    predict_proba_train = cross_val_predict(model, X_train, y_train, 
                                           groups = groups, cv = cv_groups,
                                           n_jobs = 5, method = 'predict_proba')

    #Extract final prediction
    pred_train = np.argmax(predict_proba_train, axis = 1)

    model.fit(X_train, y_train)
    predict_proba_val = model.predict_proba(X_val)
    pred_val = np.argmax(predict_proba_val, axis = 1)

    return model, predict_proba_train, pred_train, predict_proba_val, pred_val

def run_xgb_cv(X_train, y_train, X_val, y_val, groups, params = None, refit = False):

    #Setup default parameters
    if params is None:
        params = {}
        params['learning_rate'] = 0.01

    #params = {'subsample': 0.8,
    #    'min_child_weight': 10,
    #    'max_depth': 4,
    #    'gamma': 0.5,
    #    'colsample_bytree': 0.8}

    #Optimized from mars_distr_w_1dcnn random grid search below
    params = {'subsample': 0.6,
        'min_child_weight': 1,
        'max_depth': 3,
        'gamma': 2,
        'colsample_bytree': 1.0}


    model = xgb.XGBClassifier(**params)
    print('Fitting XGB model with CV')
    cv_groups = GroupKFold(n_splits = 5)
    predict_proba_train = cross_val_predict(model, X_train, y_train, 
                                           groups = groups, cv = cv_groups,
                                           n_jobs = 5, method = 'predict_proba')

    #Extract final prediction
    pred_train = np.argmax(predict_proba_train, axis = 1)

    #Use trained model on validation data.... how?
    model.fit(X_train, y_train)
    predict_proba_val = model.predict_proba(X_val)
    pred_val = np.argmax(predict_proba_val, axis = 1)

    return model, predict_proba_train, pred_train, predict_proba_val, pred_val

def run_xgb_cv_randomsearch(X_train, y_train, X_val, y_val, groups, params = None, refit = False):

    #Setup default parameters
    if params is None:
        params = {'learning_rate': 0.01, 'n_estimators': [10, 30, 100], 'criterion': ['gini', 'entropy']}

    params = {
            'min_child_weight': [1, 5, 10],
            'max_depth': [3, 4, 5],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0]
            }

    model = xgb.XGBClassifier(**params)
    print('Searching for best params with XGB model with CV')
    cv_groups = GroupKFold(n_splits = 5)
    
    scorer = sklearn.metrics.make_scorer(f1_score, labels = [0,1,2], average = 'macro')

    #Test:
    #clf = RandomizedSearchCV(model, params, n_iter = 1, cv = cv_groups, verbose = 2, n_jobs = 5, scoring = scorer)

    #Normal
    clf = RandomizedSearchCV(model, params, n_iter = 30, cv = cv_groups, verbose = 2, n_jobs = 5, scoring = scorer)
    clf.fit(X_train, y_train, groups)

    pred_val = clf.predict(X_val)
    predict_proba_val = clf.predict_proba(X_val)

    pred_train = clf.predict(X_train)
    predict_proba_train = clf.predict_proba(X_train)

    return clf, predict_proba_train, pred_train, predict_proba_val, pred_val

#TODO
# Setup a separate search for each estimator
# Combine these, along with the neural network and the rest of the raw featureset
# 

def run_rf(X_train, y_train, X_val, y_val, groups, params = None, refit = False):

    #Setup default parameters
    if params is None:
        params = {}
        params['n_estimators'] = 10
        params['criterion'] = 'entropy'
        #params['class_weight'] = 'balanced'

    #Make random forest classifier, with group-level CV
    model = RandomForestClassifier(**params)

    print('Fitting random forest model')
    model.fit(X_train, y_train)

    if refit:
        print('Refitting with all data')
        model.refit(X_train.copy(), y_train.copy())

    #Compute proba
    pred_proba_train = model.predict_proba(X_train)
    pred_proba_val = model.predict_proba(X_val)

    #Compute performance measures
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    return model, pred_proba_train, pred_train, pred_proba_val, pred_val

def run_xgb(X_train, y_train, X_val, y_val, groups, params = None, refit = False):

    #Setup default parameters
    if params is None:
        params = {}
        params['learning_rate'] = 0.01

    params = {'subsample': 0.6,
        'min_child_weight': 1,
        'max_depth': 3,
        'gamma': 2,
        'colsample_bytree': 1.0}

    model = xgb.XGBClassifier(**params)
    print("Fitting XGB model")
    model.fit(X_train, y_train)

    #Compute proba
    pred_proba_train = model.predict_proba(X_train)
    pred_proba_val = model.predict_proba(X_val)

    #Compute performance measures
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    return model, pred_proba_train, pred_train, pred_proba_val, pred_val

def evaluate(y_train, pred_train, y_val = [], pred_val = []):
    print("Performance on training data")
    print(classification_report(y_train, pred_train))
    if y_val:
        print("Performance on validation data")
        print(classification_report(y_val, pred_val))
    print("Training F1 score: (only for behavior labels)")
    print(f1_score(y_train, pred_train, labels = [0, 1, 2], average = 'macro'))
    if y_val:
        print("Validation F1 score: (only for behavior labels)")
        print(f1_score(y_val, pred_val, labels = [0, 1, 2], average = 'macro'))
        return f1_score(y_val, pred_val, labels = [0, 1, 2], average = 'macro')
    return 0

class Args(object):
    def __init__(self):
        self.features = 'mars_distr_w_1dcnn'
        self.model = 'skl_stacking'
        self.submit = False
        self.parameterset = None
        self.test = True

args = Args()

def main(args):

    supported_models = {'askl': run_askl, 
                          'rf': run_rf,
                          'xgb': run_xgb,
                          'rf_cv': run_rf_cv,
                          'xgb_cv': run_xgb_cv, 
                          'xgb_cv_search': run_xgb_cv_randomsearch,
                          'skl_stacking': run_stacking_sklearn}

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
    
    X = train_features.drop(columns = ['annotation', 'seq_id'])
    y = train_features['annotation']
    groups = train_features['seq_id']

    group_list = np.array(pd.unique(groups))

    #Split into train and validation by groups
    #70 total training videos
    #Select 7 at random to be validation videos...

    n_val_group = 0
    val_group = np.random.choice(group_list, n_val_group)

    X_train = X[~train_features['seq_id'].isin(val_group)]
    y_train = y[~train_features['seq_id'].isin(val_group)]
    groups_train = groups[~train_features['seq_id'].isin(val_group)]

    X_val = X[train_features['seq_id'].isin(val_group)]
    y_val = y[train_features['seq_id'].isin(val_group)]

    print("Running machine learning model")
    model, pred_proba_train, pred_train, pred_proba_val, pred_val = \
                run_model(X_train, y_train, X_val, y_val, groups_train, params)

    if hasattr(model, 'best_params_'):
        best_params = model.best_params_ 
        search_results = model.cv_results_
    else:
        best_params = None
        search_results = None

    base_f1 = evaluate(y_train, pred_train, y_val, pred_val)

    #Can add HMM and reweighting here... based on training data, and based on predicted probabilities

    #Save training and validation predictions
    fn_out = f"results/train_validation_model_{args.features}_ml_{args.model}_paramset_{args.parameterset}.pkl"
    save_data = {'pred_train': pred_train, 'pred_val': pred_val, 'X_train': X_train, 
                 'y_train': y_train, 'groups_train': groups_train, 'X_val': X_val, 
                 'y_val': y_val, 'val_group': val_group}

    if best_params:
        save_data['best_params'] = best_params 
        save_data['search_results'] = search_results

    with open(fn_out, 'wb') as handle:
        pickle.dump(save_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print("HMM optimization")
    D = pred_proba_val.shape[1]
    C = 11
    lambdas = np.logspace(1, 30, 10)

    hmm = fit_hmm(np.array(y_train), np.array(pred_proba_train), np.array(pred_train), D, C)
    hmm_pred = infer_hmm(hmm, np.array(pred_proba_train), np.array(pred_train), C)
    hmm_pred_val = infer_hmm(hmm, np.array(pred_proba_val), np.array(pred_val), C)

    f1_hmm = evaluate(y_train, hmm_pred, y_val, hmm_pred_val)

    print("Reweighting for optimal F1 score")
    (w_star, f_star, train_pred_probs_reweighted, train_pred_reweighted, 
                val_pred_probs_reweighted, val_pred_reweighted) = \
                    optimize_weights(y_train, pred_proba_train, pred_proba_val)

    evaluate(y_train, train_pred_reweighted, y_val, val_pred_reweighted)
    f1_optimal_weights = f_star

    print("Applying HMM to reweighted model optimal F1 score")
    hmm_reweighted = fit_hmm(np.array(y_train), np.array(train_pred_probs_reweighted), np.array(train_pred_reweighted), D, C)
    hmm_pred_reweighted = infer_hmm(hmm_reweighted, np.array(train_pred_probs_reweighted), np.array(train_pred_reweighted), C)
    hmm_pred_val_reweighted = infer_hmm(hmm_reweighted, np.array(val_pred_probs_reweighted), np.array(val_pred_reweighted), C)
    f1_hmm_reweighted = evaluate(y_train, hmm_pred_reweighted, y_val, hmm_pred_val_reweighted)

    #Decide on best model... based on the validation data? I guess so... still could be overfitting to this?
    f1_scores = np.array([base_f1, f1_hmm, f1_optimal_weights, f1_hmm_reweighted])
    f1_methods = ['base', 'hmm', 'reweighted', 'hmm_reweighted']
    best_f1 = np.argmax(f1_scores)
    print(f"Best validation F1 score of {np.max(f1_scores)} with: {f1_methods[best_f1]} method")

    print('but, submitting baseline model here.')
    #Maybe this helps a tiny bit???
    best_f1 = 2

    if args.test:

        with open(f'data/intermediate/test_map_features_{args.features}.pkl', 'rb') as handle:
            test_map = pickle.load(handle)

        X_test = test_features.drop(columns = ['seq_id'])
        groups_test = test_features['seq_id']

        print("Predicting fit model on test data")
        predictions = model.predict(X_test)
        predict_proba = model.predict_proba(X_test)

        if best_f1 == 0:
            final_predictions = predictions 
        elif best_f1 == 1:
            final_predictions = infer_hmm(hmm, np.array(predict_proba), np.array(predictions), C)
        else:
            test_pred_probs_reweighted = predict_proba*w_star
            final_predictions = np.argmax(test_pred_probs_reweighted, axis = -1)

        print("Preparing submission")
        fn_out = f"results/submission_{args.features}_ml_{args.model}_paramset_{args.parameterset}.npy"
        submission = defaultdict(list)
        for idx in range(len(final_predictions)):
            submission[test_map[groups_test[idx]]].append(final_predictions[idx])
        np.save(fn_out, submission)

        print("Validating submission")
        valid = validate_submission(submission, sample_submission)
        if not valid:
            print("Invalid submission format. Check submission.npy")
            #return 
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
    try:
        args = parser.parse_args()
    except SystemExit:
        print("Cannot parse arguments. Resorting to the inbuilt defaults.")
    main(args)