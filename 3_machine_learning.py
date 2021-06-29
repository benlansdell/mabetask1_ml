import numpy as np
import os
import pandas as pd
import pickle 
import argparse

from collections import defaultdict

from sklearn.metrics import f1_score

import xgboost as xgb

#Support for HMMs
import ssm

from joblib import dump

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

def run_xgb(X_train, y_train, X_val, y_val, groups, params = None, refit = False):

    #This is our 'winning' submission. Along with the reweighted HMM
    #Optimized from mars_distr_stacked_w_1dcnn random grid search below
    params = {'subsample': 0.6,
        'min_child_weight': 1,
        'max_depth': 3,
        'gamma': 1.5,
        'colsample_bytree': 1.0}

    model = xgb.XGBClassifier(**params)
    print("Fitting XGB model")
    model.fit(X_train, y_train)

    #Save the model
    dump(model, f'./results/level_1_model_xgb.joblib')

    #Compute proba
    pred_proba_train = model.predict_proba(X_train)
    pred_proba_val = model.predict_proba(X_val)

    #Compute performance measures
    pred_train = model.predict(X_train)
    pred_val = model.predict(X_val)

    return model, pred_proba_train, pred_train, pred_proba_val, pred_val

def main(args):

    supported_models = {'xgb': run_xgb}

    if args.model not in supported_models:
        print("Model not found. Select one of", list(supported_models.keys()))
        return

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

    print("Reweighting for optimal F1 score")
    (w_star, _, train_pred_probs_reweighted, train_pred_reweighted, 
                val_pred_probs_reweighted, val_pred_reweighted) = \
                    optimize_weights(y_train, pred_proba_train, pred_proba_val)

    print("Applying HMM to reweighted model optimal F1 score")
    D = pred_proba_train.shape[1]
    C = 11
    hmm_reweighted = fit_hmm(np.array(y_train), np.array(train_pred_probs_reweighted), np.array(train_pred_reweighted), D, C)

    hmm = fit_hmm(np.array(y_train), np.array(pred_proba_train), np.array(pred_train), D, C)

    #Save the models too
    dump(hmm, f'./results/level_1_hmm_model.joblib')

    if args.test:

        with open(f'data/intermediate/test_map_features_{args.features}.pkl', 'rb') as handle:
            test_map = pickle.load(handle)

        X_test = test_features.drop(columns = ['seq_id'])
        groups_test = test_features['seq_id']

        print("Predicting fit model on test data")
        predict_proba = model.predict_proba(X_test)
        
        test_pred_probs_reweighted = predict_proba*w_star
        reweighted_predictions = np.argmax(test_pred_probs_reweighted, axis = -1)
        final_predictions = infer_hmm(hmm_reweighted, np.array(test_pred_probs_reweighted), np.array(reweighted_predictions), C)

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
            return 
        print("Submission validated.")

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