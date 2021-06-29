import numpy as np
import os
import pandas as pd
import pickle 
import argparse
from collections import defaultdict
from joblib import load

from sklearn.metrics import f1_score

import xgboost as xgb

#Support for HMMs
import ssm

from lib.utils import seed_everything, validate_submission
from lib.helper import API_KEY

seed_everything()

parser = argparse.ArgumentParser() 
parser.add_argument('features', type = str) #The feature set to use in model
parser.add_argument('model', type = str) #The model to use

def infer_hmm(hmm, emissions_raw, preds_raw, C):
    emissions = np.hstack(((emissions_raw*(C-1)).astype(int), np.atleast_2d((preds_raw).astype(int)).T))
    return hmm.most_likely_states(emissions)

def main(args):

    supported_models = {'xgb': None}

    if args.model not in supported_models:
        print("Model not found. Select one of", list(supported_models.keys()))
        return

    params = None

    test_features = pd.read_csv(f'data/intermediate/test_features_{args.features}.csv')

    sample_submission = np.load('data/sample_submission.npy',allow_pickle=True).item()
    
    #Load in models from joblib files
    print("Running machine learning model")
    #model, pred_proba_train, pred_train, pred_proba_val, pred_val = \
    #            run_model(X_train, y_train, X_val, y_val, groups_train, params)
    model = load('./results/level_1_model_xgb.joblib')

    print("Applying HMM")
    #hmm = fit_hmm(np.array(y_train), np.array(train_pred_probs_reweighted), np.array(train_pred_reweighted), D, C)
    hmm = load('./results/level_1_hmm_model.joblib')

    with open(f'data/intermediate/test_map_features_{args.features}.pkl', 'rb') as handle:
        test_map = pickle.load(handle)

    X_test = test_features.drop(columns = ['seq_id'])
    groups_test = test_features['seq_id']

    print("Predicting fit model on test data")
    predict_proba = model.predict_proba(X_test)
    predict = model.predict(X_test)

    C = 11
    final_predictions = infer_hmm(hmm, np.array(predict_proba), np.array(predict), C)

    print("Preparing submission")
    fn_out = f"results/submission_{args.features}_ml_{args.model}_paramset_default_hmm.npy"
    submission = defaultdict(list)
    for idx in range(len(final_predictions)):
        submission[test_map[groups_test[idx]]].append(final_predictions[idx])
    np.save(fn_out, submission)

    fn_out = f"results/submission_{args.features}_ml_{args.model}_paramset_default.npy"
    submission = defaultdict(list)
    for idx in range(len(predict)):
        submission[test_map[groups_test[idx]]].append(predict[idx])
    np.save(fn_out, submission)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)