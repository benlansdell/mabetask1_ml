#!/bin/sh
conda activate ml_mabe

#Initialization

#Change to your own key...
API_KEY="0ba231d61506b40a4ae00df011cf0cb9" aicrowd login --api-key $API_KEY
aicrowd dataset download --challenge mabe-task-1-classical-classification

mkdir data
mkdir data/intermediate
mkdir results
 
mv train.npy data/train.npy
mv test-release.npy data/test.npy
mv sample-submission.npy data/sample_submission.npy

#Step 0
# Format data
# ~5 minutes
python 0_basic_data_formatting.py 

#Step 1
# Train DL 1D CNN model
# ~90 minutes
python 1_deep_learning_stacking.py

#Step 2
# Create features for model stacking
# ~3-4 hours, note also that the test features csv file is ~100GB
python 2_feature_engineering_stacking.py 

#Step 3 
# Train final ML model (XGB) with these features
# ~20 minutes
python 3_machine_learning.py mars_distr_stacked_w_1dcnn xgb --test

#Submit:
aicrowd submission create -c mabe-task-1-classical-classification \
    -f results/submission_mars_distr_stacked_w_1dcnn_ml_xgb_paramset_default.npy