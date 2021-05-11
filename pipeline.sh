#!/bin/sh

#Initialization
#Data should be placed in ./data/

#Step 0
#Format data
python 0_basic_data_formatting.py 

#Step 1a
#Train DL 1D CNN model
python XXXX

#Step 1b
#Create features for model stacking
python 1_feature_engineering_stacking.py 

#Step 2
#Train ML model (XGB) with these features
python 2_machine_learning.py mars_distr_stacked_w_1dcnn xgb --test

#Submit:
aicrowd submission create -c mabe-task-1-classical-classification \
                    -f results/submission_mars_distr_stacked_w_1dcnn_ml_xgb_paramset_default.npy

################################
#Check the answers are the same#
################################

#Step 0 -- checks out
md5sum ./data/intermediate/test_df.csv
md5sum ../../mabe/mabetask1_ml/data/intermediate/test_df.csv

md5sum ./data/intermediate/train_df.csv
md5sum ../../mabe/mabetask1_ml/data/intermediate/train_df.csv

#Step 1a


#Step 1b

#Step 2
# Don't have these saved :( Pretty stupid...

# Instead, here are some other sanity checks:

# Check 