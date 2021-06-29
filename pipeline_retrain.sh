#!/bin/sh

#Step 0
# Format data
# ~5 minutes
#python 0_basic_data_formatting.py 

#Step 1
# Train DL 1D CNN model
# ~90 minutes
python 1_deep_learning_stacking.py

#Step 2
# Create features for model stacking
# ~3-4 hours, note also that the test features csv file is ~100GB -- need both the RAM and HD space to run
python 2_feature_engineering_stacking.py 

#Step 3 
# Train final ML model (XGB) with these features
# ~20 minutes
python 3_machine_learning.py mars_distr_stacked_w_1dcnn xgb --test