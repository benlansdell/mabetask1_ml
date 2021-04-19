######################
## Basic processing ##
######################

#Convert input data to tables for easier user in Pandas
dvc run -n format_data -d 0_basic_data_formatting.py -d data/train.npy \
-d data/test.npy \
-d data/sample_submission.npy \
-o data/intermediate/test_df.csv \
-o data/intermediate/train_df.csv \
python3 0_basic_data_formatting.py

#########################
## Feature engineering ##
#########################

#Create data matrix with feature set based on differences between coordinates
dvc run -n features_differences -d 1_feature_engineering.py -d data/intermediate/test_df.csv \
-d data/intermediate/train_df.csv \
-o data/intermediate/train_features_differences.csv \
-o data/intermediate/test_features_differences.csv \
-o data/intermediate/test_map_features_differences.pkl \
python3 1_feature_engineering.py differences

#Create data matrix with feature set based on distances between coordinates
dvc run -n features_distances -d 1_feature_engineering.py -d data/intermediate/test_df.csv \
-d data/intermediate/train_df.csv \
-o data/intermediate/train_features_distances.csv \
-o data/intermediate/test_features_distances.csv \
-o data/intermediate/test_map_features_distances.pkl \
python3 1_feature_engineering.py distances

#Create data matrix with feature set based on distances, at a range of time points
dvc run -n features_distances_shifted -d 1_feature_engineering.py -d data/intermediate/test_df.csv \
-d data/intermediate/train_df.csv \
-o data/intermediate/train_features_distances_shifted.csv \
-o data/intermediate/test_features_distances_shifted.csv \
-o data/intermediate/test_map_features_distances_shifted.pkl \
python3 1_feature_engineering.py distances_shifted

######################
## Machine learning ##
######################

# Autosklearn

## Slow and doesn't perform well...... fughedabowdit

# Basic random forest model on differences feature set
dvc run -n ml_rf_differences -d 2_machine_learning.py -d data/intermediate/train_features_differences.csv \
-d data/intermediate/test_features_differences.csv \
-d data/intermediate/test_map_features_differences.pkl \
-o results/submission_features_differences_ml_rf_paramset_default.npy \
python3 2_machine_learning.py features_differences rf

#Run RF on distances feature set
dvc run -n ml_rf_distances -d 2_machine_learning.py -d data/intermediate/train_features_distances.csv \
-d data/intermediate/test_features_distances.csv \
-d data/intermediate/test_map_features_distances.pkl \
-o results/submission_features_distances_ml_rf_paramset_default.npy \
python3 2_machine_learning.py features_distances rf

#Run RF on distances feature set with adjacent temporal frames
dvc run -n ml_rf_distances_shifted -d 2_machine_learning.py -d data/intermediate/train_features_distances_shifted.csv \
-d data/intermediate/test_features_distances_shifted.csv \
-d data/intermediate/test_map_features_distances_shifted.pkl \
-o results/submission_features_distances_shifted_ml_rf_paramset_default.npy \
python3 2_machine_learning.py features_distances_shifted rf

#Recall that, if a change is made to an earlier step, you can rerun the whole thing with:
dvc run -n ml_gridsearchcv_distances_shifted -d 2_machine_learning_gridsearch_cv.py \
-d data/intermediate/train_features_distances_shifted.csv \
-d data/intermediate/test_features_distances_shifted.csv \
-d data/intermediate/test_map_features_distances_shifted.pkl \
-o results/submission_features_distances_shifted_ml_gridsearch_cv_paramset_default.npy \
-m results/summary_gridsearch_cv.json \
python3 2_machine_learning_gridsearch_cv.py features_distances_shifted

#################
# Deep learning #
#################

#Baseline model

dvc run -n dl_baseline -d 3_deep_learning.py \
-d data/train.npy \
-d data/test.npy \
-d data/sample_submission.npy \
-o results/submission_dl_baseline_paramset_default.npy \
-m results/summary_dl_baseline_paramset_default.json \
-p dl_baseline_settings.json \
python3 3_deep_learning.py baseline 

#UNet model
dvc run -n dl_unet -d 3_deep_learning.py \
-d data/train.npy \
-d data/test.npy \
-d data/sample_submission.npy \
-o results/submission_dl_unet_paramset_default.npy \
-m results/summary_dl_unet_paramset_default.json \
-p dl_unet_settings.json:epochs \
python3 3_deep_learning.py unet 

#LSTM model
dvc run -n dl_lstm -d 3_deep_learning.py \
-d data/train.npy \
-d data/test.npy \
-d data/sample_submission.npy \
-o results/submission_dl_lstm_paramset_default.npy \
-m results/summary_dl_lstm_paramset_default.json \
-p dl_lstm_settings.json:epochs \
python3 3_deep_learning.py lstm