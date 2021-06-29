# Multi-agent behavioral analysis AIcrowd challenge -- 

Ben Lansdell (ben.lansdell@stjude.org)

## Inference on DLC and BORIS datasets

Contains code to run behavior prediction on poses extracted by DLC and compare with behavior annotations extracted by BORIS.

## Setup

This python code is most easily reproduced using a conda environment:

```
conda env create -f ml_env.yml
conda activate ml_mabe

#Install Linderman's HMM code (in the SSM package):
git clone git@github.com:slinderman/ssm.git
cd ssm
pip install -e .
```

## How to run

1. Extract pose info from a h5 DLC file, save in a format the MABE competition understands. DONE
2. Run 0_basic_data_formatting.py to format this file for ML. DONE
3. Run 1_deep_learning_stacking.py to run inference of previously trained network and extract deep learning prediction features. DONE
4. Run 2_feature_engineering_stacking.py to use these features along with other previously trained ML features to create final feature set of hand-crafted features along with DL and classic ML prediction features. DONE
5. Finally, run a previously trained XGB model on all of these features. DONE
6. Perform some final tweaks (e.g. filter predictions with a HMM). DONE

After we have the predictions:
* Extract from test prediction dictionary, per video. Have to link our silly md5 key to the video...
* Make a video that displays the prediction for each frame
* If time: load in BORIS data and compare performance

We want to compare these predictions to hand-collected annotations.

## The ML approach

The idea is a combination of a 1DCNN deep learning model, whose output probabilities are used as input for more classical ML approaches, alongside a set of handcrafted features. In more detail:

1. Generate predictions of a 1DCNN neural network, using the set of distances between all key-points as input to the network, rather than the raw positions. Predictions are made through a 5-fold cross validation prediction procedure. Denote the predicted probabilities $X_{CNN}$.
2. Generate a set of handcrafted features based on the MARS model [1]. Denote the handcrafted features $X_{HC}$.
3. Through a separate 5-fold cross validation prediction procedure, generate predictions of a set of basic ML models, trained on $[X_{CNN}, X_{HC}]$. Call the prediction of these models a new set of features, $X_{ML}$.
4. Train an XGB model on the features $[X_{CNN}, X_{HC}, X_{ML}]$.
5. Perform some final tweaks of the output of the XGB model to finalize predictions.

## References

[1] Segalin et al 2020. The Mouse Action Recognition System (MARS): a software pipeline for automated analysis of social behaviors in mice. bioRxiv. `https://www.biorxiv.org/content/10.1101/2020.07.26.222299v1.full`