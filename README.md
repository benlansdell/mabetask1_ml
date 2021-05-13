# Multi-agent behavioral analysis AIcrowd challenge -- Task 1 (classical classification)

Ben Lansdell (ben.lansdell@stjude.org)

## Overview

Here is code to reproduce the submission that produced my highest score on the 70% public testset (with ID 133118).

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

## The basic approach

The idea is a combination of a 1DCNN deep learning model (based on baseline code), whose output probabilities are used as input for more classical ML approaches, alongside a set of handcrafted features. In more detail:

1. Generate predictions of a 1DCNN neural network, using the set of distances between all key-points as input to the network, rather than the raw positions. Predictions are made through a 5-fold cross validation prediction procedure. Denote the predicted probabilities $X_{CNN}$.
2. Generate a set of handcrafted features based on the MARS model [1]. Denote the handcrafted features $X_{HC}$.
3. Through a separate 5-fold cross validation prediction procedure, generate predictions of a set of basic ML models, trained on $[X_{CNN}, X_{HC}]$. Call the prediction of these models a new set of features, $X_{ML}$.
4. Train an XGB model on the features $[X_{CNN}, X_{HC}, X_{ML}]$.
5. Perform some final tweaks of the output of the XGB model to finalize predictions.

A more detailed description can be found in `MABE_challenge_writeup.pdf`

## Running the pipeline

All the commands to go from downloading the data to submitting the final predictions are detailed in `pipeline.sh`

## References

[1] Segalin et al 2020. The Mouse Action Recognition System (MARS): a software pipeline for automated analysis of social behaviors in mice. bioRxiv. `https://www.biorxiv.org/content/10.1101/2020.07.26.222299v1.full`