# Multi-agent behavioral analysis AIcrowd challenge -- task 1 (classical classification)

Ben Lansdell (ben.lansdell@stjude.org)

## Overview

Here is code to reproduce the submission that produced my highest score on the 70% public testset (with ID 133118).

## Setup

This python code is most easily reproduced using a conda environment:

asdf

## The basic approach

The idea is a combination of a 1DCNN deep learning model (based on baseline code), whose output probabilities are used as input for more classical ML approaches, alongside a set of handcrafted features. In more detail:

1. Generate predictions of 1DCNN model, using the set of distances between all keypoints as input to the network, rather than the raw positions. Predictions are made through a 5-fold cross validation prediction procedure. 
2. Concatenate these 1DCNN predictions to a set of handcrafted features inspired by the MARS model [1].
3. Through a separate 5-fold cross validation prediction procedure, generate predictions of a set of basic ML models, to build a stacking classifier. 
4. Concatentate the probability predictions of these models, along with the handcrafted features and the 1DCNN features.
5. Train an XGB model for the final predictor

A more detailed description can be found in `model.pdf`

## Running the pipeline

Commands to go from downloading the data to submitting the final predictions are detailed in `pipeline.sh`

## Contents

* requirements.txt -- short list of package requirements (should be sufficient for reproducibility)
* ml_env.yaml -- yaml file containing full list of packages for reproducibility
* pipeline.sh -- the whole pipeline -- from downloading data to submitting -- can be reproduced with pipeline.sh
* model.pdf -- a more detailed description of the model 
* LICENSE.txt -- public license

## References

[1] Segalin et al 2020. The Mouse Action Recognition System (MARS): a software pipeline for automated analysis of social behaviors in mice. bioRxiv. `https://www.biorxiv.org/content/10.1101/2020.07.26.222299v1.full`