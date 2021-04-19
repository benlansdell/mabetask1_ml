#Feature engineering:
# Difference between each body part

import pandas as pd 
import numpy as np
import argparse
import pickle 
from lib.helper import xy_ids, bodypart_ids, mouse_ids, colnames

from sklearn.impute import SimpleImputer

parser = argparse.ArgumentParser() 
parser.add_argument('features', type = str)

def boiler_plate(features_df):
    hashmap = {k: i for (i,k) in enumerate(list(set(features_df['seq_id'])))}
    reversemap = {i: k for (i,k) in enumerate(list(set(features_df['seq_id'])))}
    features_df['seq_id_'] = [hashmap[i] for i in features_df['seq_id']]
    features_df['seq_id'] = features_df['seq_id_']
    features_df = features_df.drop(columns = ['seq_id_', 'Unnamed: 0'])
    return features_df, reversemap

def make_features_differences(df):

    feature_set_name = 'features_differences'

    features_df = df.copy()

    ##Make the features
    for xy in xy_ids:
        for i, bp1 in enumerate(bodypart_ids):
            for j, bp2 in enumerate(bodypart_ids):
                if i < j:
                    for mouse_id in mouse_ids:
                        #We can compute the intra-mouse difference
                        f1 = '_'.join([mouse_id, xy, bp1])
                        f2 = '_'.join([mouse_id, xy, bp2])
                        f_new = '_'.join([mouse_id, xy, bp1, bp2])
                        features_df[f_new] = features_df[f1] - features_df[f2]
                #Inter-mouse difference
                f1 = '_'.join([mouse_ids[0], xy, bp1])
                f2 = '_'.join([mouse_ids[1], xy, bp2])
                f_new = '_'.join(['M0_M1', xy, bp1, bp2])
                features_df[f_new] = features_df[f1] - features_df[f2]
    #Remove base features
    features_df = features_df.drop(columns = colnames)

    ##Clean up seq_id columns
    features_df, reversemap = boiler_plate(features_df)

    return features_df, reversemap, feature_set_name

def make_features_distances(df):

    feature_set_name = 'features_distances'

    features_df = df.copy()

    ##Make the distance features
    for i, bp1 in enumerate(bodypart_ids):
        for j, bp2 in enumerate(bodypart_ids):
            if i < j:
                for mouse_id in mouse_ids:
                    #We can compute the intra-mouse difference
                    f1x = '_'.join([mouse_id, 'x', bp1])
                    f2x = '_'.join([mouse_id, 'x', bp2])
                    f1y = '_'.join([mouse_id, 'y', bp1])
                    f2y = '_'.join([mouse_id, 'y', bp2])
                    f_new = '_'.join([mouse_id, 'dist', bp1, bp2])
                    features_df[f_new] = \
                        np.sqrt((features_df[f1x] - features_df[f2x])**2 + 
                                (features_df[f1y] - features_df[f2y])**2)
            #Inter-mouse difference
            f1x = '_'.join([mouse_ids[0], 'x', bp1])
            f2x = '_'.join([mouse_ids[1], 'x', bp2])
            f1y = '_'.join([mouse_ids[0], 'y', bp1])
            f2y = '_'.join([mouse_ids[1], 'y', bp2])
            f_new = '_'.join(['M0_M1', 'dist', bp1, bp2])
            features_df[f_new] = \
                        np.sqrt((features_df[f1x] - features_df[f2x])**2 + 
                                (features_df[f1y] - features_df[f2y])**2)

    #Remove base features
    features_df = features_df.drop(columns = colnames)

    ##Clean up seq_id columns
    features_df, reversemap = boiler_plate(features_df)

    return features_df, reversemap, feature_set_name

def make_features_distances_shifted(df):

    feature_set_name = 'features_distances_shifted'
    features_df, reversemap, _ = make_features_distances(df)
    #Now add shifts (just not of the )

    if 'annotation' in features_df.columns: 
        labels = features_df[['annotation', 'seq_id']]
        the_rest = features_df.drop(columns = ['annotation', 'seq_id'])
    else:
        labels = features_df[['seq_id']]
        the_rest = features_df.drop(columns = ['seq_id'])

    periods = [-15, -10, -5, 5, 10, 15]
    data = [labels] + [the_rest.shift(p) for p in periods] + [the_rest]
    features_df = pd.concat(data, axis = 1)

    #Impute NAs
    features_df = features_df.apply(lambda x: x.fillna(x.mean()), axis=0)

    return features_df, reversemap, feature_set_name

def main(args):

    supported_features = {'differences': make_features_differences, 
                          'distances': make_features_distances, 
                          'distances_shifted': make_features_distances_shifted}

    if args.features not in supported_features:
        print("Features not found. Select one of", list(supported_features.keys()))
        return

    test_df = pd.read_csv('./data/intermediate/test_df.csv')
    train_df = pd.read_csv('./data/intermediate/train_df.csv')

    feature_maker = supported_features[args.features]

    train_features, train_map, _ = feature_maker(train_df)
    test_features, test_map, name = feature_maker(test_df)

    train_features.to_csv(f'./data/intermediate/train_{name}.csv', index = False)
    test_features.to_csv(f'./data/intermediate/test_{name}.csv', index = False)
    with open(f'./data/intermediate/test_map_{name}.pkl', 'wb') as handle:
        pickle.dump(test_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
