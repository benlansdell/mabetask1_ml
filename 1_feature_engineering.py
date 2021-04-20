#Feature engineering:
# Difference between each body part

import pandas as pd 
import numpy as np
import argparse
import pickle 
from lib.helper import xy_ids, bodypart_ids, mouse_ids, colnames

from itertools import product

from sklearn.impute import SimpleImputer

parser = argparse.ArgumentParser() 
parser.add_argument('features', type = str)
parser.add_argument('compute_test', action = 'store_true')

#Shift decorator
#If a feature creation function has this applied then the features are
#automatically shifted and concatenated with the rest of the table

#The decorator maker, so we can provide arguments
def shift_features(window_size = 5, n_shifts = 3):
    #The decorator
    def decorator(feature_function):
        #What is called instead of the actual function, assumes the feature making
        #function returns the names of the columns just made
        def wrapper(*args, **kwargs):
            #Compute the features
            #print(args, kwargs, args[0])
            old_cols = set(args[0].columns)
            df = feature_function(*args, **kwargs)
            new_cols = set(df.columns)
            added_cols = list(new_cols.difference(old_cols))
            periods = [-(i+1)*window_size for i in range(n_shifts)] + \
                      [(i+1)*window_size for i in range(n_shifts)]
            #Shift the features just made
            shifted_data = []
            #Rename all column names
            for p in periods:
                s_df = df[added_cols].shift(p)
                s_df = s_df.rename(columns = {k:f'{k}_shifted_{p}' for k in added_cols})
                shifted_data.append(s_df)
            #Combine with current table
            df = pd.concat([df] + shifted_data, axis = 1)
            return df
        return wrapper
    return decorator

def boiler_plate(features_df):
    hashmap = {k: i for (i,k) in enumerate(list(set(features_df['seq_id'])))}
    reversemap = {i: k for (i,k) in enumerate(list(set(features_df['seq_id'])))}
    features_df['seq_id_'] = [hashmap[i] for i in features_df['seq_id']]
    features_df['seq_id'] = features_df['seq_id_']
    to_drop = ['seq_id_', 'Unnamed: 0']
    for col in to_drop:
        if col in features_df.columns:
            features_df = features_df.drop(columns = col)
    #Impute nas
    features_df = features_df.apply(lambda x: x.fillna(x.mean()), axis=0)
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

@shift_features(window_size=5, n_shifts=3)
def _compute_centroid(df, name, body_parts = bodypart_ids):
    df = df.copy()
    for mouse_id in mouse_ids:
        part_names_x = [f'{mouse_id}_x_{i}' for i in body_parts]
        part_names_y = [f'{mouse_id}_y_{i}' for i in body_parts]
        df[f'centroid_{name}_{mouse_id}_x'] = np.mean(df[part_names_x], axis = 1)
        df[f'centroid_{name}_{mouse_id}_y'] = np.mean(df[part_names_y], axis = 1)
    return df

@shift_features(window_size=5, n_shifts=3)
def _compute_abs_angle(df, name, bps, centroid = True):
    df = df.copy()
    if len(bps) != 2:
        raise ValueError('Abs angle only works between 2 bodyparts, too many or too few specified')
    for mouse_id in mouse_ids:
        if centroid:
            diff_x = df[f'{bps[0]}_{mouse_id}_x'] - df[f'{bps[1]}_{mouse_id}_x']
            diff_y = df[f'{bps[0]}_{mouse_id}_y'] - df[f'{bps[1]}_{mouse_id}_y']
        else:
            diff_x = df[f'{mouse_id}_x_{bps[0]}'] - df[f'{mouse_id}_x_{bps[1]}']
            diff_y = df[f'{mouse_id}_y_{bps[0]}'] - df[f'{mouse_id}_y_{bps[1]}']
        df[f'angle_{name}_{mouse_id}'] = np.arctan2(diff_y,diff_x)  
    return df

@shift_features(window_size=5, n_shifts=3)
def _compute_rel_angle(df, name, bps, centroid = False):
    df = df.copy()
    if len(bps) != 3:
        raise ValueError('too many body parts to compute an absolute angle. Only works for 2')
    for mouse_id in mouse_ids:
        if centroid:
            diff_x1 = df[f'{bps[0]}_{mouse_id}_x'] - df[f'{bps[1]}_{mouse_id}_x']
            diff_y1 = df[f'{bps[0]}_{mouse_id}_y'] - df[f'{bps[1]}_{mouse_id}_y']
            diff_x2 = df[f'{bps[2]}_{mouse_id}_x'] - df[f'{bps[1]}_{mouse_id}_x']
            diff_y2 = df[f'{bps[2]}_{mouse_id}_y'] - df[f'{bps[1]}_{mouse_id}_y']
        else:
            diff_x1 = df[f'{mouse_id}_x_{bps[0]}'] - df[f'{mouse_id}_x_{bps[1]}']
            diff_y1 = df[f'{mouse_id}_y_{bps[0]}'] - df[f'{mouse_id}_y_{bps[1]}']
            diff_x2 = df[f'{mouse_id}_x_{bps[2]}'] - df[f'{mouse_id}_x_{bps[1]}']
            diff_y2 = df[f'{mouse_id}_y_{bps[2]}'] - df[f'{mouse_id}_y_{bps[1]}']

        diff1 = np.vstack((diff_x1, diff_y1)).T
        diff2 = np.vstack((diff_x2, diff_y2)).T
        cosine_angle = np.sum(diff1*diff2, axis = 1) / (np.linalg.norm(diff1, axis = 1) * np.linalg.norm(diff2, axis = 1))
        df[f'angle_{name}_{mouse_id}'] = np.arccos(cosine_angle)
    return df

def _compute_ellipsoid(df):
    df = df.copy()
    #Perform SVD
    colnames = ['_'.join([a[0], a[2], a[1]]) for a in product(mouse_ids, bodypart_ids, xy_ids)]
    data = np.array(df[colnames]).reshape(-1, 2, 7, 2)
    mean_data = np.transpose(np.tile(np.mean(data, axis = 2), (7,1,1,1)), (1,2,0,3))
    svals = np.linalg.svd(data-mean_data, compute_uv = False)
    #Not technically correct, but not sure if the square of the singular values is exaclty
    #what we want either. This keeps the scale roughly the same as the distances involved
    evals = svals
    # evals = (svals*svals)/6
    for idx,m_id in enumerate(mouse_ids):
        df[f'ellipse_major_{m_id}'] = evals[:,idx,0]
        df[f'ellipse_minor_{m_id}'] = evals[:,idx,1]
    return df

#Recall framerate is 30 fps
def _compute_kinematics(df, names, window_size = 5):
    df = df.copy()
    for mouse_id in mouse_ids:
        for name in names:
            ## Speed of centroids
            dx = df[f'centroid_{name}_{mouse_id}_x'].diff(window_size)
            dy = df[f'centroid_{name}_{mouse_id}_y'].diff(window_size)
            df[f'centroid_{name}_{mouse_id}_speed'] = np.sqrt(dx**2 + dy**2)
            colnames.append(f'centroid_{name}_{mouse_id}_speed')
            ## Acceleration of centroids
            ddx = dx.diff(window_size)
            ddy = dy.diff(window_size)
            df[f'centroid_{name}_{mouse_id}_accel_x'] = ddx/(window_size**2)
            df[f'centroid_{name}_{mouse_id}_accel_y'] = ddy/(window_size**2)
    return df

def _compute_relative_body_motions(df, window_size = 3):

    #Compute vector connecting two centroids
    dx = df[f'centroid_all_{mouse_ids[0]}_x'] - df[f'centroid_all_{mouse_ids[1]}_x']
    dy = df[f'centroid_all_{mouse_ids[0]}_y'] - df[f'centroid_all_{mouse_ids[1]}_y']
    dm = np.sqrt(dx**2 + dy**2)
    df['distance_main_centroid'] = dm

    #Compute velocity of mouse centroids
    for m_id in mouse_ids:
        vx = df[f'centroid_all_{m_id}_x'].diff(window_size)/window_size
        vy = df[f'centroid_all_{m_id}_y'].diff(window_size)/window_size
        v_tangent = (dx*vx + dy*vy)/dm
        v_perp_x = vx - dx*v_tangent/dm
        v_perp_y = vy - dy*v_tangent/dm
        v_perp = np.sqrt(v_perp_x**2 + v_perp_y**2)
        df[f'relative_vel_tanget_{m_id}'] = v_tangent
        df[f'relative_vel_perp_{m_id}'] = v_perp

        ## relative distance scaled
        #Distance between main centroids, divided by length of major axis of each mouse
        df[f'scaled_main_centroid_distance_by_ellipse_major_{m_id}'] = dm/df[f'ellipse_major_{m_id}']

    return df
    
def _compute_relative_body_angles(df):

    for idx, m_id in enumerate(mouse_ids):
        #Compute vector connecting two centroids
        dx1 = df[f'centroid_all_{mouse_ids[1-idx]}_x'] - df[f'centroid_all_{mouse_ids[idx]}_x']
        dy1 = df[f'centroid_all_{mouse_ids[1-idx]}_y'] - df[f'centroid_all_{mouse_ids[idx]}_y']

        #Relative angle between body of mouse and line connecting two centroids
        dx2 = df[f'centroid_head_{m_id}_x'] - df[f'centroid_body_{m_id}_x']
        dy2 = df[f'centroid_head_{m_id}_y'] - df[f'centroid_body_{m_id}_y']            
        diff1 = np.vstack((dx1, dy1)).T
        diff2 = np.vstack((dx2, dy2)).T
        cosine_angle = np.sum(diff1*diff2, axis = 1) / (np.linalg.norm(diff1, axis = 1) * np.linalg.norm(diff2, axis = 1))
        df[f'angle_head_body_centroid_{m_id}'] = np.arccos(cosine_angle)

        #Angle between head orientation of one mouse and line connecting two centroids
        dx2 = df[f'{m_id}_x_nose'] - df[f'{m_id}_x_neck']
        dy2 = df[f'{m_id}_x_nose'] - df[f'{m_id}_y_neck']            
        diff1 = np.vstack((dx1, dy1)).T
        diff2 = np.vstack((dx2, dy2)).T
        cosine_angle = np.sum(diff1*diff2, axis = 1) / (np.linalg.norm(diff1, axis = 1) * np.linalg.norm(diff2, axis = 1))
        df[f'angle_head_centroid_{m_id}'] = np.arccos(cosine_angle)

        #Just threshold on if the angle is less than pi/4 radians
        df[f'{mouse_ids[1-idx]}_in_view_of_{mouse_ids[idx]}'] = (cosine_angle > 1/np.sqrt(2)).astype(float)

    return df

def _compute_iou(df):

    mins = {}
    maxs = {}

    for m_id in mouse_ids:
        for xy in xy_ids:
            colnames = ['_'.join([m_id, xy, bp]) for bp in bodypart_ids]
            mins['_'.join([m_id, xy])] = np.min(df[colnames], axis = 1)
            maxs['_'.join([m_id, xy])] = np.max(df[colnames], axis = 1)

    dx = np.minimum(maxs['mouse_0_x'], maxs['mouse_1_x']) - np.maximum(mins['mouse_0_x'], mins['mouse_1_x'])
    dy = np.minimum(maxs['mouse_0_y'], maxs['mouse_1_y']) - np.maximum(mins['mouse_0_y'], mins['mouse_1_y'])
    dx = np.maximum(0, dx)
    dy = np.maximum(0, dy)

    bb1_area = (maxs['mouse_0_x'] - mins['mouse_0_x'])*(maxs['mouse_0_y'] - mins['mouse_0_y'])
    bb2_area = (maxs['mouse_1_x'] - mins['mouse_1_x'])*(maxs['mouse_1_y'] - mins['mouse_1_y'])
    intersection_area = dx*dy
    iou = intersection_area / (bb1_area + bb2_area - intersection_area)
    df['iou'] = iou
    return df

def make_features_mars(df):

    #MARS-inspired feature set to play with
    feature_set_name = 'features_mars'

    features_df = df.copy()

    #######################
    ## Position features ##
    #######################
    features_df = _compute_centroid(features_df, 'all')
    features_df = _compute_centroid(features_df, 'head', ['nose', 'l_ear', 'r_ear', 'neck'])
    features_df = _compute_centroid(features_df, 'hips', ['l_hip', 'tail_base', 'r_hip'])
    features_df = _compute_centroid(features_df, 'body', ['neck', 'l_hip', 'r_hip', 'tail_base'])

    ##Distance from centroid of the mouse to the closest vertical edge
    ## and closest horizontal edge
    ##Distance to the closest edge
    for m_id in mouse_ids:
        features_df[f'centroid_all_{m_id}_x_inverted'] = 1024 - features_df[f'centroid_all_{m_id}_x']
        features_df[f'centroid_all_{m_id}_y_inverted'] = 570 - features_df[f'centroid_all_{m_id}_y']
        features_df[f'{m_id}_closest_x'] = np.min(features_df[[f'centroid_all_{m_id}_x_inverted', f'centroid_all_{m_id}_x']], axis = 1)
        features_df[f'{m_id}_closest_y'] = np.min(features_df[[f'centroid_all_{m_id}_y_inverted', f'centroid_all_{m_id}_y']], axis = 1)
        features_df[f'{m_id}_closest'] = np.min(features_df[[f'{m_id}_closest_x', f'{m_id}_closest_y']], axis = 1)
        features_df = features_df.drop(columns = [f'centroid_all_{m_id}_x_inverted', f'centroid_all_{m_id}_y_inverted'])

    #####################
    #Appearance features#
    #####################

    ## absolute orientation of mice
    features_df = _compute_abs_angle(features_df, 'head_hips', ['centroid_head', 'centroid_hips'])
    features_df = _compute_abs_angle(features_df, 'head_nose', ['neck', 'nose'], centroid = False)
    features_df = _compute_abs_angle(features_df, 'tail_neck', ['tail_base', 'neck'], centroid = False)
    ## relative orientation of mice
    features_df = _compute_rel_angle(features_df, 'l_ear_neck_r_ear', ['l_ear', 'neck', 'r_ear'])
    ## major axis len, minor axis len of ellipse fit to mouses body
    features_df = _compute_ellipsoid(features_df)
    for m_id in mouse_ids:
        ## ratio of major and minor
        features_df[f'ellipse_ratio_{m_id}'] = features_df[f'ellipse_minor_{m_id}']/features_df[f'ellipse_major_{m_id}']
        ## area of ellipse
        features_df[f'ellipse_area_{m_id}'] = features_df[f'ellipse_minor_{m_id}']*features_df[f'ellipse_major_{m_id}']

    #####################
    #Locomotion features#
    #####################

    features_df = _compute_kinematics(features_df, ['all', 'head', 'hips', 'body'])

    #################
    #Social features#
    #################

    features_df = _compute_relative_body_motions(features_df)
    features_df = _compute_relative_body_angles(features_df)

    #Intersection of union of bounding boxes of two mice
    features_df = _compute_iou(features_df)

    ## ratio of areas of ellipses of the mice
    features_df[f'ellipse_area_ratio'] = features_df[f'ellipse_area_{mouse_ids[0]}']/features_df[f'ellipse_area_{mouse_ids[1]}']

    ## distance between all pairs of keypoints of each mouse
    features_df, _, _ = make_features_distances_shifted(features_df)

    #Remove base features
    #features_df = features_df.drop(columns = colnames)

    ##Clean up seq_id columns
    features_df, reversemap = boiler_plate(features_df)

    return features_df, reversemap, feature_set_name    

class Args(object):
    def __init__(self):
        self.features = 'mars'
        self.compute_test = False

#Create a default set of parameters if we can't parse from the command line
#i.e. we're running interactively in python
args = Args()

def main(args):

    supported_features = {'differences': make_features_differences, 
                          'distances': make_features_distances, 
                          'distances_shifted': make_features_distances_shifted,
                          'mars': make_features_mars}

    if args.features not in supported_features:
        print("Features not found. Select one of", list(supported_features.keys()))
        return

    feature_maker = supported_features[args.features]

    train_df = pd.read_csv('./data/intermediate/train_df.csv')
    train_features, train_map, _ = feature_maker(train_df)
    train_features.to_csv(f'./data/intermediate/train_{name}.csv', index = False)

    if args.compute_test:
        test_df = pd.read_csv('./data/intermediate/test_df.csv')
        test_features, test_map, name = feature_maker(test_df)
        test_features.to_csv(f'./data/intermediate/test_{name}.csv', index = False)
        with open(f'./data/intermediate/test_map_{name}.pkl', 'wb') as handle:
            pickle.dump(test_map, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == "__main__":

    try:
        args = parser.parse_args()
    except SystemExit:
        print("Cannot parse arguments. Resorting to the inbuilt defaults.")

    main(args)
