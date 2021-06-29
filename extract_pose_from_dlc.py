import numpy as np
import pandas as pd
import hashlib

files_in = ['./data/dlc/e3v813a-20210610T120637-121213DLC_resnet_50_pilot_studyJun15shuffle1_40000_bx_filtered.csv',
            './data/dlc/e3v813a-20210610T121558-122141DLC_resnet_50_pilot_studyJun15shuffle1_40000_bx_filtered.csv',
            './data/dlc/e3v813a-20210610T122332-122642DLC_resnet_50_pilot_studyJun15shuffle1_40000_bx_filtered.csv',
            './data/dlc/e3v813a-20210610T122758-123309DLC_resnet_50_pilot_studyJun15shuffle1_40000_bx_filtered.csv',
            './data/dlc/e3v813a-20210610T123521-124106DLC_resnet_50_pilot_studyJun15shuffle1_40000_bx_filtered.csv']

fn_out = '/home/blansdell/projects/mabetask1_ml/data/test_inference.npy'

#Load tracks from DLC
test_out = {'sequences':{}}
for fn_in in files_in:
    print('Loading', fn_in)
    dlc_tracks = np.array(pd.read_csv(fn_in, header = [0,1,2,3]))[:,1:]
    vid_name = hashlib.md5(fn_in.encode()).hexdigest()[:8]

    #Adult, then juvenile. Same as resident then intruder... or should be flipped?
    n_rows = dlc_tracks.shape[0]
    selected_cols = [[3*i, 3*i+1] for i in range(14)]
    selected_cols = [j for i in selected_cols for j in i]
    dlc_tracks = dlc_tracks[:,selected_cols]

    #Put in shape:
    # (frame, mouse_id, x/y coord, body part)
    dlc_tracks = dlc_tracks.reshape((n_rows, 2, 7, 2))
    keypoints = dlc_tracks.transpose([0, 1, 3, 2])

    test_out['sequences'][vid_name] = {'annotator_id': 0, 'keypoints': keypoints}

#Save as a numpy file
np.save(fn_out, test_out)