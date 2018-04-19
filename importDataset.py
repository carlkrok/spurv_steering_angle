import glob
import os
import pandas as pd
import numpy as np




'''
    Function to read the dataset.

    dataset_dir: The directory of the dataset.

    test_set_regex: Used to mark a subset of the files as the test set, you
    can set the glob here. 'filename*.csv'

    Example: If you want to use the images from track4 as the test set:
    train, test = get_dataset_from_folder('lanes_training_data', 'track4*.csv')
'''

def get_dataset_from_folder(dataset_dir, test_set_regex):
    all_files = glob.glob(os.path.join(dataset_dir, "*.csv"))
    test_set = glob.glob(os.path.join(dataset_dir, test_set_regex))
    training_set = [x for x in all_files if x not in test_set]
    training_dfs = (pd.read_csv(f) for f in training_set)
    test_dfs = (pd.read_csv(f) for f in test_set)
    return pd.concat(training_dfs, ignore_index=True), pd.concat(test_dfs, ignore_index=True)


def load_simulator_data():

    csv_path = 'driving_log.csv'

    data_files_s = pd.read_csv(csv_path,
                             #names=['center','left','right','steering','throttle','break'],
                            index_col = False)

    data_files_s['direction'] = pd.Series('s', index=data_files_s.index)

    data_files_s.columns = ['center', 'left', 'right', 'steer', 'throttle', 'brake', 'speed','direction']

    rev_steer_s = np.array(data_files_s.steer,dtype=np.float32)

    steer_sm_s = rev_steer_s

    data_files_s['steer_sm'] = pd.Series(steer_sm_s, index=data_files_s.index)

    Print("Length of dataset: ",len(data_files_s))

    return data_files_s
