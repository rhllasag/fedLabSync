"""
Script for running preprocessing on a all or a subset of data described in
a U-Time project directory, loading all selected files with specified:
    - scaler
    - strip function
    - quality control function
    - channel selection
    - re-sampling

Loaded (and processed) files according to those settings are then saved to a parquet.

The produced, pre-processed parquet archive of data may be consumed by the 'train.py' script setting flag --preprocessed.

This script should be called form within a fedLabSync (utime) project directory
"""

import logging
import os
import numpy as np
import h5py
from tqdm import trange
from sklearn.cluster import KMeans
from sklearn import preprocessing
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import pyarrow as pa
import pyarrow.parquet as pq
from numpy.random import seed
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from fedLabSync import Defaults
from fedLabSync.utils import flatten_lists_recursively
from fedLabSync.hyperparameters import YAMLHParams
from fedLabSync.utils.scriptutils import assert_project_folder, get_splits_from_all_datasets, add_logging_file_handler

from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.models import model_from_json





logger = logging.getLogger(__name__)

def add_RUL(cycle, SoF, EoL, Rc):
    if cycle <= SoF:
      return Rc
    elif SoF <= EoL:
      return EoL - cycle
    

def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='TODO')  # TODO
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Dataset path")
    parser.add_argument("--elbow_point", type=int, required=True,
                        help="Elbow point value RC")
    parser.add_argument("--FD00x", type=int, required=True,
                        help="FD00x split")
    parser.add_argument("--operating_regimes", type=int, required=True,
                        help="Operating Regimes")
    parser.add_argument("--out_path", type=str, required=True,
                        help="Path to output parquet file")
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite previous pre-processed data')
    return parser


def copy_dataset_hparams(hparams, hparams_out_path):
    groups_to_save = ('select_channels',
                      'alternative_select_channels',
                      'channel_sampling_groups')
    hparams = hparams.save_current(hparams_out_path, return_copy=True)
    groups = list(hparams.keys())
    for group in groups:
        if group not in groups_to_save:
            hparams.delete_group(group)
    hparams.save_current()


def add_dataset_entry(hparams_out_path, h5_path,
                      split_identifier, period_length_sec):
    field = f"{split_identifier}_data:\n" + \
            f"  data_dir: {h5_path}\n" + \
            f"  period_length: {period_length_sec}\n" + \
            f"  time_unit: SECOND\n" + \
            f"  identifier: {split_identifier.upper()}\n\n"
    with open(hparams_out_path, "a") as out_f:
        out_f.write(field)


def preprocess_study(h5_file_group, study):
    """
    TODO

    Args:
        h5_file_group:
        study:

    Returns:
        None
    """
    # Create groups
    study_group = h5_file_group.create_group(study.identifier)
    psg_group = study_group.create_group("PSG")
    with study.loaded_in_context(allow_missing_channels=True):
        X, y = study.get_all_periods()
        for chan_ind, channel_name in enumerate(study.select_channels):
            # Create PSG channel datasets
            psg_group.create_dataset(channel_name.original_name,
                                     data=X[..., chan_ind].ravel())
        # Create hypnogram dataset
        study_group.create_dataset("hypnogram", data=y)

        # Create class --> index lookup groups
        cls_to_indx_group = study_group.create_group('class_to_index')
        dtype = np.dtype('uint16') if len(y) <= 65535 else np.dtype('uint32')
        classes = study.hypnogram.classes
        for class_ in classes:
            inds = np.where(y == class_)[0].astype(dtype)
            cls_to_indx_group.create_dataset(
                str(class_), data=inds
            )

        # Set attributes, currently only sample rate is (/may be) used
        study_group.attrs['sample_rate'] = study.sample_rate

def preprocess_fd00x(dataset_path, out_path ,number_of_dataset,RC, clusters):
    # Read training data and sort by id and cycle
    train_df = pd.read_csv(dataset_path+'train_FD00'+str(number_of_dataset)+'.txt', sep=" ", header=None)
    train_df.drop(train_df.columns[[26, 27]], axis=1, inplace=True)
    train_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                        's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                        's15', 's16', 's17', 's18', 's19', 's20', 's21']
    train_df = train_df.sort_values(['id','cycle'])


    # Read testing data - It is the aircraft engine operating data without failure events recorded.
    test_df = pd.read_csv(dataset_path+'test_FD00'+str(number_of_dataset)+'.txt', sep=" ", header=None)
    test_df.drop(test_df.columns[[26, 27]], axis=1, inplace=True)
    test_df.columns = ['id', 'cycle', 'setting1', 'setting2', 'setting3', 's1', 's2', 's3',
                        's4', 's5', 's6', 's7', 's8', 's9', 's10', 's11', 's12', 's13', 's14',
                        's15', 's16', 's17', 's18', 's19', 's20', 's21']

    #-------------------DATA PREPROCESSING-----------------------#


    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    #Obtaining data of the first trajectory
    first_trajectory=train_df.loc[train_df['id'] == 1]
    #Select only the first 32-th cycles to display 
    altitude = pd.DataFrame(first_trajectory,columns=['cycle','setting1'])
    altitude=altitude[:]
    mach_number = pd.DataFrame(first_trajectory,columns=['cycle','setting2'])
    mach_number=mach_number[:]
    throttle_resolver_angle = pd.DataFrame(first_trajectory,columns=['cycle','setting3'])
    throttle_resolver_angle=throttle_resolver_angle[:]


    # Piece-wise degradation function 

    # Data Labeling training set - generate column RUL(Remaining Usefull Life or Time to Failure)
    rul = pd.DataFrame(train_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'EoL']
    rul['Rc']=RC
    rul['SoF']=rul['EoL']-rul['Rc']
    train_df = train_df.merge(rul, on=['id'], how='left')
    train_df['RUL'] = train_df['EoL'] - train_df['cycle']
    train_df['RUL'] = train_df.apply(lambda row: add_RUL(row['cycle'], row['SoF'], row['EoL'], row['Rc']), axis=1)
    train_df.drop('EoL', axis=1, inplace=True)
    train_df.drop('SoF', axis=1, inplace=True)
    train_df.drop('Rc', axis=1, inplace=True)


    # Read ground truth data - It contains the information of true remaining cycles for each engine in the testing data.
    truth_df = pd.read_csv(dataset_path+'RUL_FD00'+str(number_of_dataset)+'.txt', sep=" ", header=None)
    truth_df.drop(truth_df.columns[[1]], axis=1, inplace=True)
    truth_df.columns = ['RUL']

    truth_df['id']=truth_df.index +1
    test_df = test_df.merge(truth_df, on=['id'], how='left')

    #test_df['RUL'] = test_df['RUL']
    max_cycle = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    max_cycle.columns = ['id', 'max_cycle']
    test_df = test_df.merge(max_cycle, on=['id'], how='left')
    test_df['RUL']= test_df['max_cycle']+test_df['RUL']-test_df['cycle']
    test_df.drop('max_cycle', axis=1, inplace=True)
    test_df['RUL'] = test_df['RUL'].apply(lambda x: RC if x >= RC else x)

    # Labeling regimes
    kmeans = KMeans(n_clusters=clusters).fit(train_df)
    centroids = kmeans.cluster_centers_

    regime_labels = pd.DataFrame(kmeans.labels_)
    train_df['regime']= regime_labels


    kmeans = KMeans(n_clusters=clusters).fit(test_df)
    centroids = kmeans.cluster_centers_

    regime_labels = pd.DataFrame(kmeans.labels_)
    test_df['regime']= regime_labels

    # Data normalization per regime: Training set

    indx = train_df['regime'].copy()
    id_ = train_df['id'].copy()
    rul_ = train_df['RUL'].copy()
    cycle_ = train_df['cycle'].copy()
    setting1_ = train_df['setting1'].copy()
    setting2_ = train_df['setting2'].copy()
    setting3_ = train_df['setting3'].copy()
    for indices in train_df.groupby('regime').groups.values():
        train_df.loc[indices] = (train_df.loc[indices]-train_df.loc[indices].mean())/train_df.loc[indices].std()
    train_df['regime'] = indx
    train_df['id'] = id_
    train_df['RUL'] = rul_
    train_df['cycle'] = cycle_
    train_df['setting1'] = setting1_
    train_df['setting2'] = setting2_
    train_df['setting3'] = setting3_
    train_df.drop('regime', axis=1, inplace=True)

    # Data normalization per regime: Testing set

    indx = test_df['regime'].copy()
    id_ = test_df['id'].copy()
    rul_ = test_df['RUL'].copy()
    cycle_ = test_df['cycle'].copy()
    setting1_ = test_df['setting1'].copy()
    setting2_ = test_df['setting2'].copy()
    setting3_ = test_df['setting3'].copy()
    for indices in test_df.groupby('regime').groups.values():
        test_df.loc[indices] = (2*(test_df.loc[indices]-test_df.loc[indices].min()))/(test_df.loc[indices].max()-test_df.loc[indices].min())-1
    test_df['regime'] = indx
    test_df['id'] = id_
    test_df['RUL'] = rul_
    test_df['cycle'] = cycle_
    test_df['setting1'] = setting1_
    test_df['setting2'] = setting2_
    test_df['setting3'] = setting3_
    test_df.drop('regime', axis=1, inplace=True)

    # Drop Inf and Nan columns and data normalization: Training set
    train_df=train_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    cols_normalize = train_df.columns.difference(['id','cycle','s1','s2','s3','s4','s5','s6','s7', 's8','s9','s10','s11','s12','s13','s14','s15','s16','s17','s18','s19','s20','s21'])
    min_max_scaler = preprocessing.MinMaxScaler()
    norm_train_df = pd.DataFrame(min_max_scaler.fit_transform(train_df[cols_normalize]), 
                                columns=cols_normalize, 
                                index=train_df.index)
    join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df) 
    train_df = join_df.reindex(columns = train_df.columns)

    # Drop Inf and Nan columns and data normalization: Testing set
    test_df=test_df.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

    norm_test_df = pd.DataFrame(min_max_scaler.transform(test_df[cols_normalize]), 
                                columns=cols_normalize, 
                                index=test_df.index)
    test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
    test_df = test_join_df.reindex(columns = test_df.columns)
    test_df = test_df.reset_index(drop=True)

    
    table_test = pa.Table.from_pandas(test_df)
    table_train = pa.Table.from_pandas(train_df)

    pq.write_table(table_test, out_path+'test_fd00'+str(number_of_dataset)+'.parquet')
    pq.write_table(table_train, out_path +'train_fd00'+str(number_of_dataset)+'.parquet')

def run(args):
    """
    Run the script according to args - Please refer to the argparser.

    args:
        args:    (Namespace)  command-line arguments
    """
    project_dir = os.path.abspath("./")
    # Check if file exists, and overwrite if specified
    if os.path.exists(project_dir+args.out_path):
        if args.overwrite:
            os.remove(project_dir+args.out_path)
            os.mkdir(project_dir+args.out_path)
        else:
            from sys import exit
            logger.info(f"Out file at {project_dir+args.out_path} exists, and --overwrite was not set")
            exit(0)
    else:
        os.mkdir(project_dir+args.out_path)
    # Check if file exists, and overwrite if specified
    if os.path.exists(project_dir+args.dataset_path):
        print(project_dir+args.dataset_path)
        preprocess_fd00x(project_dir+args.dataset_path, project_dir+args.out_path, args.FD00x, args.elbow_point, args.operating_regimes)

def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = parser.parse_args(args)
    run(args)


if __name__ == "__main__":
    entry_func()
