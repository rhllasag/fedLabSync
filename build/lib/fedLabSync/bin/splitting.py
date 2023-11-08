import pyarrow as pa
import pyarrow.parquet as pq
from argparse import ArgumentParser
import logging
import os
from sys import exit
logger = logging.getLogger(__name__)
import h5py
import pandas as pd
import random
import numpy as np

def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='TODO')  # TODO
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Dataset path")
    parser.add_argument("--FD00x", type=int, required=True,
                        help="FD00x split")
    parser.add_argument("--model", type=str, required=True,
                        help="model", default="mlp")
    parser.add_argument("--val_percentage", type=float, required=True,
                        help="val_percentage", default=0.80)
    parser.add_argument("--features_percentage", type=float,
                        help="features_percentage", default=0.80)
    parser.add_argument("--nodes", type=int, required=True,
                        help="nodes", default=1)
    parser.add_argument("--seed", type=int,
                        help="seed", default=1)
    parser.add_argument("--sequence_length", type=int,
                        help="sequence_length", default=55)
    parser.add_argument("--overwrite", action='store_true',
                        help='Overwrite previous pre-processed data')
    return parser

def gen_labels(id_df, seq_length, label):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # For one id I put all the labels in a single matrix.
    # For example:
    # [[1]
    # [4]
    # [1]
    # [5]
    # [9]
    # ...
    # [200]] 
    data_matrix = id_df[label].values
    num_elements = data_matrix.shape[0]
    # I have to remove the first seq_length labels
    # because for one id the first sequence of seq_length size have as target
    # the last label (the previus ones are discarded).
    # All the next id's sequences will have associated step by step one label as target.
    return data_matrix[seq_length:num_elements, :]

def gen_sequence(id_df, seq_length, seq_cols):
    """ Only sequences that meet the window-length are considered, no padding is used. This means for testing
    we need to drop those which are below the window-length. An alternative would be to pad sequences so that
    we can use shorter ones """
    # for one id I put all the rows in a single matrix
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    # Iterate over two lists in parallel.
    # For example id1 have 192 rows and sequence_length is equal to 50
    # so zip iterate over two following list of numbers (0,112),(50,192)
    # 0 50 -> from row 0 to row 50
    # 1 51 -> from row 1 to row 51
    # 2 52 -> from row 2 to row 52
    # ...
    # 111 191 -> from row 111 to 191
    for start, stop in zip(range(0, num_elements-seq_length), range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]
    
def save_h5_files(out_path, dataset, name):
    # Save numpy array 
    with h5py.File(out_path+name+'-centralized.h5', 'w') as f:
        f.create_dataset(name, data=dataset)


def split_data(out_path, number_of_dataset, model, nodes, sequence_length, seed, val_percentage, features_percentage):
    train_df = pq.read_table(out_path + 'train_fd00' +
                             str(number_of_dataset)+'.parquet').to_pandas()
    test_df = pq.read_table(out_path + 'test_fd00' +
                            str(number_of_dataset)+'.parquet').to_pandas()
    
    # Naming columns to training the model 
    sensor_cols = ['s2','s3','s4','s6','s7', 's8','s9','s10','s11','s12','s13','s14','s15','s17','s20','s21']
    sequence_cols = []
    sequence_cols.extend(sensor_cols)
    
    if nodes==1:
        engines = train_df['id'].max()
        ids = [*range(1,engines)]
        random.shuffle(ids)
        training_ids = ids[:int(len(ids)*val_percentage)] 
        validation_ids = ids[int(len(ids)*val_percentage):int(len(ids))]
        if model =="mlp":
            routes_train = {}
            routes_val = {}
            for x in range(len(training_ids)):
                routes_train[x] = train_df.loc[train_df['id'] == training_ids[x]]
            for x in range(len(validation_ids)):
                routes_val[x] = train_df.loc[train_df['id'] == training_ids[x]]
                
            X_train=routes_train[0][sequence_cols]
            y_train=routes_train[0]['RUL']
            X_val=routes_val[0][sequence_cols]
            y_val=routes_val[0]['RUL']
            
            for route_train in routes_train:
                if route_train != 0:
                    X_train=X_train.append(routes_train[route_train][sequence_cols],ignore_index=True)
                    y_train=y_train.append(routes_train[route_train]['RUL'],ignore_index=True)
            for route_val in routes_val:
                if route_val != 0:
                    X_val=X_val.append(routes_val[route_val][sequence_cols],ignore_index=True)
                    y_val=y_val.append(routes_val[route_val]['RUL'],ignore_index=True)
            
            if os.path.exists(out_path+"data-centralized-"+str(model)+"/"):
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",X_train.to_numpy(), "X_train")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",y_train.to_numpy(), "y_train")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",X_val.to_numpy(), "X_val")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",y_val.to_numpy(), "y_val")
            else:
                os.mkdir(out_path+"data-centralized-"+str(model)+"/")
                # Save .h5 
                save_h5_files(out_path+"data-centralized-"+str(model)+"/", X_train.to_numpy(), "X_train")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/", y_train.to_numpy(), "y_train")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",X_val.to_numpy(), "X_val")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",y_val.to_numpy(), "y_val")
        if model =="cnn":
            training_df = train_df.loc[train_df["id"].isin(training_ids)]
            validation_df = train_df.loc[train_df["id"].isin(validation_ids)]

            seq_gen_train = (list(gen_sequence(training_df[training_df['id']==id], sequence_length, sequence_cols)) for id in training_df['id'].unique())
            seq_gen_val = (list(gen_sequence(validation_df[validation_df['id']==id], sequence_length, sequence_cols)) for id in validation_df['id'].unique())

            # generate sequences and convert to numpy array
            seq_array_train = np.concatenate(list(seq_gen_train)).astype(np.float64)
            seq_array_val = np.concatenate(list(seq_gen_val)).astype(np.float64)

            seq_array_train = seq_array_train.reshape(seq_array_train.shape[0],seq_array_train.shape[1] , seq_array_train.shape[2],1)
            seq_array_val = seq_array_val.reshape(seq_array_val.shape[0],seq_array_val.shape[1] , seq_array_val.shape[2],1)

            print(seq_array_train.shape[1], seq_array_train.shape[2],1)

            label_gen_train = [gen_labels(training_df[training_df['id']==id], sequence_length, ['RUL']) 
                               for id in training_df['id'].unique()]
            label_gen_val = [gen_labels(validation_df[validation_df['id']==id], sequence_length, ['RUL']) 
                             for id in validation_df['id'].unique()]

            label_array_train = np.concatenate(label_gen_train).astype(np.float64)
            label_array_val = np.concatenate(label_gen_val).astype(np.float64)

            if os.path.exists(out_path+"data-centralized-"+str(model)+"/"):
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",seq_array_train, "X_train")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",seq_array_val, "X_val")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",label_array_train, "y_train")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",label_array_val, "y_val")
            else:
                os.mkdir(out_path+"data-centralized-"+str(model)+"/")
                # Save .h5 
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",seq_array_train, "X_train")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",seq_array_val, "X_val")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",label_array_train, "y_train")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",label_array_val, "y_val")
        if model =="utime":
            training_df = train_df.loc[train_df["id"].isin(training_ids)]
            validation_df = train_df.loc[train_df["id"].isin(validation_ids)]
            
            # generate sequences X
            seq_gen_X_train = (list(gen_sequence(training_df[training_df['id']==id], sequence_length, sequence_cols)) for id in training_df['id'].unique())
            seq_gen_X_val = (list(gen_sequence(validation_df[validation_df['id']==id], sequence_length, sequence_cols)) for id in validation_df['id'].unique())

            # convert X to numpy array
            seq_X_train = np.concatenate(list(seq_gen_X_train)).astype(np.float64)
            seq_X_val = np.concatenate(list(seq_gen_X_val)).astype(np.float64)

            seq_X_train = seq_X_train.reshape(seq_X_train.shape[0],seq_X_train.shape[1] , seq_X_train.shape[2],1)
            seq_X_val = seq_X_val.reshape(seq_X_val.shape[0],seq_X_val.shape[1] , seq_X_val.shape[2],1)
            
            # generate sequences Y
            seq_gen_Y_train = (list(gen_sequence(training_df[training_df['id']==id], sequence_length, ['RUL'])) for id in training_df['id'].unique())
            seq_gen_Y_val = (list(gen_sequence(validation_df[validation_df['id']==id], sequence_length, ['RUL'])) for id in validation_df['id'].unique())

            # convert Y to numpy array
            seq_Y_train = np.concatenate(list(seq_gen_Y_train)).astype(np.float64)
            seq_Y_val = np.concatenate(list(seq_gen_Y_val)).astype(np.float64)

            seq_Y_train = seq_Y_train.reshape(seq_Y_train.shape[0],seq_Y_train.shape[1] , seq_Y_train.shape[2])
            seq_Y_val = seq_Y_val.reshape(seq_Y_val.shape[0],seq_Y_val.shape[1] , seq_Y_val.shape[2])
            
            if os.path.exists(out_path+"data-centralized-"+str(model)+"/"):
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",seq_X_train, "X_train")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",seq_X_val, "X_val")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",seq_Y_train, "y_train")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",seq_Y_val, "y_val")
            else:
                os.mkdir(out_path+"data-centralized-"+str(model)+"/")
                # Save .h5 
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",seq_X_train, "X_train")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",seq_X_val, "X_val")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",seq_Y_train, "y_train")
                save_h5_files(out_path+"data-centralized-"+str(model)+"/",seq_Y_val, "y_val")
    if nodes!=1:
        random.seed(seed)
        engines = train_df['id'].max()
        ids = [*range(1,engines)]
        random.shuffle(ids)
        data_nodes=[ids[x:x+len(ids)//nodes] for x in range(0, len(ids), len(ids)//nodes)]  
        if model =="mlp":
            for data_node in data_nodes:
                routes_train = {}
                routes_val = {}
                random.shuffle(sequence_cols)
                sequence_cols_ = sequence_cols[:int(len(sequence_cols)*features_percentage)]
                training_ids = ids[:int(len(data_node)*val_percentage)] 
                validation_ids = ids[int(len(data_node)*val_percentage):int(len(data_node))]
                
                for x in range(len(training_ids)):
                    routes_train[x] = train_df.loc[train_df['id'] == training_ids[x]]
                for x in range(len(validation_ids)):
                    routes_val[x] = train_df.loc[train_df['id'] == training_ids[x]]
                    
                # Get X and Y data
                X_train=routes_train[0][sequence_cols_]
                y_train=routes_train[0]['RUL']
                X_val=routes_val[0][sequence_cols_]
                y_val=routes_val[0]['RUL']
                for route_train in routes_train:
                    if route_train != 0:
                        X_train=X_train.append(routes_train[route_train][sequence_cols_],ignore_index=True)
                        y_train=y_train.append(routes_train[route_train]['RUL'],ignore_index=True)
                for route_val in routes_val:
                    if route_val != 0:
                        X_val=X_val.append(routes_val[route_val][sequence_cols_],ignore_index=True)
                        y_val=y_val.append(routes_val[route_val]['RUL'],ignore_index=True)
                if os.path.exists(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/"):
                    save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",X_train, "X_train")
                    save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",X_val, "X_val")
                    save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",y_train, "y_train")
                    save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",y_val, "y_val")
                else:
                    if os.path.exists(out_path+"data-decentralized-"+str(model)+"/"):
                        if os.path.exists(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))):
                            save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",X_train, "X_train")
                            save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",X_val, "X_val")
                            save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",y_train, "y_train")
                            save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",y_val, "y_val")
                        else:
                            os.mkdir(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/")
                            save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",X_train, "X_train")
                            save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",X_val, "X_val")
                            save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",y_train, "y_train")
                            save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",y_val, "y_val")
                    else: 
                        os.mkdir(out_path+"data-decentralized-"+str(model)+"/")
                        os.mkdir(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/")
                        save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",X_train, "X_train")
                        save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",X_val, "X_val")
                        save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",y_train, "y_train")
                        save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",y_val, "y_val")
        #
        if model =="utime":
            for data_node in data_nodes:
                random.shuffle(sequence_cols)
                sequence_cols_ = sequence_cols[:int(len(sequence_cols)*features_percentage)]
                training_ids = ids[:int(len(data_node)*val_percentage)] 
                validation_ids = ids[int(len(data_node)*val_percentage):int(len(data_node))] 

                training_df = train_df.loc[train_df["id"].isin(training_ids)]
                validation_df = train_df.loc[train_df["id"].isin(validation_ids)]
                
                # generate sequences X
                seq_gen_X_train = (list(gen_sequence(training_df[training_df['id']==id], sequence_length, sequence_cols_)) for id in training_df['id'].unique())
                seq_gen_X_val = (list(gen_sequence(validation_df[validation_df['id']==id], sequence_length, sequence_cols_)) for id in validation_df['id'].unique())

                # convert X to numpy array
                seq_X_train = np.concatenate(list(seq_gen_X_train)).astype(np.float64)
                seq_X_val = np.concatenate(list(seq_gen_X_val)).astype(np.float64)

                seq_X_train = seq_X_train.reshape(seq_X_train.shape[0],seq_X_train.shape[1] , seq_X_train.shape[2],1)
                seq_X_val = seq_X_val.reshape(seq_X_val.shape[0],seq_X_val.shape[1] , seq_X_val.shape[2],1)
                
                # generate sequences Y
                seq_gen_Y_train = (list(gen_sequence(training_df[training_df['id']==id], sequence_length, ['RUL'])) for id in training_df['id'].unique())
                seq_gen_Y_val = (list(gen_sequence(validation_df[validation_df['id']==id], sequence_length, ['RUL'])) for id in validation_df['id'].unique())

                # convert Y to numpy array
                seq_Y_train = np.concatenate(list(seq_gen_Y_train)).astype(np.float64)
                seq_Y_val = np.concatenate(list(seq_gen_Y_val)).astype(np.float64)

                seq_Y_train = seq_Y_train.reshape(seq_Y_train.shape[0],seq_Y_train.shape[1] , seq_Y_train.shape[2])
                seq_Y_val = seq_Y_val.reshape(seq_Y_val.shape[0],seq_Y_val.shape[1] , seq_Y_val.shape[2])
                if os.path.exists(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/"):
                    save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",seq_X_train, "X_train")
                    save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",seq_X_val, "X_val")
                    save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",seq_Y_train, "y_train")
                    save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",seq_Y_val, "y_val")
                else:
                    if os.path.exists(out_path+"data-decentralized-"+str(model)+"/"):
                        if os.path.exists(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))):
                            save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",seq_X_train, "X_train")
                            save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",seq_X_val, "X_val")
                            save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",seq_Y_train, "y_train")
                            save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",seq_Y_val, "y_val")
                        else:
                            os.mkdir(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/")
                            save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",seq_X_train, "X_train")
                            save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",seq_X_val, "X_val")
                            save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",seq_Y_train, "y_train")
                            save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",seq_Y_val, "y_val")
                    else: 
                        os.mkdir(out_path+"data-decentralized-"+str(model)+"/")
                        os.mkdir(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/")
                        save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",seq_X_train, "X_train")
                        save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",seq_X_val, "X_val")
                        save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",seq_Y_train, "y_train")
                        save_h5_files(out_path+"data-decentralized-"+str(model)+"/node"+str(data_nodes.index(data_node))+"/",seq_Y_val, "y_val")
                    # Save .h5 
def run(args):
    """
    Run the script according to args - Please refer to the argparser.

    args:
        args:    (Namespace)  command-line arguments
    """
    
    project_dir = os.path.abspath("./")
    # Check if file exists, and overwrite if specified
    if os.path.exists(project_dir+args.dataset_path):
        if args.overwrite:
            os.remove(project_dir+args.dataset_path)
            os.mkdir(project_dir+args.dataset_path)
        else:
            split_data(project_dir+args.dataset_path, args.FD00x, args.model, args.nodes, args.sequence_length, args.seed, args.val_percentage, args.features_percentage)            
    else:
        os.mkdir(project_dir+args.dataset_path)
        logger.info(
                f"Out file at {project_dir+args.dataset_path} exists, and --overwrite was not set")
        exit(0)


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = parser.parse_args(args)
    run(args)


if __name__ == "__main__":
    entry_func()
