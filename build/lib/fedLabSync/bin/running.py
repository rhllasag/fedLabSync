import pyarrow as pa
import pyarrow.parquet as pq
from argparse import ArgumentParser
import logging
import os
from sys import exit
logger = logging.getLogger(__name__)
import h5py
import numpy as np
import tensorflow as tf
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv1D, Conv2D, MaxPooling2D, Reshape
from keras.optimizers import SGD
from keras import optimizers
from tensorflow.keras import layers
from tensorflow import  keras
import keras
import keras.backend as K
from keras.utils import plot_model


def get_argparser():
    """
    Returns an argument parser for this script
    """
    parser = ArgumentParser(description='TODO')  # TODO
    parser.add_argument("--dataset_path", type=str, required=True,
                        help="Dataset path")
    parser.add_argument("--model", type=str, required=True,
                        help="model", default="mlp")
    parser.add_argument("--nodes", type=int, required=True,
                        help="nodes", default=1)
    parser.add_argument("--input_signals", type=int, required=True,
                        help="features", default=16)
    return parser

def read_h5_file(resources_path, name):
    # Read numpy array 
    hf = h5py.File(resources_path+name+"-centralized.h5", 'r')
    return np.array(hf[name][:])

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 


def add_RUL(cycle, SoF, EoL, Rc):
    if cycle <= SoF:
      return Rc
    elif SoF <= EoL:
      return EoL - cycle

def create_model_mlp(input_signals):
    model = Sequential()
    model.add(Dense(10, activation="sigmoid", input_shape=(input_signals,)))
    model.add(Dense(1, activation='relu'))
    optimizer = SGD(lr=0.001)
    model.compile(loss=rmse,optimizer=optimizer,metrics=['mae','mse',rmse])
    return model

def create_model_cnn(input_shape):
    optimizer = optimizers.Adam(lr=0.0001)
    model = Sequential()
    model.add(Conv2D(10, (10,1),padding='same', activation='tanh', input_shape=input_shape))
    model.add(Conv2D(10, (10,1),padding='same', activation='tanh'))
    model.add(Conv2D(10, (10,1),padding='same', activation='tanh'))
    model.add(Conv2D(10, (10,1),padding='same', activation='tanh'))
    model.add(Conv2D(1, (3,1),padding='valid', activation='tanh'))
    model.add(Flatten())
    model.add(Dense(100, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='relu'))
    model.compile(loss=rmse,optimizer=optimizer,metrics=['mae','mse',rmse])
    return model

def save_model_weights(resources_path, model):
    model.save_weights(resources_path+"model.h5")
    # serialize model to JSON
    model_json = model.to_json()
    with open(resources_path+'model.json', 'w') as json_file:
        json_file.write(model_json)

def load_data_and_run(dataset_path, model, nodes, input_signals):
    
    if nodes==1:
        resources_path = dataset_path+"data-centralized-"+str(model)+"/"
        X_train=read_h5_file(resources_path, "X_train")
        y_train=read_h5_file(resources_path, "y_train")
        if model =="mlp":
            model = create_model_mlp(input_signals)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_rmse', patience=8)
            early_history = model.fit(X_train, y_train, 
                        epochs=580, validation_split=0.15,
                        callbacks=[early_stop])
            save_model_weights(resources_path, model)
        if model=="cnn":
            X_val=read_h5_file(resources_path, "X_val")
            y_val=read_h5_file(resources_path, "y_val")

            input_shape = (X_val.shape[1], X_val.shape[2],1)
            model = create_model_cnn(input_shape)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_rmse', patience=10)

            early_history = model.fit(X_train, y_train,
                                      epochs=580,  validation_data=(X_val,y_val),callbacks=[early_stop])
            save_model_weights(resources_path, model)


def run(args):
    """
    Run the script according to args - Please refer to the argparser.

    args:
        args:    (Namespace)  command-line arguments
    """
    
    project_dir = os.path.abspath("./")
    # Check if file exists, and overwrite if specified
    if os.path.exists(project_dir+args.dataset_path):
            load_data_and_run(project_dir+args.dataset_path, args.model, args.nodes, args.input_signals)            
    else:
        logger.info(
                f"Out file at {project_dir+args.dataset_path} exists")
        exit(0)
    
    
    


def entry_func(args=None):
    # Get the script to execute, parse only first input
    parser = get_argparser()
    args = parser.parse_args(args)
    run(args)


if __name__ == "__main__":
    entry_func()
