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
import pandas as pd
from tqdm import trange
import random
import json
from argparse import ArgumentParser
from sklearn import preprocessing
from tensorflow.keras.models import Model
from tensorflow.keras import regularizers
from tensorflow.keras.layers import (Input, BatchNormalization, Cropping2D,
                                     Concatenate, MaxPooling2D,
                                     UpSampling2D, ZeroPadding2D, Lambda,
                                     Conv2D, AveragePooling2D)
from psg_utils.utils import ensure_list_or_tuple
from typing import List
from tensorflow_addons import optimizers as addon_optimizers
from tensorflow_addons import activations as addon_activations
from tensorflow_addons import losses as addon_losses
from tensorflow_addons import metrics as addon_metrics
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

def create_encoder(in_,
                       depth,
                       pools,
                       filters,
                       kernel_size,
                       activation,
                       dilation,
                       padding,
                       complexity_factor,
                       regularizer=None,
                       dense_classifier_activation=None,
                       l2_reg = False,
                       n_crops = 0,
                       n_classes = 2,
                       cf=2.00,
                       transition_window = 1,
                       name="encoder",
                       name_prefix=""):
    
    
        name = "{}{}".format(name_prefix, name)
        residual_connections = []
        for i in range(depth):
            l_name = name + "_L%i" % i
            conv = Conv2D(int(filters*complexity_factor), (kernel_size, 1),
                          activation=activation, padding=padding,
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          dilation_rate=dilation,
                          name=l_name + "_conv1")(in_)
            bn = BatchNormalization(name=l_name + "_BN1")(conv)
            conv = Conv2D(int(filters*complexity_factor), (kernel_size, 1),
                          activation=activation, padding=padding,
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          dilation_rate=dilation,
                          name=l_name + "_conv2")(bn)
            bn = BatchNormalization(name=l_name + "_BN2")(conv)
            in_ = MaxPooling2D(pool_size=(pools[i], 1),
                               name=l_name + "_pool")(bn)

            # add bn layer to list for residual conn.
            residual_connections.append(bn)
            filters = int(filters * 2)

        # Bottom
        name = "{}bottom".format(name_prefix)
        conv = Conv2D(int(filters*complexity_factor), (kernel_size, 1),
                      activation=activation, padding=padding,
                      kernel_regularizer=regularizer,
                      bias_regularizer=regularizer,
                      dilation_rate=1,
                      name=name + "_conv1")(in_)
        bn = BatchNormalization(name=name + "_BN1")(conv)
        conv = Conv2D(int(filters*complexity_factor), (kernel_size, 1),
                      activation=activation, padding=padding,
                      kernel_regularizer=regularizer,
                      bias_regularizer=regularizer,
                      dilation_rate=1,
                      name=name + "_conv2")(bn)
        encoded = BatchNormalization(name=name + "_BN2")(conv)

        return encoded, residual_connections, filters

def create_dense_modeling(in_=None,
                              in_reshaped=None,
                              filters=None,
                              dense_classifier_activation=None,
                              regularizer=None,
                              complexity_factor=None,
                              name_prefix="",
                              n_periods=0,
                              input_dims=0,
                              n_crops=0,
                              **kwargs):
        cls = Conv2D(filters=int(filters*complexity_factor),
                     kernel_size=(1, 1),
                     kernel_regularizer=regularizer,
                     bias_regularizer=regularizer,
                     activation=dense_classifier_activation,
                     name="{}dense_classifier_out".format(name_prefix))(in_)
        s = (n_periods * input_dims) - cls.get_shape().as_list()[1]
        out = crop_nodes_to_match(
            node1=ZeroPadding2D(padding=[[s // 2, s // 2 + s % 2], [0, 0]])(cls),
            node2=in_reshaped, n_crops=n_crops
        )
        return out

def create_seq_modeling(in_,
                            input_dims,
                            data_per_period,
                            n_periods,
                            n_classes,
                            transition_window,
                            activation,
                            regularizer=None,
                            name_prefix=""):
        cls = AveragePooling2D((data_per_period, 1),
                               name="{}average_pool".format(name_prefix))(in_)
        out = Conv2D(filters=n_classes,
                     kernel_size=(transition_window, 1),
                     activation=activation,
                     kernel_regularizer=regularizer,
                     bias_regularizer=regularizer,
                     padding="same",
                     name="{}sequence_conv_out_1".format(name_prefix))(cls)
        out = Conv2D(filters=n_classes,
                     kernel_size=(transition_window, 1),
                     activation="softmax",
                     kernel_regularizer=regularizer,
                     bias_regularizer=regularizer,
                     padding="same",
                     name="{}sequence_conv_out_2".format(name_prefix))(out)
        s = [-1, n_periods, input_dims//data_per_period, n_classes]
        if s[2] == 1:
            s.pop(2)  # Squeeze the dim
        out = Lambda(lambda x: tf.reshape(x, s),
                     name="{}sequence_classification_reshaped".format(name_prefix))(out)
        return out

def create_upsample(in_,
                        res_conns,
                        depth,
                        pools,
                        filters,
                        kernel_size,
                        activation,
                        dilation,  # NOT USED
                        padding,
                        complexity_factor,
                        regularizer=None,
                        name="upsample",
                        name_prefix="",
                        n_crops=0,
                        dense_classifier_activation=None,
                        l2_reg = False,
                        n_classes = 2,
                        cf=2.00,
                        transition_window = 1):
        name = "{}{}".format(name_prefix, name)
        residual_connections = res_conns[::-1]
        for i in range(depth):
            filters = int(filters/2)
            l_name = name + "_L%i" % i

            # Up-sampling block
            fs = pools[::-1][i]
            up = UpSampling2D(size=(fs, 1),
                              name=l_name + "_up")(in_)
            conv = Conv2D(int(filters*complexity_factor), (fs, 1),
                          activation=activation,
                          padding=padding,
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          name=l_name + "_conv1")(up)
            bn = BatchNormalization(name=l_name + "_BN1")(conv)

            # Crop and concatenate
            cropped_res = crop_nodes_to_match(residual_connections[i], bn, n_crops)
            # cropped_res = residual_connections[i]
            merge = Concatenate(axis=-1,
                                name=l_name + "_concat")([cropped_res, bn])
            conv = Conv2D(int(filters*complexity_factor), (kernel_size, 1),
                          activation=activation, padding=padding,
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          name=l_name + "_conv2")(merge)
            bn = BatchNormalization(name=l_name + "_BN2")(conv)
            conv = Conv2D(int(filters*complexity_factor), (kernel_size, 1),
                          activation=activation, padding=padding,
                          kernel_regularizer=regularizer,
                          bias_regularizer=regularizer,
                          name=l_name + "_conv3")(bn)
            in_ = BatchNormalization(name=l_name + "_BN3")(conv)
        return in_
    
def crop_nodes_to_match(node1, node2, n_crops):
        """
        If necessary, applies Cropping2D layer to node1 to match shape of node2
        """
        s1 = np.array(node1.get_shape().as_list())[1:-2]
        s2 = np.array(node2.get_shape().as_list())[1:-2]

        if np.any(s1 != s2):
            n_crops += 1
            c = (s1 - s2).astype(np.int)
            cr = np.array([c // 2, c // 2]).flatten()
            cr[n_crops % 2] += c % 2
            cropped_node1 = Cropping2D([list(cr), [0, 0]])(node1)
        else:
            cropped_node1 = node1
        return cropped_node1
       
def get_activation_function(activation_string):
    """
    Same as 'init_losses', but for optimizers.
    Please refer to the 'init_losses' docstring.
    """
    activation = _get_classes_or_funcs(
        activation_string,
        func_modules=[tf.keras.activations, addon_activations]
    )
    assert len(activation) == 1, f'Received unexpected number of activation functions ({len(activation)}, expected 1)'
    return activation[0]

def _get_classes_or_funcs(string_list: list, func_modules: list) -> List[callable]:
    """
    Helper for 'init_losses' or 'init_metrics'.
    Please refer to their docstrings.

    Args:
        string_list:  (list)   List of strings, each giving a name of a metric
                               or loss to use for training. The name should
                               refer to a function or class in either tf_funcs
                               or custom_funcs modules.
        func_modules: (module or list of modules) A Tensorflow.keras module of losses or metrics,
                                                  or a list of various modules to look through.

    Returns:
        A list of len(string_list) of classes/functions of losses/metrics/optimizers/activation functions etc.
    """
    functions_or_classes = []
    func_modules = ensure_list_or_tuple(func_modules)
    for func_or_class_str in ensure_list_or_tuple(string_list):
        found = False
        for module in func_modules:
            found = getattr(module, func_or_class_str, False)
            if found:
                print(f"Found requested class '{func_or_class_str}' in module '{module}'")
                functions_or_classes.append(found)  # return the first found
                break
        if not found:
            raise AttributeError(f"Did not find loss/metric function {func_or_class_str} "
                                 f"in the module(s) '{func_modules}'")
    return functions_or_classes

def create_model_utime(input_shape):
    return input_shape

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
        X_val=read_h5_file(resources_path, "X_val")
        y_val=read_h5_file(resources_path, "y_val")
        if model =="mlp":
            model = create_model_mlp(input_signals)
            early_stop = keras.callbacks.EarlyStopping(monitor='val_rmse', patience=8)
            early_history = model.fit(X_train, y_train, 
                        epochs=580, validation_data=(X_val,y_val),
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
        if model=="utime":
            model = create_model_utime(input_signals)
            
            seq_X_train=read_h5_file(resources_path, "X_train")
            seq_Y_train=read_h5_file(resources_path, "y_train")
            seq_X_val=read_h5_file(resources_path, "X_val")
            seq_Y_val=read_h5_file(resources_path, "y_val")
            
            # Imput Shape
            n_periods=seq_X_train.shape[1]
            input_dims=seq_X_train.shape[2]
            n_channels=1
                
            # Hyperparametes
            settings = {
                        "depth": 4,
                        "pools": [8, 6, 4, 2],
                        "filters": 4,
                        "kernel_size": 5,
                        "activation": "elu",
                        "cf": 2.000,
                        "dense_classifier_activation": "tanh",
                        "dilation": 2,
                        "padding": "same",
                        "name_prefix": "",
                        "complexity_factor": 2.000,
                        "l2_reg": False,
                        "n_crops":0,
                        "n_classes":2,
                        "transition_window": 1
            }
            fit = {
                        "balanced_sampling": True,
                        "use_multiprocessing": True,
                        "channel_mixture": False,
                        "margin": 17,
                        "loss": 'SparseCategoricalCrossentropy',
                        "metrics": 'SparseCategoricalAccuracy',
                        "ignore_out_of_bounds_classes": True,
                        "batch_size": 12,
                        "patience":8,
                        "n_epochs": 100,
                        "verbose": True,
                        "optimizer": "Adam",
                        "optimizer_kwargs": {"learning_rate": 5.0e-06, "decay": 0.0, "beta_1": 0.9, "beta_2": 0.999, "epsilon": 1.0e-8}
            }

            inputs = Input(shape=[n_periods, input_dims, n_channels])
            reshaped = [-1, n_periods*input_dims, 1, n_channels]
            in_reshaped = Lambda(lambda x: tf.reshape(x, reshaped))(inputs) #  shape=(None, periods*input_signals, 1, n_channels)

            # Apply regularization if not None or 0
            regularizer = regularizers.l2(settings['l2_reg']) if settings['l2_reg'] else None
            
            settings['regularizer']=regularizer
            # Get activation func from tf or tfa
            activation = get_activation_function(activation_string=settings['activation'])

            """
            Encoding path
            """
            enc, residual_cons, filters = create_encoder(in_=in_reshaped,**settings)

            """
            Decoding path
            """
            settings["filters"] = filters

            up = create_upsample(enc, residual_cons, **settings)

            """
            Dense class modeling layers
            """
            cls = create_dense_modeling(in_=up,
                                                    in_reshaped=in_reshaped,
                                                    filters=settings['n_classes'],
                                                    dense_classifier_activation=settings['dense_classifier_activation'],
                                                    regularizer=regularizer,
                                                    complexity_factor=settings['cf'],
                                                    name_prefix=settings['name_prefix'],
                                                    n_periods=n_periods,
                                                    input_dims=input_dims,
                                                    n_crops=settings['n_crops'])

            """
            Sequence modeling
            """
            data_per_prediction = input_dims

            out = create_seq_modeling(in_=cls,
                                                input_dims=input_dims,
                                                data_per_period=data_per_prediction,
                                                n_periods=n_periods,
                                                n_classes=settings['n_classes'],
                                                transition_window=settings['transition_window'],
                                                activation=activation,
                                                regularizer=regularizer,
                                                name_prefix=settings['name_prefix'])
            model = tf.keras.Model(inputs=inputs, outputs=out)
            train_dataset = tf.data.Dataset.from_tensor_slices((seq_X_train, seq_Y_train))
            train_dataset = train_dataset.shuffle(buffer_size=len(seq_Y_train)).batch(fit['batch_size'])
            
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
            metric = tf.keras.metrics.SparseCategoricalAccuracy()


            model.compile(fit['optimizer'],loss,metric)

            callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=fit['patience'], restore_best_weights = True)

            history = model.fit(train_dataset, epochs=fit['n_epochs'], validation_data=(seq_X_val, seq_Y_val), callbacks=[callback], verbose=fit['verbose'])
            
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
