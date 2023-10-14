import tensorflow as tf
from tensorflow.keras import layers
from tensorflow import  keras
import keras.backend as K

from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, Flatten
from keras.optimizers import SGD
from keras.models import model_from_json

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from tqdm import trange
from sklearn.cluster import KMeans
from sklearn import preprocessing
import numpy as np
import random
import json
import os
import pandas as pd
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from numpy.random import seed






def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true))) 

optimizer = SGD(lr=0.001)


def add_RUL(cycle, SoF, EoL, Rc):
    if cycle <= SoF:
      return Rc
    elif SoF <= EoL:
      return EoL - cycle
    

print(tf.__version__)

dataset_path = 'data/cmapss/'
number_of_dataset = 4
seed(5)
#-------------------DATA GATHERING-----------------------#

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
RC=120

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
clusters=3

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

#-------------------DATA SPLITS-----------------------#

routes_train = {}
routes_test = {}
for unit_nr in train_df['id'].unique():
    routes_train[unit_nr-1] = train_df.loc[train_df['id'] == unit_nr]
for unit_nr in test_df['id'].unique():
    routes_test[unit_nr-1] = test_df.loc[test_df['id'] == unit_nr]

# Naming columns to training the model 
sensor_cols = ['s2','s3','s4','s6','s7', 's8','s9','s10','s11','s12','s13','s14','s15','s17','s20','s21']
sequence_cols = []
sequence_cols.extend(sensor_cols)

X_train=routes_train[0][sequence_cols]
y_train=routes_train[0]['RUL']
for route_train in routes_train:
    if route_train != 0:
        X_train=X_train.append(routes_train[route_train][sequence_cols],ignore_index=True)
        y_train=y_train.append(routes_train[route_train]['RUL'],ignore_index=True)

#-------------------NETWORK-----------------------#

model = Sequential()
model.add(Dense(10, activation="sigmoid", input_shape=(len(sequence_cols),)))
model.add(Dense(1, activation='relu'))

model.compile(loss=rmse,optimizer=optimizer,metrics=['mae','mse',rmse])
early_stop = keras.callbacks.EarlyStopping(monitor='val_rmse', patience=8)

# Training model

early_history = model.fit(X_train.to_numpy(), y_train.to_numpy(), 
                    epochs=580, validation_split=0.15,
                    callbacks=[early_stop])
model.save_weights(dataset_path+"model.h5")


# serialize model to JSON
model_json = model.to_json()
with open(dataset_path+'model.json', 'w') as json_file:
    json_file.write(model_json)


json_file = open(dataset_path+'model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)

#-------------------TESTING-----------------------#

#Kalman Filter
routes_kalman = []
rul_predicted=np.array([])
rul_truth=np.array([])

for route_test in routes_test:
    # Define a dictionary containing Students data 
    data = {} 
    # Convert the dictionary into DataFrame 
    df = pd.DataFrame(data)
    #Obtaining rul - groud truth
    truth_dataframe=routes_test[route_test]['RUL']
    truth=np.array(truth_dataframe.values.tolist())*RC
    rul_truth=np.append(rul_truth,truth)
    #Obtaining rul - predicted
    trajectory = routes_test[route_test][sequence_cols].to_numpy()
    rul_trajectory=model.predict(trajectory)
    rul_trajectory = pd.DataFrame(rul_trajectory, columns=['rul'])
    z = np.array(rul_trajectory.values.tolist())
    # intial parameters
    n_iter = len(rul_trajectory)
    sz = (len(rul_trajectory),) # size of array
    # process variance
    #Q=1/209
    Q = 1/209
    # allocate space for arrays
    xhat=np.zeros(sz)      # a posteri estimate of x
    P=np.zeros(sz)         # a posteri error estimate
    xhatminus=np.zeros(sz) # a priori estimate of x
    Pminus=np.zeros(sz)    # a priori error estimate
    K=np.zeros(sz)         # gain or blending factor

    R = 0.3**2 # estimate of measurement variance, change to see effect
    # intial guesses
    #xhat[0] = 1
    xhat[0]=truth[0]/RC
    P[0] = 0.0

    for k in range(1,n_iter):
        # time update
        xhatminus[k] = xhat[k-1]
        Pminus[k] = P[k-1]+Q
        # measurement update
        K[k] = Pminus[k]/( Pminus[k]+R )
        xhat[k] = xhatminus[k]+K[k]*(z[k]-xhatminus[k])
        P[k] = (1-K[k])*Pminus[k]
    xhat=xhat*RC
    rul_predicted=np.append(rul_predicted,xhat)
    routes_kalman.insert((unit_nr-1),xhat[(len(xhat)-1)])


# Statistics
mae=mean_absolute_error(rul_predicted,rul_truth)
mse=mean_squared_error(rul_predicted,rul_truth)
rmse=np.sqrt(mse)
print("GAUSSIAN KALMAN")
print("MAE:",mae)
print("MSE:",mse)
print("RMSE:",rmse)