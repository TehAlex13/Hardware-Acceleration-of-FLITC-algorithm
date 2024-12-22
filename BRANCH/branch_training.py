import os
import joblib
import tensorflow
import numpy as np
import pandas as pd
import create_topology as top
from tensorflow.keras.utils import plot_model
from ast import literal_eval
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dropout, Dense, Flatten, Conv2D, MaxPooling2D
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.regularizers import L2
from tensorflow.keras.backend import clear_session
import matplotlib.pyplot as plt
import logging
from tqdm.keras import TqdmCallback
import random
import time

random.seed()
rand_num = int(100 * random.random())
print("seed=", rand_num)
cce = tensorflow.keras.losses.CategoricalCrossentropy()

# Load data
directory = os.path.abspath(os.path.dirname(__file__))

tree, nodes, grid_length = top.create_grid()
leaf_nodes = top.give_leaves(nodes[0])

# Find all leaf nodes
grid_path = []
branch_distance = []
for leaf in leaf_nodes:
    arr, dist = top.give_path(nodes[0], leaf)
    grid_path.append(arr)
    branch_distance.append(dist)
feeders_num = len(nodes[0].children)  # Number of root's children: 3
branches_num = len(grid_path)  # Number of grid's branches: 9
metrics_num = len(nodes)  # Number of voltage meters: 33
distance_sections = 5  # Number of sections a branch is divided: 5

best_models_loc = [[] for _ in range(3)]
best_models_loc[0] = directory + r'/best_dmdcwt_branch_id_1_not_full_v2.csv'
best_models_loc[1] = directory + r'/best_dmdcwt_branch_id_2_not_full_v2.csv'
best_models_loc[2] = directory + r'/best_dmdcwt_branch_id_3_not_full_v1.csv'

tree, nodes, grid_length = top.create_grid()
leaf_nodes = top.give_leaves(nodes[0])

# Find all leaf nodes
grid_path = []
branch_distance = []
for leaf in leaf_nodes:
    arr, dist = top.give_path(nodes[0], leaf)
    grid_path.append(arr)
    branch_distance.append(dist)
feeder_num = len(nodes[0].children)  # Number of root's children: 3
branches_num = len(grid_path)  # Number of grid's branches: 9
metrics_num = len(nodes)  # Number of voltage meters: 33
distance_sections = 5  # Number of sections a branch is divided: 5

V_feeder_CWT, Branch_Output_sorted, best_models = [[] for _ in range(feeders_num)], [[] for _ in range(feeders_num)], \
                                                  [[] for _ in range(feeders_num)]
Rs, Duration = [[] for _ in range(feeders_num)], [[] for _ in range(feeders_num)]
for index in range(feeder_num): 
    data_path = directory + r'/V_feeder_DMDCWT_reduced_' + str(index + 1) + '_not_full.joblib'
    class_path = directory + r'/Branch_Output_sorted_' + str(index + 1) + '.joblib'
    Rs_path = directory + r'/Rs_FBNN_' + str(index + 1) + '.joblib'
    Duration_path = directory + r'/Duration_FBNN_' + str(index + 1) + '.joblib'
    V_feeder_CWT[index] = joblib.load(data_path)
    Branch_Output_sorted[index] = np.array(joblib.load(class_path))
    Rs[index] = joblib.load(Rs_path)
    Duration[index] = joblib.load(Duration_path)
    best_models[index] = pd.read_csv(best_models_loc[index], index_col=0)


def shuffle_dataset(dataset, output_class, rs, dur):

    x_train, x_test, y_train, y_test, rs_train, rs_test, dur_train, dur_test = train_test_split(dataset, output_class,
                                                                                                rs, dur, test_size=0.2,
                                                                                                random_state=rand_num)
    return x_train, x_test, y_train, y_test, rs_train, rs_test, dur_train, dur_test


def create_model(input_shapes, rates, kernel_initializer, no_outputs, no_layer, unit):
    # Keras model
    model = Sequential()
    # First layer specifies input_shape
    model.add(Conv2D(64, 5, activation='relu', padding='same', input_shape=input_shapes,
                     kernel_initializer=kernel_initializer, data_format='channels_last'))
    model.add(Dropout(rate=rates))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, 5, activation='relu', padding='same', kernel_initializer=kernel_initializer,
                     data_format='channels_last'))
    model.add(Dropout(rate=2*rates))
    model.add(MaxPooling2D())
    model.add(Flatten())

    # Max 5 Full connected hidden layers
    for idx in range(no_layer - 1):
        model.add(Dense(units=unit[idx], activation='relu', kernel_initializer=kernel_initializer))
        rt = rates*(3 + float(idx))
        model.add(Dropout(rate=rt))

    model.add(Dense(no_outputs, activation='softmax'))

    return model


def create_model_v2(input_shapes, rates, kernel_initializer, kernel_regularizer, no_outputs, no_layer, unit):
    # Keras model
    model = Sequential()
    # First layer specifies input_shape
    model.add(Conv2D(16, (3, 3), activation='relu', padding='same', input_shape=input_shapes,
                     data_format='channels_last', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer))
    model.add(MaxPooling2D())
    model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer))
    model.add(MaxPooling2D())
    model.add(Conv2D(64, (3, 3), activation='relu', padding='same',
                     kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer))
    model.add(MaxPooling2D())
    model.add(Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer=kernel_initializer,
                     kernel_regularizer=kernel_regularizer))
    model.add(MaxPooling2D())
    model.add(Flatten())

    # Max 5 Full connected hidden layers
    for idx in range(no_layer - 1):
        model.add(Dense(units=unit[idx], activation='relu', kernel_initializer=kernel_initializer))
        rt = rates/(1 + float(idx))
        model.add(Dropout(rate=rt))

    model.add(Dense(no_outputs, activation='softmax'))

    return model


def evaluate_model(number, dataset, output, rs, dur, model_df):
    epochs = 200
    history = np.empty((len(model_df), 4, epochs))
    x_axis = range(1, epochs + 1)
    for i in range(3):
        model = model_df.iloc[i]
        print(model)
        
        batch_size, layers = int(model['batch_size']), int(model['layers'])
        rate, units = int(model['rate']), literal_eval(model['units'])
        x_train, x_test, y_train, y_test, rs_train, rs_test, dur_train, dur_test = shuffle_dataset(dataset,
                                                                                                   output, rs, dur)
        x_train, x_test = tensorflow.convert_to_tensor(x_train), tensorflow.convert_to_tensor(x_test)
        y_train, y_test = tensorflow.convert_to_tensor(y_train), tensorflow.convert_to_tensor(y_test)
        # define model
        num_features, num_outputs = x_train.shape[1], y_train.shape[1]
        # reshape into subsequences (samples, time steps, rows, cols, channels)
        shape_input = (x_train.shape[1], x_train.shape[2], x_train.shape[3])

        # define model
        verbose = 1

        # define model
        if (number == 1 or number == 2):
            best_model = create_model_v2(shape_input, rate, "he_normal", L2(l=0.01), num_outputs, layers, units)
        else:
            best_model = create_model(shape_input, rate, "he_normal", num_outputs, layers, units)

        best_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=["accuracy"])
        best_model.summary()

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=25)

        history_i = best_model.fit(x_train, y_train, validation_split=0.2, epochs=epochs,
                                   batch_size=batch_size, verbose=verbose, callbacks=[es, TqdmCallback(verbose=0)])
                                   
        np.save('x_tst' + str(number) + '.npy', x_test)
        np.save('y_tst' + str(number) + '.npy', y_test)
        np.save('x_trn' + str(number) + '.npy', x_train)
        np.save('y_trn' + str(number) + '.npy', y_train)
      
        best_model.save('model_f' + str(number) + '_' + str(i+1) + '.h5')
       
# Load Data
best_models_f1 = pd.read_csv(best_models_loc[0], index_col=0)
best_models_f2 = pd.read_csv(best_models_loc[1], index_col=0)
best_models_f3 = pd.read_csv(best_models_loc[2], index_col=0)
"""
history_f1 = evaluate_model(1, V_feeder_CWT[0], Branch_Output_sorted[0], Rs[0], Duration[0], best_models_f1)
joblib.dump(history_f1, directory + r'/history_f1.joblib')
"""
history_f2 = evaluate_model(2, V_feeder_CWT[1], Branch_Output_sorted[1], Rs[1], Duration[1], best_models_f2)
joblib.dump(history_f2, directory + r'/history_f2.joblib')
"""
history_f3 = evaluate_model(3, V_feeder_CWT[2], Branch_Output_sorted[2], Rs[2], Duration[2], best_models_f3)
joblib.dump(history_f3, directory + r'/history_f3.joblib')
"""
