import tensorflow as tf
import joblib
import numpy as np
from sklearn.utils import shuffle
from keras.models import load_model
from keras.callbacks import EarlyStopping
import random

from tensorflow_model_optimization.quantization.keras import vitis_quantize
from sklearn.model_selection import train_test_split


# Load data
I_CWT_path = r'/workspace/FEEDER/Models/I_CWT.joblib'
Class_path = r'/workspace/FEEDER/Models/Feeder_Output_4_Outputs.joblib'
Rs_path = r'/workspace/FEEDER/Models/Rs_FFNN.joblib'
Duration_path = r'/workspace/FEEDER/Models/Duration_FFNN.joblib'


# Change load file for different dataset
I_CWT = joblib.load(I_CWT_path)
Feeder_Output = joblib.load(Class_path)
Rs = joblib.load(Rs_path)
Duration = joblib.load(Duration_path)

random.seed()
rand_num = int(100* random.random())

#create x_train
def shuffle_dataset(dataset, output_class, rs, dur):
    x_trn, x_tst, y_trn, y_tst = [], [], [], []
    rs_trn, rs_tst, dur_trn, dur_tst = [], [], [], []

    scenario_length = [743, 1907, 1907, 1907, 205, 205, 205, 477, 477, 477, 163, 245]
    array = np.array(scenario_length) * 3
    scenario_length = list(array)

    for idx in range(len(scenario_length)):
        
        print(idx)

        x_train, x_test, y_train, y_test, rs_train, rs_test, dur_train, dur_test = train_test_split(
            dataset[sum(scenario_length[:idx]):sum(scenario_length[:idx + 1])],
            output_class[sum(scenario_length[:idx]):sum(scenario_length[:idx + 1])],
            rs[sum(scenario_length[:idx]):sum(scenario_length[:idx + 1])],
            dur[sum(scenario_length[:idx]):sum(scenario_length[:idx + 1])],
            test_size=0.2, random_state=rand_num)

        for j in x_train:
            x_trn.append(j)
        for j in x_test:
            x_tst.append(j)
        for j in y_train:
            y_trn.append(j)
        for j in y_test:
            y_tst.append(j)
        for j in rs_train:
            rs_trn.append(j)
        for j in rs_test:
            rs_tst.append(j)
        for j in dur_train:
            dur_trn.append(j)
        for j in dur_test:
            dur_tst.append(j)

    temp1 = list(zip(x_trn, y_trn, rs_trn, dur_trn))
    temp2 = list(zip(x_tst, y_tst, rs_tst, dur_tst))

    random.shuffle(temp1)
    random.shuffle(temp2)

    x_trn, y_trn, rs_trn, dur_trn = zip(*temp1)
    x_tst, y_tst, rs_tst, dur_tst = zip(*temp2)

    x_trn = np.stack(x_trn, axis=0)
    x_tst = np.stack(x_tst, axis=0)
    y_trn = np.stack(y_trn, axis=0)
    y_tst = np.stack(y_tst, axis=0)
    rs_trn = np.stack(rs_trn, axis=0)
    rs_tst = np.stack(rs_tst, axis=0)
    dur_trn = np.stack(dur_trn, axis=0)
    dur_tst = np.stack(dur_tst, axis=0)

    return x_trn, x_tst, y_trn, y_tst, rs_trn, rs_tst, dur_trn, dur_tst



def evaluate_model(dataset, output, rs, dur):
    
    # Φόρτωση του quantized μοντέλου από το αρχείο 'quantized_model.h5'
    quantized_model = tf.keras.models.load_model('quantized_model_1.h5')

    x_trn, x_tst, y_trn, y_tst, rs_trn, rs_tst, dur_trn, dur_tst = shuffle_dataset(I_CWT, Feeder_Output, Rs, Duration)
   
    print("\n-----ACCURACY-----\n")
    from sklearn.metrics import accuracy_score

    prediction_digits = []
    for test_image in x_tst:
        test_image = np.expand_dims(test_image, axis=0)
        predictions = quantized_model.predict(test_image)
        digit = np.argmax(predictions)
        prediction_digits.append(digit)
    
    digits2 = []
    for i in range(len(y_tst)):  
        digit2 = np.argmax(y_tst[i])
        digits2.append(digit2)
    
    accuracy = accuracy_score(digits2, prediction_digits)

    print("Accuracy:", accuracy)

evaluate_model(I_CWT, Feeder_Output, Rs, Duration)
