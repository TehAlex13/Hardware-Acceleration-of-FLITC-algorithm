import tensorflow as tf
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
import random
  


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

def shuffle_dataset(dataset, output_class, rs, dur):
    x_trn, x_tst, y_trn, y_tst = [], [], [], []
    rs_trn, rs_tst, dur_trn, dur_tst = [], [], [], []

    scenario_length = [743, 1907, 1907, 1907, 205, 205, 205, 477, 477, 477, 163, 245]
    array = np.array(scenario_length) * 3
    scenario_length = list(array)

    for idx in range(len(scenario_length)):

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

x_trn, x_tst, y_trn, y_tst, rs_trn, rs_tst, dur_trn, dur_tst = shuffle_dataset(I_CWT, Feeder_Output, Rs, Duration)
model_path = r'/workspace/FEEDER/Models/quantized_model.tflite' 
interpreter = tf.lite.Interpreter(model_path=str(model_path))
interpreter.allocate_tensors()
def evaluate_model(interpreter, x_tst, y_tst):
    input_index = interpreter.get_input_details()[0]["index"]
    output_index = interpreter.get_output_details()[0]["index"]

    # Run predictions on every image in the "test" dataset.
    prediction_digits = []
    digits2 = []
    for test_image in x_tst:
        # Pre-processing: add batch dimension and convert to float32 to match with
        # the model's input data format.
        test_image = np.expand_dims(test_image, axis=0).astype(np.float32)
        interpreter.set_tensor(input_index, test_image)

        # Run inference.
        interpreter.invoke()

        # Post-processing: remove batch dimension and find the digit with highest
        # probability.
        output = interpreter.tensor(output_index)
        digit = np.argmax(output()[0])
        prediction_digits.append(digit)
        
    for i in range(len(y_tst)):  
        digit2 = np.argmax(y_tst[i])
        digits2.append(digit2)
    
    # Compare prediction results with ground truth labels to calculate accuracy.
    
    accurate_count = 0
    for index in range(len(prediction_digits)):
        if prediction_digits[index] == digits2[index]:
            accurate_count += 1
            accuracy = accurate_count * 1.0 / len(prediction_digits)

    print(accuracy)
    return 

# Evaluate models and print accuracies
evaluate_model(interpreter, x_tst, y_tst)





