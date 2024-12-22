import tensorflow as tf
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
from sklearn.model_selection import train_test_split
import random
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

x_trn, x_tst, y_trn, y_tst, rs_trn, rs_tst, dur_trn, dur_tst = shuffle_dataset(I_CWT, Feeder_Output, Rs, Duration)
# Helper function to run inference on a TFLite model
def run_tflite_model(test_image_indices):
  global x_tst
  global y_tst
  tflite_file = r'/workspace/FEEDER/Models/quantized_model2.tflite'
  # Initialize the interpreter
  interpreter = tf.lite.Interpreter(model_path=str(tflite_file))
  interpreter.allocate_tensors()

  input_details = interpreter.get_input_details()[0]
  output_details = interpreter.get_output_details()[0]

  predictions = np.zeros((len(test_image_indices),), dtype=int)
  for i, test_image_index in enumerate(test_image_indices):
    test_image = x_tst[test_image_index]
    
    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
      input_scale, input_zero_point = input_details["quantization"]
      test_image = test_image / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]
    
    predictions[i] = output.argmax()

  return predictions

def evaluate_model():
    global x_tst
    global y_tst
    
    digits2 = []
    for i in range(len(y_tst)):  
        digit2 = np.argmax(y_tst[i])
        digits2.append(digit2)
    
    test_image_indices = range(x_tst.shape[0])
    predictions = run_tflite_model(test_image_indices)
    
    accuracy = (np.sum(digits2 == predictions) * 100) / len(x_tst)

    print('model accuracy is %.4f%% (Number of test samples=%d)' % (
      accuracy, len(x_tst)))
  
evaluate_model()