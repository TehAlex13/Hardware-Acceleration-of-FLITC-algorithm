import numpy as np
import time
from tflite_runtime.interpreter import Interpreter  # Use TensorFlow Lite runtime for Raspberry Pi

# Helper function to run inference on a TFLite model
def run_tflite_model(test_image_indices, model_path, x_tst):
    time_start = time.time()
    # Initialize the interpreter for the Raspberry Pi
    interpreter = Interpreter(model_path=model_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = np.zeros((len(test_image_indices),), dtype=int)
    for i, test_image_index in enumerate(test_image_indices):
        test_image = x_tst[test_image_index]

        # Rescale input data to uint8 if quantized
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details['quantization']
            test_image = test_image / input_scale + input_zero_point

        test_image = np.expand_dims(test_image, axis=0).astype(input_details['dtype'])
        interpreter.set_tensor(input_details['index'], test_image)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details['index'])[0]

        predictions[i] = output.argmax()
        
        
    time_end = time.time()
    print("Total Time:", time_end - time_start)

    return predictions

# Helper function to evaluate a TFLite model on all images
def evaluate_model(x_tst, y_tst, model_path):
    digits2 = [np.argmax(label) for label in y_tst]

    test_image_indices = range(x_tst.shape[0])
    predictions = run_tflite_model(test_image_indices, model_path, x_tst)
    
    accuracy = (np.sum(digits2 == predictions) * 100) / len(x_tst)
    print('Model accuracy is %.4f%% (Number of test samples=%d)' % (accuracy, len(x_tst)))

# Timing and evaluation
tflite_model_path = '/workspace/FEEDER/Models/quantized_model2.tflite'

x_tst = np.load('x_tst_file.npy')
y_tst = np.load('y_tst_file.npy')

x_test = x_tst[0:5000]
y_test = y_tst[0:5000]
evaluate_model(x_test, y_test, tflite_model_path)



