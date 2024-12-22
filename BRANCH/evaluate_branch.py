import tensorflow as tf
import numpy as np
from keras.models import load_model
import time
from sklearn.metrics import accuracy_score
from tensorflow_model_optimization.quantization.keras import vitis_quantize


# Φόρτωση του quantized μοντέλου από το αρχείο 'quantized_model.h5'
#model = tf.keras.models.load_model('model_f1.h5')
model = tf.keras.models.load_model('quantized_branch_f3.h5')
    
x_tst = np.load('x_trn3.npy')
y_tst = np.load('y_trn3.npy')

# Limit data to the first samples for testing
x_test = x_tst
y_test = y_tst
print(x_test.shape)
print(y_test.shape)
#print(x_test[0:1])
print(y_test[0:3])
time_start = time.time()
# Predict with batch size 32
predictions = model.predict(x_test, batch_size=128)
print(predictions)

time_end = time.time()
timetotal = time_end - time_start

# Extract predicted class labels
prediction_digits = np.argmax(predictions, axis=1)

# Extract true class labels
digits2 = np.argmax(y_test, axis=1)

# Calculate accuracy
accuracy = accuracy_score(digits2, prediction_digits)
print("Accuracy:", accuracy)

timetotal = time_end - time_start
print("Total Time:", timetotal)


