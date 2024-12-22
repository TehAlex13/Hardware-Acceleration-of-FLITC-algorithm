import tensorflow as tf
import numpy as np

trained_model = tf.keras.models.load_model('model_f3.h5')
    
x_trn = np.load('x_trn3.npy')

# Import quantization libraries
from tensorflow_model_optimization.quantization.keras import vitis_quantize

# Create VitisQuantizer object
quantizer = vitis_quantize.VitisQuantizer(trained_model)

# Quantize the model
quantized_model = quantizer.quantize_model(calib_dataset=x_trn, calib_steps=7000)

# Save the quantized model
quantized_model.save('quantized_branch_f3.h5')
