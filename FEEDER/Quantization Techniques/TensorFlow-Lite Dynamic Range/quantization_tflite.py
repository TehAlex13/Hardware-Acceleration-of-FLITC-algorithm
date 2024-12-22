import tensorflow as tf

# Load the Keras model from the HDF5 file
model_path = r'/workspace/FEEDER/Models/model_1.h5'
loaded_model = tf.keras.models.load_model(model_path)

# Convert the Keras model to TensorFlow Lite
converter = tf.lite.TFLiteConverter.from_keras_model(loaded_model)

# Set optimization options (if needed)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

# Convert the model to TensorFlow Lite format
tflite_quant_model = converter.convert()

# Save the TensorFlow Lite model to a file
with open('quantized_model.tflite', 'wb') as f:
    f.write(tflite_quant_model)
    





