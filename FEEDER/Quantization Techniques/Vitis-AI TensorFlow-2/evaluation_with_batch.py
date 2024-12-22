def evaluate_model():
    # Load model
    quantized_model = tf.keras.models.load_model('model_1.h5')

    # Shuffle data
    x_trn, x_tst, y_trn, y_tst, rs_trn, rs_tst, dur_trn, dur_tst = shuffle_dataset(I_CWT, Feeder_Output, Rs, Duration)

    # Limit data to the first samples for testing
    x_test = x_tst[0:5000]
    y_test = y_tst[0:5000]

    # Measure time
    time_start = time.time()

    # Predict with batch size 32
    predictions = quantized_model.predict(x_test, batch_size=128)
    print(predictions)
    # Extract predicted class labels
    prediction_digits = np.argmax(predictions, axis=1)
    time_end = time.time()
    timetotal = time_end - time_start
    # Extract true class labels
    digits2 = np.argmax(y_test, axis=1)

    # Calculate accuracy
    accuracy = accuracy_score(digits2, prediction_digits)

    # Print results
    print("Accuracy:", accuracy)
    print("Total Time:", timetotal)


# Run evaluation
evaluate_model()
