import os
import sys

# Ensure UTF-8 output encoding for Windows compatibility
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

import tensorflow as tf

h5_path = 'Team3model.h5'
tflite_path = 'Team3model.tflite'

if not os.path.exists(h5_path):
    print(f"Error: {h5_path} not found for conversion.")
    sys.exit(1)

print(f"Loading 709MB Keras H5 model from {h5_path}...")
model = tf.keras.models.load_model(h5_path, compile=False)

print("Converting model to quantized TensorFlow Lite (TFLite) format...")
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

print(f"Saving TFLite model to {tflite_path}...")
with open(tflite_path, 'wb') as f:
    f.write(tflite_model)

h5_size = os.path.getsize(h5_path) / (1024 * 1024)
tflite_size = os.path.getsize(tflite_path) / (1024 * 1024)

print("--------------------------------------------------")
print("TFLITE MODEL CONVERSION SUCCESSFUL!")
print(f"Original Model Size: {h5_size:.2f} MB")
print(f"Quantized TFLite   : {tflite_size:.2f} MB")
print(f"RAM Savings        : {h5_size / tflite_size:.1f}x smaller!")
print("--------------------------------------------------")
