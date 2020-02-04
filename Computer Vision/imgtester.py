import h5py
import os
import tensorflow as tf
import numpy as np
import argparse
from matplotlib import pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

## creates argument parser for the necessary input paths
parser = argparse.ArgumentParser()
parser.add_argument("input_path", help="input your testing directory")
parser.add_argument("model", help="input your testing directory")
args = parser.parse_args()

## loads model from model path and preprocesses it
model = load_model(args.model)
test_dir = args.input_path
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_dir,
        batch_size=8,
        target_size=(200, 200),
        color_mode="rgb",
        shuffle = "false",
        class_mode='categorical')

## predicts and evaluates test data to generate accuracy scores
loss, acc = model.evaluate(test_generator)
preds = model.predict(test_generator)
predicted_class_indices=np.argmax(preds,axis=1)
labels = (test_generator.class_indices)
labels2 = dict((v,k) for k,v in labels.items())
predictions = [labels2[k] for k in predicted_class_indices]
print("\n%s: %.2f%%" % (model.metrics_names[1], acc * 100))
print(predictions)
