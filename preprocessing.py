import numpy as np
import tensorflow as tf
import os

class Preprocessor:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def mnist_preprocess(X_train, X_test, y_train, y_test, lb, train_size=0.7):
        ## splits dataset into train/val/test
        training_set_size = X_train.shape[0]
        X_train = X_train.reshape(X_train.shape[0], (X_train.shape[1] * X_train.shape[2]))
        X_train, X_val = X_train[:training_set_size*train_size-1], X_train[training_set_size*train_size:]
        X_test = X_test.reshape(X_test.shape[0], (X_test.shape[1] * X_test.shape[2]))
        y_train = lb.fit_transform(y_train)
        y_train, y_val = y_train[:-10000], y_train[-10000:]
        y_test = lb.transform(y_test)
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def normalize(self):
        pass

    def reshape_image_for_convnet(size):
        ## takes in a tuple of image dimensions to return the value
        return np.reshape(self.x, size)
