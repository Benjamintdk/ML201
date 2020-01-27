import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
import os

class Preprocessor:

    lb = LabelBinarizer()

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def mnist_preprocess(X_train, X_test, y_train, y_test, train_size=0.7):
        ## splits dataset into train/val/test
        training_set_size = round((X_train.shape[0]) * train_size)
        X_train = X_train.reshape(X_train.shape[0], (X_train.shape[1] * X_train.shape[2]))
        X_train, X_val = X_train[:training_set_size], X_train[training_set_size:]
        X_test = X_test.reshape(X_test.shape[0], (X_test.shape[1] * X_test.shape[2]))
        temp1 = np.zeros((y_train.shape[0], y_train.max() + 1))
        for i, j in enumerate(y_train):
            temp1[i][j] = 1
        y_train, y_val = temp1[:training_set_size], temp1[training_set_size:]
        temp2 = np.zeros((y_test.shape[0], y_test.max() + 1))
        for i, j in enumerate(y_test):
            temp2[i][j] = 1
        y_test = temp2
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)

    def normalize(self):
        pass

    def reshape(size):
        ## takes in a tuple of image dimensions to return the value
        return np.reshape(self.x, size)
