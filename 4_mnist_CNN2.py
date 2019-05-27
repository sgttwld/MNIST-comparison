"""
A high-level implementation of an advanced convolutional neural network 
for MNIST classification using tensorflow.keras.models (~0.9% test error)
Author: Sebastian Gottwald
Project: https://github.com/sgttwld/classification
Date: 2019-05-25
"""

import numpy as np
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import math, os
import matplotlib.pyplot as plt
import seaborn as sns
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

## import MNIST data
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()
x_train = x_train.reshape((60000,28,28,1))
x_test = x_test.reshape((10000,28,28,1))
x_train, x_test = x_train/255, x_test/255

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))
model.summary()
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

print('test data:')
test_loss, test_acc = model.evaluate(x_test, y_test)










