# Importing classes and functions
import tensorflow as tf
import keras
import numpy as np
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.utils import np_utils

#Fix random seed for Reproductability
seed = 7
np.random.seed(seed)

# Loading the data
(x_train,y_train),(x_test, y_test) = mnist.load_data()

# Flatten 28*28 images to a 784 vector for each image
# Since the dataset is 3 dimension , So it is needed to be converted into a vector
#  3-dimensional array of instance, image width and image height
num_pixels  = x_train.shape[1] * x_train.shape[2]
x_train = x_train.reshape(x_train.shape[0], num_pixels).astype('float32')
x_test = x_test.reshape(x_test.shape[0], num_pixels).astype('float32')

# Normalize inputs from 0-255 to 0-1
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255

# Using one hot encoding : to transform the vector of class integers into a binary matrix
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Defining the base line mode;
def baseline_model():
    #Creating the model
    model = Sequential()
    model.add(Dense(num_pixels, input_dim = num_pixels, kernel_initializer = 'normal', activation = 'relu'))
    model.add(Dense(num_classes, kernel_initializer = 'normal', activation = 'softmax'))
    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


# build the model
model = baseline_model()
# Fit the model
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=5, batch_size=200, verbose=2)
# Final evaluation of the model
scores = model.evaluate(x_test, y_test, verbose=0)
print("Baseline Error: %.2f%%" % (100-scores[1]*100))
























