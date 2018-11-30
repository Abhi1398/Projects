import numpy as np
import tensorflow as tf
import keras
from keras import backend as K
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dropout
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
K.set_image_dim_ordering('th')

# Fix the random seed
seed = 7
np.random.random(seed)

# Loading the data
(x_train,y_train), (x_test, y_test)  = mnist.load_data()
# reshape to be [samples][pixels][width][hieght]
x_train = x_train.reshape(x_train.shape[0],1,28,28).astype('float32')
x_test = x_test.reshape(x_test.shape[0],1,28,28).astype('float32')

# Normalize inputs from 0-255 to 0-1
x_train = (x_train/255).astype('float32')
x_test = (x_test/255).astype('float32')
# one hot encoding
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes  = y_test.shape[1]

'''
Convolutional neural networks are more complex than standard multi-layer perceptrons, so we will start by using a simple structure to begin with that uses all of the elements for state of the art results. Below summarizes the network architecture.

The first hidden layer is a convolutional layer called a Convolution2D. The layer has 32 feature maps, which with the size of 5×5 and a rectifier activation function. This is the input layer, expecting images with the structure outline above [pixels][width][height].
Next we define a pooling layer that takes the max called MaxPooling2D. It is configured with a pool size of 2×2.
The next layer is a regularization layer using dropout called Dropout. It is configured to randomly exclude 20% of neurons in the layer in order to reduce overfitting.
Next is a layer that converts the 2D matrix data to a vector called Flatten. It allows the output to be processed by standard fully connected layers.
Next a fully connected layer with 128 neurons and rectifier activation function.
Finally, the output layer has 10 neurons for the 10 classes and a softmax activation function to output probability-like predictions for each class.
'''

# Defining the model
def baseline_model():
    # Creation of model
    model = Sequential()
    model.add(Conv2D(32,(5,5),input_shape=(1,28,28),activation = 'relu'))
    model.add(MaxPooling2D(pool_size=(2,2)))
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128,activation = 'relu'))
    model.add(Dense(num_classes, activation = 'softmax' ))
    # Compiling the model
    model.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])
    return model

# Running the model
model = baseline_model()
# Fitting the model
model.fit(x_train,y_train,validation_data = (x_test,y_test), epochs = 5, batch_size = 200, verbose = 2)
model.evaluate(x_test,y_test, verbose = 0)
print("CNN Error: %.2f%%" % (100-scores[1]*100))


























