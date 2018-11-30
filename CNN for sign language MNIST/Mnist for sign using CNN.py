import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#Libraries for implementing a CNN
from keras.layers import Conv2D, Flatten, MaxPooling2D, Dense, Dropout
from keras.models import Sequential
from keras.utils import to_categorical


# Loading the dataset
train = pd.read_csv('sign_mnist_train.csv')
test = pd.read_csv('sign_mnist_test.csv')

# Resizing the label column
labels = train.pop('label')
labels = to_categorical(labels)
train = train.values
train = np.array([np.reshape(i,(28,28)) for i in train])
train = train/255

# Creating a train and test data
X_train, X_val, y_train, y_val = train_test_split(train, labels, test_size = 0.3, random_state = 34)
# Reshaping the data
X_train  = X_train.reshape(X_train.shape[0], 28,28,1)
X_val = X_val.reshape(X_val.shape[0] , 28,28,1)

# Creating the CNN Model
model = Sequential()
model.add(Conv2D(8 , (3,3)  , input_shape = (28,28,1) , activation = 'relu'))
model.add(MaxPooling2D( pool_size = (2,2) , strides = 2 ))
model.add(Conv2D(8 , (3,3)  , input_shape = (28,28,1) , activation = 'relu'))
model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.4))
model.add(Dense(25, activation = 'softmax'))

model.summary()

# Compiling and optimizing the CNN
model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam',  metrics = ['accuracy'])

# Preparing for training Model
history = model.fit(X_train, y_train, validation_data = (X_val,y_val), epochs = 5, batch_size = 512)

#Plotting the result
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.xlabel('epochs')
plt.title('label')
plt.ylabel('accuracy')
plt.legend(['train', 'test'])
plt.show()

# Trsining ther dtata
y_test = test.pop('label')
y_test = to_categorical(y_test)
y_test.shape
X_test = test.values
X_test = np.array([np.reshape(i, (28,28)) for i in X_test ])
X_test  = X_test / 255
X_test = X_test.reshape(X_test.shape[0], 28,28,1)
X_test.shape

# Predicting the accuracy and recognizing the image
prediction  = model.predict(X_test)
test_accuracy = accuracy_score(np.argmax(y_test, axis = 1), np.argmax(prediction , axis = 1))
print ("the accuracy is : ", test_accuracy)

# In This parameters the accuracy score is 76%





























































