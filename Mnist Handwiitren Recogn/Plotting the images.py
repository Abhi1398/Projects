import tensorflow as tf
import matplotlib.pyplot as plt

mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Plotting the images as gray scale
plt.subplot(221)
plt.imshow(x_train[0], cmap = plt.get_cmap('gray'))
plt.subplot(222)
plt.imshow(x_train[1], cmap = plt.get_cmap('gray'))
plt.subplot(223)
plt.imshow(x_train[2], cmap = plt.get_cmap('gray'))
plt.subplot(224)
plt.imshow(x_train[3], cmap = plt.get_cmap('gray'))
# Showing hte plot
plt.show()

