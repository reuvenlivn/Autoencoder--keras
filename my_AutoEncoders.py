# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 11:20:50 2019

@author: reuve
"""
#  https://blog.keras.io/building-autoencoders-in-keras.html

from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt

# Define random seed for reproducibility
np.random.seed(7)

# 1. Load the mnist dataset using Keras
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#preprocess the images by normalizing values
#to 0-255 and flattenning the images before sending them into the network
x_train = x_train.astype('float32') / 255.
x_test  = x_test.astype('float32') / 255.
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1] * x_train.shape[2])
x_test  = x_test.reshape(x_test.shape[0], x_test.shape[1] * x_test.shape[2])

n_features = x_train.shape[1]

# 2. Build the encoding layer using a fully connected Keras layer (Dense)
encoding_dim = 32
encoder = Sequential()
encoder.add(Dense(encoding_dim, input_dim=n_features, activation="relu"))

# 3. Build the decoding layer using another fully connected Keras layer
decoder = Sequential()
decoder.add(Dense(n_features, input_dim=encoding_dim, activation="relu"))

# 4. Build a Keras Model object:
autoencoder = Sequential()

autoencoder.add(encoder)
autoencoder.add(decoder)

# 5. Comile and fit the network
autoencoder.compile(loss="binary_crossentropy" ,
                    optimizer="adam", 
                    metrics=["accuracy"])
autoencoder.fit(x_train, 
                x_train, 
                epochs=50, 
                batch_size=256, 
                shuffle=True, 
                validation_data=(x_test, x_test))

# 6. Save another Keras object for the encoding step and for the decoding step
# 7. Encode a test image using the encoder you have built and then decode it
encoded_imgs = encoder.predict(x_test)
decoded_imgs = decoder.predict(encoded_imgs)

# Plot using Matplotlib the original and the reconstructed digits
n = 10  # how many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(x_test[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
