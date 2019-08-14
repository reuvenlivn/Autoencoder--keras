# Autoencoder--keras

1. Load the mnist dataset using Keras
2. Build the encoding layer using a fully connected Keras layer (Dense)
3. Build the decoding layer using another fully connected Keras layer
4. Build a Keras Model object:
https://keras.io/models/model/
5. Comile and fit the network, don’t forget to preprocess the images by normalizing values
to 0-255 and flattenning the images before sending them into the network.
6. Save another Keras object for the encoding step and for the decoding step
7. Encode a test image using the encoder you have built and then decode it
8. Plot using Matplotlib the original and the reconstructed digits
You can use the following reference for the exercise:
https://blog.keras.io/building-autoencoders-in-keras.html
