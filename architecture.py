from keras.models import Sequential
from keras.layers.core import Dense, Activation, Flatten, Dropout, Lambda
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.advanced_activations import ReLU


def net():
    # ------------------------ Convolutional Neural Network Architecture ------------------------
    model = Sequential()
    kernel_size = (5, 5)
    # Normalize
    model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))

    # Add three 5x5 convolution layers (output depth 24, 36, and 48), each with 2x2 stride
    model.add(Conv2D(24, kernel_size, strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ReLU())
    model.add(Conv2D(36, kernel_size, strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ReLU())
    model.add(Conv2D(48, kernel_size, strides=(2, 2), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ReLU())

    model.add(Dropout(0.10))

    # Add two 3x3 convolution layers (output depth 64, and 64)
    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ReLU())
    model.add(Conv2D(64, (3, 3), padding='valid', kernel_regularizer=l2(0.001)))
    model.add(ReLU())

    # Add a flatten layer
    model.add(Flatten())

    # Add three fully connected layers (depth 100, 50, 10), tanh activation (and dropouts)
    model.add(Dense(100, kernel_regularizer=l2(0.001)))
    model.add(ReLU())
    model.add(Dropout(0.10))
    model.add(Dense(50, kernel_regularizer=l2(0.001)))
    model.add(ReLU())
    model.add(Dropout(0.10))
    model.add(Dense(10, kernel_regularizer=l2(0.001)))
    model.add(ReLU())
    model.add(Dropout(0.10))

    # Add a fully connected output layer
    model.add(Dense(1))

    # ------------------------------------- End Architecture ------------------------------------
