
from keras.models import Sequential
from keras.layers import Flatten, Dropout, Dense, ELU
from keras.losses import mse
from keras.optimizers import adam


def first_go_model(img_height, img_width):

    # Initialize new model
    model = Sequential()


    # Add convolutional layers
    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), input_shape=( img_width, img_height, 3), padding='same'))

    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))

    model.add(Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), padding='same'))

    model.add(ELU())

    model.add(MaxPooling2D(pool_size=(5, 5), strides=(2, 2)))


    # Add fully connected layers
    model.add(Flatten())

    model.add(Dense(100))

    model.add(ELU())

    model.add(Dense(1))


    # Compile the model
    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])


    # Print model properties
    model.summary()


    return model
