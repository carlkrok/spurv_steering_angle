import os
from keras.models import Model
from keras.models import load_model
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Input
from keras.losses import mean_squared_error
from keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def model_vgg16_v1(nr_of_untrainable_layers):

    #base_model = VGG16(weights='imagenet', include_top=False, input_shape=(64, 64, 3), pooling='max')

    counter = 0

    #for this_layer in base_model.layers:
    #    if counter < nr_of_untrainable_layers:
    #        this_layer.trainable = False
    #        counter += 1



    #x = Model(inputs=base_model.input, outputs=base_model.output)
    #

    #x.compile(loss='mean_squared_error',
    #                  optimizer='adam',
    #                  metrics=['accuracy'])

    input = Input(shape=(64, 64, 3))
    vgg16 = VGG16(weights="imagenet", include_top=False)

    for this_layer in vgg16.layers:
        if counter < nr_of_untrainable_layers:
            this_layer.trainable = False
            counter += 1

    x = vgg16(input)
    x = Flatten()(x)
    #x = Flatten(input_shape=base_model.output_shape[1:])(x)
    #x.add(Flatten())


    # Regression part
    fc1 = Dense(100, activation='relu')(x)
    fc2 = Dense(50, activation='relu')(fc1)
    fc3 = Dense(10, activation='relu')(fc2)
    prediction = Dense(1)(fc3)

    model = Model(inputs=input, outputs=prediction)

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    return model
