import os
from keras.models import Model
from keras.models import load_model, Sequential
from keras import applications
from keras.layers import Flatten, Dense, Input #, Sequential
from keras.losses import mean_squared_error
from keras.optimizers import Adam

def model_vgg16_v1(nr_of_untrainable_layers):

    #input_tensor = Input(shape=(128, 128, 3))

    #base_model = VGG16(weights='imagenet', include_top=False, input_tensor=input_tensor, pooling='max')
    base_model = applications.VGG16(weights='imagenet', include_top=False)
    print('Model loaded.')

    # build a classifier model to put on top of the convolutional model
    #top_model = Sequential()
    #top_model.add(Flatten(input_shape=model.output_shape[1:]))
    #counter2 = 0

    #new_model = Sequential()
    #for l in base_model.layers:
    #    if counter2 > 30:
    #        break
    #    else:
    #        counter2 += 1
    #    new_model.add(l)

    #new_model.pop()

    #counter = 0

    #x = Model(inputs=base_model.input, outputs=base_model.output)
    #

    #x.compile(loss='mean_squared_error',
    #                  optimizer='adam',
    #                  metrics=['accuracy'])

    input = Input(shape=(64, 64, 3))

    #vgg16 = VGG16(weights="imagenet", include_top=False)

    #for this_layer in new_model.layers:
    #    if counter < nr_of_untrainable_layers:
    #        this_layer.trainable = False
    #        counter += 1

    x = base_model(input)
    #x = Flatten()(base_model.output)
    #base_model.add(Flatten())
    #x = Flatten(input_shape=base_model.output_shape[1:])(x)
    x.add(Flatten())

    #top_model = Sequential()
    #top_model.add(Flatten(input_shape=model.output_shape))


    # Regression part
    fc1 = Dense(100, activation='relu')(x)
    fc2 = Dense(50, activation='relu')(fc1)
    fc3 = Dense(10, activation='relu')(fc2)
    prediction = Dense(1)(fc3)

    #new_model.add(prediction)

    model = Model(inputs=model.input, outputs=prediction)

    for this_layer in model.layers[:nr_of_untrainable_layer]:
        #if counter < nr_of_untrainable_layers:
        this_layer.trainable = False
        #counter += 1

    model.compile(loss='mean_squared_error',
                  optimizer='adam',
                  metrics=['accuracy'])

    model.summary()

    return model
