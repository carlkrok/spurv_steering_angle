

def model_vgg16_v1(nr_of_trainable_layers):

    base_model = VGG16(weights='imagenet', include_top=False, input_shape=[64, 64, 3], pooling='max')

    counter = 0

    for this_layer in base_model.layers:
        if counter < nr_of_trainable_layers:
            this_layer.trainable = False
            counter += 1

    x = Model(inputs=base_model.input, outputs=base_model.output)
    x.compile(loss=categorical_crossentropy,
              optimizer=SGD(lr=0.05),
              metrics=['accuracy'])

    x = Flatten()(x.output)

    # Regression part
    fc1 = Dense(100, activation='relu')(x)
    fc2 = Dense(50, activation='relu')(fc1)
    fc3 = Dense(10, activation='relu')(fc2)
    predictions = Dense(1)(fc3)

    model = Model(inputs=base_model.input, outputs=predictions)

    model.compile(loss=binary_crossentropy,
                  optimizer=SGD(lr=0.05),
                  metrics=['accuracy'])

    model.summary()

    return model
