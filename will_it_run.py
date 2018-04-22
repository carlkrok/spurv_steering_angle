import string
import utilityFunctions.createModel
import utilityFunctions.importDataset
import utilityFunctions.loadModel
import preprocessing
import VGG16_v1_test
import h5py
import matplotlib.pyplot as plt
from IPython import get_ipython
get_ipython().run_line_magic('matplotlib', 'inline')

from keras.models import Sequential
import pandas

height = 64
width = 64


def main():

    print("Creating model...")

    model = VGG16_v1_test.model_vgg16_v1(16)

    print("Loading dataset...")

    pd_dataset = utilityFunctions.importDataset.load_simulator_data()

    print("Processing data... Length of data: ", len(pd_dataset))
    print("Dataset['steer_sm'][0]: ", pd_dataset['image'][0].strip())


    np_images, np_steering = preprocessing.np_from_pd(pd_dataset, width, height)

    plt.hist(np_steering, bins=100)
    plt.show()

    print("Training the model...")

    #history = model.fit(np_images, np_steering, epochs=1, batch_size=32)

    print("Saving the model...")

    #save_model(model, "model")

    print("Finished!")

    return 0;


def save_model(model, model_name):

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name+".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save(model_name+".h5")

    print("Saved model to disk")

    return;

if __name__== "__main__":
    main()
