import string
import createModel
import importDataset
import loadModel
import preprocessing


from keras.models import Sequential
import pandas

height = 64
width = 64


def main():

    print("Creating model...")

    model = createModel.first_go_model(height, width)

    print("Loading dataset...")

    pd_dataset = importDataset.load_simulator_data()

    print("Processing data...")

    np_images, np_steering = preprocessing.np_from_pd(pd_dataset, width, height)

    print("Training the model...")

    history = model.fit(np_images, np_steering, epochs=10, batch_size=32)

    print("Saving the model...")

    save_model("first_go")

    print("Finished!")

    return 0;



if __name__== "__main__":
    main()



def save_model(model_name):

    # serialize model to JSON
    model_json = model.to_json()
    with open(model_name+".json", "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(model_name+".h5")

    print("Saved model to disk")
