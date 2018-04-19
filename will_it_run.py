import createModel
import importDataset
import loadModel
import preprocessing

from keras.models import Model
import pandas

height = 64
width = 64

def main():

    model = createModel.first_go_model(height, width)

    pd_dataset = importDataset.load_simulator_data()

    np_images, np_steering = preprocessing.np_from_pd(pd_dataset, width, height)

    history = model.fit(np_images, np_steering, epochs=10, batch_size=32)

    loadModel.save_model(model, "first_go")
