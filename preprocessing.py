import pandas as pd
import numpy as np
import cv2
import math
import matplotlib.pyplot as plt

def np_from_pd(data, new_size_row, new_size_col):

    data_size = len(data)

    np_images = np.zeros((data_size, new_size_row, new_size_col, 3))
    np_steering = np.zeros(data_size)

    for i_elem in range(data_size):

        #line_data = data.iloc[[i_elem]].reset_index()

        image = cv2.imread(data['image'][i_elem].strip())
        image = preprocessImage(image, new_size_row, new_size_col)
        image = np.array(image)

        steer = np.array([[line_data['steer_sm'][0]]])

        np_images[i_elem] = image

        np_steering[i_elem] = steer

        #if steer

        #image2, steer2 = randomflip(image, steer)
        #image3 = image

    np.histogram(np_steering)

    return np_images, np_steering;


def preprocessImage(image, new_size_row, new_size_col):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row), interpolation=cv2.INTER_AREA)
    image = image/255.-.5
    return image;
