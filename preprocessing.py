import pandas as pd
import numpy as np
import cv2

def np_from_pd(data pd.DataFrame, new_size_row int, new_size_col int):

    data_size = len(data)

    np_images = np.zeros((data_size, new_size_row, new_size_col, 3))
    np_steering = np.zeros(data_size)

    for i_elem in range(data_size):

        line_data = data.iloc[[i_elem]].reset_index()

        image = cv2.imread(line_data['center'][0].strip())

        image = preprocessImage(image, new_size_row, new_size_col)

        image = np.array(image)

        steer = np.array([[line_data['steer_sm'][0]]])

        np_images[i_elem] = image
        np_steering[i_elem] = steer


    return np_images, np_steering


def preprocessImage(image, new_size_row, new_size_col):
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/4):shape[0]-25, 0:shape[1]]
    image = cv2.resize(image,(new_size_col,new_size_row), interpolation=cv2.INTER_AREA)
    image = image/255.-.5
    return image
