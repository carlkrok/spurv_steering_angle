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
        #print("Inside preprocessing! imread:", data['image'][i_elem]) #.strip()
        image = cv2.imread("/home/student/Desktop/Syndata/spurv_steering_angle/"+data['image'][i_elem].strip())

        if image is None:
            break

        image = preprocessImage(image, new_size_row, new_size_col)
        image = np.array(image)

        steer = np.array([[data['steer_sm'][0]]])

        np_images[i_elem] = image

        np_steering[i_elem] = steer

        #if steer

        #image2, steer2 = randomflip(image, steer)
        #image3 = image

    print("Now plotting.. ")
    plt.hist(np_steering, bins=100)
    plt.show()


    return np_images, np_steering;


def preprocessImage(image, new_size_row, new_size_col):
    #print("Inside process image! Trying to find shape of: ", image)
    shape = image.shape
    # note: numpy arrays are (row, col)!
    image = image[math.floor(shape[0]/4):, 0:shape[1]] #removed shape[0]-25 in row
    image = cv2.resize(image,(new_size_col,new_size_row), interpolation=cv2.INTER_AREA)
    image = image/255.-.5
    return image;
