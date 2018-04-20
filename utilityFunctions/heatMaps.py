
import numpy as np

from matplotlib import pyplot as plt

from skimage import transform
import math
import loadModel
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from keras.layers import Activation
from vis.utils import utils

FRAME_H = 64
FRAME_W = 64

model = load_model('model.h5')
img = utils.load_img('IMG2/image3782.jpg')

shape = img.shape
img = img[math.floor(shape[0]/4):, 0:shape[1]] #removed shape[0]-25 from row

target_size=(FRAME_H, FRAME_W)
img = transform.resize(img, target_size, preserve_range=True).astype('uint8')

plt.figure()
plt.subplot()
plt.imshow(img)
plt.show()

# Convert to BGR, create input with batch_size: 1.
bgr_img = utils.bgr2rgb(img)
img_input = np.expand_dims(img_to_array(bgr_img), axis=0)
pred = model.predict(img_input)[0][0]
print('Predicted {}'.format(pred))

for layer in model.layers:
    layer.trainable = True

from vis.visualization import visualize_saliency, overlay

titles = ['right steering', 'left steering', 'maintain steering']
modifiers = [None, 'negate', 'small_values']

layer_idx = utils.find_layer_idx(model, 'dense_4')

# Swap softmax with linear
model.layers[layer_idx].activation = Activation('linear')

for i, modifier in enumerate(modifiers):
    heatmap = visualize_saliency(model, layer_idx=-1, filter_indices=0,
                                 seed_input=bgr_img, grad_modifier=modifier)
    plt.figure()
    plt.subplot()
    plt.title(titles[i])
    # Overlay is used to alpha blend heatmap onto img.
    plt.imshow(overlay(img, heatmap, alpha=0.7))
    plt.show()

from vis.visualization import visualize_cam

for i, modifier in enumerate(modifiers):
    heatmap = visualize_cam(model, layer_idx=-1, filter_indices=0,
                            seed_input=bgr_img, grad_modifier=modifier)
    plt.figure()
    plt.title(titles[i])
    # Overlay is used to alpha blend heatmap onto img.
    plt.imshow(overlay(img, heatmap, alpha=0.7))
    #plt.imshow(overlay(img, heatmap, alpha=0.7))
    plt.show()

#plt.show()
