
import numpy as np

from matplotlib import pyplot as plt

import loadModel
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from vis.utils import utils

FRAME_H = 64
FRAME_W = 64

model = load_model('Really_bad_model/model.h5')
img = utils.load_img('drunk_driving/IMG/center_2018_04_20_11_50_15_356.jpg', target_size=(FRAME_H, FRAME_W))
plt.figure()
plt.subplot()
plt.imshow(img)
plt.show()

# Convert to BGR, create input with batch_size: 1.
bgr_img = utils.bgr2rgb(img)
img_input = np.expand_dims(img_to_array(bgr_img), axis=0)
pred = model.predict(img_input)[0][0]
print('Predicted {}'.format(pred))

from vis.visualization import visualize_saliency, overlay

titles = ['right steering', 'left steering', 'maintain steering']
modifiers = [None, 'negate', 'small_values']
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
