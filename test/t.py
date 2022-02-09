import numpy as np
import cv2
import io
import requests
from PIL import Image
import matplotlib.pyplot as plt
# Using Keras implementation from tensorflow
from tensorflow.python.keras import applications
from tensorflow.python.keras.preprocessing.image import load_img, img_to_array
from tensorflow.python.keras import backend as K


# Get the layer of the last conv layer
fianlconv = model.get_layer('final_conv')
# Get the weights matrix of the last layer
weight_softmax = model.layers[-1].get_weights()[0]

# Function to generate Class Activation Mapping
HEIGHT = 28
WIDTH = 28

def returnCAM(feature_conv, weight_softmax, class_idx):
    size_upsample = (WIDTH, HEIGHT)

    # Keras default is channels last, hence nc is in last
    bz, h, w, nc = feature_conv.shape

    output_cam = []
    for idx in class_idx:
        cam = np.dot(weight_softmax[:, idx], np.transpose(feature_conv.reshape(h*w, nc)))
        cam = cam.reshape(h, w)
        cam = cam - np.min(cam)
        cam_img = cam / np.max(cam)
        cam_img = np.uint8(255 * cam_img)

        output_cam.append(cv2.resize(cam_img, size_upsample))

    return output_cam

x = x_test_resized[0,:,:,0]

plt.imshow(x)
plt.show()

classes = {1:'1', 2: '2', 3: '3', 4:'4', 5:'5', 6:'6', 7:'7', 8:'8', 9:'9', 0:'0'}

probs_extractor = K.function([model.input], [model.output])
features_conv_extractor = K.function([model.input], [fianlconv.output])

probs = probs_extractor([np.expand_dims(x, 0).reshape(1,28,28,1)])[0]

features_blob = features_conv_extractor([np.expand_dims(x, 0).reshape(1,28,28,1)])[0]

features_blobs = []
features_blobs.append(features_blob)

idx = np.argsort(probs)
probs = np.sort(probs)

for i in range(-1, -6, -1):
    print('{:.3f} -> {}'.format(probs[0, i], classes[idx[0, i]]))


CAMs = returnCAM(features_blobs[0], weight_softmax, [idx[0, -1]])

heatmap = cv2.applyColorMap(cv2.resize(CAMs[0], (28, 28)), cv2.COLORMAP_JET)

result = heatmap[:,:,0] * 0.3 + x * 0.5

print(result.shape)

plt.imshow(result)
plt.show()