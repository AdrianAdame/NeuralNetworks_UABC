import os
from Layer_definitions.FCLayer import FCLayer
import utils.utilities as utilities
import NeuralNetwork as nn
import numpy as np
import pandas as pd
import requests
from matplotlib.image import imread

np.set_printoptions(precision= 20)

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.2989, 0.5870, 0.1140])

# path = "Data"
# for i in range(204):
#     url = "https://picsum.photos/64/64/?random"
#     response = requests.get(url)
#     if response.status_code == 200:
#         file_name = "not_cat_{}.jpg".format(i)
#         file_path = os.path.join(path, file_name)
#         with open(file_path, 'wb') as f:
#             f.write(response.content)
    
#     print("Downloaded image {}".format(i))

# Generate labels for the images
inputs = np.zeros((204, 64 * 64))
targets = np.ones((204, 1))

for img, index in zip(os.listdir("Data"), range(204)):
    if not img.startswith('.') and os.path.isfile(os.path.join("Data", img)):
        img_path = os.path.join("Data", img)
        if "not_cat_" in img_path:
            targets[index] = 0
        img_data = imread(img_path)
        gray = rgb2gray(img_data)
        inputs[index] = gray.flatten()

print(targets)

#NUMBER OF INPUTS/OUTPUTS THE NN WILL HAVE
#THE LAYER LIST WORKS AS: 
# FIRST HIDDEN LAYER
# N HIDDEN LAYERS
# OUTPUT LAYER
layers = [
    FCLayer(inputs.shape[1], 100, 0, 'ReLU'),
    FCLayer(100, 100, 0, 'ReLU'),
    FCLayer(100, 100, 0, 'ReLU'),
    FCLayer(100, targets.shape[1], 0, 'sigmoid'),
]

datasets = utilities.train_valid_test_split(inputs, targets, train_size = .8, valid_size=.1, test_size=.1)

nn_test = nn.NeuralNetwork(
    layers,
    initialize_weight_bias = 'Widrow',
    num_epoch_train = 10000,
    performance_function = 'cross-entropy',
    optimizer= 'Adam',
    optimParams= {'Beta1' : 0.9, 'Beta2' : 0.999, 'Alpha' : 0.01},
    ming_grad = 1e-10,
    min_perf = 1e-5,
    learning_method = 'batch',
    batch_size = 30
)

nn_test.train(datasets['Train'][0].T, datasets['Train'][1].T, validation_data = None, verbose = True)

res = nn_test.predict(datasets['Test'][0].T).T

input(datasets['Test'][1])
print(res)


utilities.plot_classification(datasets['Test'][1], res)