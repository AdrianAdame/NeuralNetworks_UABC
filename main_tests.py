from Layer_definitions.FCLayer import FCLayer
import utils.utilities as utilities
import NeuralNetwork as nn
import numpy as np
import pandas as pd


from sklearn.preprocessing import MinMaxScaler

from sklearn.datasets import load_iris

# data_file = pd.read_csv("/Users/adrianadame/Git/NeuralNetworks_UABC/Examples/Classification_Example/ovarian_dataset.dat", sep="\t", header=None)
# inputs = np.array(data_file.iloc[:, :-1])
# Y = np.array(data_file.iloc[:,-1:])
# targets = utilities.getClasses_Classification(Y)

dataset = load_iris()
inputs = dataset.data
targets = dataset.target.reshape(-1,1)
targets = utilities.getClasses_Classification(targets)

#NUMBER OF INPUTS/OUTPUTS THE NN WILL HAVE
#THE LAYER LIST WORKS AS: 
# FIRST HIDDEN LAYER
# N HIDDEN LAYERS
# OUTPUT LAYER
layers = [
    FCLayer(inputs.shape[1], 4, 0, 'ReLU'),
    FCLayer(4, 10, 0, 'ReLU'),
    FCLayer(10, 4, 0, 'ReLU'),
    FCLayer(4, targets.shape[1], 0, 'softmax')
]

print("Number of inputs: ", inputs.shape[1])
print("Number of outputs: ", targets.shape[1])

#TO PERFORM A GOOD TRAINING WE NEED TO NORMALIZE DATA
scaler = MinMaxScaler(feature_range = (0,1))

inputs_normalized = scaler.fit_transform(inputs)

datasets = utilities.train_valid_test_split(inputs, targets, train_size = .6, valid_size=.2, test_size=.2)


nn_test = nn.NeuralNetwork(
    layers,
    initialize_weight_bias = 'Widrow', 
    num_epoch_train = 300,
    performance_function = 'cross-entropy',
    optimizer= 'Adam',
    optimParams= {'Beta1' : 0.9, 'Beta2' : 0.999, 'Alpha' : 0.001},
    ming_grad = 1e-10,
    min_perf = 1e-5,
    learning_method = 'mini-batch',
    batch_size = 32
)

nn_test.train(datasets['Train'][0].T, datasets['Train'][1].T, verbose = True)

res = nn_test.predict(datasets['Test'][0].T).T

input(datasets['Test'][1])
print(res)

utilities.plot_classification(datasets['Test'][1], res)