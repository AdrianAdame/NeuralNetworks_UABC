from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from Layer_definitions.FCLayer import FCLayer
import utils.utilities as utilities
import NeuralNetwork as nn
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.neural_network import MLPRegressor

# data_file_a = pd.read_csv("reaction_dataset.dat", sep=" ", header=None)

# inputs = np.array(data_file_a.iloc[:, 1:-1])
# targets = np.array(data_file_a.iloc[:,-1:])

inputs, targets = make_regression(n_samples=1000, n_features=5, noise=0.1)
targets = targets.reshape(-1,1)

#NUMBER OF INPUTS/OUTPUTS THE NN WILL HAVE
#THE LAYER LIST WORKS AS: 
# FIRST HIDDEN LAYER
# N HIDDEN LAYERS
# OUTPUT LAYER
layers = [
    FCLayer(inputs.shape[1], 100, 0, 'tanh'),
    FCLayer(100,100,0.2, 'tanh'),
    FCLayer(100,100,0.2, 'tanh'),
    FCLayer(100, targets.shape[1], 0, 'tanh')
]

#TO PERFORM A GOOD TRAINING WE NEED TO NORMALIZE DATA
scaler = MinMaxScaler(feature_range=(-1,1))

inputs_normalized = scaler.fit_transform(inputs)
targets_normalized = scaler.fit_transform(targets)

#NOW NORMALIZED WE CAN SPLIT INTO TRAIN - VALIDATION - TEST
#datasets = utilities.train_valid_test_split(inputs_normalized, targets_normalized, train_size = .6, valid_size=.2, test_size=.2)

#Split in train, test and validation
X_train, X_test, y_train, y_test = train_test_split(inputs_normalized, targets_normalized, test_size=0.2) # Try with random_state = 42
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25) # Try with random_state = 42

nn_test = nn.NeuralNetwork(
    layers,
    initialize_weight_bias = 'Widrow',
    num_epoch_train = 300,
    performance_function = 'mse',
    optimizer = 'Adam',
    optimParams = {'Beta1' : 0.9, 'Beta2' : 0.999, 'Alpha' : 0.001},
    ming_grad = 1e-5,
    min_perf = 1e-10,
    learning_method = 'mini-batch',
    batch_size = 32
)

nn_test.train(X_train.T, y_train.T, verbose = True)

res = nn_test.predict(X_test.T)
nn_predict = scaler.inverse_transform(res.T)

perf_values = nn_test.create_performance_graph()
utilities.plot_generic_graph(perf_values[0], perf_values[1])

utilities.plot_regression_graph(y_test, nn_predict)