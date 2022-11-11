from Layer_definitions.FCLayer import FCLayer
import utils.utilities as utilities
import NeuralNetwork as nn
import numpy as np

from sklearn.preprocessing import MinMaxScaler

inputs = np.array([
    [4.7 , 6.0],
    [6.1 , 3.9],
    [2.9 , 4.2],
    [7.0 , 5.5],
])

targets = np.array([
    [3.52 , 4.02],
    [5.43 , 6.23],
    [4.95 , 5.76],
    [4.70 , 4.28],
])

#NUMBER OF INPUTS/OUTPUTS THE NN WILL HAVE
#THE LAYER LIST WORKS AS: 
# FIRST HIDDEN LAYER
# N HIDDEN LAYERS
# OUTPUT LAYER
layers = [
    FCLayer(inputs.shape[1], 3, 0, 'tanh'),
    FCLayer(3, 3, 0, 'ReLU'),
    FCLayer(3, targets.shape[1], 0, 'tanh')
]

#TO PERFORM A GOOD TRAINING WE NEED TO NORMALIZE DATA
scaler = MinMaxScaler(feature_range=(0,1))

inputs_normalized = scaler.fit_transform(inputs)
targets_normalized = scaler.fit_transform(targets)

nn_test = nn.NeuralNetwork(
    layers,
    initialize_weight_bias = 'Widrow',
    num_epoch_train = 300,
    performance_function = 'mse',
    optimizer = 'Adam',
    optimParams = {'Beta1' : 0.9, 'Beta2' : 0.999, 'Alpha' : 0.01},
    ming_grad = 1e-10,
    min_perf = 1e-5,
    learning_method = 'batch',
    batch_size = 20
)

#WE TRAIN THE NEURAL NETWORK GIVEN THE INPUT AND TARGETS NORMALIZED
nn_test.train(inputs_normalized.T, targets_normalized.T, verbose = True)

#WE CALCULATE A PREDICTION AFTER TRAINING
res = nn_test.predict(inputs_normalized.T)
nn_predict = scaler.inverse_transform(res.T)

#OUTPUT
print(targets)
print(nn_predict)

#CREATION OF PERFORMACE GRAPH - LOSS PER EPOCH
perf_values = nn_test.create_performance_graph()
utilities.plot_generic_graph(perf_values[0], perf_values[1])

#IN THE CASE OF REGRESSION WE PLOT THE CORRELATION COEFICIENT
utilities.plot_regression_graph(targets, nn_predict)
