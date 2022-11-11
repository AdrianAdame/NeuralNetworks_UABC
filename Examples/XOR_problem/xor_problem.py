from Layer_definitions.FCLayer import FCLayer
import utils.utilities as utilities
import NeuralNetwork as nn
import numpy as np

#TEST XOR DATA

inputs = np.array([
    [0 , 0], 
    [0 , 1],
    [1 , 0],
    [1 , 1]
])

targets = np.array([
    [0], 
    [1],
    [1],
    [0]
])


#NUMBER OF INPUTS/OUTPUTS THE NN WILL HAVE
#THE LAYER LIST WORKS AS: 
# FIRST HIDDEN LAYER
# N HIDDEN LAYERS
# OUTPUT LAYER
layers = [
    FCLayer(inputs.shape[1], 30, 0.2, 'tanh'),
    FCLayer(30, 30, 0.2, 'tanh'),
    FCLayer(30, 30, 0.2, 'tanh'),
    FCLayer(30, targets.shape[1], 0, 'tanh')
]

#TO PERFORM A GOOD TRAINING WE NEED TO NORMALIZE DATA
inputs_normalized, settingsX = utilities.mapminmax(inputs)
targets_normalized, settingsY = utilities.mapminmax(targets)

#NOW NORMALIZED WE CAN SPLIT INTO TRAIN - VALIDATION - TEST
#datasets = utilities.train_valid_test_split(inputs_normalized, targets_normalized, train_size = .6, valid_size=.2, test_size=.2)

nn_test = nn.NeuralNetwork(
    layers,
    initialize_weight_bias = 'Widrow', 
    num_epoch_train = 1000,
    performance_function = 'mse',
    optimizer = 'Adam',
    optimParams = {'Beta1' : 0.9, 'Beta2' : 0.999, 'Alpha' : 0.001},
    ming_grad = 1e-10,
    min_perf = 1e-5,
    learning_method = 'batch',
    batch_size = 2
)

nn_test.train(inputs_normalized.T, targets_normalized.T, verbose = True)

res = nn_test.predict(inputs_normalized.T)

nn_predict,_ = utilities.mapminmax(res.T, PS = settingsY, method='inverted')

print(nn_predict)

perf_values = nn_test.create_performance_graph()

utilities.plot_generic_graph(perf_values[0], perf_values[1])
