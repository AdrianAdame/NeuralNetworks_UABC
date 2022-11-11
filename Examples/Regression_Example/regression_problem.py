from Layer_definitions.FCLayer import FCLayer
import utils.utilities as utilities
import NeuralNetwork as nn
import numpy as np
import pandas as pd


data_file_a = pd.read_csv('bodyfat_dataset.dat', sep="\t", header=None)

inputs = np.array(data_file_a.iloc[:, :-1])
targets = np.array(data_file_a.iloc[:,-1:])

#NUMBER OF INPUTS/OUTPUTS THE NN WILL HAVE
#THE LAYER LIST WORKS AS: 
# FIRST HIDDEN LAYER
# N HIDDEN LAYERS
# OUTPUT LAYER
layers = [
    FCLayer(inputs.shape[1], 5, 'tanh'),
    FCLayer(5, 3, 'tanh'),
    FCLayer(3, 3, 'tanh'),    
    FCLayer(3, 3, 'tanh'),    
    FCLayer(3, 3, 'tanh'),    
    FCLayer(3, 3, 'tanh'),    
    FCLayer(3, 3, 'tanh'),    
    FCLayer(3, 3, 'tanh'),
    FCLayer(3, targets.shape[1], 'tanh')
]

#TO PERFORM A GOOD TRAINING WE NEED TO NORMALIZE DATA
inputs_normalized, settingsX = utilities.mapminmax(inputs)
targets_normalized, settingsY = utilities.mapminmax(targets)

#NOW NORMALIZED WE CAN SPLIT INTO TRAIN - VALIDATION - TEST
datasets = utilities.train_valid_test_split(inputs_normalized, targets_normalized, train_size = .6, valid_size=.2, test_size=.2)

nn_test = nn.NeuralNetwork(
    layers,
    initialize_weight_bias = 'Widrow', 
    num_epoch_train = 5000,
    performance_function = 'mse',
    optimizer = 'NAdam',
    optimParams = {'Beta1' : 0.9, 'Beta2' : 0.999, 'Alpha' : 0.001},
    ming_grad = 1e-5,
    min_perf = 1e-5,
    learning_method = 'mini-batch',
    batch_size = 40
)

nn_test.train(datasets['Train'][0].T, datasets['Train'][1].T, verbose = True)

res = nn_test.predict(datasets['Test'][0].T)
nn_predict,_ = utilities.mapminmax(res.T, PS = settingsY, method='inverted')

print(nn_predict)

perf_values = nn_test.create_performance_graph()
utilities.plot_generic_graph(perf_values[0], perf_values[1])
