from Layer_definitions.FCLayer import FCLayer
import utils.utilities as utilities
import NeuralNetwork as nn
import numpy as np
import pandas as pd


from sklearn.preprocessing import MinMaxScaler

data_file = pd.read_csv("ex2data2.txt", sep=",", header=None)

inputs = np.array(data_file.iloc[:, :-1])
Y = np.array(data_file.iloc[:,-1:])
targets = utilities.getClasses_Classification(Y)


#NUMBER OF INPUTS/OUTPUTS THE NN WILL HAVE
#THE LAYER LIST WORKS AS: 
# FIRST HIDDEN LAYER
# N HIDDEN LAYERS
# OUTPUT LAYER
layers = [
    FCLayer(inputs.shape[1], 100, 0, 'ReLU'),
    FCLayer(100, 100, 0, 'ReLU'),
    FCLayer(100, 100, 0, 'ReLU'),
    FCLayer(100, 100, 0, 'ReLU'),
    FCLayer(100, 100, 0, 'ReLU'),
    FCLayer(100, 100, 0.2, 'ReLU'),
    FCLayer(100, targets.shape[1], 0, 'softmax')
]

#TO PERFORM A GOOD TRAINING WE NEED TO NORMALIZE DATA
scaler = MinMaxScaler(feature_range = (0,1))

inputs_normalized = scaler.fit_transform(inputs)

datasets = utilities.train_valid_test_split(inputs, targets, train_size = .6, valid_size=.2, test_size=.2)


nn_test = nn.NeuralNetwork(
    layers,
    initialize_weight_bias = 'Widrow',
    num_epoch_train = 1000,
    performance_function = 'cross-entropy',
    #optimizer = 'RMSProp',
    #optimParams = {'Beta' : 0.999, 'Alpha' : 0.0001},
    optimizer= 'Adam',
    optimParams= {'Beta1' : 0.9, 'Beta2' : 0.999, 'Alpha' : 0.01},
    ming_grad = 1e-10,
    min_perf = 1e-5,
    learning_method = 'batch',
    batch_size = 20
)

nn_test.train(datasets['Train'][0].T, datasets['Train'][1].T, verbose = True)

res = nn_test.predict(datasets['Test'][0].T).T

input(datasets['Test'][1])
print(res)

utilities.plot_classification(datasets['Test'][1], res)