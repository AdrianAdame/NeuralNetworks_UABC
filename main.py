from Layer_definitions.FCLayer import FCLayer
import utils.utilities as utilities
import NeuralNetwork as nn
import numpy as np
import pandas as pd

np.set_printoptions(precision= 20)

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
    FCLayer(inputs.shape[1], 500, 0, 'ReLU'),
    FCLayer(500,400,0, 'ReLU'),
    FCLayer(400,300,0, 'ReLU'),
    FCLayer(300,200,0, 'ReLU'),
    FCLayer(200,100,0, 'ReLU'),
    FCLayer(100,50,0, 'ReLU'),
    FCLayer(50,50,0, 'ReLU'),
    FCLayer(50, targets.shape[1], 0, 'softmax')
]

#TO PERFORM A GOOD TRAINING WE NEED TO NORMALIZE DATA
scaler = MinMaxScaler(feature_range = (0,1))

inputs_normalized = scaler.fit_transform(inputs)

datasets = utilities.train_valid_test_split(inputs, targets, train_size = .6, valid_size=.2, test_size=.2)


nn_test = nn.NeuralNetwork(
    layers,
    initialize_weight_bias = 'Xavier',
    num_epoch_train = 100,
    performance_function = 'cross-entropy',
    optimizer= 'Adam',
    optimParams= {'Beta1' : 0.9, 'Beta2' : 0.999, 'Alpha' : 0.001},
    ming_grad = 1e-10,
    min_perf = 1e-5,
    learning_method = 'batch',
    batch_size = 30
)

nn_test.train(datasets['Train'][0].T, datasets['Train'][1].T, validation_data = [datasets['Validation'][0].T, datasets['Validation'][1].T], verbose = True)

res = nn_test.predict(datasets['Test'][0].T).T

input(datasets['Test'][1])
print(res)

utilities.plot_classification(datasets['Test'][1], res)

print("Save model?")
save = input()
if save == "1":
    nn_test.save_model()