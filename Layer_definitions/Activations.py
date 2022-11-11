from scipy.special import expit as logistic_sigmoid
import numpy as np

def identity(x):
    """Simply leave the input array unchanged."""
    # Nothing to do
    return x

def d_identity(x):
    """Simply leave the input array unchanged."""
    # Nothing to do
    return x

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - (np.power(np.tanh(x), 2))

def sigmoid(x):
    return logistic_sigmoid(x, out=x)

def d_sigmoid(x):
    return (1 - sigmoid(x)) * sigmoid(x)

def ReLU(x):
    return np.maximum(x, 0)

def d_ReLU(x):
    return np.greater(x, 0).astype(int)

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

def notI(x):
    return 0

activationFunctions = {
    'tanh'    : (tanh, d_tanh),
    'sigmoid' : (sigmoid, d_sigmoid),
    'ReLU'    : (ReLU, d_ReLU),
    'softmax' : (softmax, notI),
    'identity': (identity, d_identity)
}