from .Activations import *
from utils.NeuralNetwork_Utils import *
from .Layer import Layer
import numpy as np

np.set_printoptions(precision=20)


class FCLayer(Layer):

    def __init__(self, n_inputs, n_neurons, dropout_ratio, activation_function):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.W = None
        self.b = None
        self.WE = None
        self.act, self.d_act = activationFunctions.get(activation_function)

        self.dropout_ratio = dropout_ratio
        
    def setLayerID(self, id):
        self.layerID = id
    
    def dropout(self, y):
        m, n = y.shape
        ym = np.zeros((m,n)).flatten()
        num = round(m*n*(1 - self.dropout_ratio))
        idx = np.random.choice(m * n, num, replace = False)
        ym[idx] = 1 / (1 - self.dropout_ratio)

        return ym.reshape((m,n))

    def initialize_w_b(self, method):
        if method == 'Widrow':
            beta = 0.7 * self.n_neurons ** (1/self.n_inputs)
            W = -0.5 + np.random.rand(self.n_neurons, self.n_inputs)
            for i in range(self.n_neurons):
                W[i , :] = beta * W[i , :]/np.linalg.norm(W[i , :])
            b = -beta + 2 * beta * np.random.rand(self.n_neurons, 1)
            
            self.W = W
            self.b = b

            self.WE = np.hstack((self.W, self.b))
        
        if method == 'Manual':
            beta = 0.7 * self.n_neurons ** (1/self.n_inputs)
            W = -0.5 + np.ones((self.n_neurons, self.n_inputs))
            for i in range(self.n_neurons):
                W[i , :] = beta * W[i , :]/np.linalg.norm(W[i , :])
            b = -beta + 2 * beta * np.ones((self.n_neurons, 1))
            
            self.W = W
            self.b = b

            self.WE = np.hstack((self.W, self.b))
        
        if method == 'Random':
            W = np.random.rand(self.n_neurons, self.n_inputs)
            b = np.random.rand(self.n_neurons, 1)
            self.W = W
            self.b = b

            self.WE = np.hstack((self.W, self.b))
        
        if method == 'Glorot' or method == 'Xavier':
            factor = 6.0

            if self.act.__name__ == 'sigmoid':
                factor = 2.0
            
            init_bound = np.sqrt(factor / (self.n_inputs + self.n_neurons))

            W = np.random.uniform(-init_bound, init_bound, (self.n_neurons, self.n_inputs)).astype(float)
            b = np.random.uniform(-init_bound, init_bound, (self.n_neurons, 1)).astype(float)

            self.W = W
            self.b = b
            
            self.WE = np.hstack((self.W, self.b))

    def forward_propagation(self, A_prev):
        self.A_prev = A_prev
        self.A_extended = np.vstack((self.A_prev, np.ones((1, self.A_prev.shape[1]))))
        self.Z = np.matmul(self.WE, self.A_extended)
        self.A = self.act(self.Z)
        
        #NUEVO
        if self.dropout_ratio != 0:
            self.A = self.A * self.dropout(self.A)

        return self.A
    
    def back_propagation(self, A, WE_prev, delta_prev):
        self.A_extended = np.vstack((self.A_prev, np.ones((1, self.A_prev.shape[1]))))
        df_net = self.d_act(A)
        delta = np.multiply(df_net, np.matmul(WE_prev[:,:-1].T, delta_prev))
        self.delta = delta
        self.dE_dWE = np.matmul(delta, self.A_extended.T)
        
        return self.WE, delta

    def back_propagation_lastLayer(self, A, error):
        self.A_extended = np.vstack((A, np.ones((1, A.shape[1]))))
        df_net = self.d_act(self.A)
        delta = np.multiply(df_net, (-2 * error))
        self.WE_prev = self.WE
        self.dE_dWE = np.matmul(delta, self.A_extended.T)

        return self.WE_prev, delta
    
    def back_propagation_lastLayer_crossEntropy(self, A, error):
        self.A_extended = np.vstack((A, np.ones((1, A.shape[1]))))
        delta = error
        self.WE_prev = self.WE
        self.delta = delta
        self.dE_dWE = np.matmul(delta, self.A_extended.T)
        return self.WE_prev, delta

    def getWE(self):
        return self.WE, self.WE.shape
    
    def getdWE(self):
        return self.dE_dWE, self.dE_dWE.shape