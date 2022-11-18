from .Activations import *
from .Layer import Layer
import numpy as np
from scipy import signal

np.set_printoptions(precision=20)

"""
TODO:
-> Verify if forward propagation is well implemented
-> Verify how delta and We is calculated
-> Optimize code
"""

class Conv2DLayer(Layer):
    def __init__(self, n_inputs, n_neurons, n_w3, activation_function):
        self.type = "Convolutional Layer (Conv2DLayer)"
        self.has_w = True


        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.filters = n_w3
        self.W = None
        self.b = None
        self.WE = None

        self.act, self. d_act = activationFunctions.get(activation_function)
    


    def initialize_w_b(self, method):
        if method == 'Widrow':
            beta = 0.7 * self.n_neurons ** (1/self.n_inputs)
            W = -0.5 + np.random.rand(self.n_neurons, self.n_inputs, self.filters)
            for i in range(self.n_neurons):
                W[i , :, :] = beta * W[i , :]/np.linalg.norm(W[i , :, :])
            b = -beta + 2 * beta * np.random.rand(self.n_neurons, 1, self.filters)
            
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
        self.A_extended = np.vstack((self.A, np.ones((1, self.A_prev.shape[1], self.filters))))
        self.Z = np.matmul(self.WE, self.A_extended)
        self.A = self.act(np.sum(np.sum(self.Z)))

        return self.A
    
    def back_propagation(self, A, WE_prev, delta_prev):
        self.A_extended = np.vstack((self.A_prev, np.ones((1, self.A_prev.shape[1], self.filters))))
        df_net = self.d_act(A)
        delta = np.multiply(df_net, np.matmul(np.rot90(WE_prev[:,:-1], 2), delta_prev))
        self.dE_dWE = np.multiply(np.rot90(delta, 2), self.A_extended) 

        return self.WE, delta

    def getWE(self):
        return self.WE, self.WE.shape
    
    def getdWE(self):
        return self.dE_dWE, self.dE_dWE.shape