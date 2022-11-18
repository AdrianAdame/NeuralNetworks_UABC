from .Layer import Layer
import numpy as np

np.set_printoptions(precision=20)

"""
TODO:
-> Verify if forward propagation is well implemented
-> In a flatten layer, this has no W, so how you backpropagate We and delta
-> Optimize code
"""

class Flatten(Layer):
    def __init_(self):
        self.type = "Flatten Layer (Flatten)"
        self.has_w = False
    
    def forward_propagation(self, A_prev):
        self.A_prev_shape = A_prev.shape
        self.A = A_prev.reshape(-1,1)

        return self.A

    def back_propagation(self, A, WE_prev, delta_prev):
        return A.reshape(self.A_prev_shape)
