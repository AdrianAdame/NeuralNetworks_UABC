class Layer:
    def __init__(self):
        self.n_inputs = None
        self.n_neurons = None
        self.W = None
        self.b = None
        self.WE = None
        self.act, self.d_act = None, None
    
    def initialize_w_b(self, method):
        raise NotImplementedError
    
    def forward_propagation(self, input):
        raise NotImplementedError
    
    def backward_propagation(self, input):
        raise NotImplementedError