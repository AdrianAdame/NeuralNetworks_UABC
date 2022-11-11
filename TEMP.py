import numpy as np

np.set_printoptions(precision=20)

def tanh(x):
    return np.tanh(x)

def d_tanh(x):
    return 1 - (np.power(np.tanh(x), 2))

def sigmoid(x):
    return np.where(x > 0, 1. / (1. + np.exp(-x)), np.exp(x) / (np.exp(x) + np.exp(0)))

def d_sigmoid(x):
    return (1 - x) * x

def ReLU(x):
    return np.maximum(x, 0)

def d_ReLU(x):
    return np.greater(x, 0).astype(int)

def softmax(x):
    ex = np.exp(x)
    return ex / np.sum(ex) 

def notI(x):
    return 0

activationFunctions = {
    'tanh'    : (tanh, d_tanh),
    'sigmoid' : (sigmoid, d_sigmoid),
    'ReLU'    : (ReLU, d_ReLU),
    'softmax' : (softmax, notI)
}

class FCLayer:

    def __init__(self, n_inputs, n_neurons, activation_function):
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.W = None
        self.b = None
        self.WE = None
        self.act, self.d_act = activationFunctions.get(activation_function)
        
    def setLayerID(self, id):
        self.layerID = id
    
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
            W = 2 * np.random.rand(self.n_neurons, self.n_inputs) - 1
            b = np.random.rand(self.n_neurons, 1)
            self.W = W
            self.b = b

            self.WE = np.hstack((self.W, self.b))

    def forward_propagation(self, A_prev):
        self.A_prev = A_prev
        self.A_extended = np.vstack((self.A_prev, np.ones((1, self.A_prev.shape[1]))))
        self.Z = np.dot(self.WE, self.A_extended)
        self.A = self.act(self.Z)

        return self.A
    
    def back_propagation(self, A, WE_prev, delta_prev):
        self.A_extended = np.vstack((self.A_prev, np.ones((1, self.A_prev.shape[1]))))
        df_net = self.d_act(A)
        delta = np.multiply(df_net, np.dot(WE_prev[:,:-1].T, delta_prev))

        self.delta = delta
        self.dE_dWE = np.dot(delta, self.A_extended.T)
        
        return self.WE, delta

    def back_propagation_lastLayer(self, A, error):
        self.A_extended = np.vstack((A, np.ones((1, A.shape[1]))))
        df_net = self.d_act(self.A)
        delta = np.multiply(df_net, (-2 * error))
        self.WE_prev = self.WE
        self.dE_dWE = np.dot(delta, self.A_extended.T)

        return self.WE_prev, delta
    
    def back_propagation_lastLayer_crossEntropy(self, A, error):
        self.A_extended = np.vstack((A, np.ones((1, A.shape[1]))))
        delta = error
        self.WE_prev = self.WE
        self.dE_dWE = np.dot(delta, self.A_extended.T)

        return self.WE_prev, delta

    def getWE(self):
        return self.WE, self.WE.shape
    
    def getdWE(self):
        return self.dE_dWE, self.dE_dWE.shape

def cross_entropy(Targets, Predict):

    #H = Targets * np.log(Predict) + (1 - Targets) * np.log(1 - Predict)
    #return - np.sum(np.sum(H))/(Targets.shape[1] * Predict.shape[0])

    return -np.sum(Targets * np.log(Predict))

def matrix_to_vector(matrix_arr):
    vector = matrix_arr[0].flatten()

    matrix_arr.pop(0)

    for matrix_index in range(len(matrix_arr)):
        vector = np.append(vector, matrix_arr[matrix_index].flatten())
    
    return vector.reshape((-1,1))

def vector_to_matrix(vector, w_sizes_tuples):
    vector = vector
    matrix_arr = list()
    begin = 0
    end = 0
    for matrix_tuple in w_sizes_tuples:
        end = matrix_tuple[0] * matrix_tuple[1] + end
        matrix_arr.append(vector[begin:end].reshape(matrix_tuple))
        begin = end
    
    return matrix_arr

def gradientDescent(wt, dw, optimParams):
    lr = optimParams['lr']
    wt = wt - lr * dw

    return wt

def RMSProp(wt, dw, optimParams):
    vt = 0
    eps = np.finfo(np.float64).eps
    Beta = optimParams['Beta']
    Alpha = optimParams['Alpha']
    
    vt = Beta * vt + (1 - Beta) * dw ** 2.0
    wt = wt - Alpha /(np.sqrt(vt + eps)) * dw
    
    return wt

def multiClass(layers, X, D):
    alpha = 0.9999
    N = 5

    for k in range(N):
        x = X[:,:,k].reshape(25,1)
        
        d = D[k,:].T.reshape(5,1)

        #WE GET ALL INPUTS IN A
        A = x
                    
        #FOR EACH LAYER WE PERFORM THE FORWARD PROPAGATION
        for layer in layers:
            A = layer.forward_propagation(A)

        e = A - d
        
        perf = cross_entropy(d, A)

        WE, delta = layers[-1].back_propagation_lastLayer_crossEntropy(layers[-1].A_prev, e)

        for layer in reversed(layers[:-1]):
            WE, delta = layer.back_propagation(layer.A, WE, delta)

        print(perf)

        if perf < 1e-1:
            break
        
        WE_array = list()
        dWE_array = list()
        WE_shapes = list()

        for layer in reversed(layers):
            WE, shape = layer.getWE()
            dWE, _ = layer.getdWE()
            WE_array.append(WE)
            WE_shapes.append(shape)
                    
            dWE_array.append(dWE)
        
        we_vector = matrix_to_vector(WE_array)
        dwe_vector = matrix_to_vector(dWE_array)

        optimized_vector = RMSProp(we_vector, dwe_vector, {'Beta' : 0.9, 'Alpha' : 0.001})

        matrixes = vector_to_matrix(optimized_vector, WE_shapes)
        
        matrix_size = len(matrixes)
        for index in range(matrix_size):
            layers[matrix_size - index - 1].WE = matrixes[index]


x = np.zeros((5,5,5))

x[:,:,0] = np.array([
    [0 , 1 , 1 , 0 , 0],
    [0 , 0 , 1 , 0 , 0],
    [0 , 0 , 1 , 0 , 0],
    [0 , 0 , 1 , 0 , 0],
    [0 , 1 , 1 , 1 , 0]
])

x[:,:,1] = np.array([
    [1 , 1 , 1 , 1 , 0],
    [0 , 0 , 0 , 0 , 1],
    [0 , 1 , 1 , 1 , 0],
    [1 , 0 , 1 , 0 , 0],
    [1 , 1 , 1 , 1 , 1]
])
x[:,:,2] = np.array([
    [1 , 1 , 1 , 1 , 0],
    [0 , 0 , 0 , 0 , 1],
    [0 , 1 , 1 , 1 , 0],
    [0 , 0 , 0 , 0 , 1],
    [0 , 1 , 1 , 1 , 0]
])
x[:,:,3] = np.array([
    [0 , 0 , 0 , 1 , 0],
    [0 , 0 , 1 , 1 , 0],
    [0 , 1 , 0 , 1 , 0],
    [1 , 1 , 1 , 1 , 1],
    [0 , 0 , 0 , 1 , 0]
])
x[:,:,4] = np.array([
    [1 , 1 , 1 , 1 , 1],
    [1 , 0 , 0 , 0 , 0],
    [1 , 1 , 1 , 1 , 0],
    [0 , 0 , 1 , 0 , 1],
    [1 , 1 , 1 , 1 , 0]
])

D = np.array([
    [1 , 0 , 0 , 0 , 0],
    [0 , 1 , 0 , 0 , 0],
    [0 , 0 , 1 , 0 , 0],
    [0 , 0 , 0 , 1 , 0],
    [0 , 0 , 0 , 0 , 1]
])

#W1 = 2 * np.random.rand(50,25) - 1
#W2 = 2 * np.random.rand(5,50) - 1

layers = [
    FCLayer(25, 50, 'ReLU'),
    FCLayer(50, 5, 'softmax')
]

layerID = 1
for layer in layers:
    layer.setLayerID(layerID)
    layer.initialize_w_b("Random")
    layerID += 1

for epoch in range(100):
    multiClass(layers, x, D)

N = 5

for k in range(N):
    X = x[:,:,k].reshape(25,1)
    #FORWARD PROPAGATION TO PREDICT
    #WE GET ALL INPUTS IN A
    A = X
                    
    #FOR EACH LAYER WE PERFORM THE FORWARD PROPAGATION
    for layer in layers:
        A = layer.forward_propagation(A)
    
    print(A)