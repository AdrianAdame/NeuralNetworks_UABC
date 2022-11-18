import os
from tabulate import tabulate
from utils.NeuralNetwork_Utils import *
import glob

np.set_printoptions(precision=20)

"""
TODO:
-> Add function self.add() to append layers without a list
-> Add parameters to initialize n layers generalized
-> Optimize code (Verify parts where a code repeats and place it in a separate function)
-> Verify when to update weights in validation function
"""

class NeuralNetwork:
    def __init__(self, layers, initialize_weight_bias = 'Widrow', num_epoch_train = 300,  performance_function = 'sse', optimizer = 'RMSProp', optimParams = {'Alpha' : 0.9, 'Beta' : 0.001},  ming_grad = 1e-5, min_perf = 1e-5, learning_method = 'batch', batch_size = 30):

        #NUMBER OF ITERATIONS TO TRAIN THE NETWORK
        self.num_epochs = num_epoch_train

        #LEARNING METHOD = {BATCH, MINI BATCH - > batch_size, SGD}
        self.learning_method = learning_method
        
        if learning_method == 'mini-batch':
            self.batch_size = batch_size
                
        #PERFORMANCE FUNCTION
        self.perfunc, self.perfunc_name = performance_functions.get(performance_function)
        self.min_grad = ming_grad
        self.min_perf = min_perf

        #HIDDEN LAYERS SPECIFICATIONS/NUMBER
        self.layers = layers

        #SOLVER METHOD
        self.optimizer_name = optimizer
        self.optimizer = optimizers_implemented.get(optimizer)

        self.optimParams = optimParams

        #IF HIDDEN_LAYER EXIST
        #if((layers_type != '' and hidden_layers != (100,) and n_inputs != 0 and n_targets != 0) and layers == None):
        #    layers = []

        for layer in self.layers:
            layer.initialize_w_b(initialize_weight_bias)            

    def summary(self):
        print("\t\t*** NEURAL NETWORK SUMMARY ***\n\n")
        headersLayersTable = ["Layer (type)", "Shape (IN , OUT)", "No. Weights" , "No. biases"]

        bodyLayersTable = [[
            layer.type, 
            (layer.n_inputs, layer.n_neurons), 
            layer.n_neurons * layer.n_inputs, 
            layer.n_neurons
        ] for layer in self.layers]
        
        print("\t\t << Layer configuration >>")
        print(tabulate(bodyLayersTable, headersLayersTable))

    def _update_weigths(self):
        WE_array = list()
        dWE_array = list()
        WE_shapes = list()

        for layer in reversed(self.layers):
            WE, shape = layer.getWE()
            dWE, _ = layer.getdWE()            
            WE_array.append(WE)
            WE_shapes.append(shape)
            dWE_array.append(dWE)                    
        
        
        vector_to_optimize = matrix_to_vector(WE_array)
        gradient_vector = matrix_to_vector(dWE_array)
                    
        optimized_vector = self.optimizer(vector_to_optimize, gradient_vector, self.optimParams)

        matrix_obtained = vector_to_matrix(optimized_vector, WE_shapes)

        matrix_size = len(matrix_obtained)
        for index in range(matrix_size):
            self.layers[matrix_size - index - 1].WE = matrix_obtained[index]
        
        return gradient_vector

    def _validate(self, inputs, targets, verbose = False):
        #THE VALIDATION IS VIA BATCH METHOD

        #WE STORE TRAINED WEIGHTS IN CASE THIS IS THE BEST PERFORMANCE
        best_WE = list()

        #Fast forward
        #FOR EACH LAYER WE PERFORM THE FORWARD PROPAGATION
        A = inputs
        for layer in self.layers:
            A = layer.forward_propagation(A)
        
        best_perf = self.perfunc(targets, A)

        for layer in self.layers:
            best_WE.append(layer.WE)
        
        #We train the model again, verifying that the new weights are better than the ones we already calculate
        for epoch in range(self.num_epochs):
            #WE GET ALL INPUTS IN A
            A = inputs
                    
            #FOR EACH LAYER WE PERFORM THE FORWARD PROPAGATION
            for layer in self.layers:
                A = layer.forward_propagation(A)
                
            #WE CALCULATE THE ERROR GIVEN THE BATCHED TARGETS AND A
            if self.perfunc_name == 'CROSS-ENTROPY':
                e = A - targets
            else:
                e = targets - A                    
                
            #WE CALCULATE THE PERFORMANCE FROM THE TARGETS AND RESULTS FROM THE FORWARD PROPAGATION
            perf = self.perfunc(targets, A)

            if self.perfunc_name == 'CROSS-ENTROPY':
                WE, delta = self.layers[-1].back_propagation_lastLayer_crossEntropy(layer.A_prev, e)
            else: 
                WE, delta = self.layers[-1].back_propagation_lastLayer(layer.A_prev, e)
                
            for layer in reversed(self.layers[:-1]):
                WE, delta = layer.back_propagation(layer.A, WE, delta)

            if verbose == True:
                print("Epoch {} : {} : {}".format(epoch, self.perfunc_name, perf))
                
            if perf >= best_perf:
                best_perf = perf
                _ = self._update_weigths()
            else:
                i = 0
                for layer in self.layers:
                     layer.WE = best_WE[i]
                     i += 1

    def train(self, inputs, targets, validation_data = None, verbose = False):
        self.perf = list()
        
        if self.learning_method == 'mini-batch':
            #WE TRY LEARNING WITH BATCH USING BATCH SIZE
            input_size = inputs.shape[1]
            batches_per_epoch = -(-input_size // self.batch_size)

            self.numEpochs = 0
            for epoch in range(self.num_epochs):
                #WE STORE THE PERFORMANCE OBTAINED PER BATCH IF WE WANT TO SHOW A PERFORMANCE GRAPH
                perf_per_batch = 0

                #WE INITIALIZE A NEW BATCH GIVEN THE BATCHES PER EPOCH
                for batch in range(batches_per_epoch):

                    #WE CALCULATE THE BEGIN AND END INDEXES TO GET THE BATCH
                    begin = batch * self.batch_size
                    end = min(begin + self.batch_size, inputs.shape[1]-1)
                    
                    #WE GET OUR BATCH INPUT IN A
                    A = inputs[:,begin:end]
                    
                    #FOR EACH LAYER WE PERFORM THE FORWARD PROPAGATION
                    for layer in self.layers:
                        A = layer.forward_propagation(A)
                    
                    #WE CALCULATE THE ERROR GIVEN THE BATCHED TARGETS AND A
                    if self.perfunc_name == 'CROSS-ENTROPY':
                        e = A - targets[:, begin:end]
                    else:
                        e = targets[:, begin:end] - A   
                    
                    #WE CALCULATE THE PERFORMANCE FROM THE TARGETS AND RESULTS FROM THE FORWARD PROPAGATION
                    perf = self.perfunc(targets[:,begin:end], A)

                    if perf == np.nan:
                        print("Nan value encountered!")
                        break
                    
                    perf_per_batch += perf

                    if self.perfunc_name == "CROSS-ENTROPY":
                        WE, delta = self.layers[-1].back_propagation_lastLayer_crossEntropy(layer.A_prev, e)
                    else: 
                        WE, delta = self.layers[-1].back_propagation_lastLayer(layer.A_prev, e)
                    
                    for layer in reversed(self.layers[:-1]):
                        WE, delta = layer.back_propagation(layer.A, WE, delta)

                    gradient_vector = self._update_weigths()

                self.perf.append(perf_per_batch)
                self.numEpochs += 1

                if verbose == True:
                    print("Epoch {} : {} : {} | Grad : {}".format(epoch, self.perfunc_name, perf_per_batch, np.linalg.norm(gradient_vector)))
                
                if(perf_per_batch < self.min_perf):
                    break
                
                if(np.linalg.norm(gradient_vector) < self.min_grad):
                    break
        
        elif self.learning_method == 'batch':
            self.numEpochs = 0

            for epoch in range(self.num_epochs):
                #WE GET ALL INPUTS IN A
                A = inputs
                    
                #FOR EACH LAYER WE PERFORM THE FORWARD PROPAGATION
                for layer in self.layers:
                    A = layer.forward_propagation(A)
                
                #WE CALCULATE THE ERROR GIVEN THE BATCHED TARGETS AND A
                if self.perfunc_name == 'CROSS-ENTROPY':
                    e = A - targets
                else:
                    e = targets - A                    
                
                #WE CALCULATE THE PERFORMANCE FROM THE TARGETS AND RESULTS FROM THE FORWARD PROPAGATION
                perf = self.perfunc(targets, A)

                self.perf.append(perf)

                if self.perfunc_name == 'CROSS-ENTROPY':
                    WE, delta = self.layers[-1].back_propagation_lastLayer_crossEntropy(layer.A_prev, e)
                else: 
                    WE, delta = self.layers[-1].back_propagation_lastLayer(layer.A_prev, e)
                
                for layer in reversed(self.layers[:-1]):
                    WE, delta = layer.back_propagation(layer.A, WE, delta)
                
                gradient_vector = self._update_weigths()
                
                self.numEpochs += 1

                if verbose == True:
                    print("Epoch {} : {} : {} | Grad: {} ".format(epoch, self.perfunc_name, perf, np.linalg.norm(gradient_vector)))
                    
                if(perf < self.min_perf):
                    print("Min perf reached!")
                    break
                if(np.linalg.norm(gradient_vector) < self.min_grad):
                    print("Min grad reached!")
                    break
        
        elif self.learning_method == "SGD":
            q = inputs.shape[1]

            self.numEpochs = 0
            for epoch in range(self.num_epochs):
                perf_per_instance = 0

                for instance in range(q):
                    A = inputs[:, instance].reshape(inputs[:,instance].shape[0], 1)

                    for layer in self.layers:
                        A = layer.forward_propagation(A)
                        
                    #WE CALCULATE THE ERROR GIVEN THE BATCHED TARGETS AND A
                    if self.perfunc_name == 'CROSS-ENTROPY':
                        e = A - targets[:,instance].reshape(targets[:,instance].shape[0], 1)
                    else:
                        e = targets[:,instance].reshape(targets[:,instance].shape[0], 1) - A 
                    
                    perf = self.perfunc(targets[:,instance].reshape(targets[:,instance].shape[0], 1), A)
                    perf_per_instance += perf

                    if self.perfunc_name == 'CROSS-ENTROPY':
                        WE, delta = self.layers[-1].back_propagation_lastLayer_crossEntropy(layer.A_prev, e)
                    else: 
                        WE, delta = self.layers[-1].back_propagation_lastLayer(layer.A_prev, e)
                    
                    for layer in reversed(self.layers[:-1]):
                        WE, delta = layer.back_propagation(layer.A, WE, delta)

                    gradient_vector = self._update_weigths()

                    self.perf.append(perf_per_instance)
                    self.numEpochs += 1

                    if verbose == True:
                        print("Epoch {} : {} : {} | Grad : {}".format(epoch, self.perfunc_name, perf_per_instance, np.linalg.norm(gradient_vector)))

                    if(perf_per_instance < self.min_perf):
                        break
                    
                    if(np.linalg.norm(gradient_vector) < self.min_grad):
                        break

        self.firstpass = 1

        if validation_data != None:
            print("Validation data found, we validate the model updating if necesary")
            self._validate(validation_data[0], validation_data[1], verbose= verbose)         
    
    def predict(self, test_data):
        A = test_data
        for layer in self.layers:
            A = layer.forward_propagation(A)        
        return A

    def create_performance_graph(self):
        return self.numEpochs, self.perf

    def save_model(self):
        num_layer = 0
        for layer in self.layers:
            string = f"weights_layer_{num_layer}.npy"
            np.savetxt(string, layer.WE)
            num_layer += 1

    def load_model(self):
        npyCounter = len(glob.glob1("","*.npy"))

        for i in range(npyCounter):
            string = f"weights_layer_{i}.npy"
            self.layers[0].WE = np.load(string, allow_pickle= True)

