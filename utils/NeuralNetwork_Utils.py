
from scipy.special import xlogy
import numpy as np

np.set_printoptions(precision=20)


"""
FILE THAT HAVE ALL FUNCTIONS WE USE TO TRAIN THE NEURAL NETWOK

- Performance Functions:
-> SSE
-> MSE
-> RMSE

- Gradient based optimizers:
-> Gradient Descent
-> RMSProp
-> NAdam
-> More to add
"""

#<!----------------------------------------------------------------------------------------------------------------------------->#
#PERFORMANCE FUNCTIONS

"""
Calculation of SSE ( Sum Square Error )

Params using kwargs:
-> Targets : Real values for the error
-> Predict : Predicted values while training
"""
def sse(Targets, Predict):
    return np.sum(np.sum(np.power((Targets - Predict), 2)))

"""
Calculation of MSE ( Mean Square Error )

Params:
-> Targets : Real values for the error
-> Predict : Predicted values while training
"""
def mse(Targets, Predict):
    return sse(Targets, Predict).mean()

"""
Calculation of RMSE ( Root Mean Square Error )
Params:
-> Targets : Real values for the error
-> Predict : Predicted values while training
"""
def rmse(Targets, Predict):
    return np.sqrt(mse(Targets, Predict))

"""
Calculation of SST ( Total Sum of Squares )
Params:
-> Targets : Real values for the error
-> Predict : Predicted values while training
"""
def sst(Targets):
    return np.sum(np.square(Targets - np.mean(Targets)))

"""
"""
def cross_entropy(Targets, Predict):

    #H = Targets * np.log(Predict) + (1 - Targets) * np.log(1 - Predict)
    #return - np.sum(np.sum(H))/(Targets.shape[1] * Predict.shape[0])

    #return -np.sum(Targets * np.log(Predict))

    eps = np.finfo(Predict.dtype).eps
    Predict = np.clip(Predict, eps, 1 - eps)
    if Predict.shape[1] == 1:
        Predict = np.append(1 - Predict, Predict, axis=1)

    if Targets.shape[1] == 1:
        Targets = np.append(1 - Targets, Targets, axis=1)

    return -xlogy(Targets, Predict).sum() / Predict.shape[0]

"""
Dictionary to export performance functions to the Neural Network
"""
performance_functions = {
    'sse' : (sse, 'SSE'),
    'mse' : (mse, 'MSE'),
    'rmse' : (rmse, 'RMSE'),
    'cross-entropy' : (cross_entropy, 'CROSS-ENTROPY')
}

#<!----------------------------------------------------------------------------------------------------------------------------->#
#GRADIENT BASED OPTIMIZERS

def gradientDescent(wt, dw, optimParams):
    lr = optimParams['lr']
    wt = wt - lr * dw

    return wt

def RMSProp(wt, dw, optimParams):
    eps = 1e-8
    Beta = optimParams['Beta']
    Alpha = optimParams['Alpha']
    
    if 'vt' in optimParams:
        vt = optimParams['vt']
    else:
        vt = 0

    vt = Beta * vt + (1 - Beta) * dw ** 2.0
    wt = wt - Alpha /(np.sqrt(vt + eps)) * dw

    optimParams['vt'] = vt
    
    return wt

def Adam(wt, dw, optimParams):
    beta_1 = optimParams['Beta1']
    beta_2 = optimParams['Beta2']
    alpha = optimParams['Alpha']
    eps = 1e-8
    n = wt.shape[0]

    if 't' in optimParams:
        t = optimParams['t']
    else:
        t = 1
        optimParams['t'] = t
    
    if 'mt' in optimParams and 'vt' in optimParams:
        mt = optimParams['mt']
        vt = optimParams['vt']
    else:
        mt = np.zeros((n,1))
        vt = np.zeros((n,1))
    
    #Algoritmo
    mt = beta_1 * mt + (1-beta_1) * dw
    vt = beta_2 * vt + (1-beta_2) * np.power(dw, 2)
    learning_rate = alpha * np.sqrt(1 - beta_2 ** t)/(1-beta_1**t)
    wt = wt - (learning_rate * mt /(np.sqrt(vt)+eps))
    
    optimParams['mt'] = mt
    optimParams['t'] += 1
    optimParams['vt'] = vt

    return wt

def NAdam(wt, dw, optimParams):
    beta_1 = optimParams['Beta1']
    beta_2 = optimParams['Beta2']
    alpha = optimParams['Alpha']
    eps = 1e-8
    n = wt.shape[0]

    if 't' in optimParams:
        t = optimParams['t']
    else:
        t = 1
        optimParams['t'] = t
    
    if 'mt' in optimParams and 'vt' in optimParams:
        mt = optimParams['mt']
        vt = optimParams['vt']
    else:
        mt = np.zeros((n,1))
        vt = np.zeros((n,1))

    #Algoritmo
    mt = beta_1*mt+(1-beta_1)*dw
    vt = beta_2*vt+(1-beta_2)* np.power(dw,2)
    
    mt_hat = mt/(1-beta_1**t)
    vt_hat = vt/(1-beta_2**t)

    m_bar = beta_1 * mt_hat + (1- beta_1) * dw
    
    wt = wt - (alpha * m_bar/(np.sqrt(vt_hat + eps)))
    
    optimParams['mt'] = mt
    optimParams['t'] += 1
    optimParams['vt'] = vt
    
    return wt

"""
Dictionary to export implemented optimizers to the Neural Network
"""
optimizers_implemented = {
    'GD' : gradientDescent,
    'RMSProp' : RMSProp,
    'Adam' : Adam,
    'NAdam' : NAdam
}

#<!----------------------------------------------------------------------------------------------------------------------------->#
#CALCULATION OF MATRIX AND VECTORS FOR OPTIMIZATIONS

"""
Function to transform an array of matrix to vector form (n,1)

Params:
-> Matrix_arr : Array of matrix  | list(np.array)
"""
def matrix_to_vector(matrix_arr):
    vector = matrix_arr[0].flatten()

    matrix_arr.pop(0)

    for matrix_index in range(len(matrix_arr)):
        vector = np.append(vector, matrix_arr[matrix_index].flatten())
    
    return vector.reshape((-1,1))

"""
Function to transform a vector to matrix form with reshaped size

Params:
-> Vector        : Vector of matrix flattened                  | np.array(n,1)
-> w_size_tuples : Matrix shapes to transform from the vector  | list(tuples)
"""
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