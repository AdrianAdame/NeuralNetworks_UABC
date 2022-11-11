from utils.pretty_confusion_matrix import pp_matrix_from_data
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
import numpy as np

sns.set()

#<!----------------------------------------------------------------------------------------------------------------------------->#
#DATA SEPARATION TECHNIQUES

"""
Hold out implementation for splitting into Test - Validation - Test

Params:
-> X : Input Array (n * n)
-> Y : Target Array (n * n)
"""
def train_valid_test_split(X, Y, train_size = .6, valid_size = .2, test_size = .2, random_state = None):
    if train_size + valid_size + test_size == 1:

        X_train, X_temp, Y_train, Y_temp = train_test_split(X, Y, test_size = (1.0 - train_size), random_state = random_state)

        relative_test = test_size / (valid_size + test_size)

        X_valid, X_test, Y_valid, Y_test = train_test_split(X_temp, Y_temp, test_size = relative_test, random_state = random_state)

        generated_splits = {
            'Train' : [X_train, Y_train],
            'Validation' : [X_valid, Y_valid],
            'Test' : [X_test, Y_test]
        }

        return generated_splits
    else:
        raise Exception("Train, Valid and Test size not equal to 1, data could not be splitted in the proportions given.")


#<!----------------------------------------------------------------------------------------------------------------------------->#
#CLASSIFICATION AUXILIAR FUNCTIONS

"""
Function to obtain a matrix of classes from the Targets Array | np.array(1,n)

Params:
-> Targets : Array of values
"""
def getClasses_Classification(Targets):
    if Targets.shape[1] == 1:
        obtainedClasses = np.zeros((Targets.shape[0],1))
        classes_unique = np.unique(Targets)
        for class_value in classes_unique:
            obtainedClasses = np.hstack((obtainedClasses,
                np.array(
                    Targets == class_value, dtype=np.int32
                    )
            ))
        return obtainedClasses[:,1:]
    else:
        return Targets


#<!----------------------------------------------------------------------------------------------------------------------------->#
#PLOTTING FUNCTIONS

"""
Generic plot Graphs

Params:
-> X : Array of values to horizontal dimension (1 * n)
-> Y : Array of values to vertical dimension   (1 * n)
"""
def plot_generic_graph(x, y):
    plt.plot([epoch for epoch in range(x)], y)
    plt.show()

"""
Plot Regression graphic

Params:
-> Targets
-> Predicted
"""
def plot_regression_graph(Targets, Predicted):
    r = np.corrcoef(Targets.T[0], Predicted.T[0])

    plt.plot(Targets.T[0], Predicted.T[0], 'o')
    m, b = np.polyfit(Targets.T[0], Predicted.T[0], 1)
    plt.plot(Targets.T[0], m*Targets.T[0] + b)
    plt.title("R={}".format(r[0][1]))
    str = "{}*{}={}".format(round(m, 3), "Target",round(b, 3))
    plt.ylabel(str)
    plt.show()

"""
Plot Classification confusion matrix and ROC Curve

Params:
-> Targets :  
-> Ph      :
"""
def plot_classification(Targets, Ph):
    pp_matrix_from_data(np.argmax(Targets, axis = 1), np.argmax(Ph, axis = 1))

    plt.figure()

    color = iter(cm.rainbow(np.linspace(0,1,15)))

    for class_v in range(Targets.shape[1]):
        fpr, tpr, none = roc_curve(Targets[:,class_v].reshape(-1,1), Ph[:,class_v].reshape(-1,1))
        plt.plot(fpr, tpr, color=next(color), lw=1, label=f'Class {class_v}')

    plt.xlim([-0.1, 1.05])
    plt.ylim([-0.1, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    plt.show()


#<!----------------------------------------------------------------------------------------------------------------------------->#
#NORMALIZATION ALGORITMS

"""
Normalization of data for NN as input

Params:
-> X       : Numpy array of data (n * n)
-> y_min   : Minimum value to be normalized << Min range >>
-> y_max   : Maximum value to be normalized << Max range >>
-> method  : Method variation << Normal: Raw data to normalized data | Inverted: Normalized data to Raw data >>
-> PS      : Previous setting to normalize or denormalize data, used when method = 'Inverted'
"""
def mapminmax(X, y_min = -1, y_max = 1, method = 'normal', PS = None):
    if method == 'normal':   
        xmax      = X.max(0)
        xmin      = X.min(0)
        xrange    = xmax - xmin

        ymax   = y_max
        ymin   = y_min
        yrange = 2

        gain = yrange / xrange
        fix = np.nonzero(~np.isfinite(xrange) | (xrange == 0))
        gain[fix] = 1
        xmin[fix] = ymin

        a = yrange * (X - xmin)/(xrange)

        settings = {
            'xmax' : xmax,
            'xmin' : xmin,
            'xrange' : xrange,
            'ymax' : ymax,
            'ymin' : ymin,
            'yrange' : yrange,
            'gain' : gain
        }    

        return a + ymin, settings
    else:
        if PS != None:
            xmax      = PS['xmax']
            xmin      = PS['xmin']
            xrange    = xmax - xmin

            ymax   = PS['ymax']
            ymin   = PS['ymin']
            yrange = 2

            gain = yrange / xrange
            fix = np.nonzero(~np.isfinite(xrange) | (xrange == 0))
            gain[fix] = 1
            xmin[fix] = ymin

            a = xrange * (X - ymin)/(yrange)

            settings = {
                'xmax' : xmax,
                'xmin' : xmin,
                'xrange' : xrange,
                'ymax' : ymax,
                'ymin' : ymin,
                'yrange' : yrange,
                'gain' : gain
            }    

            return a + xmin, settings

#<!----------------------------------------------------------------------------------------------------------------------------->#
#OTHER UTILITIES

"""
Generation of a random numpy matrix given a range

Params:
-> data_range : Tuple of range to generate the data from (min, max) 
-> row_size   : Number of rows in the generated matrix
-> col_size   : Number of columns in the generated matrix
-> round      : Number of decimal to round the values
"""
def generate_random_matrix(data_range = (1,10), row_size = 100, col_size = 100, round = 2):
    return np.round((data_range[1] - data_range[0]) * np.random.rand(row_size, col_size) + data_range[0], round)
