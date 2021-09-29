import numpy as np

def square(x: np.ndarray)->np.ndarray:
    '''
    Square each element in the input ndarray
    '''
    return np.power(x,2)

def leaky_relu(x:np.ndarray)->np.ndarray:
    '''
    Apply "Leaky ReLU" function to each element in ndarray
    '''
    return np.maximum(0.2*x,x)
    
def sigmoid(x:np.ndarray):
    '''
    Apply the sigmoid function to each element in the input ndarray
    '''
    return 1/(1+np.exp(-x))