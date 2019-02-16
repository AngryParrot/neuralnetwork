import numpy as np 

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def tanh(x):
    return np.tanh(x)

def softmax(x):
    """Expects x to be a column vector."""
    return np.exp(x) / np.sum(np.exp(x), axis=1)