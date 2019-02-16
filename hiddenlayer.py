from layer import Layer
#from activations import sigmoid
import numpy as np
import json

def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)

def tanh(x):
    return np.tanh(x)

class HiddenLayer(Layer):
    def __init__(self, input_shape, output_shape, activation_function):
        """
        parameters:

        input_shape : expected to be a row vector
        """

        self.input_shape = input_shape
        self.output_shape = output_shape
        self._weights = np.random.rand(self.input_shape[1], self.output_shape[1])
        self._bias = np.random.rand(1,self.output_shape[1]) 
        self.activation_function = activation_function

    def __repr__(self):
        return json.dumps({'input shape': self.input_shape})

    def activation(self, input):
        return self.activation_function(input)

    def feedforward(self, input):
        output = np.dot(self.activation(input), self._weights) + self._bias
        return output
    
    def backpropagation(self, error, learning_rate):
        raise NotImplementedError

test = HiddenLayer((5,3), (1,3), tanh)
print(test)