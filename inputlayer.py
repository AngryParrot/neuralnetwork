from layer import Layer
import numpy as np

class Inputlayer(Layer):
    def __init__(self, input_shape, output_shape):
        """
        parameters:

        input_shape : expected to be a row vector
        """

        self.input_shape = input_shape
        self.output_shape = output_shape
        self._weights = np.random.rand(self.input_shape[1], self.output_shape[1])
        self._bias = np.random.rand(1,self.output_shape[1]) 

    def feedforward(self, input):
        output = np.dot(input, self._weights) + self._bias
        return output
    
    def backpropagation(self, error, learning_rate):
        raise NotImplementedError