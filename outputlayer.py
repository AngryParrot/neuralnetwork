from layer import Layer
from activations import softmax
import numpy as np

def sigmoid(x):
    return np.exp(x) / (np.exp(x) + 1)

def tanh(x):
    return np.tanh(x)


'important comment'
class OutputLayer(Layer):
    def __init__(self, input_shape, y_shape):
        """
        Used to classify multiple and exclusive classes.

        parameters:

        input_shape : expected to be a row vector
        """
        assert input_shape == y_shape, 'Must have the same dimensions.'
        self.input_shape = input_shape
        self.y = y_shape

    def feedforward(self, input):
        """Uses softmax to ensure that probabilities over all classes sum to one. Prediction with highest probability returns 1, rest 0."""
        output = softmax(input)
        max_index = np.argmax(output, axis=1)
        output = np.zeros(output.shape)
        output[0,max_index] = 1
        return output
    
    def backpropagation(self, error, learning_rate):
        raise NotImplementedError