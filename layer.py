from abc import abstractmethod

class Layer(object):
    def __init__(self):
        self.input = None
        self.output = None
        self.input_shape = None
        self.output_shape = None

    @abstractmethod
    def feedforward(self, input):
        raise NotImplementedError
    
    @abstractmethod
    def backpropagation(self, error, learning_rate):
        raise NotImplementedError