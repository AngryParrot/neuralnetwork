from layer import Layer
from inputlayer import Inputlayer
from hiddenlayer import HiddenLayer
from outputlayer import OutputLayer
from activations import tanh, sigmoid
import numpy as np 

class NeuralNetwork(Layer):
    def __init__(self, x_training, y_training):
        self.x_training = x_training
        self.y_training = y_training
        #self.learning_rate = learning_rate
        #self.epochs = epochs
        self.layers = []

    def add(self, layer):
        """Adds a new layer to the NeuralNetwork."""
        self.layers.append(layer)

    def fit(self, X, y, epochs, learning_rate):
        #for epoch in range(epochs):
        amount_samples = X.shape[0]

        for sample in range(amount_samples):
            input = X[sample,:]

            for layer in self.layers:
                print(layer)
                print(layer.feedforward(input))

        #print(epoch) # to avoid unused variable error; delete later

# testing
np.random.seed(1)

# data
x = np.random.rand(1,10)
y = np.random.rand(1,3) 

# network
NN = NeuralNetwork(x,y)
NN.add(Inputlayer(x.shape,(1,5)))
NN.add(HiddenLayer((1,5),(1,9),tanh))
NN.add(HiddenLayer((1,9),(1,3),tanh))
NN.add(OutputLayer((1,3), y.shape))

# feedforward
output1 = NN.layers[0].feedforward(x)
output2 = NN.layers[1].feedforward(output1)
output3 = NN.layers[2].feedforward(output2)
output4 = NN.layers[3].feedforward(output3)

#print(NN.fit(x,y,1,1))

"""
print("outout1",output1)
print("outout2",output2)
print("outout3",output3)
print("outout4",output4)
"""