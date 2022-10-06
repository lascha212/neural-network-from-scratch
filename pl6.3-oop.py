import numpy as np
import nnfs
from nnfs.datasets import spiral_data

nnfs.init()

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):
        self.output = np.dot(inputs, self.weights) + self.biases


class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)
        
class Activation_Softmax:
    def forward(self, inputs): 
        exp_values = np.exp(inputs - np.max(inputs, axis=1, keepdims=True))  # Prevent overflows by subtracting max. Otherwise overflows might be caused by exponentiation (as values grow very quickly). 
        self.output = exp_values / np.sum(exp_values, axis=1, keepdims=True)


X, y = spiral_data(samples=100, classes=3)

dense1 = Layer_Dense(2, 3)  # Input is 2 because spiral data just has coordinates, so an x and a y value for every datapoint.
activation1 = Activation_ReLU()

dense2 = Layer_Dense(3, 3)  # Output layer should have 3 neurons because we have 3 classes. 
activation2 = Activation_Softmax()

dense1.forward(X)
activation1.forward(dense1.output)
dense2.forward(activation1.output)
activation2.forward(dense2.output)

print(activation2.output[:5])