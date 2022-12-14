# Website useful for review: https://cs231n.github.io/neural-networks-case-study/

import numpy as np
import nnfs

np.random.seed(0)

def create_data(points, classes): 
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y

X, y = create_data(100, 3)

class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)  
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):  # the inputs could be your training data or whatever's forwarded from the previous layer
        self.output = np.dot(inputs, self.weights) + self.biases

# The Rectified Linear Unit Function
class Activation_ReLU:
    def forward(self, inputs):
        self.output = np.maximum(0, inputs)

layer1 = Layer_Dense(2, 5)
activation1 = Activation_ReLU()

layer1.forward(X)
print(layer1.output)

print("----")

activation1.forward(layer1.output)
print(activation1.output)
