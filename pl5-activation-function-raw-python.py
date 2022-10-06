import numpy as np

np.random.seed(0)

# 'inputs' serves as our demo data, calling it 'X' is a standard in ML
X = [[1, 2, 3, 2.5],
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

inputs = [0, 2, -1, 3.3, -2.7, 1.1, 2.2, -100]
output = []

# The Rectified Linear Unit Function in raw Python
for i in inputs: 
    output.append(max(i, 0))
        
print(output)

'''
# Now we want to define the two 'hidden' layers. They are called hidden, because the programmer is not really in charge of how that layer changes. 

# How to initialise a layer
# You've got weights & biases that need to be initialised. Usually they are initialised to random values between -1 & 1. The principle is to keep values small to avoid value explosion. 
# Biases are usually initialised as 0. 
class Layer_Dense:
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)  
        self.biases = np.zeros((1, n_neurons))
        
    def forward(self, inputs):  # the inputs could be your training data or whatever's forwarded from the previous layer
        self.output = np.dot(inputs, self.weights) + self.biases

layer1 = Layer_Dense(4, 5)  # the number of output neurons can be anything you want
layer2 = Layer_Dense(5, 2)  # the number of input neurons must be the same as the output number of the previous layer. the number of output neurons can be anything you want. 

layer1.forward(X)
print(layer1.output)
print("\n")
layer2.forward(layer1.output)
print(layer2.output)
'''
