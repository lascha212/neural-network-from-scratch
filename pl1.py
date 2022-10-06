# Let's code a neuron that's somewhere in a neural network. 
# There are three neurons feeding into our neuron. 

inputs = [1, 2, 3, 2.5]  # these are outputs from 3 neurons in a previous layer

weights = [[0.2, 0.8, -0.5, 1.0],
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]
 
layer_outputs = []  # output of current layer
for neuron_weights, neuron_bias in zip(weights, biases):
    neuron_output = 0  # output of a given neuron
    for n_input, weight in zip(inputs, neuron_weights):
        neuron_output += n_input*weight
    neuron_output += neuron_bias
    layer_outputs.append(neuron_output)
    
print(layer_outputs)