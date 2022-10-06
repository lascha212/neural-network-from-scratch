import math
import numpy as np 

layer_outputs = [4.8, 1.21, 2.385]

E = math.e

# To make use of our output values, eg. turn them into a probability distribution, we need to normalize them. In addition, we also need to exponentiate them in order to get rid of the negative values first (turning negative values into probability doesn't really work).

# exp_values = []
# for output in layer_outputs: 
#     exp_values.append(E**output)
# The above turns into the line below using numpy.
exp_values = np.exp(layer_outputs)
    
print("exponentiated values: ", exp_values)

# Normalization occurs after the exponentiation
norm_values = exp_values / np.sum(exp_values)
    
print("normalized values: ", norm_values)
print("sum of normalized values: ", sum(norm_values))  # Should add up to 1. 

# The combination of the exponentiation and normalization is actually the softmax function. The formula for the softmax function is S_i,j = e^x_i,j / sum(e^x_i,j).
