import numpy as np 

layer_outputs = [[4.8, 1.21, 2.385],
                 [8.9, -1.81, 0.2],
                 [1.41, 1.051, 0.026]]

exp_values = np.exp(layer_outputs)  # Here, np.exp() handles a batch input by itself.
    
print("exponentiated values: ", exp_values)

# Normalization occurs after the exponentiation.
# To get the sum of each batch of outputs, eg. first sum would be sum([4.8, 1.21, 2.385]). That's why we pass the 'axis' parameter to the sum function. After this, to actually be able to divide each value of each batch by the sum of the same batch, we need to keep the sum in the correct matrix shape as well. We do not just want an array, we want to keep the matrix of 3 rows. That's where the 'keepdims' parameter comes in.
norm_values = exp_values / np.sum(exp_values, axis=1, keepdims=True)
    
print("normalized values: ", norm_values)
print("sum of normalized values: ", sum(norm_values))  # Should add up to 1. 
