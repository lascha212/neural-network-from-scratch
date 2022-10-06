# The NN outputs a probability distribution for the predicted classes. To catch the slight difference between eg. a 80% and a 52% prediction for a certain class, we use a loss function to update the weights. 
# The usual loss function used is Categorical Cross-Entropy loss function. The formula for it is: L_i = - sum(y_i,j * log(y-hat_i,j)). So it is the negative sum of the target value multiplied by the log of the predicted value for each of the values in the distribution. 

''' Re-Fresher of logarithm
solving for x

e ** x = b
'''
import numpy as np
import math 

b = 5.2

# Euler's number raised to what actually equals b? 
print(np.log(b))
print(math.e ** 1.6486586255873816)