# The criteria here is to be true (equal to 1) if the input contains a sequence of [0,1] as in [1,0,1,0] 
#  and 0 if it has not as in [1,0,0,0]
# Deu certo!!

import itertools
import numpy as np

# defining the input dataset
n = 4 
x = np.array(list(itertools.product([0, 1], repeat=n)))
x = np.delete(x, 13, axis=0) #removing to use as test
# defining the target output dataset
y = np.array([[0,1,1,1,1,1,1,1,0,1,1,1,0,0,0]]).T

# initialize weights to random values between 0 and 1
weights = np.random.random((4,1))
print(weights)

# training the model
for i in range(10000):
    # forward propagation
    z = np.dot(x, weights)
    sigmoid = 1/(1+np.exp(-z))
    # Cost function
    erro = (y - sigmoid)
    #backpropagation
    sigmoidderivative = sigmoid * (1-sigmoid)
    weights += np.dot(x.T, erro*sigmoidderivative) 


#one final forward propagation to get output
newz = np.dot(np.array([1,0,1,1]), weights)
output = 1/(1+np.exp(-newz))
print("considering [1,0,1,1]:" , output)

