import numpy as np

np.random.seed(0)

X = [[1, 2, 3, 2.5], # convention for input data to be denoted by X
     [2.0, 5.0, -1.0, 2.0],
     [-1.5, 2.7, 3.3, -0.8]]

class Layer_Dense():
    def __init__(self, n_inputs, n_neurons):
        self.weights = 0.10*np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeroes((1, n_neurons)) #np.zeroes takes first parameter as the shape (tuple of the shape)
    def forward(self):
        pass
    # when intializing weights in a layer, they are typically random values in (-1,1). biases tend to be intialized as 0, unless nothing happens with 0 as initial values. 


print(np.random.randn(4,3))

        
weights = [[0.2, 0.8, -0.5, 1.0], 
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

weights2 = [[0.1, -0.14, 0.5],
            [-0.5, 0.12, -0.33],
            [-0.44, 0.73, -0.13]]

biases2 = [-1, 2, -0.5]



layer1_outputs = np.dot(input, np.array(weights).T) + biases 
# the dot product is (3,3) but biases is (3,). however, numpy performs 'broadcasting'.
# in (p,p) + (p,), the second array acts like a (1,p) array. ith element of (p,) is added to ith column of (p,p) 
# in (m,p) + (p,), the second array acts like a (1,p) array.
# in (p,m) + (p,), the second array acts like a (p,1) array. 

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2