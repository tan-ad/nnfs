import numpy as np

inputs = [1, 2, 3, 2.5]

weights = [[0.2, 0.8, -0.5, 1.0], #each output node/neuron has its own set of weights
           [0.5, -0.91, 0.26, -0.5],
           [-0.26, -0.27, 0.17, 0.87]]

biases = [2, 3, 0.5]

# outputs with for loops
# outputs = []
# for m in range(3):
#     outNode = 0
#     for n in range(4):
#         outNode += weights[m][n]*inputs[n]
#     outNode += biases[m]
#     outputs.append(outNode)

# outputs with list comprehension
# outputs = [sum(w * i for w, i in zip(neuronWeight, inputs)) + bias for neuronWeight, bias in zip(weights, biases)]

# outputs with numpy dot product
outputs = np.dot(weights, inputs) + biases # adding two (1,3) arrays
print(outputs)
