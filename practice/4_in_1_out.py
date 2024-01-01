inputs = [1, 2, 3, 2.5] # input neurons
weights = [0.2, 0.8, -0.5, 1.0] 
bias = 2 # same across all input neurons connecting to one neuron in the next layer
output = 0 # one output node/neuron
dataSize = len(inputs)

for i in range(dataSize):
    output += inputs[i]*weights[i]
output += bias
print(output)