import numpy as np

inputs = [
    [1, 2, 3, 2.5],
    [2., 5., -1., 2],
    [-1.5, 2.7, 3.3, -0.8]
]
# No of inputs for a sample in batch should match with number of elements in weigths sub element array
# size of 'weights' is nothing but number of outputs
weights = [
    [0.2, 0.8, -0.5, 1],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87],
    [0.5, -0.91, 0.26, -0.5],
    [-0.26, -0.27, 0.17, 0.87]
]

# size of 'bias' should be equal to size of 'weights'
biases = [2, 3, 0.5, 3, 5]


weights2 = [
    [0.1, -0.14, 0.5, 2, 4],
    [-0.5, 0.12, -0.33, 2, 4]
]

biases2 = [-1, 2]

layer1_outputs = np.dot(inputs, np.array(weights).T) + biases

print(layer1_outputs)

layer2_outputs = np.dot(layer1_outputs, np.array(weights2).T) + biases2

print(layer2_outputs)

# NN
print(str(len(inputs)) + ":" + str(len(weights)) + ':' +
      str(len(weights2)) + ':' + str(len(layer2_outputs[0])))
