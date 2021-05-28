import numpy as np


def sigmoid_fn(x):
    output = 1.0 / (1.0 + np.exp(-1.0 * x)) 
    
    return output


def softmax_fn(x, axis = -1):
    denominator = np.sum(np.exp(x), axis = axis) 
    output = np.exp(x) / denominator

    return output


def relu_fn(x):
    output = np.maximum(x, 0)

    return output


def sigmoid_grad(x):
    output = sigmoid_fn(x) * (1.0 - sigmoid_fn(x)) 

    return output


def softmax_grad(x):
    x = x.reshape(-1, 1)
    output = np.diagflat(x) - np.dot(x, x.T)

    return output 

