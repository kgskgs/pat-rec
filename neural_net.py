#!/usr/bin/python3
import numpy as np
import math
import random

import mnist_reader


class multilayer_perceptron():

    def __init__(self, layers_sizes, act, act_drv, cost, cost_drv):
        self.layers = layers_sizes
        self.n_layers = len(layers_sizes)
        self.act = act
        self.act_drv = act_drv
        self.get_cost = cost
        self.cost_drv = cost_drv

        self.biases = [np.random.randn(x, 1) for x in self.layers[1:]]
        self.weights = [np.random.randn(x, y) for x, y in zip(self.layers[1:], self.layers[:-1])]

    def feed_forward(self, data):
        """
        :param data: input
        :return: last layer result
        """
        for i in range(self.n_layers - 1): #first layer is input
            data = self.act(self.weights[i] @ data + self.biases[i])

        return data

    def predict(self, data):
        """
        return a prediction as number
        :param data: input
        :return: :type int:
        """
        return np.argmax(self.feed_forward(data))

    def gradient_descent(self, data, epochs, batch_size, eta):
        """
        perform gradient decent on training data, splitting it into batches - update weights and biases for each batch
        :param data: :type list of tuples: input (parameters, label)
        :param epochs: # runs trough the whole input
        :param batch_size:
        :param eta: learning rate
        """
        for z in range(epochs):
            #split data
            random.shuffle(data)
            batches = [data[x:x+batch_size] for x in range(0,len(data), batch_size)]
            #train
            for batch in batches:
                #gradients for the whole batch
                grad_w = [np.zeros(w.shape) for w in self.weights]
                grad_b = [np.zeros(b.shape) for b in self.biases]
                for params, label in batch:
                    #parital derivatives for each example error with respect to weigts/biases
                    delta_w, delta_b = self.backpropagation(params, label)
                    grad_w = [gw + dw for gw, dw in zip(grad_w, delta_w)]
                    grad_b = [gb + dw for gb, dw in zip(grad_b, delta_b)]

                #update weights & biases; averaged across the size of the batch
                self.weights = [w - eta / len(batch) * gw for w, gw in zip(self.weights, grad_w)]
                self.biases = [b - eta / len(batch) * gb for b, gb in zip(self.biases, grad_b)]
            print("{}/{}".format(z+1, epochs))

    def backpropagation(self, params, label):
        """
        :param params: training data
        :param label: training labels
        :return: gradient for all weights and biases in the network
        """
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        #feed forward

        #outputs after activation
        activations = [params]
        activation = params
        #inputs to the activation of the layer
        inps = []

        for i in range(self.n_layers - 1):
            inp = self.weights[i] @ activation + self.biases[i]
            inps.append(inp)
            activation = self.act(inp)
            activations.append(activation)

        #backward pass
        #final layer

        #delta - how much the current layer contributes to the error/cost
        #delta at output layer dError/dActivation ⊙ dActivation/dInput -> dError/dInput
        delta = self.cost_drv(activations[-1], label) * self.act_drv(inps[-1]) #TODO final
        #output = w @ [previous activation] + b
        # dInput/dBias
        grad_b[-1] = delta
        # dInput/dWeight
        grad_w[-1] = delta @ activations[-2].transpose()
        #hidden layers
        for i in range(2, self.n_layers):
            #pull delta back to the previous layer by multiplying it by the weights 'connecting' them transposed (i.e. in reverse)
            delta = (self.weights[-i + 1].transpose() @ delta) * self.act_drv(inps[-i])
            grad_b[-i] = delta
            grad_w[-i] = delta @ activations[-i-1].transpose()

        return(grad_w, grad_b)

    def test(self, data):
        """
        check network against training data
        :param data: input data and labels - list of tuples
        :return: # of correct predictions
        """
        #label is in vector form
        results = [(self.predict(params), np.argmax(label)) for (params, label) in data]
        return sum(int(x == y) for (x, y) in results)


def sigmoid(x):
    """
    applies sigmoid
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_drv(x):
    """
    sigmoid derivative
    """
    return sigmoid(x)*(1-sigmoid(x))

def squared_loss(y, y_hat):
    """
    squared loss function
    :param y: :type numpy vector: label
    :param y_hat: :type numpy vector: observation
    :return: :type float: loss
    """
    loss = 0
    for i in range(len(y)):
        loss += math.pow(sum(y[i] - y_hat[i]), 2)/2
    return float(loss)

def squared_loss_drv(x, target):
    """
    :param x: :type numpy vector:
    :param target: :type numpy vector:
    :return: :type numpy vector:
    """
    return x - target



if __name__ == "__main__":
    trn, tst = mnist_reader.get_data()
    net = multilayer_perceptron([784, 30, 10], sigmoid, sigmoid_drv, squared_loss, squared_loss_drv)

    net.gradient_descent(trn, 30, 10, 3)