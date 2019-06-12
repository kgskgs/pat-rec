#!/usr/bin/python3
import numpy as np
import math
import random

class neural_net():


    def __init__(self, layers_sizes, eta, act, act_derv, cost, cost_derv):
        self.layers = layers_sizes
        self.n_layers = len(layers_sizes)
        self.act = act
        self.act_derv = act_derv
        self.get_cost = cost
        self.cost_derv = cost_derv

        self.biases = [np.random.rand(x, 1) for x in self.layers[1:]]
        self.weights = [np.random.rand(x, y) for x, y in zip(self.layers[1:], self.layers[:-1])]

    def feed_forward(self, data):
        """
        :param data: input
        :return: last layer result
        """
        for i in range(self.n_layers - 1): #first layer is input
            data = self.act(self.weights[i] @ data + self.biases[i])

        return data

    def train(self, data, epochs, batch_size, eta):
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
                    #parital derivatives for each example with respect to weigts/biases
                    delta_w, delta_b = self.backpropagation(params, label)
                    grad_w = [gw + dw for gw, dw in zip(grad_w, delta_w)]
                    grad_b = [gb + dw for gb, dw in zip(grad_b, delta_b)]

                #update weights & biases
                self.weights = [w - eta/len(batch) * gw for w, gw in zip(self.weights, grad_w)]
                self.weights = [b - eta / len(batch) * gb for b, gb in zip(self.biases, grad_b)]
            print("{}/{}".format(z, epochs))

    def backpropagation(self, params, label):
        grad_b = [np.zeros(b.shape) for b in self.biases]
        grad_w = [np.zeros(w.shape) for w in self.weights]

        #feed forward
        activation = params
        activations = [params]
        #outputs before activation
        outs = []
        for i in range(len(self.n_layers - 1)):
            out = self.weights[i] @ activation + self.biases[i]
            outs.append(out)
            activation = self.act(out)
            activations.append(activation)

        #backward pass
        #output layer
        delta = self.cost_derv(activations[-1], y) * self.act_derv(outs[-1]) #TODO final
        grad_b[-1] = delta
        grad_w[-1] = delta @ activations[-2].transpose()

        for i in range(2, self.n_layers):
            #out =
            delta = (self.weights[-i + 1].transpose() @ delta) * self.act_derv(outs[-i])
            grad_b[-i] = delta
            grad_w[-i] = delta @ activations[-i-1].transpose()

        return(grad_w, grad_b)

    def test(self, data):
        results = [(np.argmax(self.feed_forward(params), label) for (params, label) in data)]
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



a = neural_net([3,2,1], 0, test, test, test, test)

tst = np.array([[1], [2], [3]])

print(a.feed_forward(tst))
