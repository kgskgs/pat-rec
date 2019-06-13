#!/usr/bin/python3
import numpy as np
import matplotlib.pyplot as plt
from math import sqrt


class perceptron:

    def __init__(self):
        #self.w = np.random.rand(2)
        self.w = np.array([0.5, 0.5])
        self.w = normalize_v(self.w)
        #self.b = np.random.rand()
        self.b = 1


    def get_loss(self, y, y_hat, X):
        """
        loss function for two classes
        :param y: value
        :param y_hat: predicted value
        :param X: parameters of training data
        :return: :type float: loss
        """
        loss = 0
        for i in range(len(y)):
            loss += (y_hat[i] - y[i]) * (self.w @ X[i] + self.b) / 2
        return loss

    def predict(self, x):
        """
        prediction for two classes
        :param x: observation
        :return: +1 if the observation is above the boundry -1 otherwise
        """
        res = self.w @ x + self.b
        if res > 0: return 1
        return -1

    def minimize_loss(self, eta_w, eta_b, X, y, it = 100):
        """
        minimize the loss function using gradient / partial derivates with respect to w and b
        :param eta_w: learning rate weights
        :param eta_b: learning rate biase ?
        :param X: training data
        :param y: training labels
        :param it: # iterations to optimize
        :returns: final weights and bias
        """
        for z in range(it):
            delta_w = np.array([0.0, 0.0])
            delta_b = 0.0
            y_hat = []
            for i in range(len(y)):
                prediction = (self.predict(X[i]) - y[i])
                y_hat.append(prediction)
                delta_w += prediction * X[i] / 2
                delta_b += prediction / 2

            self.w = normalize_v(self.w - eta_w*normalize_v(delta_w))
            self.b = self.b - eta_b*delta_b
            #print(z, self.w, self.b, delta_w, delta_b, self.get_loss(y, y_hat, X))
        return (self.w, self.b)


def normalize_v(v):
    """
    normalize a vector
    :param v: vector
    :return: normalized v
    """
    norm = sqrt(sum([x*x for x in v]))
    if norm == 0:
        raise ValueError("cannot normalize zero length vector")
    return np.array([x/norm for x in v])

if __name__=="__main__":
    d = np.loadtxt(open("./data/linear_datat_2class.csv", "r", encoding='utf-8-sig'), delimiter=",", skiprows=0)

    X, y = d[:,:-1], d[:,-1:]
    colors = ["red" if x==1 else "blue" for x in y]

    plt.scatter(X[:,0], X[:,1], color=colors)

    perc = perceptron()

    w, b = perc.minimize_loss(0.3, 0.3, X, y, 100)

    point1 = [0, -b/w[0]]
    point2 = [-b/w[1], 0]

    plt.axis('equal')
    #plt.axis([30, 100, 30, 100])
    plt.plot(point1, point2, '-r', color="green")

    correct = 0
    predictions = []
    for i in range(len(y)):
        predictions.append(perc.predict(X[i]))
        if perc.predict(X[i]) == y[i]:
            correct += 1

    print("{}/100".format(correct))
    plt.show()