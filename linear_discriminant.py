import numpy as np



def get_loss(y, y_hat, w, X, b):
    """
    :param y: value
    :param y_hat: predicted value
    :param w: weights
    :param X: parameters of training data
    :param b: bias ?
    :return: :type float: loss
    """
    loss = 0
    for i in range(len(y)):
        loss += (y_hat[i] - y[i]) * \
            (w * X[i] + b) / 2



d = np.loadtxt(open("./data/linear_datat_2class.csv", "r", encoding='utf-8-sig'), delimiter=",", skiprows=0)



X, y = d[:,:-1], d[:,-1:]

print(X)