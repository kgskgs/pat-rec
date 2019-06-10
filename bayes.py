import numpy as np

def simple_loss(x, y):
    """
    loss function
    :param x: prediction
    :param y: observed result
    :return: :type int: 0 if x==y, 1 otherwise
    """
    return int(x != y)



