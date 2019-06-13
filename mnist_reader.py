#!/usr/bin/python3
import numpy as np

def normalize_v(x):
    """
    normalize a vector
    :param x: input
    :return: normalized vector
    """
    mn = np.min(x)
    mx = np.max(x)
    for i in range(len(x)):
        x[i] = (x[i] - mn) / mx - mn

    return x

def vectorize_label(x, classes_n):
    """
    turn a label into a column vector of length classes_n with 1 at index x and 0-s at the other places
    :param x: input label
    :param classes_n: # classes
    :return: vectorized label
    """
    res = np.zeros((classes_n, 1))
    res[x] = 1
    return res

def get_data(normalize=True):
    """
    read the mnist datasets (http://yann.lecun.com/exdb/mnist/)
    :type normalize: flag for applying normalization
    :return: training and test data as tuples of (parameters - column vector, label, vectorized label)
    """
    f_train_lbl = open("./data/train-labels.idx1-ubyte", "rb")
    f_train_img = open("./data/train-images.idx3-ubyte", "rb")
    f_test_lbl = open("./data/t10k-labels.idx1-ubyte", "rb")
    f_test_img = open("./data/t10k-images.idx3-ubyte", "rb")

    # skip magic numbers & go to # of examples
    f_test_img.seek(4, 0)
    f_train_img.seek(4, 0)

    test_count, train_count = int.from_bytes(f_test_img.read(4), byteorder="big"), int.from_bytes(f_train_img.read(4),
                                                                                                  byteorder="big")
    # images are square
    img_side = int.from_bytes(f_train_img.read(4), byteorder="big")
    nfeatures = img_side * img_side
    #go to data location
    f_test_lbl.seek(8, 0)
    f_test_img.seek(16, 0)
    f_train_lbl.seek(8, 0)
    f_train_img.seek(16, 0)

    # read data from the files
    dt = np.dtype('>B')  # unsigned byte big edian

    arr_test_lbl = np.frombuffer(f_test_lbl.read(), dtype=dt)
    arr_test_img = np.frombuffer(f_test_img.read(), dtype=dt)
    arr_train_lbl = np.frombuffer(f_train_lbl.read(), dtype=dt)
    arr_train_img = np.frombuffer(f_train_img.read(), dtype=dt)

    # close files
    f_test_lbl.close()
    f_test_img.close()
    f_train_lbl.close()
    f_train_img.close()

    # right now each pixel is an element, split into images
    arr_train_img = np.split(arr_train_img, train_count)
    arr_test_img = np.split(arr_test_img, test_count)
    if normalize:
        arr_train_img = normalize_v(arr_train_img)
        arr_test_img = normalize_v(arr_test_img)

    # turn images from row to column vectors
    for i in range(len(arr_train_img)):
        arr_train_img[i] = arr_train_img[i].reshape(-1, 1)
    for i in range(len(arr_test_img)):
        arr_test_img[i] = arr_test_img[i].reshape(-1, 1)

    #vectorize labels
    arr_test_lbl = [vectorize_label(x, 10) for x in arr_test_lbl]
    arr_train_lbl = [vectorize_label(x, 10) for x in arr_train_lbl]

    return (list(zip(arr_train_img, arr_train_lbl)), (list(zip(arr_test_img, arr_test_lbl))))

if __name__=="__main__":
    #if you want to print an image this sets the line width so it looks nice
    np.set_printoptions(linewidth=150)


