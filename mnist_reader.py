#!/usr/bin/python3
import numpy as np



def get_data():
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

    #

    return arr_test_img


