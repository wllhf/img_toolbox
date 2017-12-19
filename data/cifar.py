""" CIFAR is a dataset of tiny images available on https://www.cs.toronto.edu/~kriz/cifar.html. """

import os
import pickle

import numpy as np


fnames_train = ["data_batch_1", "data_batch_2", "data_batch_3", "data_batch_4", "data_batch_5"]
fname_test = "test_batch"
fname_meta = "batches.meta"


class cifar10():
    """
    Each training batch and the test batch contain a 10000x3072 numpy array of uint8s. Each row
    of the array stores a 32x32 colour image. The first 1024 entries contain the red channel
    values, the next 1024 the green, and the final 1024 the blue. The image is stored in row-major
    order, so that the first 32 entries of the array are the red channel values of the first row
    of the image.

    Each batch contains a list of 10000 numbers in the range 0-9. The number at index i indicates
    the label of the ith image in the array data.
    """

    def __init__(self, path):
        self.nclasses = 10
        self.path = os.path.expanduser(path)

    def train(self, flatten=True, integral=False):
        img_batches = []
        lbl_batches = []
        for fname in fnames_train:
            with open(os.path.join(self.path, fname), 'rb') as fobj:
                dictionary = pickle.load(fobj, encoding='bytes')
                img_batches.append(dictionary[b'data'])
                lbl_batches.append(np.array(dictionary[b'labels']))

        img = np.vstack(img_batches)
        lbl = np.hstack(lbl_batches)

        if not flatten:
            img = img.reshape(-1, 3, 32, 32)
            img = img.transpose(0, 2, 3, 1)

        if not integral:
            img = np.array(img, dtype=float) / 255.0

        return (img, lbl)

    def test(self, flatten=True, integral=False):
        with open(os.path.join(self.path, fname_test), 'rb') as fobj:
            dictionary = pickle.load(fobj, encoding='bytes')
            img = dictionary[b'data']
            lbl = np.array(dictionary[b'labels'])

        if not flatten:
            img = img.reshape(-1, 3, 32, 32)
            img = img.transpose(0, 2, 3, 1)

        if not integral:
            img = np.array(img, dtype=float) / 255.0

        return (img, lbl)

    def meta(self):
        with open(os.path.join(self.path, fname_meta), 'rb') as fobj:
            dictionary = pickle.load(fobj, encoding='bytes')
        return dictionary[b'label_names']


class cifar100():
    pass


if __name__ == "__main__":
    cifar = cifar10("~/data/cifar10_py")
    img, lbl = cifar.train(flatten=False)
    print(img.shape, lbl.shape)
    img, lbl = cifar.test(flatten=False)
    print(img.shape, lbl.shape)
