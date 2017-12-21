""" MNIST is a dataset of handwritten digits available on http://yann.lecun.com/exdb/mnist/. """

import os
import struct
# > : big endian
# I : unsigned int
import numpy as np

fname_train_img = "train-images.idx3-ubyte"
fname_train_lbl = "train-labels.idx1-ubyte"
fname_test_img = "t10k-images.idx3-ubyte"
fname_test_lbl = "t10k-labels.idx1-ubyte"


class mnist():

    def __init__(self, path):
        self.path = os.path.expanduser(path)

    def _prepare_data(self, fname_lbl, fname_img, flatten=True, integral=True):
        with open(os.path.join(self.path, fname_lbl), 'rb') as fobj:
            magic, num = struct.unpack(">II", fobj.read(8))
            lbl = np.fromfile(fobj, dtype=np.int8)

        with open(os.path.join(self.path, fname_img), 'rb') as fobj:
            magic, num, cols, rows = struct.unpack(">IIII", fobj.read(16))
            if flatten:
                img = np.fromfile(fobj, dtype=np.int8).reshape(num, rows*cols)
            else:
                img = np.fromfile(fobj, dtype=np.int8).reshape(num, rows, cols)
                img = np.expand_dims(img, axis=3)

        if not integral:
            img = img / 255.0

        return (img, lbl)

    def train(self, flatten=True, integral=True):
        return self._prepare_data(fname_train_lbl, fname_train_img, flatten=flatten, integral=integral)

    def test(self, flatten=True, integral=True):
        return self._prepare_data(fname_test_lbl, fname_test_img, flatten=flatten, integral=integral)


if __name__ == "__main__":
    mnist = mnist("~/data/mnist/")
    img, lbl = mnist.train()
    print(img.shape, lbl.shape)
    img, lbl = mnist.test()
    print(img.shape, lbl.shape)
