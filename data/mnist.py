""" MNIST is a database of handwritten digits available on http://yann.lecun.com/exdb/mnist/. """

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

    def train(self, flatten=True):
        with open(os.path.join(self.path, fname_train_lbl), 'rb') as fobj:
            magic, num = struct.unpack(">II", fobj.read(8))
            lbl = np.fromfile(fobj, dtype=np.int8)

        with open(os.path.join(self.path, fname_train_img), 'rb') as fobj:
            magic, num, cols, rows = struct.unpack(">IIII", fobj.read(16))
            if flatten:
                img = np.fromfile(fobj, dtype=np.int8).reshape(num, rows*cols)
            else:
                img = np.fromfile(fobj, dtype=np.int8).reshape(num, rows, cols)

        return (img, lbl)

    def test(self, flatten=True):
        with open(os.path.join(self.path, fname_test_lbl), 'rb') as fobj:
            magic, num = struct.unpack(">II", fobj.read(8))
            lbl = np.fromfile(fobj, dtype=np.int8)

        with open(os.path.join(self.path, fname_test_img), 'rb') as fobj:
            magic, num, cols, rows = struct.unpack(">IIII", fobj.read(16))
            if flatten:
                img = np.fromfile(fobj, dtype=np.int8).reshape(num, rows*cols)
            else:
                img = np.fromfile(fobj, dtype=np.int8).reshape(num, rows, cols)

        return (img, lbl)


if __name__ == "__main__":
    mnist = mnist("~/data/mnist/", 100)
    img, lbl = mnist.train()
    print(img.shape, lbl.shape)

