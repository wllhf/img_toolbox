"""
This file contains basic functionality for the extraction of label information from data sets.
"""
import os
import numpy as np

from skimage.io import imread

from .io import get_full_names


def one_hot_repr(labels, max_label=None):
    """ Returns one-hot representation of labels.

    Parameters:
    -----------
    labels: numpy array (n,)
      Containing label information.
    max_label: int (default: None)
      Highest label in dataset. If None max_label is labels.max().

    Return:
    -------
    onehot: numpy array (n, max_label+1)
    """
    max_label = labels.max()+1 if max_label is None else max_label+1
    onehot = np.zeros((labels.shape[0], max_label), dtype='uint8')
    onehot[np.arange(labels.shape[0]), np.squeeze(labels)] = 1
    return onehot


def class_labels(limg, omit=[], min_area=0.1, bboxes=None, coords=None, patch_size=[25, 25], dtype='uint16'):
    """ Returns dominating class of image or image regions given by bboxes xor coords and patch_size.

    Parameters:
    -----------
    limg: numpy array
      Containing label information.
    omit: list of int
      Classes to ignore, e.g. background.
    min_area:
      Relative area that must be occupied by class.
    bboxes: tuple or list of tuples (y0, x0, h, w) (optional)
      Bounding boxes of image region.
    coords: numpy array (n,2) (optional)
      Sample coordinates (patch centers).
    patch_size: array like (2,) (optional)

    Return:
    -------
    labels: numpy array (n,1)
    """
    if bboxes is None and coords is None:
        counts = [np.bincount(limg.ravel())]
    elif bboxes is not None:
        bboxes = [bboxes] if type(bboxes) == tuple else bboxes
        counts = [np.bincount(limg[b[0]:b[0]+b[2], b[1]:b[1]+b[3]].ravel()) for b in bboxes]
    elif coords is not None:
        coords = np.expand_dims(coords, axis=0) if len(coords.shape) <= 1 else coords
        counts = [np.bincount(limg[c[0]:c[0]+patch_size[0], c[1]:c[1]+patch_size[1]].ravel()) for c in coords]

    for c in counts:
        if min_area:
            c[c < c.sum()*min_area] = 0

        c[omit] = 0

    labels = np.stack([np.argmax(c) for c in counts]).astype(dtype)
    labels = np.expand_dims(labels, axis=1) if len(labels.shape) < 2 else labels
    return labels


def generate_class_labels(path, file_names, omit=[], min_area=0.1, samples=None, patch_size=[25, 25], dtype='uint16'):
    """ Get dominating class of images or image regions given by samples and patch_size for multiple images.

    Parameters:
    -----------
    path: string
      Path to label images.
    file_names: list of strings
      List of file names of label images.
    omit: list of int
      Classes to ignore, e.g. background.
    min_area:
      Relative area that must be occupied by class.
    samples: numpy array (n,3) (optional)
      Sample coordinates (image index and patch centers).
    patch_size: array like (2,)
      Patch size.
    dtype: string
      Data type of returned numpy array.

    Return:
    -------
    labels: numpy array (n,1)
      Atomic labels corresponding to patch centers.
    """
    file_names = get_full_names(path, file_names)
    nlabels = len(file_names) if samples is None else samples.shape[0]
    labels = np.empty((nlabels, 1), dtype=dtype)
    for i, name in enumerate(file_names):
        limg = imread(os.path.join(path, name))
        indices = samples[:, 0] == i
        labels[indices] = class_labels(limg, omit=omit, min_area=min_area,
                                       coords=samples[indices, 1:], patch_size=patch_size, dtype=dtype)
    return labels


def local_labels(limg, coords, dtype='uint16'):
    """ Get the label at the patch center coordinates.

    Parameters:
    -----------
    limg: numpy array
      Containing label information.
    coords: numpy array (n,2)
      Sample coordinates (patch centers).
    dtype: string
      Data type of returned numpy array.

    Return:
    -------
    labels: numpy array (n,1)
      Atomic labels corresponding to patch centers.
    """
    labels = limg[coords[:, 0], coords[:, 1]]
    labels = np.expand_dims(labels, axis=1) if len(labels.shape) < 2 else labels
    return labels.astype(dtype)


def generate_local_labels(path, file_names, samples, dtype='uint16'):
    """ Get the label at the patch center coordinates for multiple images.

    Parameters:
    -----------
    path: string
      Path to label images.
    file_names: list of strings
      List of file names of label images.
    samples: numpy array (n,3)
      Sample coordinates (image index and patch centers).
    dtype: string
      Data type of returned numpy array.

    Return:
    -------
    labels: numpy array (n,1)
      Atomic labels corresponding to patch centers.

    """
    file_names = get_full_names(path, file_names)
    labels = np.empty((samples.shape[0], 1), dtype=dtype)
    for i, name in enumerate(file_names):
        limg = imread(os.path.join(path, name))
        indices = samples[:, 0] == i
        labels[indices] = local_labels(limg, samples[indices, 1:], dtype=dtype)
    return labels
