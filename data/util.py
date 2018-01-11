"""
This file contains various utility functions.
"""
import os
import multiprocessing as mp

import numpy as np

from skimage.io import imread


def get_shared_numpy_array(shape, dtype):
    """
    shape: tuple
    dtype: np.dtype
    """
    ctype = dtype
    while not isinstance(ctype, str):
        ctype = ctype._type_
    mp_array = mp.Array(ctype, np.prod(shape))
    np_array = np.frombuffer(mp_array.get_obj(), dtype=dtype)
    return np.reshape(np_array, shape)


def array_from_list(imgs):
    row_max = max([img.shape[0] for img in imgs])
    col_max = max([img.shape[1] for img in imgs])
    nchannels = imgs[0].shape[2] if len(imgs[0].shape) > 2 else 1

    array = np.zeros([len(imgs), row_max, col_max, nchannels])
    for i, img in enumerate(imgs):
        rows, cols, chans = img.shape
        img = np.expand_dims(img, axis=3) if len(img.shape) < 4 else img
        array[i, :rows, :cols, :chans] = img

    return array


def color_to_class(img, cmap):
    """ Color ground truth to matrix of classes. """
    max_class = max([cmap[key][0] for key in cmap])
    new = np.empty(img.shape[:2], dtype=np.min_scalar_type(max_class))
    for key in cmap:
        new[np.all(img == cmap[key][1], axis=2)] = cmap[key][0]
    return new


def get_full_names(path, files):
    """ Get names of files including extension from file names without extension.

    Parameter:
    ----------
    path: string
      Path to directory
    files: string or list of strings
      File names
    Return:
    -------
      names: list
    """
    names = []
    files = files if type(files) == list else [files]
    files = [os.path.splitext(f)[0] for f in files]
    full = sorted(os.listdir(path))
    if full:
        base = [os.path.splitext(f)[0] for f in full]
        names = [full[base.index(f)] for f in files]
    return names


def load_images(path, files=None):
    """ Loads images by name. File extension is not considered.

    Parameter:
    ----------
    path: string
      Path to directory
    files: string or list of strings
      File names
    Return:
    -------
      images: list
    """
    if files is None:
        load = sorted(os.listdir(path))
    else:
        load = get_full_names(path, files)

    return [imread(os.path.join(path, f)) for f in load]


def load_images_into_array(array, path, files=None, func=None):
    """ Loads images by name. File extension is not considered.

    Parameter:
    ----------
    array: array
      Array to store images in. It is assumed that all dimensions are adequate.
    path: string
      Path to directory
    files: string or list of strings
      File names
    """
    if files is None:
        load = sorted(os.listdir(path))
    else:
        load = get_full_names(path, files)

    for i, f in enumerate(load):
        img = imread(os.path.join(path, f))
        img = img if func is None else func(img)
        if len(array.shape) == 4 and len(img.shape) == 2:
            img = np.expand_dims(img, axis=2)
        rows, cols, chans = img.shape
        array[i, :rows, :cols, :chans] = img
