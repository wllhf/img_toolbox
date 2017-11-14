"""
This file contains basic functionality for the adjustment of images.
"""
from __future__ import division

import numpy as np


def per_image_standardization(images):
    """ Scales images to zero mean and unit norm (image - mean) / capped_stddev.

    Parameters:
    -----------
    images: numpy array (n, k, l, c), (n, k, l) or (n, k*l*c)
      Images.

    Return:
    -------
    images: numpy array (n, k, l, c), (n, k, l) or (n, k*l*c)

    Note:
    -----
    Standard deviation is capped to handle uniform images (division by zero) capped_stddev = max(stddev, 1.0/(k*l*c)).
    If images is of integral dtype it is converted to float and the maximum of images.dtype is used for normalization.
    This assumes smallest possible dtype was chosen.

    Raises:
    -------
    ValueError: if shape of images is incompatible with this function
    """
    if np.issubdtype(images.dtype, np.integer):
        images = images / np.iinfo(images.dtype).max

    if len(images.shape) == 2:
        mean, std = images.mean(axis=1, keepdims=True), images.std(axis=1, keepdims=True)
    elif len(images.shape) == 3:
        mean, std = images.mean(axis=(1, 2), keepdims=True), images.std(axis=(1, 2), keepdims=True)
    elif len(images.shape) == 4:
        mean, std = images.mean(axis=(1, 2, 3), keepdims=True), images.std(axis=(1, 2, 3), keepdims=True)
    else:
        raise ValueError("given image shape is not compatible")

    images = images - mean
    images = images / np.maximum(std, 1.0 / np.prod(images.shape[1:]))
    return images


def crop(images, size, coords=None):
    """ Crops images.

    Parameters:
    -----------
    images: numpy array (n, k, l, c)
      Images.
    size: array like (2,)
      Size of new images [u, v].
    coords: array like (2,) (default: None)
      Center coordinates of resulting images. Defaults to center of original images.

    Return:
    -------
    images: numpy array (n, u, v, c)

    Raises:
    -------
    ValueError: if shape of images, coords or size is incompatible with this function or with each other.
    """
    coords = np.array(images.shape[1:3])//2 if coords is None else np.array(coords)
    size = np.array(size)

    if len(images.shape) != 4:
        raise ValueError("given image shape is not compatible (!=4)")
    if size.shape[0] != 2:
        raise ValueError("given size is not compatible (!=2)")
    if coords is not None and coords.shape[0] != 2:
        raise ValueError("given coords are not compatible (!=2)")

    s = coords - size//2
    e = coords + (size-1)//2 + 1

    valid = ((0, 0) <= s) * (e <= images.shape[1:3])
    if not valid.all():
        raise ValueError("given images, coords and size incompatible")

    return images[:, s[0]:e[0], s[1]:e[1], :]


def crop_random(images, size, ncrops=1):
    """ Crops images at random center coordinates.

    Parameters:
    -----------
    images: numpy array (n, k, l, c)
      Images.
    size: array like (2,)
      Size of new images [u, v].
    ncrops: int (default: 1)
      If > 1 multiple stacked crops are returned

    Return:
    -------
    images: numpy array (n*ncrops, u, v, c)

    Raises:
    -------
    ValueError: if shape of images, coords or size is incompatible with this function or with each other.
    """
    size = np.array(size)
    s = size//2
    e = images.shape[1:3] - (size-1)//2 - 1

    y = np.random.random_integers(s[0], e[0], size=ncrops)
    x = np.random.random_integers(s[1], e[1], size=ncrops)
    coords = np.stack([y, x], axis=1)

    new = []
    for c in coords:
        new.append(crop(images=images, size=size, coords=c))

    return np.concatenate(new, axis=0)
