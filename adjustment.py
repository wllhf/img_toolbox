"""
This file contains basic functionality for the adjustment of images.
"""
from __future__ import division

import numpy as np


def per_image_standardization(images):
    """ Scales images to zero mean and unit norm (image - mean) / capped_stddev.

    Parameters:
    -----------
    images: numpy array (n, k, l, c)
      Images.

    Return:
    -------
    images: numpy array (n, k, l, c)

    Note:
    -----
    Standard deviation is capped to handle uniform images (division by zero) capped_stddev = max(stddev, 1.0/(k*l*c)).
    If images is of integral dtype it is converted to float and the maximum of images.dtype is used for normalization.
    This assumes smallest possible dtype was chosen.

    Raises:
    -------
    ValueError: if shape of images is incompatible with this funciton (len(images.shape) != 4)
    """
    if len(images.shape) != 4:
        raise ValueError("given image shape is not compatible (!=4)")

    if np.issubdtype(images.dtype, np.integer):
        images = images / np.iinfo(images.dtype).max

    mean, std = images.mean(axis=(1, 2, 3), keepdims=True), images.std(axis=(1, 2, 3), keepdims=True)
    images = images - mean
    images = images / np.maximum(std, 1.0 / np.prod(images.shape[1:]))
    return images
