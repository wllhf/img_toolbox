"""
This file contains basic functionality for the extraction of label information from data sets.
"""
import numpy as np


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
