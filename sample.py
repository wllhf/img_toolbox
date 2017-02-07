"""
This file contains basic functionality for the handling of data sets and sampling data points from data sets.
"""
import os
import numpy as np

from skimage.io import imread

import io as tbx_io


def grid_sample_coords(img_shape, grid_size, max_patch_size=np.array([0, 0])):
    """ Get the patch coordinates using a regular grid of an image given the sample parameters.

    Parameters:
    -----------
    img_shape: numpy array (2,)
      Image shape.
    grid_size: numpy array (2,)
      Distance in y and x direction between samples.
    max_patch_size: numpy array (2,)
      Maximum size of sample/feature patch (defines border size of unused pixels).

    Return:
    -------
    coords: numpy array (n,2) dtype `uint16`
      Array of matrix coordinates (y, x).
    """
    start = max_patch_size/2
    end = img_shape[:2] - max_patch_size/2
    x = np.arange(start[1], end[1], grid_size[1])
    y = np.arange(start[0], end[0], grid_size[0])
    rx = np.expand_dims(np.repeat(x, y.shape[0]), axis=1)
    ty = np.expand_dims(np.tile(y, x.shape[0]), axis=1)
    return np.hstack([ty, rx]).astype('uint16')


def generate_grid_samples(path, file_names, grid_size, max_patch_size):
    """ Get the patch coordinates using a regular grid of a list of images given the sample parameters.

    Parameters:
    -----------
    path: string
      Path to image files.
    file_names: list of strings
      List of file names of images.
    grid_size: numpy array (2,)
      Distance in y and x direction between samples.
    max_patch_size: numpy array (2,)
      Maximum size of sample patch (defines border size of unused pixels).

    Return:
    -------
    samples: numpy array (n, 3)
      The first column is the image index, second and third are the pixel coordinates.
    """
    file_names = tbx_io.get_full_names(path, file_names)
    samples = []
    for i, name in enumerate(file_names):
        h, w = imread(os.path.join(path, name)).shape[:2]
        coords = grid_sample_coords(np.array((h, w)), grid_size, max_patch_size)
        samples.append(np.hstack([np.ones((coords.shape[0], 1), dtype='uint16')*i, coords]))

    return np.vstack(samples).astype('uint16')


def filter_set(classes, samples, labels, arrays=None):
    """ Filter out data points by class label.

    Parameters:
    -----------
    classes: list or int
      Labels of classes to be filtered out of the data set.
    labels: numpy array (n,)
      Class labels.
    arrays: list numpy arrays, each (n, _)
      Arrays to subsample in the same way as labels.

    Return:
    -------
    (labels, arrays): numpy array and list of numpy arrays
      Filtered labels array and list of arrays.
    """
    if classes:
        classes = classes if type(classes) == list else [classes]
        indices = np.squeeze(np.logical_not(sum([labels == l for l in classes])))
        if arrays is not None:
            return samples[indices], labels[indices], [a[indices] for a in arrays]
        else:
            return samples[indices], labels[indices]
    else:
        return samples, labels


def balance(samples, labels, arrays=None):
    """ Balance data set by subsampling.

    Parameters:
    -----------
    samples: numpy array (n, _)
      Data points.
    labels: numpy array (n,)
      Class labels.
    arrays: list numpy arrays, each (n, _)
      Arrays to subsample in the same way as labels.

    Return:
    -------
    (samples, labels, arrays): tuple of numpy arrays
      Subsampled data points, labels and optional list of arrays.
    """
    indices = [np.where(labels == l)[0] for l in np.unique(labels)]
    n_samples = min([idx.shape[0] for idx in indices])
    indices = np.hstack([np.random.choice(idx, n_samples, replace=False) for idx in indices])

    if arrays is not None:
        return samples[indices], labels[indices], [a[indices] for a in arrays]
    else:
        return samples[indices], labels[indices]


def reshape(coords, values, shape):
    """ Reshape values to matrix.

    Parameters:
    -----------
    coords: numpy array (n, 2)
      Coordinates of values.
    values: numpy array (n, _)
    shape: array like (2,)
      Shape of image.

    Return:
    -------
    matrix: numpy array
    """
    shape = (shape[0], shape[1])
    if len(values.shape) > 1:
        shape = (shape[0], shape[1], values.shape[1])
    matrix = np.zeros(shape, dtype=values.dtype)
    matrix[coords[:, 0], coords[:, 1]] = values
    return np.squeeze(matrix)


def reshape_set(samples, values, shapes):
    """ Reshape samples to matrices.

    Parameters:
    -----------
    samples: numpy array (n, 3)
    values: numpy array (n, _)
    shapes: list of numpy arrays (2,)
      Shapes of images. len(shapes) == max(samples[:,0])

    Return:
    -------
    matrices: list
    """
    img_indices = np.unique(samples[:, 0])
    matrices = []
    for idx in img_indices:
        indices = samples[:, 0] == idx
        matrices.append(reshape(samples[indices, 1:], values[indices], shapes[idx]))
    return matrices


def stich(coords, patches, shape, psize, mode='sum'):
    """ Stich patches back together.

    Parameters:
    -----------
    coords: numpy array (n, 2)
      Coordinates of patches.
    patches: numpy array (n, prod(psize))
    shape: array like (2,)
      Shape of image.
    psize: array like (2|3,)

    Return:
    -------
    matrix: numpy array (shape[0], shape[1], psize[2])
    """
    px, py = psize[:2]

    if len(psize) == 2:
        shape = (shape[0], shape[1])
    elif len(psize) == 3:
        shape = (shape[0], shape[1], psize[2])

    matrix = np.zeros(shape, dtype=patches.dtype)
    count = np.zeros(shape, dtype=patches.dtype)
    for i, (cx, cy) in enumerate(coords):
        sx = cx - px/2
        sy = cy - py/2
        ex = cx + px/2
        ey = cy + py/2
        matrix[sx:ex+1, sy:ey+1] += patches[i].reshape(psize)
        count[sx:ex+1, sy:ey+1] += np.ones(psize)

    if mode == 'mean':
        matrix /= count

    return np.squeeze(matrix)


def color_to_class(img, cmap):
    """ Color ground truth to matrix of classes. """
    new = np.empty(img.shape[:2], dtype='uint')
    for i, color in enumerate(cmap):
        new[np.all(img == color, axis=2)] = i
    return new


def class_to_color(gt, cmap):
    """ Color ground truth with color map. """
    new = np.empty((gt.shape[0], gt.shape[1], cmap.shape[1]), dtype=cmap.dtype)
    for i, color in enumerate(cmap):
        new[gt == i] = color
    return new
