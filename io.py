"""
This file contains various utility functions.
"""
import os

from skimage.io import imread


def get_full_names(path, files):
    """ Get name of file including extension from file name without extension.

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
      names: list
    """
    if files is None:
        load = sorted(os.listdir(path))
    else:
        load = get_full_names(path, files)

    return [imread(os.path.join(path, f)) for f in load]
