import os
import multiprocessing as mp

import numpy as np
from skimage.io import imread


def recall(confusion, classes):
    recall = []
    for i in classes:
        den = np.sum(confusion[i, classes])
        if den > 0.0:
            recall.append(np.true_divide(confusion[i, i], den))
        else:
            recall.append(np.nan)

    return recall


def pascal(confusion, classes):
    """ Also known as Jaccard index or intersection over union."""
    pascal = []
    for i in classes:
        den = np.sum(confusion[i, classes]) + np.sum(confusion[classes, i]) - confusion[i, i]
        if den > 0.0:
            pascal.append(np.true_divide(confusion[i, i], den))
        else:
            pascal.append(np.nan)

    return pascal


def average_recall(confusion, classes):
    x = np.array(recall(confusion, classes))
    x = x[np.isfinite(x)]
    return x.sum()/x.shape[0]


def average_pascal(confusion, classes):
    x = np.array(pascal(confusion, classes))
    x = x[np.isfinite(x)]
    return x.sum()/x.shape[0]


def global_recall(confusion, classes):
    den = sum([confusion[i, classes].sum() for i in classes])
    return float(sum([confusion[i, i] for i in classes]))/den


def convert(img, cmap):
    """ Converts an color coded image to array of classes.

    Parameters:
    -----------
    img: array
      Image.
    cmap: array like
      List or array of colors. Index corresponds to class.

    Return:
    -------
    converted: array
    """
    new = np.empty(img.shape[:2], dtype='uint')
    for i, color in enumerate(cmap):
        new[np.all(img == color, axis=2)] = i
    return new


def compare_files_worker(args):
    compare_files(args[0], args[1], args[2])


def compare_files(files, classes, cmap=None):
    """ Evaluates an image and returns a confusion matrix.

    Parameters:
    -----------
    files: tuple of strings
      Full path to both ground truth and result file.
    classes: array like
      List of classes in the ground truth.
    cmap: array like
      List or array of colors. Index corresponds to class.

    Return:
    -------
    confusion: array
       Confusion matrix.
    """
    gt = np.squeeze(imread(files[0]))
    res = np.squeeze(imread(files[1]))
    nclasses = len(classes)

    if len(gt.shape) > 2 and cmap is not None:
        gt = convert(gt, cmap)
    if len(res.shape) > 2 and cmap is not None:
        res = convert(res, cmap)

    classes = np.ones((gt.shape[0], gt.shape[1], nclasses), dtype='uint')*range(nclasses)
    gt_mask = gt[:, :, np.newaxis] == classes
    confusion = np.empty((nclasses, nclasses), dtype='uint')
    for i in range(nclasses):
        for j in range(nclasses):
            confusion[i, j] = (res[gt_mask[:, :, i]] == j).sum()

    return confusion


def evaluate_dir(file_list_res, file_list_gt, classes, cmap=None, processes=None):
    path_tuples = zip(file_list_res, file_list_gt)
    args = [(t, classes, cmap) for t in path_tuples]
    confusions = []

    processes = min(processes, mp.cpu_count()) if processes is None else processes
    pool = mp.Pool(processes=processes)
    for i, r in enumerate(pool.imap_unordered(compare_files_worker, args)):
        confusions.append(r)

    pool.close()
    pool.join()

    return confusions


def write_to_file(path, confusion, classes):
    np.savetxt(os.path.join(path, "confusion.txt"), confusion)
    with open(os.path.join(path, "summary.txt"), "w") as summary_file:
        summary_file.write("Classes: " + str(classes) + "\n"
                           "GR     : " + str(global_recall(confusion, classes)) + "\n"
                           "AR     : " + str(average_recall(confusion, classes)) + "\n"
                           "PR     : " + str(average_pascal(confusion, classes)))
