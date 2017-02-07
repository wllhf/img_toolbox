"""
This file contains basic functionality for the handling of bounding boxes.
"""
import numpy as np

from skimage.morphology import convex_hull_image


def bbox_center(bbox):
    """ Center of bounding box.

    Parameters:
    -----------
    bbox: tuple (y0, x0, h, w)

    Return:
    -------
    center: tuple (y, x)
    """
    y, x, h, w = bbox
    return int(y + h/2), int(x + w/2)


def coords_to_bbox(coords):
    """ Bounding box of coords.

    Parameters:
    -----------
    coords: numpy array (n, 2)
      Coordinates.

    Return:
    -------
    bbox: tuple (y0, x0, h, w)
    """
    min_y, min_x, max_y, max_x = coords[0].min(), coords[1].min(), coords[0].max(), coords[1].max()
    return min_y, min_x, max_y - min_y, max_x - min_x


def mask_to_bbox(mask, label=None):
    """ Bounding box of pixel mask.

    Parameters:
    -----------
    mask: numpy array
      Binary mask or label image.
    label: int (optional)
      Label if mask is label image.

    Return:
    -------
    bbox: tuple (y0, x0, h, w)
    """
    mask = mask if label is None else mask == label
    coords = np.where(mask)
    return coords_to_bbox(coords)


def coords_to_chull(coords, shape):
    """ Convex hull of coords.

    Parameters:
    -----------
    coords: numpy array (n, 2)
      Coordinates.
    shape: array like (2,)
      Shape of image.

    Return:
    -------
    matrix: numpy array with input shape
    """
    matrix = np.zeros(shape[:2], dtype='uint8')
    matrix[coords[:, 0], coords[:, 1]] = 1
    return convex_hull_image(matrix)


def iou(bboxa, bboxb):
    """ Intersection over union of bounding boxes bboxa and bboxb.

    Parameters:
    -----------
    bboxa: tuple (y0, x0, h, w)
    bboxb: tuple (y0, x0, h, w)

    Return:
    -------
    iou: float
      area(bboxa n bboxb)/area(bboxa u bboxb)
    """
    yamin, xamin, ha, wa = bboxa
    ybmin, xbmin, hb, wb = bboxb
    yamax, xamax = yamin+ha, xamin+wa
    ybmax, xbmax = ybmin+hb, xbmin+wb
    xs, ys = max(xamin, xbmin), max(yamin, ybmin)
    xe, ye = min(xamax, xbmax), min(yamax, ybmax)
    areai = (xe-xs)*(ye-ys) if xe > xs and ye > ys else 0.0
    areau = ha*wa + hb*wb - areai
    return float(areai)/areau


def has_intersection(bboxa, bboxb):
    """ Returns True if bboxa has an intersection with bboxb.

    Parameters:
    -----------
    bboxa: tuple (y0, x0, h, w)
    bboxb: tuple (y0, x0, h, w)

    Return:
    -------
    iou: bool
    """
    yamin, xamin, ha, wa = bboxa
    ybmin, xbmin, hb, wb = bboxb
    yamax, xamax = yamin+ha, xamin+wa
    ybmax, xbmax = ybmin+hb, xbmin+wb
    xs, ys = max(xamin, xbmin), max(yamin, ybmin)
    xe, ye = min(xamax, xbmax), min(yamax, ybmax)
    return xe > xs and ye > ys


def evaluate_object_category(ground_truth, prediction, min_iou=0.5):
    """

    Parameters:
    -----------
    ground_truth: list of tuples (y0, x0, h, w) (bounding boxes)
    prediction: list of tuples (y0, x0, h, w) (bounding boxes)
    min_iou: float
      Intersection over union must be greater than min_iou.

    Return:
    -------
    (precision, recall, tp, fp) : float, float, int, int
    """
    # edge cases
    if len(prediction) == 0 and len(ground_truth) > 0:
        return 0.0, 0.0, 0.0, 0.0
    if len(prediction) == 0 and len(ground_truth) == 0:
        return 1.0, 1.0, 0.0, 0.0
    if len(prediction) > 0 and len(ground_truth) == 0:
        return 0.0, 1.0, 0.0, len(prediction)

    # predictions that intersect with multiple objects are associated by greatest iou
    ious = [[iou(gt, p) for gt in ground_truth] for p in prediction]
    association = [max(enumerate(lst), key=lambda x:x[1]) for lst in ious]  # generates (key, value) pairs

    # only predictions with iou > min_iou are associated
    association = [a if a[1] > min_iou else (-1, a[1]) for a in association]

    # only prediction with greatest iou is associated with the ground truth
    for gt in range(len(ground_truth)):
        indices = [(i, a[1]) for i, a in enumerate(association) if a[0] == gt]
        if len(indices) > 1:
            indices.pop(max(enumerate(indices), key=lambda x: x[1][1])[0])
            for elem in indices:
                association[elem[0]] = (-1, association[elem[0]][1])

    # return precision, recall, tp, fp
    tp = sum([1 for a in association if a[0] >= 0])
    return float(tp)/len(prediction), float(tp)/len(ground_truth), tp, len(prediction) - tp
