"""
This file contains functionality to generate low-level features as derivatives and HOG-like features.
"""
import numpy as np
from scipy.ndimage.filters import convolve
from skimage.color import rgb2lab, rgb2gray


def noise(imgs, dtype='float32'):
    return [np.random.normal(0, 1, size=img.shape).astype(dtype) for img in imgs]


def lab(imgs, dtype='float32'):
    return [rgb2lab(img).astype(dtype) for img in imgs]


def derivatives(imgs, dtype='float32'):
    ders = []
    for img in imgs:
        gray = rgb2gray(img)
        gx = np.empty(gray.shape, dtype=np.double)
        gx[:, 0] = 0
        gx[:, -1] = 0
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy = np.empty(gray.shape, dtype=np.double)
        gy[0, :] = 0
        gy[-1, :] = 0
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]

        gxx = np.empty(gray.shape, dtype=np.double)
        gxx[:, :2] = 0
        gxx[:, -2:] = 0
        gxx[:, 2:-2] = gx[:, 3:-1] - gy[:, 1:-3]
        gyy = np.empty(gray.shape, dtype=np.double)
        gyy[:2, :] = 0
        gyy[-2:, :] = 0
        gyy[2:-2, :] = gy[3:-1, :] - gy[1:-3, :]

        der = np.empty((gray.shape[0], gray.shape[1], 2), dtype=np.double)
        der[:, :, 0] = np.sqrt(gx**2 + gy**2)
        der[:, :, 1] = np.sqrt(gxx**2 + gyy**2)
        ders.append(der.astype(dtype))

    return ders


def dense_hoglike_simple(imgs, cell_size=[8, 8], orientations=9, dtype='float32'):
    bins_lower = np.linspace(0, 180, orientations+1)[:-1]
    bins_upper = np.linspace(0, 180, orientations+1)[1:]

    hds = []
    for img in imgs:
        img = rgb2gray(img)
        gx = np.empty(img.shape, dtype='float64')
        gx[:, 0] = 0
        gx[:, -1] = 0
        gx[:, 1:-1] = img[:, 2:] - img[:, :-2]
        gy = np.empty(img.shape, dtype='float64')
        gy[0, :] = 0
        gy[-1, :] = 0
        gy[1:-1, :] = img[2:, :] - img[:-2, :]
        magnitude = np.sqrt(gx**2 + gy**2)

        orientation = np.arctan2(gx, gy) * (180 / np.pi) % 180
        orientation = np.repeat(np.expand_dims(orientation, axis=2), orientations, axis=2)
        orientation = np.logical_and((bins_lower < orientation), (orientation <= bins_upper))
        magnitude = np.repeat(np.expand_dims(magnitude, axis=2), orientations, axis=2)
        magnitude = magnitude*orientation

        histogram = np.empty((img.shape[0], img.shape[1], orientations))
        for i in range(orientations):
            convolve(magnitude[:, :, i], np.ones(cell_size), histogram[:, :, i], mode='reflect')

        hds.append(histogram.astype(dtype))

    return hds


def dense_hoglike(imgs, nblock=[2, 2], cell_size=[8, 8], orientations=9, norm=True, eps=1e-5, dtype='float32'):
    bins_lower = np.linspace(0, 180, orientations+1)[:-1]
    bins_upper = np.linspace(0, 180, orientations+1)[1:]

    hds = []
    for img in imgs:
        gray = rgb2gray(img)
        gx = np.empty(gray.shape, dtype='float64')
        gx[:, 0] = 0
        gx[:, -1] = 0
        gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
        gy = np.empty(gray.shape, dtype='float64')
        gy[0, :] = 0
        gy[-1, :] = 0
        gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
        magnitude = np.sqrt(gx**2 + gy**2)

        orientation = np.arctan2(gx, gy) * (180 / np.pi) % 180
        orientation = np.repeat(np.expand_dims(orientation, axis=2), orientations, axis=2)
        orientation = np.logical_and((bins_lower < orientation), (orientation <= bins_upper))
        magnitude = np.repeat(np.expand_dims(magnitude, axis=2), orientations, axis=2)
        magnitude = magnitude*orientation

        hist = np.empty((gray.shape[0], gray.shape[1], orientations), dtype='float32')
        for i in range(orientations):
            convolve(magnitude[:, :, i], np.ones(cell_size), hist[:, :, i], mode='reflect')

        if np.prod(nblock) > 1:
            px, py = np.array(cell_size)*(np.array(nblock)-1)
            indices = np.zeros((px, py, hist.shape[2]), dtype='bool')
            indices[[0, 0, -1, -1], [0, -1, 0, -1], :] = 1

            hog = np.zeros((hist.shape[0], hist.shape[1], hist.shape[2]*np.prod(nblock)), dtype='float32')
            for i in range(px, hist.shape[0]-px):
                for j in range(py, hist.shape[1]-py):
                    hog[i, j, :] = hist[i-px:i, j-py:j, :][indices].ravel()

        if norm:
            den = np.expand_dims(np.sqrt(np.sum(hog**2, axis=2) + eps**2), axis=2).astype('float32')
            with np.errstate(divide='ignore', invalid='ignore'):
                hog /= den
                hog[~ np.isfinite(hog)] = 0  # -inf inf nan

        hds.append(hog.astype(dtype))

    return hds


def convert_img_worker(args):
    return convert_img(args[0], args[1], args[2])


def convert_img(img, channels, dtype='float32'):
    """
    Parameter:
    ----------
    path: numpy array
      Images
    channels: list of strings
      Low-level feature channels.
    Return:
    -------
      img: numpy array
    """
    features = []
    img = [img]
    if 'rgb' in channels:
        features.append(img[0])
    if 'lab' in channels:
        features.append(lab(img, dtype=dtype)[0])
    if 'der' in channels:
        features.append(derivatives(img, dtype=dtype)[0])
    if 'hog_simple' in channels:
        features.append(dense_hoglike_simple(img, dtype=dtype)[0])
    if 'hog' in channels:
        features.append(dense_hoglike(img, dtype=dtype)[0])
    if 'noise' in channels:
        features.append(noise(img, dtype=dtype)[0])

    return np.concatenate(features, axis=2)


def convert_list(imgs, channels, dtype='float32'):
    """
    Parameter:
    ----------
    path: numpy array or list of numpy arrays
      Images
    channels: list of strings
      Low-level feature channels.
    Return:
    -------
      imgs: list of images
    """

    features = []
    imgs = imgs if type(imgs) == list else [imgs]
    if 'rgb' in channels:
        features.append(imgs)
    if 'lab' in channels:
        features.append(lab(imgs, dtype=dtype))
    if 'der' in channels:
        features.append(derivatives(imgs, dtype=dtype))
    if 'hog_simple' in channels:
        features.append(dense_hoglike_simple(imgs, dtype=dtype))
    if 'hog' in channels:
        features.append(dense_hoglike(imgs, dtype=dtype))
    if 'noise' in channels:
        features.append(noise(imgs, dtype=dtype))

    return [np.concatenate(chans, axis=2) for chans in zip(*features)]
