""" MSRCv2. Using TextonBoostSplits from jamie.shotton.org/work/data/TextonBoostSplits.zip"""
import os

from skimage.io import imread
from skimage import img_as_float

from ..adjustment import per_image_standardization
from .util import load_images, load_images_into_array, color_to_class, class_to_color
from .sample import generate_grid_samples

dir_img = "Images"
dir_lbl = "GroundTruth"
dir_spl = "TextonBoostSplits"
f_train = "Train.txt"
f_val   = "Validation.txt"
f_test  = "Test.txt"

color_map = {
    'void':      (0, (0, 0, 0)),
    'building':  (1, (128, 0, 0)),
    'grass':     (2, (0, 128, 0)),
    'tree':      (3, (128, 128, 0)),
    'cow':       (4, (0, 0, 128)),
    'horse':     (5, (128, 0, 128)),
    'sheep':     (6, (0, 128, 128)),
    'sky':       (7, (128, 128, 128)),
    'mountain':  (8, (64, 0, 0)),
    'aeroplane': (9, (192, 0, 0)),
    'water':     (10, (64, 128, 0)),
    'face':      (11, (192, 128, 0)),
    'car':       (12, (64, 0, 128)),
    'bicycle':   (13, (192, 0, 128)),
    'flower':    (14, (64, 128, 128)),
    'sign':      (15, (192, 128, 128)),
    'bird':      (16, (0, 64, 0)),
    'book':      (17, (128, 64, 0)),
    'chair':     (18, (0, 192, 0)),
    'road':      (19, (128, 64, 128)),
    'cat':       (20, (0, 192, 128)),
    'dog':       (21, (128, 192, 128)),
    'body':      (22, (64, 64, 0)),
    'boat':      (23, (192, 64, 0))
}


def to_class(img):
    return color_to_class(img, color_map)


def to_color(img):
    return class_to_color(img, color_map)


def file_list(path, subset='train'):
    if subset == 'train':
        fname = f_train
    if subset == 'val':
        fname = f_val
    if subset == 'test':
        fname = f_test

    return sorted(open(os.path.join(path, dir_spl, fname), 'r').read().splitlines())


class msrc_gen:
    def __init__(self, path, subset='train', lbl_type=None, integral=True, standardize=False):
        self.lbl_type = lbl_type
        self.integral = integral
        self.standardize = standardize
        self.path = os.path.expanduser(path)
        self.flist_img = file_list(self.path, subset)
        self.flist_lbl = [os.path.splitext(n)[0]+'_GT'+os.path.splitext(n)[1] for n in self.flist_img]
        self.cur = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return len(self.flist_img)

    def next(self):
        if self.cur < len(self.flist_img):
            self.cur = self.cur + 1
            img = imread(os.path.join(self.path, dir_img, self.flist_img[self.cur-1]))
            img = img if self.integral else img_as_float(img)
            img = per_image_standardization(img) if self.standardize else img

            if self.lbl_type is None:
                return img
            else:
                lbl = imread(os.path.join(self.path, dir_lbl, self.flist_lbl[self.cur-1]))
                lbl = color_to_class(lbl, color_map) if self.lbl_type == 'class' else lbl
                return img, lbl
        else:
            raise StopIteration()


class msrc:

    def __init__(self, path):
        self.path = os.path.expanduser(path)

    def _prepare_data(self, flist, arrays=None, lbl_type='class', integral=True, standardize=False):
        flist_lbl = [os.path.splitext(n)[0]+'_GT'+os.path.splitext(n)[1] for n in flist]

        if arrays is not None:
            func = None if integral else img_as_float
            func = per_image_standardization if standardize else func
            load_images_into_array(arrays[0], os.path.join(self.path, dir_img), flist, func=func)
            func = None if lbl_type == 'color' else to_class
            load_images_into_array(arrays[1], os.path.join(self.path, dir_lbl), flist_lbl, func=func)
        else:
            imgs = load_images(os.path.join(self.path, dir_img), flist)
            lbls = load_images(os.path.join(self.path, dir_lbl), flist_lbl)

            if lbl_type == 'class':
                lbls = [to_class(lbl, color_map) for lbl in lbls]

            if not integral:
                imgs = [img_as_float(img) for img in imgs]

            if standardize:
                imgs = [per_image_standardization(img) for img in imgs]

            return (imgs, lbls)

    def generate_grid_samples(self, subset='train', grid_size=(5, 5), max_patch_size=(11, 11)):
        flist = file_list(self.path, subset)
        return generate_grid_samples(os.path.join(self.path, dir_img), flist, grid_size, max_patch_size)

    def data(self, subset='train', arrays=None, lbl_type='class', integral=True, standardize=False):
        """
        If arrays are given they should have the shape (n, 320, 320, c) with
        n=276 for train, 59 for val and 256 for test.
        """
        flist = file_list(self.path, subset)
        return self._prepare_data(flist,
                                  arrays=arrays,
                                  lbl_type=lbl_type,
                                  integral=integral,
                                  standardize=standardize)


if __name__ == "__main__":
    msrc = msrc("~/data/msrc_v2/")
    imgs, lbls = msrc.test(lbl_type='color', integral=True)
    print(imgs[0])
    print(lbls[0])
    print(imgs[0].dtype, lbls[0].dtype)
    # img, lbl = msrc.train(lbl_type='class', integral=False)
    # print(imgs[0])
    # print(lbls[0])
    # print(imgs[0].dtype, lbls[0].dtype)
    # imgs, lbls = msrc.train(lbl_type='color', integral=False)
    # print(imgs[0])
    # print(lbls[0])
    # print(imgs[0].dtype, lbls[0].dtype)
