""" MSRCv2. Using TextonBoostSplits from jamie.shotton.org/work/data/TextonBoostSplits.zip"""
import os

from skimage import img_as_float

from .util import load_images, load_images_into_array, color_to_class
from ..sample import generate_grid_samples

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


class msrc:

    def __init__(self, path):
        self.path = os.path.expanduser(path)

    def _color_to_class(self, lbl):
        return color_to_class(lbl, color_map)

    def _file_list(self, subset='train'):
        if subset == 'train':
            fname = f_train
        if subset == 'val':
            fname = f_val
        if subset == 'test':
            fname = f_test

        return sorted(open(os.path.join(self.path, dir_spl, fname), 'r').read().splitlines())

    def _prepare_data(self, file_list, arrays=None, lbl_type='class', integral=True):
        file_list_lbl = [os.path.splitext(n)[0]+'_GT'+os.path.splitext(n)[1] for n in file_list]

        if arrays is not None:
            func = None if integral else img_as_float
            load_images_into_array(arrays[0], os.path.join(self.path, dir_img), file_list, func=func)
            func = None if lbl_type == 'color' else self._color_to_class
            load_images_into_array(arrays[1], os.path.join(self.path, dir_lbl), file_list_lbl, func=func)
        else:
            imgs = load_images(os.path.join(self.path, dir_img), file_list)
            lbls = load_images(os.path.join(self.path, dir_lbl), file_list_lbl)

            if lbl_type == 'class':
                lbls = [self._color_to_class(lbl) for lbl in lbls]

            if not integral:
                imgs = [img_as_float(img) for img in imgs]

            return (imgs, lbls)

    def generate_grid_samples(self, subset='train', grid_size=(5, 5), max_patch_size=(11, 11)):
        file_list = self._file_list(subset)
        return generate_grid_samples(os.path.join(self.path, dir_img), file_list, grid_size, max_patch_size)

    def data(self, subset='train', arrays=None, lbl_type='class', integral=True):
        """
        If arrays are given they should have the shape (n, 320, 320, c) with
        n=276 for train, 59 for val and 256 for test.
        """
        file_list = self._file_list(subset)
        return self._prepare_data(file_list, arrays=arrays, lbl_type=lbl_type, integral=integral)


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
