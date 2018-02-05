""" CK+ """
import os

import numpy as np
from skimage import img_as_float
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.transform import resize

from ..adjustment import per_image_standardization
from ..label import one_hot_repr

dir_img = "cohn-kanade-images"
dir_emo = "Emotion"
dir_fac = "FACS"
dir_lan = "Landmarks"


def emotion_imgs(path):
    imgs = []
    lbls = []
    subs = []

    path_emo = os.path.join(path, dir_emo)
    for root, dirs, files in os.walk(path_emo):
        if files:
            subs.append(root.split('/')[-2])
            with open(os.path.join(root, files[0]), 'r') as lbl_file:
                lbls.append(int(float(lbl_file.read().replace('\n', '').replace(' ', ''))))

            path_img = os.path.join(root.replace(dir_emo, dir_img), files[0].replace("_emotion.txt", ".png"))
            imgs.append(imread(path_img))

    return imgs, lbls, subs


def prepare_data(imgs, lbls, subs, img_size=[64, 64], one_hot=True, standardize=True):
    imgs = [np.expand_dims(rgb2gray(img), axis=2) for img in imgs]
    imgs = [img_as_float(img) for img in imgs]
    if standardize:
        imgs = [per_image_standardization(img) for img in imgs]
    imgs = [resize(img, img_size, mode='reflect') for img in imgs]
    imgs = np.array(imgs).astype('float32')
    lbls, subs = np.array(lbls), np.array(subs)
    if one_hot:
        lbls = one_hot_repr(lbls)

    return imgs, lbls, subs


class leave_one_out_gen:

    def __init__(self, path, img_size=[64, 64], lbl_type='emo', one_hot=True, standardize=True):
        if lbl_type == 'emo':
            self.imgs, self.lbls, self.subs = emotion_imgs(path)
            self.imgs, self.lbls, self.subs = prepare_data(self.imgs, self.lbls, self.subs,
                                                           img_size=[64, 64], one_hot=True, standardize=True)
            self.unique = np.unique(self.subs)
        else:
            return NotImplementedError

        self.cur = 0

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self.unique.shape[0]

    def next(self):
        if self.cur < len(self):
            indices = self.subs == self.unique[self.cur]
            trn = self.imgs[~indices], self.lbls[~indices]
            tst = self.imgs[indices], self.lbls[indices]
            self.cur = self.cur + 1
            return trn, tst
        else:
            raise StopIteration()


def get_data(path, img_size=[64, 64], lbl_type='emo', one_hot=True, standardize=True):
        if lbl_type == 'emo':
            imgs, lbls, subs = emotion_imgs(path)
            imgs, lbls, subs = prepare_data(imgs, lbls, subs,
                                            img_size=[64, 64], one_hot=True, standardize=True)
            return imgs, lbls, subs
        else:
            return NotImplementedError


if __name__ == "__main__":
    # emotion_imgs("/home/mw/data/ck+/")
    gen = leave_one_out_gen("/home/mw/data/ck+/")
    for elem in gen:
        trn, tst = elem
        imgs, lbls = trn
        print(imgs.dtype)
        print(imgs.shape, lbls.shape)
        imgs, lbls = tst
        print(imgs.shape, lbls.shape)
