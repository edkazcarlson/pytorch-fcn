#!/usr/bin/env python

import collections
import os.path as osp

import numpy as np
import PIL.Image
import scipy.io
import torch
from torch.utils import data
import cv2
import torchvision

class VOCClassSegBase(data.Dataset):

    class_names = np.array([
        'background',
        'aeroplane',
        'bicycle',
        'bird',
        'boat',
        'bottle',
        'bus',
        'car',
        'cat',
        'chair',
        'cow',
        'diningtable',
        'dog',
        'horse',
        'motorbike',
        'person',
        'potted plant',
        'sheep',
        'sofa',
        'train',
        'tv/monitor',
    ])
    #bgr values not ABSV
    mean_bgr = np.array([104.00698793, 116.66876762, 122.67891434])
    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform

        # VOC2011 and others are subset of VOC2012
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2012')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(
                dataset_dir, 'ImageSets/Segmentation/%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
                lbl_file = osp.join(
                    dataset_dir, 'SegmentationClass/%s.png' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __len__(self):
        return len(self.files[self.split])

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        lbl = PIL.Image.open(lbl_file)
        lbl = np.array(lbl, dtype=np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

    #img is 0-255
    def transform(self, img, lbl):
        #original transform:
        img = img[:, :, ::-1]  # RGB -> BGR
        img = img.astype(np.float64)
        img -= self.mean_bgr
        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).float()
        # #bgr -> hsv
        # img = np.float32(img)

        # img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        # img = torch.tensor(img)
        # img = img.transpose(2,1)
        # img = img.transpose(0,1)
        # #hsv -> absv
        # img = hsv2absv(img, False)
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def untransform(self, img, lbl):
        # #absv -> hsv
        # img = absv2hsv(img, False)
        # #hsv -> bgr
        # img = img.transpose(0, 1).transpose(1, 2)
        # img = img.numpy()
        # img = cv2.cvtColor(img, cv2.COLOR_HSV2BGR)

        # # img = img.transpose(1, 2, 0)
        # img = img.astype(np.uint8)
        # img = img[:, :, ::-1]
        # lbl = lbl.numpy()
        img = img.numpy()
        img = img.transpose(1, 2, 0)
        img += self.mean_bgr
        img = img.astype(np.uint8)
        img = img[:, :, ::-1]
        lbl = lbl.numpy()
        return img, lbl


class VOC2011ClassSeg(VOCClassSegBase):

    def __init__(self, root, split='train', transform=False):
        super(VOC2011ClassSeg, self).__init__(
            root, split=split, transform=transform)
        pkg_root = osp.join(osp.dirname(osp.realpath(__file__)), '..')
        imgsets_file = osp.join(
            pkg_root, 'ext/fcn.berkeleyvision.org',
            'data/pascal/seg11valid.txt')
        dataset_dir = osp.join(self.root, 'VOC/VOCdevkit/VOC2012')
        for did in open(imgsets_file):
            did = did.strip()
            img_file = osp.join(dataset_dir, 'JPEGImages/%s.jpg' % did)
            lbl_file = osp.join(dataset_dir, 'SegmentationClass/%s.png' % did)
            self.files['seg11valid'].append({'img': img_file, 'lbl': lbl_file})


class VOC2012ClassSeg(VOCClassSegBase):

    url = 'http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar'  # NOQA

    def __init__(self, root, split='train', transform=False):
        super(VOC2012ClassSeg, self).__init__(
            root, split=split, transform=transform)


class SBDClassSeg(VOCClassSegBase):

    # XXX: It must be renamed to benchmark.tar to be extracted.
    url = 'http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz'  # NOQA

    def __init__(self, root, split='train', transform=False):
        self.root = root
        self.split = split
        self._transform = transform

        dataset_dir = osp.join(self.root, 'VOC/benchmark_RELEASE/dataset')
        self.files = collections.defaultdict(list)
        for split in ['train', 'val']:
            imgsets_file = osp.join(dataset_dir, '%s.txt' % split)
            for did in open(imgsets_file):
                did = did.strip()
                img_file = osp.join(dataset_dir, 'img/%s.jpg' % did)
                lbl_file = osp.join(dataset_dir, 'cls/%s.mat' % did)
                self.files[split].append({
                    'img': img_file,
                    'lbl': lbl_file,
                })

    def __getitem__(self, index):
        data_file = self.files[self.split][index]
        # load image
        img_file = data_file['img']
        img = PIL.Image.open(img_file)
        img = np.array(img, dtype=np.uint8)
        # load label
        lbl_file = data_file['lbl']
        mat = scipy.io.loadmat(lbl_file)
        lbl = mat['GTcls'][0]['Segmentation'][0].astype(np.int32)
        lbl[lbl == 255] = -1
        if self._transform:
            return self.transform(img, lbl)
        else:
            return img, lbl

# https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html#color_convert_rgb_hsv
def hsv2absv(x, normalizeSB):
    sMean = -0.4567486643791199
    vMean =  0.0775344967842102

    sStd = 0.42398601770401
    vStd = 0.47892168164253235

    x = x.float()
    # print(f'hsv2absv: {x}')
    h = x[0]
    s = x[1]
    v = x[2]
    # print(f'h shape:  {h.shape}')
    h = torch.pi * 2 * (h / 360)
    a = torch.sin(h)
    b = torch.cos(h)
    s /= 255
    v /= 255

    s = (s - .5) * 2
    v = (v - .5) * 2

    if normalizeSB:
        s = (s - sMean)/sStd
        v = (v - vMean)/vStd
    absv = torch.stack((a,b,s,v))
    return absv

# https://docs.opencv.org/4.x/de/d25/imgproc_color_conversions.html#color_convert_rgb_hsv
def absv2hsv(x, normalizeSB):
    sMean = -0.4567486643791199
    vMean =  0.0775344967842102

    sStd = 0.42398601770401
    vStd = 0.47892168164253235

    x = x.float()
    a = x[0]#sin
    b = x[1]#cos
    s = x[2]
    v = x[2]
    # print(f'h shape:  {h.shape}')
    t = torch.arctan2(a, b) #radians
    h = 360 * (t / (2 * torch.pi)) #degrees
    h = torch.where( h < 0 , h + 360, h)
    s /= 255
    v /= 255

    s = (s / 2) + .5
    v = (v / 2) + .5

    if normalizeSB:
        s = s * sStd + sMean
        v = v * vStd + vMean

    hsv = torch.stack((h,s,v))
    return hsv