from torch.utils.data import Dataset
import imageio as io
import numpy as np
import cv2
import os
import torch
import json
from collections import defaultdict
from scipy.ndimage import filters
from numpy import *
import matplotlib.pyplot as plt
import scipy.io as sio

# shape = (180, 320)

def transform(x, y):
    if np.random.uniform() < 0.5:
        x = x[:, ::-1]
        y = y[:, ::-1]
    return x, y

class ImageList(Dataset):
    def __init__(self, root, imgs, for_train=False):
        self.root = root
        self.imgs = imgs
        self.for_train = for_train

    def __getitem__(self, index):
        img_name = self.imgs[index]
        vid_index = int(img_name[0:2])
        frame_index = int(img_name[3:9])
        # print img_name
        # print vid_index
        # print frame_index
        # print index
        # exit(0)
        image_name = os.path.join(self.root, img_name)
        img = io.imread(image_name)
        img = cv2.resize(img, (320, 192), interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32')/255.0
        # img -= 0.5
        # plt.imshow(img)
        # plt.show()
        # print img.shape
        # exit(0)

        fixdatafile = ('./data/fixdata/fixdata'+str(vid_index)+'.mat')
        # print fixdatafile
        # exit()
        data = sio.loadmat(fixdatafile)
        # print data
        # exit(0)
        fix_x = data['fixdata'][frame_index-1][0][:,3]
        fix_y = data['fixdata'][frame_index-1][0][:,2]
        mask = np.zeros((720, 1280), dtype='float32')

        for i in range(len(fix_x)):
            mask[fix_x[i],fix_y[i]]=1

        mask = filters.gaussian_filter(mask,20)
        mask = np.array(mask, dtype='float32')
        mask = cv2.resize(mask, (320, 192), interpolation=cv2.INTER_CUBIC)
        mask = mask.astype('float32') / 255.0

        if mask.max()==0:
            print mask.max()
            print img_name
        else:
            mask = mask/mask.max()
        # mask= (mask-mask.min())/(mask.max()-mask.min())
        # mask -= 0.5
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(mask)
        # plt.show()
        # exit(0)


        if self.for_train:
            img, mask = transform(img, mask)
        # plt.imshow(mask)
        # plt.show()
        # exit(0)
        img = img.transpose(2, 0, 1)
        mask = mask[None, ...]
        img = np.ascontiguousarray(img)

        mask = np.ascontiguousarray(mask)

        return torch.from_numpy(img), torch.from_numpy(mask)

    def __len__(self):
        return len(self.imgs)

