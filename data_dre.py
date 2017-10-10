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

        vid_index = img_name[0:2]
        frame_index = img_name[-9:-4]
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

        mask_name = os.path.join(self.root,vid_index,'video_saliency','%s.jpg'%(frame_index,) )
        # print mask_name
        # exit(0)
        mask = io.imread(mask_name)

        mask = np.array(mask, dtype='float32')
        mask = cv2.resize(mask, (320, 192), interpolation=cv2.INTER_CUBIC)
        mask = mask.astype('float32') / 255.0
        mask = mask[:,:,0]
        # plt.imshow(mask)
        # plt.show()
        # exit(0)
        if mask.max()==0:
            print mask.max()
            print img_name
        else:
            mask = mask/mask.max()
        # mask= (mask-mask.min())/(mask.max()-mask.min())
        # mask -= 0.5
        # plt.imshow(img)
        # plt.show()
        # plt.imshow(mask[:,:,0])
        # plt.show()
        # # print mask[:,:,0]
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

