from torch.utils.data import Dataset
import imageio as io
import numpy as np
import cv2
import os
import pandas as pd
import torch
import json
import matlab.engine
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
        # gt = self.imgs[index]
        # gt_lanes = gt['lanes']
        # y_samples = gt['h_samples']
        # raw_file = gt['raw_file']
        print index#,frame_index
        exit(0)
        # img_name = os.path.join(self.root, raw_file)
        frame_index = self.imgs[index]+1

        img_str = str(frame_index).zfill(6)+ '.jpg'
        img_name = os.path.join(self.root, img_str)
        # print img_name,frame_index
        # exit(0)
        img = io.imread(img_name)
        img = cv2.resize(img, (320, 192), interpolation=cv2.INTER_CUBIC)
        img = img.astype('float32')/255.0
        # img -= 0.5
        # plt.imshow(img)
        # plt.show()
        # exit(0)

        dataFile = './data/fixdata/fixdata9.mat'
        data = sio.loadmat(dataFile)
        fix_x = data['fixdata'][frame_index][0][:,3]
        fix_y = data['fixdata'][frame_index][0][:,2]
        mask = np.zeros((720, 1280), dtype='float32')
        # print len(fix_x)
        # exit(0)
        for i in range(len(fix_x)):
            mask[fix_x[i],fix_y[i]]=1
        # print mask
        # plt.imshow(mask)
        # plt.show()
        # exit(0)

        # eng = matlab.engine.start_matlab('-nodisplay -nojvm -nosplash -nodesktop')
        # exit(0)

        mask = filters.gaussian_filter(mask,40)
        mask = np.array(mask, dtype='float32')
        mask = cv2.resize(mask, (320, 192), interpolation=cv2.INTER_CUBIC)
        mask = mask.astype('float32') / 255.0
        mask= (mask-mask.min())/(mask.max()-mask.min())
        # print mask.max()
        # plt.imshow(mask)
        # plt.show()
        # exit(0)
        # gt_lanes_vis = [[(x, y) for (x, y) in
        #     zip(lane, y_samples) if x >= 0] for lane in gt_lanes]

        # mask = np.zeros((192, 320), dtype='float32')
        # for lane in gt_lanes_vis:
        #     if not lane: continue
        #     lane = np.array([lane], dtype='float32')
        #     lane *= [320, 192]
        #     lane /= [1280, 720]
        #     lane = lane.astype('int32')
        #     cv2.polylines(mask, lane, isClosed=False, color=1, thickness=1)

        if self.for_train:
            img, mask = transform(img, mask)

        img = img.transpose(2, 0, 1)
        mask = mask[None, ...]
        img = np.ascontiguousarray(img)

        mask = np.ascontiguousarray(mask)

        return torch.from_numpy(img), torch.from_numpy(mask)

    def __len__(self):
        return len(self.imgs)

#root = label_file = '/home/thuyen/Data/dataset/'
#label_file = root + 'label_data.json'
#json_gt = [json.loads(line) for line in open(label_file)]
#dset = ImageList(root, json_gt)
#dset[0]
