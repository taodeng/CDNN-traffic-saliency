import scipy.io as scio
import matlab.engine

# print label
# print type(label)
# plt.imshow(label)
# plt.show()

# dataFile = './data/fixdata/fixdata'+str(video_index)+'.mat'
# data = scio.loadmat(dataFile)
# fix_x = data['fixdata'][frame_index][0][:,2]
# fix_y = data['fixdata'][frame_index][0][:,3]
from torch.utils.data import Dataset
import imageio as io
import numpy as np
import cv2
import os
import torch
import imageio

#shape = (180, 320)

def transform(x, y):
    if np.random.uniform() < 0.5:
        x = x[:, ::-1]
        y = y[:, ::-1]
    return x, y
# def get_frame(vid):
#     for i, im in enumerate(vid):
#         frame = vid.get_data(i)
#         return frame

class ImageList(Dataset):
    def __init__(self, root, vids, for_train=False):
        self.root = root
        self.vids = vids
        self.for_train = for_train

    def __getitem__(self, index):

        ## get image
        vid = self.vids[index]
        vid = imageio.get_reader(self.root+'out'+str(index)+'.avi','ffmpeg')
        for frame_index, im in enumerate(vid):
            img = vid.get_data(frame_index)
            img = cv2.resize(img, (320, 192), interpolation=cv2.INTER_CUBIC)
            img = img.astype('float32')/255.0
            img -= 0.5

            eng = matlab.engine.start_matlab('-nodisplay -nojvm -nosplash -nodesktop')
            mask = eng.getTrainLabel(frame_index + 1, index + 1)
            mask = cv2.resize(mask, (320, 192), interpolation=cv2.INTER_CUBIC)
            mask = mask.astype('float32')/255.0
            # mask -= 0.5

            if self.for_train:
                img, mask = transform(img, mask)
            img = img.transpose(2, 0, 1)
            mask = mask[None, ...]
            img = np.ascontiguousarray(img)
            mask = np.ascontiguousarray(mask)

            return torch.from_numpy(img), torch.from_numpy(mask)



        # gt = self.imgs[index]
        # gt_lanes = gt['lanes']
        # y_samples = gt['h_samples']
        # raw_file = gt['raw_file']
        #
        # img_name = os.path.join(self.root, raw_file)
        # img = io.imread(img_name)
        # img = cv2.resize(img, (320, 192), interpolation=cv2.INTER_CUBIC)
        # img = img.astype('float32')/255.0
        # img -= 0.5


        # ##get label
        # video_index = 0
        # frame_index = 100
        # eng = matlab.engine.start_matlab('-nodisplay -nojvm -nosplash -nodesktop')
        # label = eng.getTrainLabel(frame_index + 1, video_index + 1)
        #
        # gt_lanes_vis = [[(x, y) for (x, y) in
        #     zip(lane, y_samples) if x >= 0] for lane in gt_lanes]
        #
        # mask = np.zeros((192, 320), dtype='float32')
        # for lane in gt_lanes_vis:
        #     if not lane: continue
        #     lane = np.array([lane], dtype='float32')
        #     lane *= [320, 192]
        #     lane /= [1280, 720]
        #     lane = lane.astype('int32')
        #     cv2.polylines(mask, lane, isClosed=False, color=1, thickness=1)
        #
        # if self.for_train:
        #     img, mask = transform(img, mask)
        #
        # img = img.transpose(2, 0, 1)
        # mask = mask[None, ...]
        # img = np.ascontiguousarray(img)
        # mask = np.ascontiguousarray(mask)
        #
        # return torch.from_numpy(img), torch.from_numpy(mask)

    def __len__(self):
        return len(self.vids)

#root = label_file = '/home/thuyen/Data/dataset/'
#label_file = root + 'label_data.json'
#json_gt = [json.loads(line) for line in open(label_file)]
#dset = ImageList(root, json_gt)
#dset[0]
