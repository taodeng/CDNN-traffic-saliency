import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from skimage import io
import cPickle as pickle
from PIL import Image
import scipy.io
import json
import matplotlib.pyplot as plt

exp_dir = './dre/test/'
data_dir = '/media/tao/A3D8-E2F7/Video_frames/'
save_dir = '/media/tao/A3D8-E2F7/DREYEVE_DATA/'
npf = np.load(exp_dir + 'test_dre.npz')
# json_gt = pickle.load(open(exp_dir + 'test_dre.pkl'))
# exit(0)
preds = npf['p']
targets = npf['t']
# preds = np.load(exp_dir+'p.npy')
# targets = np.load(exp_dir+'t.npy')
# fourcc = cv2.VideoWriter_fourcc(*'XVID')
# out = cv2.VideoWriter('test_14.avi',fourcc, 25.0, (1280,720))
test_imgs = [json.loads(line) for line in open(save_dir + 'test_dre_new.json')]
# print test_imgs[19140],len(test_imgs)
# exit(0)
for i in range(len(test_imgs)):
    pred = preds[i]
    target = targets[i]
    image_name = test_imgs[i]
    vid_index = image_name[0:2]
    frame_index = image_name[3:-4]
    # print vid_index,frame_index
    # exit(0)
# overlay images with label

    img = io.imread(save_dir + image_name)
    # print data_dir + '/02/video_garmin/%05d.jpg'%(frame_index+1,)
    # exit(0)
    # print img.shape
    # exit(0)

    img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_CUBIC)
    target = cv2.resize(target, (1280, 720), interpolation=cv2.INTER_CUBIC)
    pred = cv2.resize(pred, (1280, 720), interpolation=cv2.INTER_CUBIC)
    # img = cv2.resize(img, (1280, 720), interpolation=cv2.INTER_CUBIC)
    # plt.imshow(img, cmap='gray')
    # plt.imshow(pred, cmap='jet', alpha=0.5)
    # plt.title('image overlaid with pred')
    # plt.figure(num='demo', figsize=(860,540))
    # fig = plt.figure()
    plt.subplot(2,1,1)
    plt.title('image overlaid with target')
    plt.imshow(img)
    plt.imshow(img, cmap='gray')
    plt.imshow(target, cmap='jet', alpha=0.5)
    plt.axis('off')

    plt.subplot(2,1,2)
    plt.title('image overlaid with pred')
    plt.imshow(img)
    plt.imshow(img, cmap='gray')
    plt.imshow(pred, cmap='jet', alpha=0.5)
    plt.axis('off')
    # plt.show()
    # exit(0)
    # frame = plt.gcf()
    plt.savefig('./dre/plot/'+vid_index+'/'+frame_index+'.jpg')
    plt.close()
    # frame.canvas.draw()  # draw the canvas, cache the renderer
    # image = np.fromstring(canvas.tostring_rgb(), dtype='uint8')
    #
    # print image
    # exit(0)

    # imgg = io.imread('test14.jpg')

    # out.write(imgg)
    print('processing------NO.------')
    print((i + 1),frame_index)
    # plt.show()
    # exit(0)
# out.release()
# cv2.destroyAllWindows()