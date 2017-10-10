# import imageio
# from skimage import io
# import os
# import cPickle as pickle
# import json
#
# outputPath = '/media/tao/A3D8-E2F7/Video_frames/'
#
# filename = '/media/tao/A3D8-E2F7/traffic_videos/'
# # train_index = 1,3,5,6,8,10,11,13,15,16
# train_index = 1,3,6,8,10,11,13,16,21,22,23,26,35
# validat_index = 2,5,12,15,27,34,37
# test_index = 4,7,9,14,32
# for j,k in enumerate(train_index):
#     video_name = os.path.join(filename + "out%d.avi" % (k,))
#     vid = imageio.get_reader(video_name, 'ffmpeg')
#     for i, im in enumerate(vid):
#
#         img_name = os.path.join("%02d/%06d.jpg" % (k,i+1,))
#         # print img_name
#         # exit()
#         # io.imsave(img_name, im)
#         # with open(outputPath + 'valid.txt', 'ab') as f:
#         #     pickle.dump(img_name, f)
#         jsObj = json.dumps(img_name)
#         fileObject = open(outputPath + 'test.json', 'ab')
#         fileObject.write(jsObj)
#         fileObject.write("\n")
#         fileObject.close()
#         print('processing------NO.------')
#         print(k,i+1)
#         # exit()

import imageio
from skimage import io
import os
import cPickle as pickle
import json

file = open("/home/tao/Apps/DR(eye)VE/DREYEVE_DATA/missing_gt.txt")
lines = [line for line in file]
# a = "03\t000263\n"
# print lines[0],a
# print a==lines[0]
# exit(0)
vid_index = [line[0:2] for line in file]
frame_index = [line[3:-1] for line in file]

outputPath = '/media/tao/A3D8-E2F7/DREYEVE_DATA/'

# filename1 = '/media/tao/A3D8-E2F7/traffic_videos/'
filename2 = '/media/tao/A3D8-E2F7/DREYEVE_DATA/'
# train_index = 1,3,5,6,8,10,11,13,15,16
train_index = 1,3,5,7,9
validat_index = 4,8
test_index = 2,6,10
for j,k in enumerate(validat_index):
    # # if k>16:
    # #     filename = filename2
    # video_name = os.path.join(filename2+ '/'+str(k)+'/'+ "video_garmin.avi")
    #     # print video_name
    #     # exit(0)
    # # else:
    # #     video_name = os.path.join(filename1 + "out%d.avi" % (k,))
    #
    # vid = imageio.get_reader(video_name, 'ffmpeg')
    # for i, im in enumerate(vid):
    #
    #     img_name = os.path.join("%02d/%06d.jpg" % (k,i+1,))
    #     # print img_name
    #     # print outputPath+img_name
    #     # exit()
    #     io.imsave(outputPath+img_name, im)
    #
    #     # jsObj = json.dumps(img_name)
    #     # fileObject = open(outputPath + 'test_new.json', 'ab')
    #     # fileObject.write(jsObj)
    #     # fileObject.write("\n")
    #     # fileObject.close()
    img_names = os.listdir(filename2+'%02d/video_garmin/'%(k,))
    fix_names = os.listdir(filename2+'%02d/video_saliency/'%(k,))
    for i,img_name in enumerate(img_names):
        # print '%06d'%(int(img_name[0:-4]))
        # print vid_index
        # exit(0)
        kj = (str('%02d'%(k))+'\t%06d\n'%(int(img_name[0:-4])))
        # print kj,lines[0]
        # exit(0)
        if kj not in lines:

            img_path = os.path.join('%02d/video_garmin/'%(k,),img_name)
            # fix_path = os.path.join('%02d/video_saliency/'%(k,),img_name)
            jsObj = json.dumps(img_path)
            fileObject = open(outputPath + 'valid_dre_new.json', 'ab')
            fileObject.write(jsObj)
            fileObject.write("\n")
            fileObject.close()

            # print img_path
            # exit(0)
            print('processing------NO.------')
            print(k,i+1)

# file = open("/home/tao/Apps/DR(eye)VE/DREYEVE_DATA/missing_gt.txt")
# vid_index = [line[0:2] for line in file]
# frame_index = [line[3:-1] for line in file]
# print vid_index
# exit(0)
# for line in file:
#     vid_index = line[0:2]
#     frame_index = line[3:-1]
#     print line[0:2],line[3:-1]
#     exit(0)
# gt_lanes_vis = [[(x, y) for (x, y) in
#     zip(lane, y_samples) if x >= 0] for lane in gt_lanes]

# print lines
