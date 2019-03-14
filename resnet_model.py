import torch.nn as nn
import math
import torch.utils.model_zoo as model_zoo
import torch

def conv3x3(in_planes, out_planes):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_planes, out_planes, 3, 1, 1),
            nn.BatchNorm2d(out_planes))


class Model(nn.Module):
    def __init__(self):
        n, m = 45, 3
        # n, m = 8, 3
        super(Model, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self.maxpool = nn.MaxPool2d(2, 2)


        self.convd1 = conv3x3(1*m, 1*n)
        self.convd1r = nn.Conv2d(1*m, 1*n, 1,padding=0,stride=1)
        self.convd2 = conv3x3(1*n, 2*n)
        self.convd2r = nn.Conv2d(1*n, 2*n, 1, padding=0, stride = 1)
        self.convd3 = conv3x3(2*n, 4*n)
        self.convd3r = nn.Conv2d(2* n, 4 * n, 1, padding=0, stride=1)
        self.convd4 = conv3x3(4*n, 4*n)
        self.convd4r = nn.Conv2d(4 * n, 4 * n, 1, padding=0, stride=1)

        self.convu3 = conv3x3(8*n, 4*n)
        self.convd3ur = nn.Conv2d(8 * n, 4 * n, 1, padding=0, stride=1)
        self.convu2 = conv3x3(6*n, 2*n)
        self.convd2ur = nn.Conv2d(6 * n, 2 * n, 1, padding=0, stride=1)
        self.convu1 = conv3x3(3*n, 1*n)
        self.convd1ur = nn.Conv2d(3 * n, 1 * n, 1, padding=0, stride=1)

        self.convu0 = nn.Conv2d(n, 1, 3, 1, 1)
        # self.conv1x1 = nn.Conv2d(1,1,3,1,1)

    def forward(self, x):
        x1 = x
        # residual = x
        residual = self.convd1r(x1)
        x1 = self.convd1(x1)
        #print(x1.size())
        #print('.....')
        #print(residual.size())

        x1 += residual
        x1 = self.relu(x1)

        x2 = self.maxpool(x1)
        # residual = x2
        residual = self.convd2r(x2)
        x2 = self.convd2(x2)
        #print(x2.size())
        #print('........')
        #print residual.size()
        #exit(0)
        x2 += residual
        x2 = self.relu(x2)

        x3 = self.maxpool(x2)
        # residual = x3
        residual = self.convd3r(x3)
        x3 = self.convd3(x3)
        # print(x3.size())
        x3 += residual
        x3 = self.relu(x3)

        x4 = self.maxpool(x3)
        residual = self.convd4r(x4)
        x4 = self.convd4(x4)
        # print(x4.size())
        x4 += residual
        x4 = self.relu(x4)

        y3 = self.upsample(x4)
        y3 = torch.cat([x3, y3], 1)
        residual = self.convd3ur(y3)
        y3 = self.convu3(y3)
        # print(y3.size())
        y3 += residual
        y3 = self.relu(y3)

        y2 = self.upsample(y3)
        y2 = torch.cat([x2, y2], 1)
        residual = self.convd2ur(y2)
        y2 = self.convu2(y2)
        # print(y2.size())
        y2 += residual
        y2 = self.relu(y2)

        y1 = self.upsample(y2)
        y1 = torch.cat([x1, y1], 1)
        residual = self.convd1ur(y1)
        y1 = self.convu1(y1)
        # print(y1.size())
        y1 += residual
        y1 = self.relu(y1)

        y1 = self.convu0(y1)
        y1 = self.sigmoid(y1)
        # print(y1.size())
        # exit(0)
        return y1



#import torch
#from torch.autograd import Variable
##from preresnet import resnet18, resnet34, resnet50, resnet101
##from model import resnet18, resnet34, resnet50, resnet101
#
#model = Model().cuda()
#
##images = Variable(torch.randn(2, 1, 48, 48, 48).cuda())
#images = Variable(torch.randn(2, 3, 320, 320).cuda())
#output = model(images)
#print(output.size())
