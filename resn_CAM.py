import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def upconv(in_planes, out_planes, kernel_size=4):
    return nn.Sequential(
        nn.ConvTranspose2d(in_planes, out_planes, kernel_size=kernel_size, stride=2, padding=1),
        nn.BatchNorm2d(out_planes),
        nn.LeakyReLU(inplace=True)
    )

def iconv24(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes, 24, kernel_size=3, stride=1, padding=1),
        nn.LeakyReLU(inplace=True)
    )

def pred(out_planes):
    return nn.Sequential(
        nn.Conv2d(24, out_planes, kernel_size=3, padding=1),
    )

def CAM_Convs(in_planes):
    return nn.Sequential(
        nn.Conv2d(in_planes+6, in_planes, kernel_size=3, padding=1),
        nn.LeakyReLU(inplace=True)
    )

def crop_like(input, ref):
    assert(input.size(2) >= ref.size(2) and input.size(3) >= ref.size(3))
    return input[:, :, :ref.size(2), :ref.size(3)]

Bridge_planes = [64, 256, 512, 1024, 2048]

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=16, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()

        norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResCAMDispNet(nn.Module):

    def __init__(self, block, layers, h_size, w_size,f_rate, Use_CAM=True, groups=1, width_per_group=64):
        super(ResCAMDispNet, self).__init__()
        
        norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.use=Use_CAM
        self.inplanes = 64
        self.dilation = 1

        self.groups = groups
        self.base_width = width_per_group
        
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)

        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0])
        
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=False)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=False)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=False)

        self.upconv4 = upconv(Bridge_planes[4], Bridge_planes[3])
        self.upconv3 = upconv(Bridge_planes[3]*2+5, Bridge_planes[2])
        self.upconv2 = upconv(Bridge_planes[2]*2+5, Bridge_planes[1])
        self.upconv1 = upconv(Bridge_planes[1]*2+5, Bridge_planes[0])
        self.upconv0 = upconv(Bridge_planes[0]*2+2, 32)

        self.iconv4 = iconv24(Bridge_planes[3])
        self.iconv3 = iconv24(Bridge_planes[2])
        self.iconv2 = iconv24(Bridge_planes[1])
        self.iconv1 = iconv24(Bridge_planes[0])
        self.iconv0 = iconv24(32)

        self.pred4=pred(5)
        self.pred3=pred(5)
        self.pred2=pred(5)
        self.pred1=pred(2)
        self.pred0=pred(2)

        if (self.use):
            self.CAM_Convs4=CAM_Convs(Bridge_planes[4])
            self.CAM_Convs3=CAM_Convs(Bridge_planes[3])
            self.CAM_Convs2=CAM_Convs(Bridge_planes[2])
            self.CAM_Convs1=CAM_Convs(Bridge_planes[1])
            self.CAM_Convs0=CAM_Convs(Bridge_planes[0])

        self.cc_u=self.cc_generator(h_size, w_size, -1, -1)
        self.nc_u=self.nc_generator(h_size, w_size)
        self.fov_u=self.fov_generator(self.cc_u, f_rate)
        
        self.cc=torch.zeros(16,2,h_size, w_size)
        self.nc=torch.zeros(16,2,h_size, w_size)
        self.fov=torch.zeros(16,2,h_size, w_size)

        for i in range(16):
            self.cc[i]=self.cc_u
            self.nc[i]=self.nc_u
            self.fov[i]=self.fov_u

        # initialazation
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # deleted a zero-initiation here.

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None

        previous_dilation = self.dilation

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def cc_generator(self, h, w, cx, cy):
        if (cx==-1):
            cx=w/2
            cy=h/2

        ccx=np.arange(0,w).reshape(w,1)-cx
        ccx=ccx.dot(np.ones([1,h]))
        ccx=ccx.T
        ccx=ccx.reshape(1,w,h)
        
        ccy=np.arange(0,h).reshape(h,1)-cy
        ccy=ccy.dot(np.ones([1,w]))
        ccy=ccy.reshape(1,h,w)

        ccx=torch.from_numpy(ccx)
        ccy=torch.from_numpy(ccy)

        return torch.cat((ccx, ccy), 0)

    def fov_generator(self, cc, f_rate):
        return np.arctan(cc*f_rate/72)

    def nc_generator(self, h,w):
        ccx=np.arange(0,w).reshape(w,1)-w/2
        ccx/=w
        ccx=ccx.dot(np.ones([1,h]))
        ccx=ccx.T
        ccx=ccx.reshape(1,w,h)
        
        ccy=np.arange(0,h).reshape(h,1)-h/2
        ccy/=h
        ccy=ccy.dot(np.ones([1,w]))
        ccy=ccy.reshape(1,h,w)
        ccx=torch.from_numpy(ccx)
        ccy=torch.from_numpy(ccy)
        
        return torch.cat((ccx, ccy), 0)

    def pre_intern_generator(self, I):
        _fov=F.upsample(self.fov, list(I.size()[-2:]), mode='bilinear', align_corners=True)
        _cc=F.upsample(self.cc, list(I.size()[-2:]), mode='bilinear', align_corners=True)
        _nc=F.upsample(self.nc, list(I.size()[-2:]), mode='bilinear', align_corners=True)

        return torch.cat((_fov, _cc, _nc), 1).cuda()

    def forward(self, image):
        print(image.size())
        # resnet-encoder
        image = self.conv1(image)
        image = self.bn1(image)
        conv1 = self.relu(image)
   
        pool = self.maxpool(conv1)

        res2c = self.layer1(pool)
   
        res3d = self.layer2(res2c)
        
        res4f = self.layer3(res3d)
        
        res5c = self.layer4(res4f)

        if (self.use):
            # skip connection
            conv1=torch.cat((conv1, self.pre_intern_generator(conv1)), 1)
            conv1=self.CAM_Convs0(conv1)

            res2c=torch.cat((res2c, self.pre_intern_generator(res2c)), 1)
            res2c=self.CAM_Convs1(res2c)

            res3d=torch.cat((res3d, self.pre_intern_generator(res3d)), 1)
            res3d=self.CAM_Convs2(res3d)

            res4f=torch.cat((res4f, self.pre_intern_generator(res4f)), 1)
            res4f=self.CAM_Convs3(res4f)

            res5c=torch.cat((res5c, self.pre_intern_generator(res5c)), 1)
            res5c=self.CAM_Convs4(res5c)


        # decoder
        upconv4=self.upconv4(res5c)
        inconv4=self.iconv4(upconv4)
        LR_1=self.pred4(inconv4)
        up_pre4 = torch.cat((LR_1, upconv4), 1)

        in3=torch.cat((res4f, up_pre4), 1)
        upconv3=self.upconv3(in3)
        inconv3=self.iconv3(upconv3)
        MR_1=self.pred3(inconv3)
        up_pre3 = torch.cat((MR_1, upconv3), 1)

        in2=torch.cat((res3d, up_pre3), 1)
        upconv2=self.upconv2(in2)
        inconv2=self.iconv2(upconv2)
        MR_2=self.pred2(inconv2)
        up_pre2 = torch.cat((MR_2, upconv2), 1)

        in1=torch.cat((res2c, up_pre2), 1)
        upconv1=self.upconv1(in1)
        inconv1=self.iconv1(upconv1)
        HR_1=self.pred1(inconv1)
        up_pre1 = torch.cat((HR_1, upconv1), 1)

        in0=torch.cat((conv1, up_pre1), 1)
        upconv0=self.upconv0(in0)
        inconv0=self.iconv0(upconv0)
        HR_2=self.pred0(inconv0)

        if self.training:
            return LR_1, MR_1, MR_2, HR_1, HR_2
        else:
            return HR_2


def model_generator(h_size, w_size,f_rate, UseCAMconvs=True):
    """Constructs a ResNet-50 model.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = ResCAMDispNet(Bottleneck, [2, 3, 5, 2], h_size, w_size,f_rate, Use_CAM=UseCAMconvs)
    model.load_state_dict(torch.load("./resnet50-imagenet.pth"), strict=False)
    return model
