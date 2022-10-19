import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..', '..', 'external'))

import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import pdb
from collections import OrderedDict
from .resnet_encoder import *
from inverse_warp import *


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                        padding=padding, dilation=dilation, bias=True),
            nn.ReLU(inplace=True)
            )

def deconv(in_planes, out_planes, kernel_size=4, stride=2, padding=1):
    return nn.ConvTranspose2d(in_planes, out_planes, kernel_size, stride, padding, bias=True)
    
class PoseDecoder(nn.Module):
    def __init__(self, num_ch_enc, num_input_features=1, num_frames_to_predict_for=1, stride=1):
        super(PoseDecoder, self).__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs = OrderedDict()
        self.convs[("squeeze")] = nn.Conv2d(self.num_ch_enc[-1], 256, 1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, 3, stride, 1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, 3, stride, 1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, 1)

        self.relu = nn.ReLU()

        self.net = nn.ModuleList(list(self.convs.values()))

    def forward(self, input_features):
        last_features = [f[-1] for f in input_features]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, 1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(3).mean(2)

        pose = 0.01 * out.view(-1, 6)

        return pose


class PoseNetPWC2(nn.Module):

    def __init__(self, num_layers = 18, pretrained = True, pose_num=2):
        super(PoseNetPWC2, self).__init__()
        self.w_r = torch.nn.Parameter(torch.tensor([-2.5]), requires_grad=True)
        self.w_t = torch.nn.Parameter(torch.tensor([0.0]), requires_grad=True)
        self.pose_num = pose_num
        self.nets = OrderedDict()
        for i in range(self.pose_num):
            self.nets[("encoder", i)] = ResnetEncoder(num_layers = num_layers, pretrained = pretrained, num_input_images=2)
            self.nets[("decoder", i)] = PoseDecoder(self.nets[("encoder", i)].num_ch_enc)

        self.net = nn.ModuleList(list(self.nets.values()))

    def init_weights(self):
        pass

        
    def merge(self, pose_c, pose_d):
    
        mat_c = euler2mat(pose_c[:,3:])
        mat_new = torch.matmul(mat_c, pose_d[:,:3].reshape(-1,3,1)).squeeze(-1)
        eu_angle = pose_c[:,3:] + pose_d[:,3:]
        t_mat = mat_new + pose_c[:,:3]
        
        return torch.cat((t_mat,eu_angle),1)
        
    def forward(self, img1, img2, depth1, intrinsics, T_aug = None):

        pose_list = []
        warped_img = img2

        for i in range(self.pose_num):

            if i > 0:
                last_pose = pose_list[-1]
                if T_aug == None:
                    warped_img = inverse_warp3(img2, depth1[0].detach(), last_pose, intrinsics)
                else:
                    warped_img = inverse_warp4(img2, depth1[0].detach(), last_pose, T_aug, intrinsics)

            x = torch.cat([img1, warped_img], 1)
            features = self.nets[("encoder", i)](x)
            pose = self.nets[("decoder", i)]([features])

            if i > 0:
                pose = self.merge(pose, last_pose)
            
            pose_list.append(pose)

        return pose_list[::-1], self.w_t, self.w_r
            