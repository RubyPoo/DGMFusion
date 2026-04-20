import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class get_cam_feats(nn.Module):
    def __init__(self, K_graph, step):
        super(get_cam_feats, self).__init__()
        
        self.K_graph = K_graph
        self.step = step
        
        
        # 只有深度
        # self.dtransform_Conv1 = nn.Sequential(
        #     nn.Conv2d(1, 8, 1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(True),
        # )
        
        # 最大深度+深度
        self.dtransform_Conv1 = nn.Sequential(
            nn.Conv2d(2, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
        )
        
        # 邻居深度
        self.dtransform_Conv2 = nn.Sequential(
            nn.Conv2d(self.K_graph, 8, 1),
            nn.BatchNorm2d(8),
            nn.ReLU(True),
        )
        
        # 双深度
        self.dtransform = nn.Sequential(
            nn.Conv2d(16, 32, 9, stride=4, padding=4),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
            nn.Conv2d(32, 64, 5, stride=2, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
        )
    
    def dual_input_dtransform(self,input1, input2):
        x1 = self.dtransform_Conv1(input1)
        x2 = self.dtransform_Conv2(input2)
        x = torch.cat([x1, x2], dim=1)
        #x = torch.cat([x1.contiguous(), x2.contiguous()], dim=1)
        # print("x1:\n",x1.shape)
        # print("x2:\n",x2.shape)
        # print("x cat dual depth:\n",x.shape)
        x = self.dtransform(x)
        return x

    def forward(self, img, depth, neighbors_depth):
        '''
        img:[1, 6, 256, 32, 88],
        depth:[1, 6, 1, 256, 704]
        '''
        N, C, fH, fW = img.shape
        # print("img:\n", img.shape)
        # print("depth:\n", depth.shape)
        # print("neighbors_depth:\n", neighbors_depth.shape)
        depth = depth.unsqueeze(0)  # 在批次维度上扩展2D图像特征
        #print("depth unsqueeze:\n", depth.shape)
        neighbors_depth = neighbors_depth.unsqueeze(0)  # 在批次维度上扩展2D图像特征
        #print("neighbors_depth unsqueeze:\n", neighbors_depth.shape)
        
        # 最大深度
        step = self.step
        N, C, H, W = depth.shape
        pad = (step - 1) // 2
        depth_tmp = F.pad(depth, [pad, pad, pad, pad], mode='constant', value=0)
        patches = depth_tmp.unfold(dimension=2, size=step, step=1)
        patches = patches.unfold(dimension=3, size=step, step=1)
        #print("patches:\n", patches.shape)
        max_depth, _ = patches.reshape(N, C, H, W, -1).max(dim=-1)
        #print("max_depth:\n", max_depth.shape)
        depth = torch.cat([depth, max_depth], dim=1)
        #print("max_depth+:\n", max_depth.shape)
        #####################
        
        
        depth = self.dual_input_dtransform(depth, neighbors_depth)
        #print("dual_input_dtransform depth:\n", depth.shape)
        
        #print("img unsqueeze:\n", img.shape)
        
        _, _,depth_height, depth_width = depth.shape
        if depth_height != fH or depth_width != fW:
            #print("depth_height != fH or depth_width != fW:\n")
            depth = nn.functional.interpolate(depth, (fH, fW), mode='bilinear')
        
        img = depth + img
        #img = torch.cat([depth, img], dim=1)
        #print("img + dual depth:\n", img.shape)
        
        return img