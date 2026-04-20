import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2
from .deform_fusion import DeformTransLayer
from .point_to_image_projectionv2 import Point2ImageProjectionV2
from al3d_det.models.image_modules.ifn.basic_blocks import BasicBlock1D
from scipy.spatial import cKDTree
from al3d_det.utils.attention_utils import DWT_2D, IDWT_2D
from .image_depth import get_cam_feats

class VoxelWithPointProjectionV2KITTI(nn.Module):
    def __init__(self, 
                fuse_mode, 
                interpolate, 
                voxel_size, 
                pc_range, 
                image_list, 
                image_scale=1, 
                depth_thres=0, 
                mid_channels = 16,
                double_flip=False, 
                dropout_ratio=0,
                layer_channel=None,
                activate_out=True,
                fuse_out=False):
        """
        Initializes module to transform frustum features to voxel features via 3D transformation and sampling
        Args:
            voxel_size: [X, Y, Z], Voxel grid size
            pc_range: [x_min, y_min, z_min, x_max, y_max, z_max], Voxelization point cloud range (m)
        """
        super().__init__()
        self.voxel_size = voxel_size
        self.pc_range = pc_range
        self.point_projector = Point2ImageProjectionV2(voxel_size=voxel_size,
                                                     pc_range=pc_range,
                                                     depth_thres=depth_thres,
                                                     double_flip=double_flip)
        self.fuse_mode = fuse_mode
        self.image_interp = interpolate
        self.image_list = image_list
        self.image_scale = image_scale
        self.double_flip = double_flip
        self.mid_channels = mid_channels
        self.dropout_ratio = dropout_ratio
        self.activate_out = activate_out
        self.fuse_out = fuse_out
        
        ###############################################
        self.K_graph = 8
        self.step = 3
        self.n_heads = 4
        
        # self.dtransform = nn.Sequential(
        #     nn.Conv2d(16, 32, 9, stride=4, padding=4),
        #     nn.BatchNorm2d(32),
        #     nn.ReLU(True),
        #     nn.MaxPool2d(kernel_size=2, stride=2, padding=0),
        #     nn.Conv2d(32, 64, 5, stride=2, padding=2),
        #     nn.BatchNorm2d(64),
        #     nn.ReLU(True),
        # )
        # 只有深度
        # self.dtransform_Conv1 = nn.Sequential(
        #     nn.Conv2d(1, 8, 1),
        #     nn.BatchNorm2d(8),
        #     nn.ReLU(True),
        # )
        
#         # 最大深度+深度
#         self.dtransform_Conv1 = nn.Sequential(
#             nn.Conv2d(2, 8, 1),
#             nn.BatchNorm2d(8),
#             nn.ReLU(True),
#         )
#         self.dtransform_Conv2 = nn.Sequential(
#             nn.Conv2d(self.K_graph, 8, 1),
#             nn.BatchNorm2d(8),
#             nn.ReLU(True),
#         )
        
        
        
#         def dual_input_dtransform(self,input1, input2):
#             x1 = self.dtransform_Conv1(input1)
#             x2 = self.dtransform_Conv2(input2)
#             x = torch.cat([x1, x2], dim=1)
#             #x = torch.cat([x1.contiguous(), x2.contiguous()], dim=1)
#             # print("x1:\n",x1.shape)
#             # print("x2:\n",x2.shape)
#             # print("x cat dual depth:\n",x.shape)
#             x = self.dtransform(x)
#             return x
        
        self.get_cam_feats = get_cam_feats(self.K_graph, self.step)
        ###############################################
        
        self.fc = nn.Sequential(
            nn.Linear(2, self.mid_channels),
            nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
        )
        
        
        # 定义离散小波变换（DWT）和逆小波变换（IDWT）
        # self.dwt = DWT_2D(wave='haar')  # 使用Haar小波进行2D离散小波变换
        # self.idwt = IDWT_2D(wave='haar')  # 使用Haar小波进行2D逆小波变换
        
        self.dwt = DWT_2D(wave='db8')  # 使用Haar小波进行2D离散小波变换
        self.idwt = IDWT_2D(wave='db8')  # 使用Haar小波进行2D逆小波变换
        
        # self.dwt = DWT_2D(wave='sym8')  # 使用Haar小波进行2D离散小波变换
        # self.idwt = IDWT_2D(wave='sym8')  # 使用Haar小波进行2D逆小波变换
        
        
        # 定义特征维度的降维操作
        self.reduce = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1),  # 1x1卷积进行降维
            nn.BatchNorm2d(64),  # 批归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
        )
        # 定义特征的过滤操作
        self.filter = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, groups=1),  # 3x3卷积进行特征过滤
            nn.BatchNorm2d(256),  # 批归一化
            nn.ReLU(inplace=True),  # ReLU激活函数
        )
        ###############################################
        if self.fuse_mode == 'concat':
            self.fuse_blocks = nn.ModuleDict()
            for _layer in layer_channel.keys():
                block_cfg = {"in_channels": layer_channel[_layer]*2,
                             "out_channels": layer_channel[_layer],
                             "kernel_size": 1,
                             "stride": 1,
                             "bias": False}
                self.fuse_blocks[_layer] = BasicBlock1D(**block_cfg)
        elif self.fuse_mode == 'crossattention_deform':
            self.pts_key_proj = nn.Sequential(
                nn.Linear(self.mid_channels, self.mid_channels),
                nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
                # nn.ReLU()
            )
            self.pts_transform = nn.Sequential(
                nn.Linear(self.mid_channels, self.mid_channels),
                nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
                # nn.ReLU()
            )
            self.fuse_blocks = DeformTransLayer(d_model=self.mid_channels, \
                    n_levels=1, n_heads=self.n_heads, n_points=4)
            if self.fuse_out:
                self.fuse_conv = nn.Sequential(
                    nn.Linear(self.mid_channels*2, self.mid_channels),
                    # For pts the BN is initialized differently by default
                    # TODO: check whether this is necessary
                    nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
                    nn.ReLU())

    def fusion_back(self, voxel_feat, layer_name):
        """
        Fuses voxel features and image features
        Args:
            image_feat: (C, H, W), Encoded image features
            voxel_feat: (N, C), Encoded voxel features
            image_grid: (N, 2), Image coordinates in X,Y of image plane
        Returns:
            voxel_feat: (N, C), Fused voxel features
        """
        fuse_feat = torch.zeros(voxel_feat.shape).to(voxel_feat.device)
        concat_feat = torch.cat([fuse_feat.permute(1,0).contiguous(), voxel_feat.permute(1,0).contiguous()], dim=0)
        voxel_feat = self.fuse_blocks[layer_name](concat_feat.unsqueeze(0))[0]
        voxel_feat = voxel_feat.permute(1,0).contiguous()
        return voxel_feat


    def fusion(self, image_feat, voxel_feat, image_grid, layer_name=None):
        """
        Fuses voxel features and image features
        Args:
            image_feat: (C, H, W), Encoded image features
            voxel_feat: (N, C), Encoded voxel features
            image_grid: (N, 2), Image coordinates in X,Y of image plane
        Returns:
            voxel_feat: (N, C), Fused voxel features
        """
        image_grid = image_grid[:,[1,0]] # X,Y -> Y,X

        if self.fuse_mode == 'sum':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            voxel_feat = voxel_feat + fuse_feat.permute(1,0).contiguous()
        elif self.fuse_mode == 'mean':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            voxel_feat = (voxel_feat + fuse_feat.permute(1,0).contiguous()) / 2
        elif self.fuse_mode == 'concat':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]]
            concat_feat = torch.cat([fuse_feat, voxel_feat.permute(1,0).contiguous()], dim=0)
            voxel_feat = self.fuse_blocks[layer_name](concat_feat.unsqueeze(0))[0]
            voxel_feat = voxel_feat.permute(1,0).contiguous()
        elif self.fuse_mode == 'crossattention':
            fuse_feat = image_feat[:,image_grid[:,0],image_grid[:,1]].permute(1,0).contiguous()
            voxel_feat = self.fuse_blocks(fuse_feat.unsqueeze(0), voxel_feat.unsqueeze(0))
        else:
            raise NotImplementedError
        
        return voxel_feat
    def fusion_withdeform(self, img_pre_fuse, voxel_feat):
        if self.training and self.dropout_ratio > 0:
            img_pre_fuse = F.dropout(img_pre_fuse, self.dropout_ratio)
        pts_pre_fuse = self.pts_transform(voxel_feat)

        # fuse_out = img_pre_fuse + pts_pre_fuse
        fuse_out = torch.cat([pts_pre_fuse, img_pre_fuse], dim=-1)
        if self.activate_out:
            fuse_out = F.relu(fuse_out)
        if self.fuse_out:
            fuse_out = self.fuse_conv(fuse_out)

        return fuse_out
    
    
#     def get_cam_feats(self, img, depth, neighbors_depth):
#         '''
#         img:[1, 6, 256, 32, 88],
#         depth:[1, 6, 1, 256, 704]
#         '''
#         _, C, fH, fW = img.shape
#         # print("img:\n", img.shape)
#         # print("depth:\n", depth.shape)
#         # print("neighbors_depth:\n", neighbors_depth.shape)
#         depth = depth.unsqueeze(0)  # 在批次维度上扩展2D图像特征
#         #print("depth unsqueeze:\n", depth.shape)
#         neighbors_depth = neighbors_depth.unsqueeze(0)  # 在批次维度上扩展2D图像特征
#         #print("neighbors_depth unsqueeze:\n", neighbors_depth.shape)
        
#         # 最大深度
#         step=self.step
#         N, C, H, W = depth.shape
#         pad = (step - 1) // 2
#         depth_tmp = F.pad(depth, [pad, pad, pad, pad], mode='constant', value=0)
#         patches = depth_tmp.unfold(dimension=2, size=step, step=1)
#         patches = patches.unfold(dimension=3, size=step, step=1)
#         #print("patches:\n", patches.shape)
#         max_depth, _ = patches.reshape(N, C, H, W, -1).max(dim=-1)
#         #print("max_depth:\n", max_depth.shape)
#         depth = torch.cat([depth, max_depth], dim=1)
#         #print("max_depth+:\n", max_depth.shape)
#         #####################
        
        
#         depth = self.dual_input_dtransform(depth, neighbors_depth)
#         #print("dual_input_dtransform depth:\n", depth.shape)
#         img = img.unsqueeze(0)
#         #print("img unsqueeze:\n", img.shape)
        
#         _, _,depth_height, depth_width = depth.shape
#         if depth_height != fH or depth_width != fW:
#             #print("depth_height != fH or depth_width != fW:\n")
#             depth = nn.functional.interpolate(depth, (fH, fW), mode='bilinear')
        
#         img = depth+img
#         #img = torch.cat([depth, img], dim=1)
#         #print("img + dual depth:\n", img.shape)
        
#         return img
    
    def cKDTree_neighbor(self,masked_coords):
        masked_coords_numpy=masked_coords.cpu().numpy()
        #print("masked_coords_numpy:\n",masked_coords_numpy)
        # 使用 np.unique 去除重复元素，同时返回唯一坐标及其在原数组中的索引
        unique_coords, unique_indices = np.unique(masked_coords_numpy, axis=0, return_index=True)
        #print("unique_indices:\n",unique_indices)

        # 构建 KD 树
        kdtree = cKDTree(unique_coords)

        # 查询示例：获取最近的8个邻居 +1是本身
        distances, neighbor_indices = kdtree.query(masked_coords_numpy, k=self.K_graph+1)
        #print("distances:\n",distances)
        # import pdb; pdb.set_trace()
        # print("neighbor_indices.max():\n",neighbor_indices.max())
        # print("neighbor_indices.min():\n",neighbor_indices.min())
        # print("neighbor_indices:\n",neighbor_indices,neighbor_indices.shape)
        # 假设unique_indices的大小至少为19
        valid_indices = np.all(neighbor_indices < len(unique_indices), axis=1)
        if not np.all(valid_indices):
        #if all(np.any(neighbor_index >= len(unique_indices)) for neighbor_index in neighbor_indices):
            print("neighbor_indices中存在无效索引")
        else:
            neighbor_indices = unique_indices[neighbor_indices]
            # 继续执行您的其余代码

        # neighbor_indices = unique_indices[neighbor_indices]  # 使用映射还原到原始数组中的索引
        neighbor_indices = neighbor_indices[:, 1:]  # 去掉本身
        # print("neighbor_indices:\n",neighbor_indices,neighbor_indices.shape)
        # print("neighbor_indices.max():\n",neighbor_indices.max())
        neighbor_values = masked_coords_numpy[neighbor_indices]


        # 将NumPy数组转换为PyTorch张量
        neighbor_indices_torch = torch.from_numpy(neighbor_indices)
        
        neighbor_values_torch = torch.from_numpy(neighbor_values)
  
        return neighbor_indices_torch,neighbor_values_torch
    ###############################################
    
    def forward(self, batch_dict, point_features, point_coords, layer_name=None, img_conv_func=None):
        """
        Generates voxel features via 3D transformation and sampling
        Args:
            batch_dict:
                voxel_coords: (N, 4), Voxel coordinates with B,Z,Y,X
                lidar_to_cam: (B, 4, 4), LiDAR to camera frame transformation
                cam_to_img: (B, 3, 4), Camera projection matrix
                image_shape: (B, 2), Image shape [H, W]
            encoded_voxel: (N, C), Sparse Voxel featuress
        Returns:
            batch_dict:
                voxel_features: (B, C, Z, Y, X), Image voxel features
            voxel_features: (N, C), Sparse Image voxel features
        """
        #print("point_features:\n", point_features.shape)
        #print("point_coords:\n", point_coords.shape)
        voxel_fusefeatlist = []
        final_img_voxels = point_features.new_zeros((point_features.shape[0], self.mid_channels))
        pts_feats_org = self.pts_key_proj(point_features)

        calibs = batch_dict['calib']
        batch_size = batch_dict['batch_size']
        h, w = batch_dict['images'].shape[2:]
        image_feat = batch_dict['image_features'][layer_name+'_feat2d']
        #print("image_feat:\n", image_feat.shape)
        if self.image_interp:
            image_feat = nn.functional.interpolate(image_feat, (h, w), mode='bilinear')
        image_with_voxelfeatures = []
        filter_idx_list = []
        for b in range(batch_size):
            #print("bs:\n",b)
            image_feat_batch = image_feat[b]
            #print("image_feat_batch:\n",image_feat_batch, image_feat_batch.shape)
            #print("image_feat_batch:\n", image_feat_batch.shape)
            calib = calibs[b]
             
            
            index_mask = point_coords[:,0]==b
            point_grid_batch = point_coords[index_mask][:, 1:]
            # print("point_grid_batch:\n", point_grid_batch.shape)
            
            ###############################################
            dist = point_grid_batch[:, 2]
            # print("dist:\n",dist, dist.shape)
            ###############################################
            
            
            
            voxel_features_sparse = pts_feats_org[index_mask]

            if 'aug_matrix_inv' in batch_dict.keys():
                aug_matrix_inv = batch_dict['aug_matrix_inv'][b]
                for aug_type in ['translate', 'rescale', 'rotate', 'flip']:
                    if aug_type in aug_matrix_inv:
                        if aug_type == 'translate':
                            point_grid_batch = point_grid_batch + torch.Tensor(aug_matrix_inv[aug_type]).to(point_grid_batch.device)
                        else:
                            point_grid_batch = point_grid_batch @ torch.Tensor(aug_matrix_inv[aug_type]).to(point_grid_batch.device)
        
            voxels_2d, _ = calib.lidar_to_img(point_grid_batch[:, :].cpu().numpy())

            voxels_2d_int = torch.Tensor(voxels_2d).to(image_feat_batch.device).long()
            #print("voxels_2d:\n", voxels_2d.shape)

            filter_idx = (0<=voxels_2d_int[:, 1]) * (voxels_2d_int[:, 1] < h) * (0<=voxels_2d_int[:, 0]) * (voxels_2d_int[:, 0] < w)

            filter_idx_list.append(filter_idx)
            image_grid = voxels_2d_int[filter_idx]
            # print("image_grid:\n", image_grid.shape)
            
            ###############################
            # 相对位置嵌入
            # 计算质心坐标的平均值（中心）
            ###############################
            center = image_grid.float().mean(dim=0)
            relative_positions = image_grid - center
                          
            max_pos = relative_positions.abs().max(dim=0)[0]

            max_pos[max_pos == 0] = 1
            normalized_positions = relative_positions / max_pos
            # print("normalized_positions:\n", normalized_positions.shape)
            normalized_positions = self.fc(normalized_positions)
            # print("normalized_positions:\n", normalized_positions.shape)
            
            ###############################################
            # 初始化深度张量，邻居深度张量
            #print("init depth:\n")
            depth = torch.zeros(1, h*4, w*4).to(point_grid_batch[0].device)  
            #print("depth:\n",depth, depth.shape)
            neighbors_depth = torch.zeros(self.K_graph, h*4, w*4).to(point_grid_batch[0].device)
            #print("neighbors_depth:\n",neighbors_depth, neighbors_depth.shape)
            
            ##################
            #print("cKDTree_neighbor:\n")
            #print("image_grid.max():\n",image_grid.max())
            #print("image_grid.min():\n",image_grid.min())
            _, image_grid_neighbors = self.cKDTree_neighbor(image_grid)
            #print("end cKDTree_neighbor:\n")
            #print("image_grid_neighbors:\n",image_grid_neighbors, image_grid_neighbors.shape)
            masked_dist = dist[filter_idx]
            #print("masked_dist:\n",masked_dist, masked_dist.shape)
            
            
            # print("real depth:\n")
            # print("image_grid[:, 0]:\n",image_grid[:, 0],image_grid[:, 0].shape)
            # print("image_grid[:, 0]:\n",image_grid[:, 1],image_grid[:, 1].shape)
            depth[0, image_grid[:, 0], image_grid[:, 1]] = masked_dist
            # print("depth end:\n")
            # print("depth.shape:\n",depth.shape)

            neighbors_depth[:, image_grid_neighbors[:, :, 0].unsqueeze(-1),image_grid_neighbors[:, :, 1].unsqueeze(-1)] = masked_dist.view(-1, 1, 1)
            # print("neighbors_depth.shape:\n", neighbors_depth, neighbors_depth.shape)
            # 初始化深度张量，邻居深度张量

            # print("ronghe depth:\n")
            # print("image_feat_batch:\n",image_feat_batch, image_feat_batch.shape)
           
            #################
            # 加入深度
            #################
            image_feat_batch = image_feat_batch.unsqueeze(0)
            image_features_batch = self.get_cam_feats(image_feat_batch, depth, neighbors_depth)
            #print("image_features_batch:\n", image_features_batch.shape)
            
            # # 不加深度
            # image_features_batch = image_feat_batch.unsqueeze(0)
            # # 不加深度
            
            bs, channel_num, f_h, f_w = image_features_batch.shape
            
            ######################################
            # 小波变换
            ######################################
            # print("image_features_batch:\n", image_features_batch.shape)
            if f_w % 2 != 0:
                image_features_batch = nn.functional.interpolate(image_features_batch, (f_h, f_w+1), mode='bilinear', align_corners=False)
            if f_h % 2 != 0:
                image_features_batch = nn.functional.interpolate(image_features_batch, (f_h+1, f_w), mode='bilinear', align_corners=False)
            
            #print("image_features_batch:\n", image_features_batch.shape)
            x_dwt = self.dwt(self.reduce(image_features_batch))
            #print("self.dwt(self.reduce(x)):\n", x_dwt.shape)
            x_dwt = self.filter(x_dwt)
            #print("self.filter(x_dwt):\n", x_dwt.shape)
            
            
            x_idwt = self.idwt(x_dwt)
            # print("self.idwt(x_dwt):\n", x_idwt.shape)
            
            _, _, x_idwt_height, x_idwt_width = x_idwt.shape
            if x_idwt_height != f_h or x_idwt_width != f_w:
                #print("x_idwt_height != f_h or x_idwt_width != f_w:\n")
                x_idwt = nn.functional.interpolate(x_idwt, (f_h, f_w), mode='bilinear')
                # print("self.idwt(x_dwt):\n", x_idwt.shape)
            # x_idwt = x_idwt.squeeze(0)
            # x_idwt = self.get_cam_feats(x_idwt, depth, neighbors_depth)
            flatten_img_feat =  x_idwt.permute(0, 2, 3, 1).reshape(1, f_h*f_w, channel_num)
            # print("flatten_img_feat", flatten_img_feat.shape)
            ######################################
            
            # # 没有小波
            # flatten_img_feat = image_features_batch.permute(0, 2, 3, 1).reshape(1, f_h*f_w, channel_num)
            # # 没有小波
            
            #print("flatten_img_feat", flatten_img_feat.shape)
            if not self.image_interp:
                raw_shape = tuple(batch_dict['image_shape'][b].cpu().numpy())
                image_grid = image_grid.float()
                image_grid[:,0] *= (f_w/raw_shape[1])
                image_grid[:,1] *= (f_h/raw_shape[0])
                image_grid = image_grid.long()
            ref_points = image_grid.float()
            ref_points[:, 0] /= f_w
            ref_points[:, 1] /= f_h
            ref_points = ref_points.reshape(1, -1, 1, 2)
            N, Len_in, _ = flatten_img_feat.shape
            pts_feats = voxel_features_sparse[filter_idx].reshape(1, -1, self.mid_channels)
            # print("voxel_features_sparse:\n", voxel_features_sparse.shape)
            # print("pts_feats:\n", pts_feats.shape)
            level_spatial_shapes = pts_feats.new_tensor([(f_h, f_w)], dtype=torch.long)
            level_start_index = pts_feats.new_tensor([0], dtype=torch.long)
            
            voxel_features_sparse[filter_idx] = self.fuse_blocks(pts_feats, ref_points, flatten_img_feat, level_spatial_shapes, level_start_index,None,normalized_positions).squeeze(0)
            # voxel_features_sparse[filter_idx] = self.fuse_blocks(pts_feats, ref_points, flatten_img_feat, level_spatial_shapes, level_start_index).squeeze(0)
            image_with_voxelfeatures.append(voxel_features_sparse)

        image_with_voxelfeatures = torch.cat(image_with_voxelfeatures, dim= 0)
        #print("image_with_voxelfeatures:\n", image_with_voxelfeatures.shape)
        final_voxelimg_feat = self.fusion_withdeform(image_with_voxelfeatures, point_features)
        #print("final_voxelimg_feat:\n", final_voxelimg_feat.shape)

        return final_voxelimg_feat
        