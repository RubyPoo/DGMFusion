import torch
import torch.nn as nn
import torch.nn.functional as F

from .deform_fusion import DeformTransLayer
from .point_to_image_projectionv2 import Point2ImageProjectionV2
from al3d_det.models.image_modules.ifn.basic_blocks import BasicBlock1D

#####################
# 新增
from .image_depth import get_cam_feats
from scipy.spatial import cKDTree
import numpy as np

from .wavelet import WaveletTrans

class VoxelWithPointProjectionV2(nn.Module):
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
        
  
        self.fc = nn.Sequential(
            nn.Linear(2, self.mid_channels),
            nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
        )
        
 
        self.K_graph = 8
        self.step = 3
        self.get_cam_feats = get_cam_feats(self.K_graph, self.step)
        
  
        self.wavetrans = WaveletTrans()
    
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
                    n_levels=1, n_heads=4, n_points=4)
            if self.fuse_out:
                self.fuse_conv = nn.Sequential(
                    nn.Linear(self.mid_channels*2, self.mid_channels),
                    # For pts the BN is initialized differently by default
                    # TODO: check whether this is necessary
                    nn.BatchNorm1d(self.mid_channels, eps=1e-3, momentum=0.01),
                    nn.ReLU())

    def cKDTree_neighbor(self, masked_coords):
        masked_coords_numpy=masked_coords.cpu().numpy()
        
        unique_coords, unique_indices = np.unique(masked_coords_numpy, axis=0, return_index=True)
        #print("unique_indices:\n",unique_indices)

        
        kdtree = cKDTree(unique_coords)

       
        distances, neighbor_indices = kdtree.query(masked_coords_numpy, k=self.K_graph+1)
        #print("distances:\n",distances)
        # import pdb; pdb.set_trace()
        # print("neighbor_indices.max():\n",neighbor_indices.max())
        # print("neighbor_indices.min():\n",neighbor_indices.min())
        # print("neighbor_indices:\n",neighbor_indices,neighbor_indices.shape)
        
        valid_indices = np.all(neighbor_indices < len(unique_indices), axis=1)
        if not np.all(valid_indices):
        #if all(np.any(neighbor_index >= len(unique_indices)) for neighbor_index in neighbor_indices):
            print("neighbor_indices中存在无效索引")
        else:
            neighbor_indices = unique_indices[neighbor_indices]
           

        # neighbor_indices = unique_indices[neighbor_indices]  
        neighbor_indices = neighbor_indices[:, 1:]  
        # print("neighbor_indices:\n",neighbor_indices,neighbor_indices.shape)
        # print("neighbor_indices.max():\n",neighbor_indices.max())
        neighbor_values = masked_coords_numpy[neighbor_indices]


       
        neighbor_indices_torch = torch.from_numpy(neighbor_indices)
        
        neighbor_values_torch = torch.from_numpy(neighbor_values)
  
        return neighbor_indices_torch,neighbor_values_torch
    

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

    def forward(self, batch_dict, point_features, point_coords, layer_name=None, img_conv_func=None):
        voxel_fusefeatlist = []
        final_img_voxels = point_features.new_zeros((point_features.shape[0], self.mid_channels))
        pts_feats_org = self.pts_key_proj(point_features)

        for cam_key in self.image_list:
            # Generate sampling grid for frustum volume
            projection_dict = self.point_projector(voxel_coords=point_coords.float(),
                                                   image_scale=self.image_scale,
                                                   batch_dict=batch_dict, 
                                                   cam_key=cam_key)
            batch_size = len(batch_dict['image_shape'][cam_key])
            if not self.training and self.double_flip:
                tta_ops = batch_dict["tta_ops"]
                tta_num = len(tta_ops)
                batch_size = batch_size * tta_num

            voxel_featlist = []
            bs = 1 

            for _idx in range(batch_size):
                _idx_key = _idx//tta_num if self.double_flip else _idx
                image_feat = batch_dict['image_features'][layer_name+'_feat2d'][cam_key][_idx_key]
                # print("image_feat:\n", image_feat.shape)

                if img_conv_func:
                    image_feat = img_conv_func(image_feat.unsqueeze(0))[0]

                raw_shape = tuple(batch_dict['image_shape'][cam_key][_idx_key].cpu().numpy())
                feat_shape = image_feat.shape[-2:]
                # print("feat_shape:\n", feat_shape)

                if self.image_interp:
                    image_feat = F.interpolate(image_feat.unsqueeze(0), size=raw_shape[:2], mode='bilinear', align_corners=False)[0]

                index_mask = point_coords[:,0]==_idx
                voxel_feat = pts_feats_org[index_mask]
                image_grid = projection_dict['image_grid'][_idx]
                voxel_grid = projection_dict['batch_voxel'][_idx]
                point_mask = projection_dict['point_mask'][_idx]
                image_depth = projection_dict['image_depths'][_idx]
                img_voxels = final_img_voxels[index_mask]
                voxel_mask = point_mask[:len(voxel_feat)]


                image_grid_for_voxel = image_grid[:len(voxel_feat)]  
                
                
                center = image_grid_for_voxel.float().mean(dim=0)
                relative_positions = image_grid_for_voxel - center
                max_pos = relative_positions.abs().max(dim=0)[0]
                max_pos[max_pos == 0] = 1
                normalized_positions_all = relative_positions / max_pos
                normalized_positions_all = self.fc(normalized_positions_all)
                
               
                normalized_positions = normalized_positions_all[voxel_mask]
                
               
                h, w = image_feat.shape[-2:]
                # print("h:\n", h)
                # print("w:\n", w)
                depth = torch.zeros(1, h * 2, w * 2).to(image_grid.device)
                neighbors_depth = torch.zeros(self.K_graph, h * 2, w * 2).to(image_grid.device)
                # print("neighbors_depth:\n", neighbors_depth.shape)

               
                _, image_grid_neighbors = self.cKDTree_neighbor(image_grid)
                # print("image_grid_neighbors:\n", image_grid_neighbors.shape)

                dist = point_coords[:, 2]
                # print("dist:\n", dist, dist.shape)
                masked_dist = dist[index_mask]
                # print("masked_dist:\n", masked_dist, masked_dist.shape)

              
                raw_h, raw_w = raw_shape
                target_h, target_w = h * 2, w * 2

                
                scale_h = target_h / raw_h
                scale_w = target_w / raw_w

                # print(f"DEBUG: raw_shape = {raw_h}x{raw_w}, target_shape = {target_h}x{target_w}")
                # print(f"DEBUG: scale_h = {scale_h:.4f}, scale_w = {scale_w:.4f}")

                
                image_grid_scaled = image_grid.clone().float()
                image_grid_scaled[:, 0] = image_grid_scaled[:, 0] * scale_h
                image_grid_scaled[:, 1] = image_grid_scaled[:, 1] * scale_w
                image_grid_scaled = torch.round(image_grid_scaled).long()
                image_grid_scaled[:, 0] = torch.clamp(image_grid_scaled[:, 0], 0, target_h - 1)
                image_grid_scaled[:, 1] = torch.clamp(image_grid_scaled[:, 1], 0, target_w - 1)

               
                depth[0, image_grid_scaled[:, 0], image_grid_scaled[:, 1]] = masked_dist

                
                image_grid_neighbors_scaled = image_grid_neighbors.clone().float()
                image_grid_neighbors_scaled[:, :, 0] = image_grid_neighbors_scaled[:, :, 0] * scale_h
                image_grid_neighbors_scaled[:, :, 1] = image_grid_neighbors_scaled[:, :, 1] * scale_w
                image_grid_neighbors_scaled = torch.round(image_grid_neighbors_scaled).long()
                image_grid_neighbors_scaled[:, :, 0] = torch.clamp(image_grid_neighbors_scaled[:, :, 0], 0, target_h - 1)
                image_grid_neighbors_scaled[:, :, 1] = torch.clamp(image_grid_neighbors_scaled[:, :, 1], 0, target_w - 1)

                
                
                batch_size_neighbors = image_grid_neighbors_scaled.shape[0]

                
                h_idx = image_grid_neighbors_scaled[..., 0].long()  
                w_idx = image_grid_neighbors_scaled[..., 1].long()  

             
                mask = (h_idx >= 0) & (h_idx < target_h) & (w_idx >= 0) & (w_idx < target_w)

                
                k_indices = torch.arange(self.K_graph, device=h_idx.device)  
                k_indices = k_indices.unsqueeze(0).expand(batch_size_neighbors, -1) 

                linear_indices = k_indices * (target_h * target_w) + h_idx * target_w + w_idx  #

              
                valid_mask = mask.flatten()
                valid_indices = linear_indices.flatten()[valid_mask]  # [num_valid]

                
                if masked_dist.dim() == 1:
                    masked_dist = masked_dist.unsqueeze(1)  # [batch_size_neighbors, 1]

                source_values = masked_dist.expand(-1, self.K_graph).flatten()[valid_mask]  # [num_valid]
                

              
                neighbors_depth_flat = neighbors_depth.view(-1)
                device = neighbors_depth_flat.device
                valid_indices = valid_indices.to(device)
                source_values = source_values.to(device)
                neighbors_depth_flat.scatter_(0, valid_indices, source_values)

               
                invalid_count = (~mask).sum().item()
                if invalid_count > 0:
                    print(f"WARNING: Found {invalid_count} invalid indices")

               
                image_feat_batch = image_feat.unsqueeze(0)

                

                image_features_batch = self.get_cam_feats(image_feat_batch, depth, neighbors_depth)
                
               
                flatten_img_feat = self.wavetrans(image_features_batch)

                
                if self.training and 'overlap_mask' in batch_dict.keys():
                    overlap_mask = batch_dict['overlap_mask'][_idx]
                    is_overlap = overlap_mask[image_grid[:,1], image_grid[:,0]].bool()
                    if 'depth_mask' in batch_dict.keys():
                        depth_mask = batch_dict['depth_mask'][_idx]
                        depth_range = depth_mask[image_grid[:,1], image_grid[:,0]]
                        is_inrange = (image_depth > depth_range[:,0]) & (image_depth < depth_range[:,1])
                        is_overlap = is_overlap & (~is_inrange)

                    image_grid = image_grid[~is_overlap]
                    voxel_grid = voxel_grid[~is_overlap]
                    point_mask = point_mask[~is_overlap]
                    voxel_mask = voxel_mask & (~is_overlap[:len(voxel_feat)])
                    
                    if normalized_positions_all.shape[0] == len(voxel_feat):
                        normalized_positions_all = normalized_positions_all[~is_overlap[:len(voxel_feat)]]
                        normalized_positions = normalized_positions_all[voxel_mask]

                if not self.image_interp:
                    image_grid = image_grid.float()
                    image_grid[:,0] *= (feat_shape[1]/raw_shape[1])
                    image_grid[:,1] *= (feat_shape[0]/raw_shape[0])
                    image_grid = image_grid.long()

                if image_grid[point_mask].shape[0]>1:
                   
                    ref_points = image_grid[point_mask].float()
                    ref_points[:, 0] /= feat_shape[1]
                    ref_points[:, 1] /= feat_shape[0]
                    ref_points = ref_points.reshape(bs, -1, 1, 2)
                    N, Len_in, _ = flatten_img_feat.shape
                    pts_feats = voxel_feat[voxel_mask].reshape(bs, -1, self.mid_channels)
                    
                    normalized_positions_reshaped = normalized_positions.unsqueeze(0)  # [1, N, 64]
                    
                    level_spatial_shapes = pts_feats.new_tensor([(h, w)], dtype=torch.long)
                    level_start_index = pts_feats.new_tensor([0], dtype=torch.long)
                    
                   
                    img_voxels[voxel_mask] = self.fuse_blocks(
                        pts_feats, 
                        ref_points, 
                        flatten_img_feat, 
                        level_spatial_shapes, 
                        level_start_index,
                        None,
                        normalized_positions_reshaped
                    ).squeeze(0)
                    

                final_img_voxels[index_mask] = img_voxels

        final_voxelimg_feat = self.fusion_withdeform(final_img_voxels, point_features)

        return final_voxelimg_feat