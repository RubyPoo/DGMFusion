import math

import torch
import torch.nn as nn
from .block import DANetHead

import time
import pywt
#import math
import numpy as np
# import torch
# import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.autograd import Variable, gradcheck


class TransformerEncoder(nn.Module):
    """
    Based on: https://pytorch.org/tutorials/beginner/transformer_tutorial.html (https://knowyourmeme.com/memes/i-made-this)
    """
    def __init__(self, attention_cfg, pos_encoder=None):
        super().__init__()
        self.attention_cfg = attention_cfg
        self.pos_encoder = pos_encoder
        encoder_layers = nn.TransformerEncoderLayer(attention_cfg.NUM_FEATURES, attention_cfg.NUM_HEADS, attention_cfg.NUM_HIDDEN_FEATURES, attention_cfg.DROPOUT)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, attention_cfg.NUM_LAYERS)

    def forward(self, point_features, positional_input, src_key_padding_mask=None):
        """
        Args:
            point_features: (b, xyz, f)
            positional_input: (b, xyz, 3)
            src_key_padding_mask: (b, xyz)
        Returns:
            point_features: (b, xyz, f)
        """
        # Clone point features to prevent mutating input arguments
        attended_features = torch.clone(point_features)
        if src_key_padding_mask is not None:
            # RoIs sometimes have all zero inputs. This results in a 0/0 division because of masking. Thus, we
            # remove the empty rois to prevent this issue(https://github.com/pytorch/pytorch/issues/24816#issuecomment-524415617)
            empty_rois_mask = src_key_padding_mask.all(-1)
            attended_features_filtered = attended_features[~empty_rois_mask]

            if self.pos_encoder is not None:
                src_key_padding_mask_filtered = src_key_padding_mask[~empty_rois_mask]
                attended_features_filtered[~src_key_padding_mask_filtered] = self.pos_encoder(attended_features_filtered,
                                                                                              positional_input[~empty_rois_mask] if positional_input is not None else None)[~src_key_padding_mask_filtered]

            # (b, xyz, f) -> (xyz, b, f)
            attended_features_filtered = attended_features_filtered.permute(1, 0, 2)
            # (xyz, b, f) -> (b, xyz, f)
            attended_features[~empty_rois_mask] = self.transformer_encoder(attended_features_filtered,
                                                                           src_key_padding_mask=src_key_padding_mask[~empty_rois_mask]).permute(1, 0, 2).contiguous()
        else:
            if self.pos_encoder is not None:
                attended_features = self.pos_encoder(attended_features, positional_input)

            # (b, xyz, f) -> (xyz, b, f)
            attended_features = attended_features.permute(1, 0, 2)
            # (xyz, b, f) -> (b, xyz, f)
            attended_features = self.transformer_encoder(attended_features).permute(1, 0, 2).contiguous()

        return attended_features


class FrequencyPositionalEncoding3d(nn.Module):
    def __init__(self, d_model, max_spatial_shape, dropout=0.1):
        """
        Sine + Cosine positional encoding based on Attention is all you need (https://arxiv.org/abs/1706.03762) in 3D. Using the same concept as DETR,
        the sinusoidal encoding is independent across each spatial dimension.
        Args:
            d_model: Dimension of the input features. Must be divisible by 6 ((cos + sin) * 3 dimensions = 6)
            max_spatial_shape: (3,) Size of each dimension
            dropout: Dropout probability
        """
        super().__init__()

        assert len(max_spatial_shape) == 3, 'Spatial dimension must be 3'
        assert d_model % 6 == 0, f'Feature dimension {d_model} not divisible by 6'
        self.max_spatial_shape = max_spatial_shape

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros([d_model] + list(max_spatial_shape))

        d_model = int(d_model / len(max_spatial_shape))

        # Equivalent to attention is all you need encoding: https://arxiv.org/abs/1706.03762
        div_term = torch.exp(torch.arange(0., d_model, 2) * -(math.log(10000.0) / d_model))

        pos_x = torch.arange(0., max_spatial_shape[0]).unsqueeze(1)
        pos_y = torch.arange(0., max_spatial_shape[1]).unsqueeze(1)
        pos_z = torch.arange(0., max_spatial_shape[2]).unsqueeze(1)

        pe[0:d_model:2, ...] = torch.sin(pos_x * div_term).transpose(0, 1)[:, :, None, None].repeat(1, 1, max_spatial_shape[1], max_spatial_shape[2])
        pe[1:d_model:2, ...] = torch.cos(pos_x * div_term).transpose(0, 1)[:, :, None, None].repeat(1, 1, max_spatial_shape[1], max_spatial_shape[2])
        pe[d_model:2*d_model:2, ...] = torch.sin(pos_y * div_term).transpose(0, 1)[:, None, :, None].repeat(1, max_spatial_shape[0], 1, max_spatial_shape[2])
        pe[d_model+1:2*d_model:2, ...] = torch.cos(pos_y * div_term).transpose(0, 1)[:, None, :, None].repeat(1, max_spatial_shape[0], 1, max_spatial_shape[2])
        pe[2*d_model:3*d_model:2, ...] = torch.sin(pos_z * div_term).transpose(0, 1)[:, None, None, :].repeat(1, max_spatial_shape[0], max_spatial_shape[1], 1)
        pe[2*d_model+1:3*d_model:2, ...] = torch.cos(pos_z * div_term).transpose(0, 1)[:, None, None, :].repeat(1, max_spatial_shape[0], max_spatial_shape[1], 1)

        self.register_buffer('pe', pe)

    def forward(self, point_features, positional_input, grid_size=None):
        """
        Args:
            point_features: (b, xyz, f)
            positional_input: (b, xyz, 3)
        Returns:
            point_features: (b, xyz, f)
        """
        assert len(point_features.shape) == 3
        num_points = point_features.shape[1]
        num_features = point_features.shape[2]
        if grid_size == None:
            grid_size = self.max_spatial_shape
        assert num_points == grid_size.prod()

        pe =  self.pe[:, :grid_size[0], :grid_size[1], :grid_size[2]].permute(1, 2, 3, 0).contiguous().view(1, num_points, num_features)
        point_features = point_features + pe
        return self.dropout(point_features)


class FeedForwardPositionalEncoding(nn.Module):
    def __init__(self, d_input, d_output):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Conv1d(d_input, d_output // 2, 1),
            nn.BatchNorm1d(d_output // 2),
            nn.ReLU(d_output // 2),
            nn.Conv1d(d_output // 2, d_output, 1),
        )
        self.DAblock = DANetHead(d_input*32, d_output)  # 4*32, 128

    def forward(self, point_features, positional_input, grid_size=None):
        """
        Args:
            point_features: (b, xyz, f)
            local_point_locations: (b, xyz, 3)
        Returns:
            point_features: (b, xyz, f)
        """
        if point_features.shape[0] == 0:
            #print("point_features.shape[0] == 0")
            pos_encoding = self.ffn(positional_input.permute(0, 2, 1))
            point_features = point_features + pos_encoding.permute(0, 2, 1)
            return point_features
        else:
            #print("point_features.shape[0] != 0")
            b, n, f = point_features.shape
            x = torch.clone(point_features)
            x = x.reshape(b*6, 6, 6, f).permute(0, 3, 1, 2)
            #print("reshape,point_features", x.shape)
            # x = self.DAblock(x)
            #print("self.DAblock1", x.shape)
            x  = x.permute(0, 2, 3, 1).reshape(b, -1, f)
            #print("reshape,point_features", x.shape)
            ##################################################
            pos_encoding = self.ffn(positional_input.permute(0, 2, 1))
            #print("pos_encoding:\n", pos_encoding.shape)
            x = x + pos_encoding.permute(0, 2, 1)
            #print("x:\n", x.shape)
            return x


def get_positional_encoder(pool_cfg):
    pos_encoder = None
    attention_cfg = pool_cfg.ATTENTION
    if attention_cfg.POSITIONAL_ENCODER == 'frequency':
        pos_encoder = FrequencyPositionalEncoding3d(d_model=attention_cfg.NUM_FEATURES,
                                                    max_spatial_shape=torch.IntTensor([pool_cfg.GRID_SIZE] * 3),
                                                    dropout=attention_cfg.DROPOUT)
    elif attention_cfg.POSITIONAL_ENCODER == 'grid_points':
        pos_encoder = FeedForwardPositionalEncoding(d_input=3, d_output=attention_cfg.NUM_FEATURES)
    elif attention_cfg.POSITIONAL_ENCODER == 'density':
        pos_encoder = FeedForwardPositionalEncoding(d_input=1, d_output=attention_cfg.NUM_FEATURES)
    elif attention_cfg.POSITIONAL_ENCODER == 'density_grid_points':
        pos_encoder = FeedForwardPositionalEncoding(d_input=4, d_output=attention_cfg.NUM_FEATURES)
    elif attention_cfg.POSITIONAL_ENCODER == 'density_centroid':
        pos_encoder = FeedForwardPositionalEncoding(d_input=7, d_output=attention_cfg.NUM_FEATURES)

    return pos_encoder


class DWT_Function(Function):
    @staticmethod
    def forward(ctx, x, w_ll, w_lh, w_hl, w_hh):
        x = x.contiguous()
        # x = x.to(dtype=torch.float16)
        # print(type(x))
        ctx.save_for_backward(w_ll, w_lh, w_hl, w_hh)
        ctx.shape = x.shape

        dim = x.shape[1]
        x_ll = torch.nn.functional.conv2d(x, w_ll.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_lh = torch.nn.functional.conv2d(x, w_lh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hl = torch.nn.functional.conv2d(x, w_hl.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x_hh = torch.nn.functional.conv2d(x, w_hh.expand(dim, -1, -1, -1), stride = 2, groups = dim)
        x = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            w_ll, w_lh, w_hl, w_hh = ctx.saved_tensors

            
            B, C_total, H, W = dx.shape 
           
            C_original = C_total // 4 

            
            dx = dx.view(B, 4, C_original, H, W) # [B, 4, C_original, H, W]
            dx = dx.transpose(1, 2)              # [B, C_original, 4, H, W]
            dx = dx.reshape(B, -1, H, W)        # [B, C_original * 4, H, W]

            
            filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0) # shape [4, 1, k, k] 
            filters = filters.repeat(C_original, 1, 1, 1) 
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C_original) # 或者 groups=1

        return dx, None, None, None, None

class IDWT_Function(Function):
    @staticmethod
    def forward(ctx, x, filters):
        ctx.save_for_backward(filters)
        ctx.shape = x.shape

        B, _, H, W = x.shape
        x = x.view(B, 4, -1, H, W).transpose(1, 2)
        C = x.shape[1]
        x = x.reshape(B, -1, H, W)
        filters = filters.repeat(C, 1, 1, 1)
        x = torch.nn.functional.conv_transpose2d(x, filters, stride=2, groups=C)
        return x

    @staticmethod
    def backward(ctx, dx):
        if ctx.needs_input_grad[0]:
            filters = ctx.saved_tensors
            filters = filters[0]
            
            B, C, H, W = ctx.shape
            C = C // 4 
            dx = dx.contiguous()
            
            # B, C, H, W = dx.shape
            # C = C
            # dx = dx.contiguous()

            w_ll, w_lh, w_hl, w_hh = torch.unbind(filters, dim=0)
            x_ll = torch.nn.functional.conv2d(dx, w_ll.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_lh = torch.nn.functional.conv2d(dx, w_lh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hl = torch.nn.functional.conv2d(dx, w_hl.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            x_hh = torch.nn.functional.conv2d(dx, w_hh.unsqueeze(1).expand(C, -1, -1, -1), stride = 2, groups = C)
            dx = torch.cat([x_ll, x_lh, x_hl, x_hh], dim=1)
        return dx, None

class IDWT_2D(nn.Module):
    def __init__(self, wave):
        super(IDWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        rec_hi = torch.Tensor(w.rec_hi)
        rec_lo = torch.Tensor(w.rec_lo)
        
        w_ll = rec_lo.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_lh = rec_lo.unsqueeze(0)*rec_hi.unsqueeze(1)
        w_hl = rec_hi.unsqueeze(0)*rec_lo.unsqueeze(1)
        w_hh = rec_hi.unsqueeze(0)*rec_hi.unsqueeze(1)

        w_ll = w_ll.unsqueeze(0).unsqueeze(1)
        w_lh = w_lh.unsqueeze(0).unsqueeze(1)
        w_hl = w_hl.unsqueeze(0).unsqueeze(1)
        w_hh = w_hh.unsqueeze(0).unsqueeze(1)
        filters = torch.cat([w_ll, w_lh, w_hl, w_hh], dim=0)
        self.register_buffer('filters', filters)
        self.filters = self.filters.to(dtype=torch.float32)

    def forward(self, x):
        # print("self.filters.shape",self.filters.shape)
        return IDWT_Function.apply(x, self.filters)

class DWT_2D(nn.Module):
    def __init__(self, wave):
        super(DWT_2D, self).__init__()
        w = pywt.Wavelet(wave)
        self.dec_hi = torch.Tensor(w.dec_hi[::-1]) 
        self.dec_lo = torch.Tensor(w.dec_lo[::-1])

        w_ll = self.dec_lo.unsqueeze(0)*self.dec_lo.unsqueeze(1)
        w_lh = self.dec_lo.unsqueeze(0)*self.dec_hi.unsqueeze(1)
        w_hl = self.dec_hi.unsqueeze(0)*self.dec_lo.unsqueeze(1)
        w_hh = self.dec_hi.unsqueeze(0)*self.dec_hi.unsqueeze(1)

        self.register_buffer('w_ll', w_ll.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_lh', w_lh.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hl', w_hl.unsqueeze(0).unsqueeze(0))
        self.register_buffer('w_hh', w_hh.unsqueeze(0).unsqueeze(0))

        self.w_ll = self.w_ll.to(dtype=torch.float32)
        self.w_lh = self.w_lh.to(dtype=torch.float32)
        self.w_hl = self.w_hl.to(dtype=torch.float32)
        self.w_hh = self.w_hh.to(dtype=torch.float32)

    def forward(self, x):
        # print("self.dec_hi.shape",self.dec_hi.shape,self.dec_hi)
        # print("self.dec_lo.shape",self.dec_lo.shape,self.dec_lo)
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)