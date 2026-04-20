import torch
import torch.nn as nn
from torch.autograd import Function
import pywt

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

            
            dx = torch.nn.functional.conv_transpose2d(dx, filters, stride=2, groups=C_original) 

            
            # dx = dx[:, :, :expected_H, :expected_W]

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
       
        return DWT_Function.apply(x, self.w_ll, self.w_lh, self.w_hl, self.w_hh)
    

class WaveletTrans(nn.Module):
    def __init__(self):
        super(WaveletTrans, self).__init__()

        self.dwt = DWT_2D(wave='db8')
        self.idwt = IDWT_2D(wave='db8')


        self.dwt = DWT_2D(wave='db8')  
        self.idwt = IDWT_2D(wave='db8') 

       
        self.reduce = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=1, padding=0, stride=1),  
            nn.BatchNorm2d(64),  
            nn.ReLU(inplace=True),  
        )
        
        self.filter = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, padding=1, stride=1, groups=1),
            nn.BatchNorm2d(256),  
            nn.ReLU(inplace=True),  
        )
        

    def forward(self, image_features_batch):
        bs, channel_num, f_h, f_w = image_features_batch.shape

        
        if f_w % 2 != 0:
            image_features_batch = nn.functional.interpolate(image_features_batch, (f_h, f_w + 1), mode='bilinear',
                                                             align_corners=False)
        if f_h % 2 != 0:
            image_features_batch = nn.functional.interpolate(image_features_batch, (f_h + 1, f_w), mode='bilinear',
                                                             align_corners=False)

        # print("image_features_batch:\n", image_features_batch.shape)
        x_dwt = self.dwt(self.reduce(image_features_batch))
        # print("self.dwt(self.reduce(x)):\n", x_dwt.shape)
        x_dwt = self.filter(x_dwt)
        # print("self.filter(x_dwt):\n", x_dwt.shape)

        x_idwt = self.idwt(x_dwt)
        # print("self.idwt(x_dwt):\n", x_idwt.shape)

        _, _, x_idwt_height, x_idwt_width = x_idwt.shape
        if x_idwt_height != f_h or x_idwt_width != f_w:
            # print("x_idwt_height != f_h or x_idwt_width != f_w:\n")
            x_idwt = nn.functional.interpolate(x_idwt, (f_h, f_w), mode='bilinear')
            # print("self.idwt(x_dwt):\n", x_idwt.shape)
        # x_idwt = x_idwt.squeeze(0)
        # x_idwt = self.get_cam_feats(x_idwt, depth, neighbors_depth)
        flatten_img_feat = x_idwt.permute(0, 2, 3, 1).reshape(1, f_h * f_w, channel_num)
        # print("flatten_img_feat", flatten_img_feat.shape)
        ######################################
        return flatten_img_feat