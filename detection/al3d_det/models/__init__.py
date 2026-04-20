from collections import namedtuple

import numpy as np
import torch
import kornia
from .centerpoint_waymo import CenterPointPC
from .centerpoint_MM_waymo import CenterPointMM
from .anchor_kitti import ANCHORKITTI
from .anchor_MM_kitti import ANCHORMMKITTI
__all__ = {
    'CenterPointPC': CenterPointPC,
    'CenterPointMM': CenterPointMM,
    'ANCHORKITTI': ANCHORKITTI,
    'ANCHORMMKITTI': ANCHORMMKITTI,
}


def build_network(model_cfg, num_class, dataset):
    model = __all__[model_cfg.NAME](
        model_cfg=model_cfg,
        num_class=num_class,
        dataset=dataset
    )
    return model

# 原始
# def load_data_to_gpu(batch_dict):
#     for key, val in batch_dict.items():
#         if key in ['frame_id', 'sequence_name', 'pose', 'tta_ops', 'aug_matrix_inv','db_flag', 'frame_id', 'metadata', 'calib', 'sequence_name']:
#             continue
#         elif key in ['images', 'extrinsic', 'intrinsic']:
#             temp = {}
#             for cam in val.keys():
#                 temp[cam] = torch.from_numpy(val[cam]).float().cuda().contiguous()
#             batch_dict[key] = temp
#         elif key in ['image_shape']:
#             temp = {}
#             for cam in val.keys():
#                 temp[cam] = torch.from_numpy(val[cam]).int().cuda()
#             batch_dict[key] = temp
#         elif isinstance(val, np.ndarray):
#             batch_dict[key] = torch.from_numpy(val).float().cuda()
#         else:
#             continue

# def load_data_to_gpu(batch_dict):
#     for key, val in batch_dict.items():
#         if key in ['frame_id', 'sequence_name', 'pose', 'tta_ops', 'aug_matrix_inv',
#                   'db_flag', 'frame_id', 'metadata', 'calib', 'sequence_name']:
#             continue
#         elif isinstance(val, dict):  # 检查是否是字典
#             if key in ['images', 'extrinsic', 'intrinsic']:
#                 temp = {}
#                 for cam, cam_val in val.items():  # 遍历字典项
#                     if isinstance(cam_val, np.ndarray):
#                         temp[cam] = torch.from_numpy(cam_val).float().cuda().contiguous()
#                     else:
#                         temp[cam] = cam_val  # 或者其他处理方式
#                 batch_dict[key] = temp
#             elif key in ['image_shape']:
#                 temp = {}
#                 for cam, cam_val in val.items():
#                     if isinstance(cam_val, np.ndarray):
#                         temp[cam] = torch.from_numpy(cam_val).int().cuda()
#                     else:
#                         temp[cam] = cam_val  # 或者其他处理方式
#                 batch_dict[key] = temp
#         elif isinstance(val, np.ndarray):
#             batch_dict[key] = torch.from_numpy(val).float().cuda()
#         else:
#             continue



# def load_data_to_gpu(batch_dict):
#     for key, val in batch_dict.items():
#         # 跳过不需要处理的键
#         if key in ['frame_id', 'sequence_name', 'pose', 'tta_ops', 'aug_matrix_inv',
#                   'db_flag', 'metadata', 'calib', 'sequence_name']:
#             continue
        
#         # 处理图像数据
#         elif key == 'images':
#             if isinstance(val, np.ndarray):
#                 # 假设输入是 [batch, height, width, channels]
#                 if val.ndim == 4 and val.shape[3] in [3, 1]:  # 检查通道数
#                     val = val.transpose(0, 3, 1, 2)  # 转换为 [batch, channels, height, width]
#                 batch_dict[key] = torch.from_numpy(val).float().cuda().contiguous()
#             elif isinstance(val, torch.Tensor):
#                 # 如果已经是张量，确保形状正确
#                 if val.dim() == 4 and val.shape[1] in [3, 1]:
#                     pass  # 形状已经正确
#                 else:
#                     # 可能需要调整形状
#                     val = val.permute(0, 3, 1, 2)  # 假设输入是 [batch, height, width, channels]
#                 batch_dict[key] = val.cuda().contiguous()
        
#         # 处理其他数组类型数据
#         elif isinstance(val, np.ndarray):
#             batch_dict[key] = torch.from_numpy(val).float().cuda() if val.dtype != np.int64 else torch.from_numpy(val).int().cuda()
        
#         # 处理已经是张量的数据
#         elif isinstance(val, torch.Tensor):
#             batch_dict[key] = val.cuda().contiguous()
        
#         # 其他情况（如列表、字典等）可以根据需要添加处理逻辑
#         else:
#             # 可以选择跳过或记录警告
#             print(f"Warning: Unhandled type for key '{key}': {type(val)}")
#             continue

#     return batch_dict

# 修改
def load_data_to_gpu(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        else:
            batch_dict[key] = torch.from_numpy(val).float().cuda()
            
def load_data_to_gpukitti(batch_dict):
    for key, val in batch_dict.items():
        if not isinstance(val, np.ndarray):
            continue
        elif key in ['frame_id', 'metadata', 'calib', 'pose', 'tta_ops', 'aug_matrix_inv','db_flag', 'frame_id', 'metadata', 'calib', 'sequence_name']:
            continue
        elif key in ['images']:
            batch_dict[key] = kornia.image_to_tensor(val).float().cuda().contiguous()
        elif key in ['image_shape']:
            batch_dict[key] = torch.from_numpy(val).int().cuda()
        elif isinstance(val, np.ndarray):
            batch_dict[key] = torch.from_numpy(val).float().cuda()
        else:
            continue
def model_fn_decorator():
    ModelReturn = namedtuple('ModelReturn', ['loss', 'tb_dict', 'disp_dict'])

    def model_func(model, batch_dict):
        if 'calib' in batch_dict.keys():
            load_data_to_gpukitti(batch_dict)
        else:
            load_data_to_gpu(batch_dict)
        ret_dict, tb_dict, disp_dict = model(batch_dict)

        loss = ret_dict['loss'].mean()
        if hasattr(model, 'update_global_step'):
            model.update_global_step()
        else:
            model.module.update_global_step()

        return ModelReturn(loss, tb_dict, disp_dict)

    return model_func
