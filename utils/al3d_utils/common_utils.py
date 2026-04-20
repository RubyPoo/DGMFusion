import os
import logging
import pickle
import random
import shutil
import subprocess

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp


# ============================== #
# === Build the Network Part === #
# ============================== #

def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    return logger


def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:
    Returns:
    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank
# multi-node multi-gpu for PAI
def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    # print('num_gpus:', num_gpus)
    # print('local_rank:', local_rank)
    # print('tcp_port:', tcp_port)
    dist.init_process_group(
        backend=backend,
        # init_method='tcp://127.0.0.1:%d' % tcp_port,
        # rank=local_rank,
        # world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank


def get_dist_info():
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1
    return rank, world_size


def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results


# ============================== #
# ====== Calculation Part ====== #
# ============================== #

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

# 旋转角度
# 输入：
#   boxes3d[:, 6]: 【：，[x, y, z, dx, dy, dz]】
# offset=0.5
# period=np.pi
def limit_period(val, offset=0.5, period=np.pi):
    # 确保为PyTorch 张量
    val, is_numpy = check_numpy_to_torch(val)
    # check_numpy_to_torch: 将numpy转为torch，并返回是否为numpy
    # val:[0.0000, 1.5700, 0.0000,  ..., 1.5700, 0.0000, 1.5700]
    #    torch.Size([70400])
    # is_numpy:False
    print('val:\n', val,val.shape)
    print('is_numpy\n', is_numpy)

    # 公式：val - floor(val / period + offset) * period，角度规范到pi/2到-pi/2范围
    ans = val - torch.floor(val / period + offset) * period  # 向下取整，不大于元素的最大整数
    # torch.floor 是 PyTorch 库中的一个函数，用于对输入张量的每个元素向下取整，即返回小于或等于每个元素的最大整数。
    # ans = 输入的每个元素-[(输入的每个元素/pi+0.5)*pi]，即将角度限制在pi/2到-pi/2范围

    return ans.numpy() if is_numpy else ans  # 结果需要转numpy


def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info


def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds


def cart2cylinder(points):
    is_numpy = type(points) == np.ndarray
    if is_numpy:
        points = torch.from_numpy(points)

    r = torch.sqrt(points[:, 0] ** 2 + points[:, 1] ** 2)
    phi = torch.atan2(points[:, 1], points[:, 0])
    points = torch.stack((r, phi, points[:, 2]), dim=1)

    if is_numpy:
        points = points.numpy()

    return points


def cylinder2cart(points):
    is_numpy = type(points) == np.ndarray
    if is_numpy:
        points = torch.from_numpy(points)

    x = torch.cos(points[:,1])*points[:,0]
    y = torch.sin(points[:,1])*points[:,0]

    points = torch.stack((x,y,points[:,2]),dim=1)

    if is_numpy:
        points = points.numpy()

    return points

# 输入：
#   points: 3D点集，形状为(N, 3)   local_roi_grid_points.clone()
#   angle: 旋转角度，单位弧度       rois[:, 6] 第7个参数是旋转角度
# 返回：
#   旋转后的点集，形状与输入points相同，类型与输入points相同（即如果输入是numpy数组，则输出也是numpy数组；如果是torch张量，则输出也是torch张量）。
def rotate_points_along_z(points, angle):
    """
    该函数用于根据给定的角度绕z轴旋转3D点集。

    参数:
        points: 形状为(B, N, 3 + C)的张量或数组，其中B表示批量大小，N表示每个批量中的点的数量，3表示每个点的x, y, z坐标，C表示每个点可能附加的特征数量。
        angle: 形状为(B)的张量或数组，表示每个批量的旋转角度（单位：弧度）。角度增加的方向是从x轴到y轴。

    返回值:
        返回旋转后的点集，形状与输入points相同，类型与输入points相同（即如果输入是numpy数组，则输出也是numpy数组；如果是torch张量，则输出也是torch张量）。
    """
    # 检查points是否为numpy数组，如果是则转换为torch张量，并记录其原始类型
    points, is_numpy = check_numpy_to_torch(points)
    # 检查angle是否为numpy数组，如果是则转换为torch张量
    angle, _ = check_numpy_to_torch(angle)

    # 如果points在GPU上，将angle也移动到GPU上
    if points.is_cuda:
        angle = angle.cuda()

    # 计算cos(angle)和sin(angle)，用于构建旋转矩阵
    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    # 创建与angle相同设备且形状为(B)的零张量和全一张量，用于构建旋转矩阵
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    # 构建绕z轴的旋转矩阵，形状为(B, 3, 3)
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).reshape(-1, 3, 3).float()
    # 使用旋转矩阵对点的x, y, z坐标部分进行旋转操作
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    # 将旋转后的坐标与原始点的特征部分（如果有的话）重新拼接
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    # 根据原始输入points的类型决定返回numpy数组还是torch张量
    return points_rot.numpy() if is_numpy else points_rot



def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask


def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:
    Returns:s
    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers

def get_pad_params(desired_size, cur_size):
    """
    Get padding parameters for np.pad function
    Args:
        desired_size: int, Desired padded output size
        cur_size: int, Current size. Should always be less than or equal to cur_size
    Returns:
        pad_params: tuple(int), Number of values padded to the edges (before, after)
    """
    assert desired_size >= cur_size

    # Calculate amount to pad
    diff = desired_size - cur_size
    pad_params = (0, diff)

    return pad_params

def lidar_to_image_strict(points, lidar_to_cam, cam_to_image):
    is_numpy = type(points) == np.ndarray
    if is_numpy:
        points = torch.from_numpy(points)
        lidar_to_cam = torch.from_numpy(lidar_to_cam)
        cam_to_image = torch.from_numpy(cam_to_image)

    points = torch.cat(
        [points, torch.ones([points.shape[0], 1],
            dtype=lidar_to_cam.dtype,
            device=lidar_to_cam.device)],
        dim=1
    )
    points_cam = torch.matmul(
        points,
        torch.transpose(lidar_to_cam, 0, 1)
    )

    points_img = torch.matmul(
        points_cam[:, :3],
        torch.transpose(cam_to_image[:, :3], 0, 1)
    )
    
    points_depth = points_img[:, [2]]
    points_img = points_img[:, :2] / points_depth

    if is_numpy:
        points_img = points_img.numpy()
        points_depth = points_depth.numpy()

    return points_img, points_depth
