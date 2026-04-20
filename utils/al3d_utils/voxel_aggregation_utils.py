import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# 将三维坐标的点（point_coords）映射到一个体素（voxel）网格中，并返回体素的索引。
# 如果点在网格的范围外，则返回 (-1, -1, -1)
# 点云xyz
# 4
# voxel_size = [0.05, 0.05, 0.1]
# point_cloud_range = [0, -40, -3, 70.4, 40, 1]
def get_overlapping_voxel_indices(point_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        point_coords: (N, 3) 形状为 (N, 3) 的张量，表示 N 个三维点的坐标
        downsample_times: (int)  一个整数，表示对体素大小的下采样倍数
        voxel_size: [x_size, y_size, z_size]  列表，包含三个值 [x_size, y_size, z_size]，表示每个体素在 x、y、z 方向上的大小。
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    This assumes that the (pc_range[3:6] - pc_range[0:3]) / voxel_size is an integer. If the pc_range
    is not entirely divisible by the voxel_size, points on the far extremes may be excluded.
    E.g.
    1)
        pc_range = [0, 0, 0, 1, 1, 1], voxel_size = [0.5, 0.5, 0.5]
        The point [0,0,0.7] will be considered inside and will return a value of (0, 0, 1)
    2)
        pc_range = [0, 0, 0, 1, 1, 1], voxel_size = [0.6, 0.6, 0.6]
        The point [0,0,0.7] will be considered outside and will return a value of (-1, -1, -1)
    Returns: voxel_indices (xyz). If the point cloud is outside the range of the voxels,
             it returns a value of (-1, -1, -1)
    """
    # 验证 point_coords 的形状，确保其第二维为 3
    # point_coords.shape[1] 返回 point_coords 张量的第二维的大小，应该等于 3，以确保每个点包含三个坐标（x, y, z）。
    assert point_coords.shape[1] == 3
    # 将 voxel_size 和 point_cloud_range 转换为张量，并针对输入的设备进行处理，确保数据在同一设备上计算（如 GPU）
    voxel_size = torch.tensor(voxel_size, device=point_coords.device).float() * downsample_times  # 体素大小*下采样倍数（4）[0.2, 0.2, 0.4]
    print("get_overlapping_voxel_indices voxel_size:", voxel_size)  # tensor([0.2000, 0.2000, 0.4000]
    pc_range = torch.tensor(point_cloud_range, device=point_coords.device).float()  # [0, -40, -3, 70.4, 40, 1]
    print("get_overlapping_voxel_indices pc_range:", pc_range)  # tensor([  0.0000, -40.0000,  -3.0000,  70.4000,  40.0000,   1.0000]

    # 使用公式 ((point_coords - pc_range[0:3]) / voxel_size) 计算每个点的体素索引。
    # 这个操作是为了根据点的坐标及其在体素网格中的位置来确定该点对应的体素。
    # pc_range[0:3] 表示网格的最小坐标，voxel_size 表示每个体素的大小。
    voxel_indices = ((point_coords - pc_range[0:3]) / voxel_size)
    print("get_overlapping_voxel_indices voxel_indices:", voxel_indices, "voxel_indices.shape:", voxel_indices.shape)
    # 输出：
    #  tensor([[ 78.8616, 127.5337,   4.0699],
    #         [ 55.9580, 235.2738,   2.9385],
    #         [112.3797, 271.1936,   3.4402],
    #         ...,
    #         [149.3189, 216.8464,   5.1276],
    #         [ 30.7538, 202.9331,   4.3735],
    #         [184.6271, 387.0577,   6.9922]]


    # Calculate number of voxels in each dimension 计算网格大小
    # 使用公式 ((pc_range[3:6] - pc_range[0:3]) / voxel_size).long() 计算在每个维度中体素的数量。
    # # 计算网格在每个维度的体素数量
    # # 公式解释：
    # # pc_range[3:6] 表示点云范围的最大坐标（x_max, y_max, z_max）
    # # pc_range[0:3] 表示点云范围的最小坐标（x_min, y_min, z_min）
    # # 因此，(pc_range[3:6] - pc_range[0:3]) 计算的是在每个维度上的总大小（x_size, y_size, z_size）
    # # 然后通过 voxel_size 将总大小除以每个体素的大小来得出每个维度的体素数量
    # # 最后使用 .long() 方法将结果转换为长整型，以表示体素的数量
    grid_size = ((pc_range[3:6] - pc_range[0:3]) / voxel_size).long()
    print("get_overlapping_voxel_indices grid_size:\n", grid_size, "grid_size.shape:\n", grid_size.shape)
    # 输出：
    # grid_size tensor([352, 400,  10]
    # grid_size.shape:torch.Size([3])

    # Check which points are in and which points are outside the point cloud range and set to -1
    # 检查点是否在范围内
    # 通过评估 voxel_indices 中是否存在小于 0 或大于等于 grid_size 的情况，来确定哪些点是超出范围的。如果某个点的体素索引小于 0 或等于或大于对应维度的网格大小，判定其为超出范围。
    # 使用 .sum(dim=-1) > 0 来判断是否有点是超出范围的，如果有，则将这些点的体素索引设为 (-1, -1, -1)。
    # ，sum(dim=-1) 的作用是对 voxel_indices 的最后一个维度进行求和。
    # 这一操作的目的是检查每个点的体素索引是否在有效范围内。
    # 具体来说，它会将每个点在各个维度的判断结果（即是否小于 0 或大于等于 grid_size）进行相加。
    # 如果结果大于 0，表示该点在至少一个维度上超出了点云范围。
    # 检查每个元素，如果大于0，表示该维度（切片）中有至少一个元素超出了范围，结果为True；否则为False
    points_out_of_range = ((voxel_indices < 0) | (voxel_indices >= grid_size)).sum(dim=-1) > 0
    # 将所有超出范围的体素索引设为 -1，表示这些点是无效的
    voxel_indices[points_out_of_range] = -1

    return voxel_indices.long() # (xyz)


def get_voxel_indices_to_voxel_list_index(x_conv):
    """
    Args:
        x_conv: (SparseConvTensor)
    Returns:
        x_conv_hash_table: (B, X, Y, Z) Dense representation of sparse voxel indices
    """
    x_conv_indices = x_conv.indices
    # Note that we need to offset the values by 1 since the dense representation has "0" to indicate an empty location
    x_conv_values = torch.arange(1, x_conv_indices.shape[0]+1, device=x_conv_indices.device)
    x_conv_shape = [x_conv.batch_size] + list(x_conv.spatial_shape)

    # TODO: Need to convert to_dense representation. Can we use rule table instead? Can try scatter_nd in spconv too
    x_conv_hash_table = torch.sparse_coo_tensor(x_conv_indices.T, x_conv_values, x_conv_shape, device=x_conv_indices.device).to_dense()
    return x_conv_hash_table


def get_nonempty_voxel_feature_indices(voxel_indices, x_conv):
    """
    Args:
        voxel_indices: (N, 4) [bxyz]
        x_conv: (SparseConvTensor)
    Returns:
        overlapping_voxel_feature_indices_nonempty: (N', 4)
        overlapping_voxel_feature_nonempty_mask: (N)
    """
    x_conv_hash_table = get_voxel_indices_to_voxel_list_index(x_conv)

    # Get corresponding voxel feature indices
    overlapping_voxel_feature_indices = torch.zeros(voxel_indices.shape[0], device=voxel_indices.device, dtype=torch.int64)
    overlapping_voxel_feature_indices = x_conv_hash_table[voxel_indices[:,0], voxel_indices[:,1],
                                                          voxel_indices[:,2], voxel_indices[:,3]]
    # Remove empty voxels features
    overlapping_voxel_feature_nonempty_mask = overlapping_voxel_feature_indices != 0
    overlapping_voxel_feature_indices_nonempty = overlapping_voxel_feature_indices[overlapping_voxel_feature_nonempty_mask] - 1
    return overlapping_voxel_feature_indices_nonempty, overlapping_voxel_feature_nonempty_mask


# 获取每个体素的质心坐标
# 输入：
# points = points_valid, Nx4
# voxel_indices = voxel_idxs_valid , Nx5
# num_points_in_voxel = None
def get_centroid_per_voxel(points, voxel_idxs, num_points_in_voxel=None):
    """
    Args:
        points: (N, 4 + (f)) [bxyz + (f)] 点的坐标和特征
        voxel_idxs: (N, 4) [bxyz] 体素的索引
        num_points_in_voxel: (N) 每个体素中的点的数量
    Returns:
        centroids: (N', 4 + (f)) [bxyz + (f)] Centroids for each unique voxel 每个独特体素的质心
        centroid_voxel_idxs: (N', 4) [bxyz] Voxels idxs for centroids 质心对应的体素索引
        labels_count: (N') Number of points in each voxel 每个体素中点的数量
    """

    # 确保点的数量与体素索引的数量相同 N == voxel_idxs.shape[0]
    # 每个点都有一个索引，可能在一个体素内，也可能在不同体素内
    assert points.shape[0] == voxel_idxs.shape[0]
    print("get_centroid_per_voxel points:\n", points, "points.shape:\n", points.shape, "voxel_idxs:\n", voxel_idxs, "voxel_idxs.shape:\n", voxel_idxs.shape)
    # get_centroid_per_voxel points:
    # tensor([[  0.0000,  15.7723, -14.4933,  -1.3720,   0.2700],
    #         [  0.0000,  11.1916,   7.0548,  -1.8246,   0.4100],
    #         [  0.0000,  22.4759,  14.2387,  -1.6239,   0.3800],
    #         ...,
    #         [  1.0000,  29.8638,   3.3693,  -0.9490,   0.1900],
    #         [  1.0000,   6.1508,   0.5866,  -1.2506,   0.0000],
    #         [  1.0000,  36.9254,  37.4115,  -0.2031,   0.0000]], device='cuda:0')
    # torch.Size([37453, 5])

    # voxel_idxs:
    #  tensor([[  0,   4, 127,  78],
    #         [  0,   2, 235,  55],
    #         [  0,   3, 271, 112],
    #         ...,
    #         [  1,   5, 216, 149],
    #         [  1,   4, 202,  30],
    #         [  1,   6, 387, 184]], device='cuda:0')
    #  torch.Size([37453, 4])

    # get_centroid_per_voxel num_points_in_voxel:
    #  None

    # 获取唯一的体素索引、反向索引和每个体素中的点的数量
    # numpy.unique()使用方法 https://blog.csdn.net/xhtchina/article/details/129025249
    print("获取唯一的体素索引、反向索引和每个体素中的点的数量\n")
    centroid_voxel_idxs, unique_idxs, labels_count = voxel_idxs.unique(dim=0, return_inverse=True, return_counts=True)
    # 函数 unique 用于在指定维度（这里是 dim=0）上查找唯一值，return_inverse=True 表示返回原始张量中每个元素对应于唯一值的索引，
    # return_counts=True 则表示返回每个唯一值在原始张量中出现的次数。
    print("get_centroid_per_voxel centroid_voxel_idxs:\n", centroid_voxel_idxs, "centroid_voxel_idxs.shape:\n", centroid_voxel_idxs.shape, "unique_idxs:\n", unique_idxs, "unique_idxs.shape:\n", unique_idxs.shape, "labels_count:\n", labels_count, "labels_count.shape:\n", labels_count.shape)
    # get_centroid_per_voxel unique_idxs:
    #  tensor([[  0,   2,  92, 169],
    #         [  0,   2,  94, 171],
    #         [  0,   2,  96, 145],
    #         ...,
    #         [  1,   9, 386, 184],
    #         [  1,   9, 387, 184],
    #         [  1,   9, 388, 330]]
    # torch.Size([13869, 4])

    #  labels_count:
    #  tensor([1, 2, 1,  ..., 1, 1, 1]
    # torch.Size([13869])

    # 将索引扩展以匹配点的维度
    # numpy.view https://blog.csdn.net/weixin_46016933/article/details/133271694
    # numpy.size https://wenku.baidu.com/view/2f3b441dedfdc8d376eeaeaad1f34693daef1017.html?_wkts_=1733384384241&bdQuery=numpy.size%E6%96%B9%E6%B3%95%E4%B8%BE%E4%BE%8B
    # numpy.expand
    print("将索引扩展以匹配点的维度\n")
    unique_idxs = unique_idxs.view(unique_idxs.size(0), 1).expand(-1, points.size(-1))
    print("unique_idxs:\n", unique_idxs, "unique_idxs.shape:\n", unique_idxs.shape)
    # 将 unique_idxs 的形状调整为 (N, 1)，并扩展到 (N, D)，其中 N 是唯一体素索引的数量，D 是点的维度。
    # 使用 view 方法将 unique_idxs 的形状调整为 (N, 1)，
    # 其中 N 是唯一体素索引的数量。这为后续的 expand 操作做准备。
    # 使用 expand 方法将 unique_idxs 扩展到 (N, D) 的形状，
    # 其中 D 是点的维度。-1 表示保持当前维度的大小不变，
    # 通过这个操作，使得每个唯一体素索引都能对应到所有点的维度，
    # 以便进行 scatter 操作时可以正确地进行广播。
    # 这样做是为了方便后面的散点累加操作，使每个唯一体素索引都对应到点的每一个维度。
    # tensor([[ 126,  126,  126,  126,  126],
    #         [ 130,  130,  130,  130,  130],
    #         [ 141,  141,  141,  141,  141],
    #         ...,
    #         [7041, 7041, 7041, 7041, 7041],
    #         [7041, 7041, 7041, 7041, 7041],
    #         [7043, 7043, 7043, 7043, 7043]]
    # torch.Size([13869, 5])

    # Scatter add points based on unique voxel idxs
    # 根据唯一的体素索引对点进行散点累加
    # .shape https://www.jb51.net/python/292085sga.htm
    # .scatter_add_ https://blog.csdn.net/peng_pi/article/details/123413701
    # .unsqueeze https://www.jianshu.com/p/48efb6831428  https://blog.51cto.com/u_16175446/6946164
    print("根据唯一的体素索引对点进行散点累加\n")
    if num_points_in_voxel is not None:
        # 计算每个体素的加权质心
        print("计算每个体素的加权质心\n")
        centroids = torch.zeros((centroid_voxel_idxs.shape[0], points.shape[-1]), device=points.device, dtype=torch.float).scatter_add_(0, unique_idxs, points * num_points_in_voxel.unsqueeze(-1))
        # 创建一个形状为 (num_centroids, D) 的全零张量，其中 num_centroids 是唯一体素的数量，D 是点的维度。
        # 该张量用来存储每个体素的加权质心
        # 运行沿着第 0 维进行 scatter 累加
        #  unique_idxs,  # 使用扩展后的唯一体素索引作为索引
        #  points * num_points_in_voxel.unsqueeze(-1)  # 累加的值为每个点乘以对应体素中的点数
        print("get_centroid_per_voxel centroids:\n", centroids, "centroids.shape:\n", centroids.shape)
        #
        num_points_in_centroids = torch.zeros((centroid_voxel_idxs.shape[0]), device=points.device, dtype=torch.int64).scatter_add_(0, unique_idxs[:,0], num_points_in_voxel)
        print("get_centroid_per_voxel num_points_in_centroids:\n", num_points_in_centroids, "num_points_in_centroids.shape:\n", num_points_in_centroids.shape)
        #
        # 创建一个形状为 (num_centroids,) 的全零张量，用于存储每个体素中的点的数量。
        # num_centroids 是唯一体素的数量，数据类型为 int64。
        #  0,  # 运行沿着第 0 维进行 scatter 累加
        #     unique_idxs[:, 0],  # 使用扩展后的唯一体素索引的第 0 列作为索引
        #     num_points_in_voxel  # 累加的值为每个体素中的点的数量

        # 计算质心
        ####################################
        # 点云质心 https://blog.csdn.net/twnkie/article/details/143311997
        # 定义：点云质心（point cloud centroid）是指在点云中所有点的质心位置，它是对点云的几何中心的一种度量。
        #   计算点云的质心涉及数学中的简单平均计算。质心是点云中所有点的平均位置。对于一个三维点云，其质心的计算公式如下:
        # 1 质心计算公式
        #   假设点云中有 N 个点，每个点的坐标为 (xi,yi,zì)，那么质心 (x,y,z)(的计算公式为:
        # 2 原理
        #   1.求和:首先计算所有点在每个坐标轴上的和。
        #   2.平均:然后将每个坐标轴的和除以点的数量 N。
        #   3.结果:结果就是点云的质心，表示点云的平均位置。
        # 3 计算步骤
        #   1.遍历点云中的所有点,
        #   2.对每个点的 x、y、z坐标分别求和。
        #   3.将求和结果分别除以点的数量，得到质心的 x、y、z 坐标。
        # 离散点质心计算公式 https://wenku.baidu.com/view/eff4fc1da3116c175f0e7cd184254b35eefd1aef.html?_wkts_=1733388201149&bdQuery=%E7%A6%BB%E6%95%A3%E7%82%B9%E8%B4%A8%E5%BF%83%E8%AE%A1%E7%AE%97%E5%85%AC%E5%BC%8F%E6%98%AF%E4%BB%80%E4%B9%88
        centroids = centroids / num_points_in_centroids.float().unsqueeze(-1)
        print("get_centroid_per_voxel centroids:\n", centroids, "centroids.shape:\n", centroids.shape)
        # 通过将累加后的点坐标 (centroids) 除以每个体素中的点的数量 (num_points_in_centroids)，
        # 计算加权质心。首先将 num_points_in_centroids 转换为浮点型，并扩展维度，以便进行广播计算。
        #
    else:
        # 计算每个体素的未加权质心
        # 创建一个形状为 (num_centroids, D) 的全零张量，其中 num_centroids 是唯一体素的数量，D 是点的维度。
        # 该张量用于存储每个体素的未加权质心。
        print("计算每个体素的未加权质心\n")
        centroids = torch.zeros((centroid_voxel_idxs.shape[0], points.shape[-1]), device=points.device, dtype=torch.float).scatter_add_(0, unique_idxs, points)
        print("get_centroid_per_voxel centroids:\n", centroids, "centroids.shape:\n", centroids.shape)
        # 0,  # 沿着第 0 维进行 scatter 累加
        #     unique_idxs,  # 使用扩展后的唯一体素索引作为索引
        #     points  # 累加的值为所有点的坐标
        # tensor([[-3.1002e+01, -1.1052e+01, -1.3441e+01],
        #         [-7.0741e+00, -2.7589e+00, -2.5658e+00],
        #         [-1.5307e+00, -5.3359e-01, -2.2235e-01],
        #         ...,
        #         [ 2.0807e-02,  6.0749e-01, -1.4805e+00],
        #         [ 1.0430e+00,  7.7341e-01, -3.3143e+00],
        #         [ 1.2066e+00,  1.2586e+00, -3.2775e+00]]
        # torch.Size([2361, 3])

        # 计算质心
        centroids = centroids / labels_count.float().unsqueeze(-1)
        print("get_centroid_per_voxel centroids:\n", centroids, "centroids.shape:\n", centroids.shape)
        # 通过将累加后的点坐标 (centroids) 除以每个体素的点数量 (labels_count)，
        # 计算未加权的质心。首先将 labels_count 转换为浮点型，并扩展维度，以便进行广播计算。
        # get_centroid_per_voxel centroids:
        #  tensor([[-1.3479, -0.4805, -0.5844],
        #         [-1.1790, -0.4598, -0.4276],
        #         [-1.5307, -0.5336, -0.2223],
        #         ...,
        #         [ 0.0104,  0.3037, -0.7403],
        #         [ 0.2608,  0.1934, -0.8286],
        #         [ 0.3016,  0.3146, -0.8194]], device='cuda:0') centroids.shape:
        #  torch.Size([2361, 3])

    # 返回质心、体素索引和每个体素中的点的数量
    print("返回最终的质心、体素索引和每个体素中的点的数量\n")
    print("get_centroid_per_voxel centroids:\n", centroids, "centroids.shape:\n", centroids.shape)
    # get_centroid_per_voxel centroids:
    #  tensor([[-1.3479, -0.4805, -0.5844],
    #         [-1.1790, -0.4598, -0.4276],
    #         [-1.5307, -0.5336, -0.2223],
    #         ...,
    #         [ 0.0104,  0.3037, -0.7403],
    #         [ 0.2608,  0.1934, -0.8286],
    #         [ 0.3016,  0.3146, -0.8194]], device='cuda:0') centroids.shape:
    #  torch.Size([2361, 3])
    print("get_centroid_per_voxel centroid_voxel_idxs:\n", centroid_voxel_idxs, "centroid_voxel_idxs.shape:\n", centroid_voxel_idxs.shape)
    # get_centroid_per_voxel centroid_voxel_idxs:
    #  tensor([[  0,   0,   0,   0],
    #         [  0,   0,   0,   1],
    #         [  0,   0,   0,   2],
    #         ...,
    #         [127,   3,   5,   0],
    #         [127,   4,   4,   0],
    #         [127,   4,   5,   0]], device='cuda:0') centroid_voxel_idxs.shape:
    #  torch.Size([2361, 4])
    print("get_centroid_per_voxel labels_count:\n", labels_count, "labels_count.shape:\n", labels_count.shape)
    # get_centroid_per_voxel labels_count:
    #  tensor([23,  6,  1,  ...,  2,  4,  4], device='cuda:0') labels_count.shape:
    #  torch.Size([2361])
    return centroids, centroid_voxel_idxs, labels_count



# 获取每个voxel的质心，并返回每个voxel的质心的坐标和voxel的坐标
# 处理点云数据，特别是通过体素聚合（voxel aggregation）来计算不同特征层次下的质心（centroids）和对应的体素索引
#
# feature_locations=[x_conv3, x_conv4]
# multi_scale_3d_strides={x_conv1: 1，x_conv2:2 ，x_conv3: 4, x_conv4: 8}
# voxel_size=[0.05, 0.05, 0.1]
# point_cloud_range=[0, -40, -3, 70.4, 40, 1]
def get_centroids_per_voxel_layer(points, feature_locations, multi_scale_3d_strides, voxel_size, point_cloud_range):
    """
    Group points that lie within the same voxel together and average their xyz location.
    将位于相同体素内的点分组，并计算它们的xyz位置的平均值。
    Details can be found here: https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335
    Args:
        points: (N, 4 + (f)) [bxyz + (f)]  点的坐标和特征数组，N是点的数量，4代表点的批次索引、x、y、z坐标，(f)代表额外的特征数量。
        feature_locations: [str] (Order matters! Needs to be xconv1 -> xconv4) (顺序很重要！需要是xconv1 -> xconv4)，特征层次的列表。
        multi_scale_3d_strides: (dict) Map feature_locations to stride 将feature_locations映射到步长，指定每个特征层次的下采样因子。
        voxel_size: [x_size, y_size, z_size] 体素在x、y、z方向上的大小。
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max] 点云数据在x、y、z方向上的最小和最大范围。
    Returns:
        centroids_all: (dict) Centroids for each feature_locations 每个特征层次的质心。
        centroid_voxel_idxs_all: (dict) Centroid voxel ids for each feature_locations  每个特征层次的质心体素索引。
    """
    # 确保输入的 points 二维数组和 feature_locations 非空。
    assert len(points.shape) == 2
    assert len(feature_locations) > 0

    # 创建两个字典 centroids_all 和 centroid_voxel_idxs_all 用于存储每个特征层次的质心和体素索引。
    centroids_all = {}
    centroid_voxel_idxs_all = {}

    print("特征层1\n")
    # Take first layer feature locations 处理第一层特征位置，并根据该特征位置获得下采样因子。
    feature_location_first = feature_locations[0]  # x_conv3
    downsample_factor_first = multi_scale_3d_strides[feature_location_first] # 4

    # Calculate centroids 获取体素索引：通过调用 get_overlapping_voxel_indices 函数，计算给定点云（去掉第一列，通常是批次索引）的体素索引。
    voxel_idxs = get_overlapping_voxel_indices(points[:, 1:4], # 取点的xyz坐标
                                               downsample_times=downsample_factor_first, # 下采样因子 4
                                               voxel_size=voxel_size, # 体素大小 [0.05, 0.05, 0.1]
                                               point_cloud_range=point_cloud_range) # 点云范围 [0, -40, -3, 70.4, 40, 1]
    print("get_centroids_per_voxel_layer voxel_idxs:\n", voxel_idxs,voxel_idxs.shape)
    # tensor([[ 78, 127,   4],
    #         [ 55, 235,   2],
    #         [112, 271,   3],
    #         ...,
    #         [149, 216,   5],
    #         [ 30, 202,   4],
    #         [184, 387,   6]], device='cuda:0')
    #        torch.Size([38803, 3])

    # Add batch_idx 添加批次索引：将第一列（通常是批次索引）添加到体素索引的左侧。
    # 提取所有行的第一列（通常是批次索引），并将其添加到左侧，形成新的体素索引。
    # torch.cat 函数用于将两个张量沿着指定的维度进行拼接。其中 dim=-1 表示在最后一个维度上进行拼接。
    # 对于这个例子来说，就是将批次索引和体素索引在列的方向上拼接起来
    print("points\n", points,points.shape)
    # points
    # tensor([[  0.0000,  15.7723, -14.4933,  -1.3720,   0.2700],
    #         [  0.0000,  11.1916,   7.0548,  -1.8246,   0.4100],
    #         [  0.0000,  22.4759,  14.2387,  -1.6239,   0.3800],
    #         ...,
    #         [  1.0000,  29.8638,   3.3693,  -0.9490,   0.1900],
    #         [  1.0000,   6.1508,   0.5866,  -1.2506,   0.0000],
    #         [  1.0000,  36.9254,  37.4115,  -0.2031,   0.0000]], device='cuda:0')
    # torch.Size([38803, 5])
    print("points[:,0:1]:\n", points[:,0:1],points[:,0:1].shape)
    # points[:,0:1]:
    # tensor([[0.],
    #         [0.],
    #         [0.],
    #         ...,
    #         [1.],
    #         [1.],
    #         [1.]])
    #  torch.Size([38803, 1])
    voxel_idxs = torch.cat((points[:,0:1].long(), voxel_idxs), dim=-1)  # 将批次索引和体素索引拼接起来
    print("get_centroids_per_voxel_layer voxel_idxs:\n", voxel_idxs,voxel_idxs.shape)
    # get_centroids_per_voxel_layer voxel_idxs:
    # tensor([[  0,  78, 127,   4],
    #         [  0,  55, 235,   2],
    #         [  0, 112, 271,   3],
    #         ...,
    #         [  1, 149, 216,   5],
    #         [  1,  30, 202,   4],
    #         [  1, 184, 387,   6]], device='cuda:0')
    #   torch.Size([38803, 4])

    # Filter out points that are outside the valid point cloud range (invalid indices have -1)
    # 过滤有效点：根据有效的体素索引（非 -1）过滤点云数据，保留有效的体素索引和点。
    # -1通常表示无效的体素索引，可能是因为点位于点云范围之外。
    # .all(-1)：这个方法会对上一步的结果沿着最后一个维度（在这里，-1表示最后一个维度）进行逻辑与（AND）操作。
    # 如果某个体素的所有索引都不等于-1（即该体素的所有维度上的索引都是有效的），则结果为True；如果任何一个索引为-1，则结果为False。
    # 这意味着，只有当体素的所有相关索引都有效时，该体素才会被视为有效。
    # ############### 需要结合代码理解
    voxel_idxs_valid_mask = (voxel_idxs != -1).all(-1)   # 检查体素索引(每个元素)是否全部不为-1 （-1）最后一个维度，也就是 1，按列相加
    print("voxel_idxs_valid_mask:\n", voxel_idxs_valid_mask,voxel_idxs_valid_mask.shape)
    # tensor([True, True, True,  ..., True, True, True]
    # torch.Size([38803])
    voxel_idxs_valid = voxel_idxs[voxel_idxs_valid_mask]  # 保留有效的体素索引
    print("voxel_idxs_valid:\n", voxel_idxs_valid,voxel_idxs_valid.shape)
    # voxel_idxs 是原始的体素索引数组，而 voxel_idxs_valid_mask 是对应的布尔掩码。
    # Python 会遍历 voxel_idxs_valid_mask，将其值为 True 的索引所对应的 voxel_idxs 中的元素提取出来
    # tensor([[  0,  78, 127,   4],
    #         [  0,  55, 235,   2],
    #         [  0, 112, 271,   3],
    #         ...,
    #         [  1, 149, 216,   5],
    #         [  1,  30, 202,   4],
    #         [  1, 184, 387,   6]], device='cuda:0')
    #  torch.Size([37453, 4])

    # Convert voxel_indices from (bxyz) to (bzyx) format for properly indexing voxelization layer
    # 将体素索引从(bxyz)格式转换为(bzyx)格式，以正确索引体素化层。
    # 这通常是因为某些体素化实现需要这种特定的索引顺序。

    voxel_idxs_valid = voxel_idxs_valid[:, [0,3,2,1]]  # 重新排列索引顺序
    print("voxel_idxs_valid:\n", voxel_idxs_valid,voxel_idxs_valid.shape)
    # tensor([[  0,   4, 127,  78],
    #         [  0,   2, 235,  55],
    #         [  0,   3, 271, 112],
    #         ...,
    #         [  1,   5, 216, 149],
    #         [  1,   4, 202,  30],
    #         [  1,   6, 387, 184]]
    #  torch.Size([37453, 4])
    # 过滤掉体素范围内的点
    points_valid = points[voxel_idxs_valid_mask]  # 保留对应的点
    print("points_valid:\n", points_valid,points_valid.shape)
    # tensor([[  0.0000,  15.7723, -14.4933,  -1.3720,   0.2700],
    #         [  0.0000,  11.1916,   7.0548,  -1.8246,   0.4100],
    #         [  0.0000,  22.4759,  14.2387,  -1.6239,   0.3800],
    #         ...,
    #         [  1.0000,  29.8638,   3.3693,  -0.9490,   0.1900],
    #         [  1.0000,   6.1508,   0.5866,  -1.2506,   0.0000],
    #         [  1.0000,  36.9254,  37.4115,  -0.2031,   0.0000]]
    # torch.Size([37453, 5])
    #################################################################

    # 质心计算：对有效点和体素索引调用 get_centroid_per_voxel 函数计算质心和对应的体素索引。
    centroids_first, centroid_voxel_idxs_first, num_points_in_centroids_first = get_centroid_per_voxel(points_valid, voxel_idxs_valid)

    # 保存第一层特征位置的质心和体素索引到字典中。
    # # feature_location_first = feature_locations[0]  # x_conv3
    centroids_all[feature_location_first] = centroids_first
    centroid_voxel_idxs_all[feature_location_first] = centroid_voxel_idxs_first

    # 处理后续特征位置：迭代后续的特征位置，计算与第一层特征层相对应的质心和体素索引，其过程会使用之前计算出的质心及体素索引。
    # 将第一层特征位置的质心和体素索引存储到字典中。
    # feature_locations=[x_conv3, x_conv4]
    # multi_scale_3d_strides={x_conv1: 1，x_conv2:2 ，x_conv3: 4, x_conv4: 8}
    for feature_location in feature_locations[1:]:  # 获取特征位置列表中的所有特征位置，除了第一个特征位置。x_conv4
        # 计算当前特征层次相对于第一层特征层次的下采样比例。
        # multi_scale_3d_strides：{x_conv1: 1，x_conv2:2 ，x_conv3: 4, x_conv4: 8}
        # downsample_factor_first：8
        print("特征层\n",feature_location)
        grid_scaling = int(multi_scale_3d_strides[feature_location] / downsample_factor_first) # 8/4=2
        print("grid_scaling:",grid_scaling)

        # 克隆第一层特征层次的体素索引，用于计算当前特征层次的体素索引。
        voxel_idxs = centroid_voxel_idxs_first.clone()

        print("voxel_idxs:\n", voxel_idxs,voxel_idxs.shape)

        # voxel_idxs[:, 1:] = centroid_voxel_idxs_first[:, 1:] // grid_scaling
        # 根据下采样比例调整体素索引，以计算当前特征层次的质心和体素索引。
        # 注意：这里使用了整除操作，确保结果为整数索引。
        # torch.div https://blog.csdn.net/zhuguiqin1/article/details/120016991
        # centroid_voxel_idxs_first[:, 1:]/2
        # rounding_mode='trunc':将除法结果向零舍入
        voxel_idxs[:, 1:] = torch.div(centroid_voxel_idxs_first[:, 1:], grid_scaling, rounding_mode='trunc')
        print("centroids_first:\n", centroids_first,centroids_first.shape)
        print("voxel_idxs:\n", voxel_idxs,voxel_idxs.shape)
        # 使用调整后的体素索引和第一层特征层次的质心（这里可能是一个错误，应该使用当前层次的有效点，但代码中没有这样做），
        # 调用get_centroid_per_voxel函数计算当前特征层次的质心和体素索引。
        # 注意：这里的实现可能存在问题，因为num_points_in_centroids_first可能不适用于当前层次。
        print("num_points_in_centroids_first:\n", num_points_in_centroids_first)
        centroids, centroid_voxel_idxs, _ = get_centroid_per_voxel(centroids_first, voxel_idxs, num_points_in_centroids_first)
        # 将当前特征位置的质心和体素索引存储到字典中。

        centroids_all[feature_location] = centroids
        centroid_voxel_idxs_all[feature_location] = centroid_voxel_idxs

    # 返回包含每个特征层次的质心和体素索引的字典。
    print("所有特征层的质心和体素索引\n")
    print("centroids_all:\n", centroids_all,"centroid_voxel_idxs_all\n",centroid_voxel_idxs_all)
    return centroids_all, centroid_voxel_idxs_all

# 没有调用该函数
def get_centroids_point_per_voxel_layer(points, feature_locations, multi_scale_3d_strides, voxel_size, point_cloud_range):
    """
    Group points that lie within the same voxel together and average their xyz location.
    Details can be found here: https://discuss.pytorch.org/t/groupby-aggregate-mean-in-pytorch/45335
    Args:
        points: (N, 4 + (f)) [bxyz + (f)]
        feature_locations: [str] (Order matters! Needs to be xconv1 -> xconv4)
        multi_scale_3d_strides: (dict) Map feature_locations to stride
        voxel_size: [x_size, y_size, z_size]
        point_cloud_range: [x_min, y_min, z_min, x_max, y_max, z_max]
    Returns:
        centroids_all: (dict) Centroids for each feature_locations
        centroid_voxel_idxs_all: (dict) Centroid voxel ids for each feature_locations
    """
    assert len(points.shape) == 2
    assert len(feature_locations) > 0

    centroids_all = {}
    centroid_voxel_idxs_all = {}

    print("点云大的尺寸\n",points.shape)

    # Take first layer feature locations
    feature_location_first = feature_locations[0]
    downsample_factor_first = multi_scale_3d_strides[feature_location_first]

    # Calculate centroids
    voxel_idxs = get_overlapping_voxel_indices(points[:, 1:4],
                                               downsample_times=downsample_factor_first,
                                               voxel_size=voxel_size,
                                               point_cloud_range=point_cloud_range)
    # Add batch_idx
    voxel_idxs = torch.cat((points[:,0:1].long(), voxel_idxs), dim=-1)

    # Filter out points that are outside the valid point cloud range (invalid indices have -1)
    voxel_idxs_valid_mask = (voxel_idxs != -1).all(-1)
    voxel_idxs_valid = voxel_idxs[voxel_idxs_valid_mask]
    # Convert voxel_indices from (bxyz) to (bzyx) format for properly indexing voxelization layer
    voxel_idxs_valid = voxel_idxs_valid[:, [0,3,2,1]]
    points_valid = points[voxel_idxs_valid_mask]

    centroids_first, centroid_voxel_idxs_first, num_points_in_centroids_first = get_centroid_per_voxel(points_valid, voxel_idxs_valid)
    centroids_all[feature_location_first] = centroids_first
    centroid_voxel_idxs_all[feature_location_first] = centroid_voxel_idxs_first

    for feature_location in feature_locations[1:]:
        grid_scaling = int(multi_scale_3d_strides[feature_location] / downsample_factor_first)
        voxel_idxs = centroid_voxel_idxs_first.clone()
        # voxel_idxs[:, 1:] = centroid_voxel_idxs_first[:, 1:] // grid_scaling
        voxel_idxs[:, 1:] = torch.div(centroid_voxel_idxs_first[:, 1:], grid_scaling, rounding_mode='trunc')
        centroids, centroid_voxel_idxs, _ = get_centroid_per_voxel(centroids_first, voxel_idxs, num_points_in_centroids_first)
        centroids_all[feature_location] = centroids
        centroid_voxel_idxs_all[feature_location] = centroid_voxel_idxs

    return centroids_all, centroid_voxel_idxs_all