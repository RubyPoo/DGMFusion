import torch

from . import common_utils, voxel_aggregation_utils
from al3d_utils.ops.roiaware_pool3d import roiaware_pool3d_utils


def find_num_points_per_part(batch_points, batch_boxes, grid_size):
    """
    Args:
        batch_points: (N, 4)
        batch_boxes: (B, O, 7)
        grid_size: G
    Returns:
        points_per_parts: (B, O, G, G, G)
    """
    assert grid_size > 0

    batch_idx = batch_points[:, 0]
    batch_points = batch_points[:, 1:4]

    points_per_parts = []
    for i in range(batch_boxes.shape[0]):
        boxes = batch_boxes[i]
        bs_mask = (batch_idx == i)
        points = batch_points[bs_mask]
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_boxes_gpu(points.unsqueeze(0), boxes.unsqueeze(0)).squeeze(0)
        points_in_boxes_mask = box_idxs_of_pts != -1
        box_for_each_point = boxes[box_idxs_of_pts.long()][points_in_boxes_mask]
        xyz_local = points[points_in_boxes_mask] - box_for_each_point[:, 0:3]
        xyz_local = common_utils.rotate_points_along_z(
            xyz_local[:, None, :], -box_for_each_point[:, 6]
        ).squeeze(dim=1)
        # Change coordinate frame to corner instead of center of box
        xyz_local += box_for_each_point[:, 3:6] / 2
        # points_in_boxes_gpu gets points slightly outside of box, clamp values to make sure no out of index values
        xyz_local = torch.min(xyz_local, box_for_each_point[:, 3:6] - 1e-5)
        xyz_local_grid = (xyz_local // (box_for_each_point[:, 3:6] / grid_size))
        xyz_local_grid = torch.cat((box_idxs_of_pts[points_in_boxes_mask].unsqueeze(-1),
                                    xyz_local_grid), dim=-1).long()
        part_idxs, points_per_part = xyz_local_grid.unique(dim=0, return_counts=True)
        points_per_part_dense = torch.sparse_coo_tensor(part_idxs.T, points_per_part, size=(boxes.shape[0], grid_size, grid_size, grid_size)).to_dense()
        points_per_parts.append(points_per_part_dense)

    return torch.stack(points_per_parts)


def find_num_points_per_part_multi(batch_points, batch_boxes, grid_size, max_num_boxes, return_centroid=False):
    """
    计算每个多边形框中每个网格单元内的点的数量，并可选地返回这些点的质心。

    Args:
        batch_points: (N, 4) 形状的张量，其中N是点的数量，第一列是点所属的batch索引，接下来三列是点的空间坐标。
        batch_boxes: (B, O, 7) 形状的张量，其中B是batch的数量，O是每个batch中的多边形框的最大数量，7列分别是框的位置(x, y, z)，尺寸(dx, dy, dz)，以及旋转角度(theta).
        grid_size: G 网格单元的数量，用于将每个框划分为GxGxG的子网格。
        max_num_boxes: M 每个batch中多边形框的最大数量。
        return_centroid: 布尔值，如果为True，则返回每个网格单元内点的质心信息。

    Returns:
        points_per_parts: (B, O, G, G, G) 形状的张量，表示每个batch中每个框的每个网格单元内的点的数量（或质心信息）。
    """
    assert grid_size > 0, "网格大小必须大于0"

    # 提取点的batch索引和空间坐标
    batch_idx = batch_points[:, 0]
    batch_points = batch_points[:, 1:4]

    points_per_parts = []
    for i in range(batch_boxes.shape[0]):  # 遍历每个batch
        boxes = batch_boxes[i]  # 获取当前batch的所有框
        bs_mask = (batch_idx == i)  # 创建一个掩码来筛选出当前batch中的点
        points = batch_points[bs_mask]  # 筛选出当前batch中的点
        # 计算每个点所在的框索引
        box_idxs_of_pts = roiaware_pool3d_utils.points_in_multi_boxes_gpu(points.unsqueeze(0), boxes.unsqueeze(0), max_num_boxes).squeeze(0)
        box_for_each_point = boxes[box_idxs_of_pts.long()]  # 对应每个点的框信息
        xyz_local = points.unsqueeze(1) - box_for_each_point[..., 0:3]  # 将点的坐标转换为相对于框中心的局部坐标
        xyz_local_original_shape = xyz_local.shape  # 记录原始形状以便后续操作
        xyz_local = xyz_local.reshape(-1, 1, 3)  # 重塑形状以便进行旋转操作

        # 对局部坐标进行旋转，使其与框的方向对齐
        xyz_local = common_utils.rotate_points_along_z(
            xyz_local, -box_for_each_point.reshape(-1, 7)[:, 6]
        )
        # 将坐标变换到以框的一个角为原点的坐标系
        xyz_local_corner = xyz_local.reshape(xyz_local_original_shape) + box_for_each_point[..., 3:6] / 2
        # 计算点在网格中的索引
        xyz_local_grid = (xyz_local_corner / (box_for_each_point[..., 3:6] / grid_size))
        # 找出超出网格范围的点
        points_out_of_range = ((xyz_local_grid < 0) | (xyz_local_grid >= grid_size) | (xyz_local_grid.isnan())).any(-1).flatten()
        # 将框索引与网格索引合并
        xyz_local_grid = torch.cat((box_idxs_of_pts.unsqueeze(-1),
                                    xyz_local_grid), dim=-1).long()
        xyz_local_grid = xyz_local_grid.reshape(-1, xyz_local_grid.shape[-1])
        # 过滤掉超出范围和无效框索引的点
        valid_points_mask = (xyz_local_grid[:, 0] != -1) & (~points_out_of_range)
        xyz_local_grid = xyz_local_grid[valid_points_mask]

        if return_centroid:
            # 获取有效点的局部坐标
            xyz_local = xyz_local[valid_points_mask].squeeze(1)
            # 计算每个网格单元内点的质心及其数量
            centroids, part_idxs, points_per_part = voxel_aggregation_utils.get_centroid_per_voxel(xyz_local, xyz_local_grid)
            points_per_part = torch.cat((points_per_part.unsqueeze(-1), centroids), dim=-1)
            # 如果没有点在框内，返回一个空张量
            if part_idxs.shape[0] == 0:
                points_per_part_dense = torch.zeros((boxes.shape[0], grid_size, grid_size, grid_size, points_per_part.shape[-1]), dtype=points_per_part.dtype, device=points.device)
            else:
                # 将稀疏张量转换为密集张量
                points_per_part_dense = torch.sparse_coo_tensor(part_idxs.T, points_per_part, size=(boxes.shape[0], grid_size, grid_size, grid_size, points_per_part.shape[-1])).to_dense()
        else:
            # 计算每个网格单元内的点的数量
            part_idxs, points_per_part = xyz_local_grid.unique(dim=0, return_counts=True)
            # 如果没有点在框内，返回一个空张量
            if part_idxs.shape[0] == 0:
                points_per_part_dense = torch.zeros((boxes.shape[0], grid_size, grid_size, grid_size), dtype=points_per_part.dtype, device=points.device)
            else:
                # 将稀疏张量转换为密集张量
                points_per_part_dense = torch.sparse_coo_tensor(part_idxs.T, points_per_part, size=(boxes.shape[0], grid_size, grid_size, grid_size)).to_dense()

        points_per_parts.append(points_per_part_dense)

    return torch.stack(points_per_parts)  # 将所有batch的结果堆叠成一个张量
