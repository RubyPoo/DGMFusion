import torch

# from ... import common_utils
# from . import iou3d_nms_cuda

from al3d_utils import common_utils
from al3d_utils.ops.iou3d_nms import iou3d_nms_cuda


def boxes_bev_iou_cpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:

    """
    boxes_a, is_numpy = common_utils.check_numpy_to_torch(boxes_a)
    boxes_b, is_numpy = common_utils.check_numpy_to_torch(boxes_b)
    assert not (boxes_a.is_cuda or boxes_b.is_cuda), 'Only support CPU tensors'
    assert boxes_a.shape[1] == 7 and boxes_b.shape[1] == 7
    ans_iou = boxes_a.new_zeros(torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    iou3d_nms_cuda.boxes_iou_bev_cpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou.numpy() if is_numpy else ans_iou


def boxes_iou_bev(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7
    ans_iou = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()

    iou3d_nms_cuda.boxes_iou_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), ans_iou)

    return ans_iou

# 3D IOU 计算
# 3D IoU框的计算有如下的步骤：
#  1、计算高度的重叠部分，boxes_height先计算最高处与最低处。(z轴坐标±高度的一半)
#  2、overlaps_h(height)存储高度重叠结果：min_of_max-max_of_min
#  3、计算水平方向的重叠部分，调用iou3d_nms_cuda.boxes_overlap_bev_gpu这个函数
#  4、计算overlaps_3d=水平重叠*高度重叠（交集）
#  5、无论是2D框还是3D框，计算公式都是 IoU=框的交集/框的并集，所以下一步计算框的并集
#  6、框的并集=两个框的体积之和(vol_a,vol_b)-重叠部分(overlaps_3d)
#  7、最后计算IoU_3d: iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)
# 原文链接：https://blog.csdn.net/weixin_44395365/article/details/132802022
def boxes_iou3d_gpu(boxes_a, boxes_b):
    """
    Args:
        boxes_a: (N, 7) [x, y, z, dx, dy, dz, heading]
        boxes_b: (N, 7) [x, y, z, dx, dy, dz, heading]

    Returns:
        ans_iou: (N, M)
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7

    # height overlap
    # 最高：z轴坐标+高度的一半
    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).reshape(-1, 1)
    # 最高：z轴坐标+高度的一半
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).reshape(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).reshape(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).reshape(1, -1)

    # bev overlap
    overlaps_bev = torch.cuda.FloatTensor(torch.Size((boxes_a.shape[0], boxes_b.shape[0]))).zero_()  # (N, M)
    # 调用函数计算bev的overlap
    iou3d_nms_cuda.boxes_overlap_bev_gpu(boxes_a.contiguous(), boxes_b.contiguous(), overlaps_bev)
    # 选择两个框中的最低处的更高的位置
    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    # 选择两个框中的最高处的更低的位置
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    # 计算垂直方向上的高度重叠部分
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)

    # 3d iou  水平*高度重叠部分
    overlaps_3d = overlaps_bev * overlaps_h

    # 计算两个框的体积
    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).reshape(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).reshape(1, -1)

    # 交集除以并集  分母为两个框的体积减去重叠的部分
    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)

    return iou3d


def nms_gpu(boxes, scores, thresh, pre_maxsize=None, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]
    if pre_maxsize is not None:
        order = order[:pre_maxsize]

    boxes = boxes[order].contiguous()
    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None


def nms_normal_gpu(boxes, scores, thresh, **kwargs):
    """
    :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
    :param scores: (N)
    :param thresh:
    :return:
    """
    assert boxes.shape[1] == 7
    order = scores.sort(0, descending=True)[1]

    boxes = boxes[order].contiguous()

    keep = torch.LongTensor(boxes.size(0))
    num_out = iou3d_nms_cuda.nms_normal_gpu(boxes, keep, thresh)
    return order[keep[:num_out].cuda()].contiguous(), None
