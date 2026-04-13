import torch 
import numpy as np


def normalize_bbox(bboxes, pc_range):

    cx = bboxes[..., 0:1]
    cy = bboxes[..., 1:2]
    cz = bboxes[..., 2:3]
    w = bboxes[..., 3:4].log()
    l = bboxes[..., 4:5].log()
    h = bboxes[..., 5:6].log()

    rot = bboxes[..., 6:7]
    if bboxes.size(-1) > 7:
        vx = bboxes[..., 7:8] 
        vy = bboxes[..., 8:9]
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos(), vx, vy), dim=-1
        )
    else:
        normalized_bboxes = torch.cat(
            (cx, cy, w, l, cz, h, rot.sin(), rot.cos()), dim=-1
        )
    return normalized_bboxes

def denormalize_bbox(normalized_bboxes, pc_range):
    # rotation 
    rot_sine = normalized_bboxes[..., 6:7]

    rot_cosine = normalized_bboxes[..., 7:8]
    rot = torch.atan2(rot_sine, rot_cosine)

    # center in the bev
    cx = normalized_bboxes[..., 0:1]
    cy = normalized_bboxes[..., 1:2]
    cz = normalized_bboxes[..., 4:5]
   
    # size
    w = normalized_bboxes[..., 2:3]
    l = normalized_bboxes[..., 3:4]
    h = normalized_bboxes[..., 5:6]

    w = w.exp() 
    l = l.exp() 
    h = h.exp() 
    if normalized_bboxes.size(-1) > 8:
         # velocity 
        vx = normalized_bboxes[:, 8:9]
        vy = normalized_bboxes[:, 9:10]
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot, vx, vy], dim=-1)
    else:
        denormalized_bboxes = torch.cat([cx, cy, cz, w, l, h, rot], dim=-1)
    return denormalized_bboxes


# Copyright (c) OpenMMLab. All rights reserved.
import warnings
from typing import Optional

import torch
from torch import Tensor

from mmcv.utils import ext_loader

ext_module = ext_loader.load_ext('_ext', [
    'iou3d_boxes_overlap_bev_forward',
    'points_in_boxes_cpu_forward'
])


def points_in_boxes_cpu(points: Tensor, boxes: Tensor) -> Tensor:
    """Find all boxes in which each point is (CPU). The CPU version of
    :meth:`points_in_boxes_all`.

    Args:
        points (torch.Tensor): [B, M, 3], [x, y, z] in
            LiDAR/DEPTH coordinate
        boxes (torch.Tensor): [B, T, 7],
            num_valid_boxes <= T, [x, y, z, x_size, y_size, z_size, rz],
            (x, y, z) is the bottom center.

    Returns:
        torch.Tensor: Return the box indices of points with the shape of
        (B, M, T). Default background = 0.
    """
    assert points.shape[0] == boxes.shape[0], \
        'Points and boxes should have the same batch size, ' \
        f'but got {points.shape[0]} and {boxes.shape[0]}'
    assert boxes.shape[2] == 7, \
        'boxes dimension should be 7, ' \
        f'but got unexpected shape {boxes.shape[2]}'
    assert points.shape[2] == 3, \
        'points dimension should be 3, ' \
        f'but got unexpected shape {points.shape[2]}'
    batch_size, num_points, _ = points.shape
    num_boxes = boxes.shape[1]

    point_indices = points.new_zeros((batch_size, num_boxes, num_points),
                                     dtype=torch.int)
    for b in range(batch_size):
        ext_module.points_in_boxes_cpu_forward(boxes[b].float().contiguous(),
                                               points[b].float().contiguous(),
                                               point_indices[b])
    point_indices = point_indices.transpose(1, 2)

    return point_indices


def boxes_overlap_bev(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Calculate boxes BEV overlap.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 7).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 7).

    Returns:
        torch.Tensor: BEV overlap result with shape (M, N).
    """
    ans_overlap = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    ext_module.iou3d_boxes_overlap_bev_forward(boxes_a.contiguous(),
                                               boxes_b.contiguous(),
                                               ans_overlap)

    return ans_overlap


def boxes_iou3d(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Calculate boxes 3D IoU.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 7).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 7).

    Returns:
        torch.Tensor: 3D IoU result with shape (M, N).
    """
    assert boxes_a.shape[1] == boxes_b.shape[1] == 7, \
        'Input boxes shape should be (N, 7)'

    boxes_a_height_max = (boxes_a[:, 2] + boxes_a[:, 5] / 2).view(-1, 1)
    boxes_a_height_min = (boxes_a[:, 2] - boxes_a[:, 5] / 2).view(-1, 1)
    boxes_b_height_max = (boxes_b[:, 2] + boxes_b[:, 5] / 2).view(1, -1)
    boxes_b_height_min = (boxes_b[:, 2] - boxes_b[:, 5] / 2).view(1, -1)

    overlaps_bev = boxes_a.new_zeros(
        torch.Size((boxes_a.shape[0], boxes_b.shape[0])))
    ext_module.iou3d_boxes_overlap_bev_forward(boxes_a.contiguous(),
                                               boxes_b.contiguous(),
                                               overlaps_bev)

    max_of_min = torch.max(boxes_a_height_min, boxes_b_height_min)
    min_of_max = torch.min(boxes_a_height_max, boxes_b_height_max)
    overlaps_h = torch.clamp(min_of_max - max_of_min, min=0)
    overlaps_3d = overlaps_bev * overlaps_h
    vol_a = (boxes_a[:, 3] * boxes_a[:, 4] * boxes_a[:, 5]).view(-1, 1)
    vol_b = (boxes_b[:, 3] * boxes_b[:, 4] * boxes_b[:, 5]).view(1, -1)
    iou3d = overlaps_3d / torch.clamp(vol_a + vol_b - overlaps_3d, min=1e-6)
    return iou3d


def _xyxyr2xywhr(boxes: Tensor) -> Tensor:
    """Convert [x1, y1, x2, y2, heading] box to [x, y, dx, dy, heading] box.

    Args:
        box (torch.Tensor): Input boxes with shape (N, 5).

    Returns:
        torch.Tensor: Converted boxes with shape (N, 7).
    """
    warnings.warn(
        'This function is deprecated and will be removed in the future.',
        DeprecationWarning)
    return torch.stack(
        ((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2,
         boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1], boxes[:, 4]),
        dim=-1)


def boxes_iou_bev(boxes_a: Tensor, boxes_b: Tensor) -> Tensor:
    """Calculate boxes IoU in the Bird's Eye View.

    Args:
        boxes_a (torch.Tensor): Input boxes a with shape (M, 5)
            ([x1, y1, x2, y2, ry]).
        boxes_b (torch.Tensor): Input boxes b with shape (N, 5)
            ([x1, y1, x2, y2, ry]).

    Returns:
        torch.Tensor: IoU result with shape (M, N).
    """
    from .box_iou_rotated import box_iou_rotated

    warnings.warn(
        '`iou3d.boxes_iou_bev` is deprecated and will be removed in'
        ' the future. Please, use `box_iou_rotated.box_iou_rotated`.',
        DeprecationWarning)

    return box_iou_rotated(_xyxyr2xywhr(boxes_a), _xyxyr2xywhr(boxes_b))


def nms_bev(boxes: Tensor,
            scores: Tensor,
            thresh: float,
            pre_max_size: Optional[int] = None,
            post_max_size: Optional[int] = None) -> Tensor:
    """NMS function GPU implementation (for BEV boxes).

    The overlap of two boxes for IoU calculation is defined as the exact
    overlapping area of the two boxes. In this function, one can also
    set ``pre_max_size`` and ``post_max_size``.

    Args:
        boxes (torch.Tensor): Input boxes with the shape of (N, 5)
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of boxes with the shape of (N,).
        thresh (float): Overlap threshold of NMS.
        pre_max_size (int, optional): Max size of boxes before NMS.
            Default: None.
        post_max_size (int, optional): Max size of boxes after NMS.
            Default: None.

    Returns:
        torch.Tensor: Indexes after NMS.
    """
    from .nms import nms_rotated

    warnings.warn(
        '`iou3d.nms_bev` is deprecated and will be removed in'
        ' the future. Please, use `nms.nms_rotated`.', DeprecationWarning)
    assert boxes.size(1) == 5, 'Input boxes shape should be (N, 5)'
    order = scores.sort(0, descending=True)[1]

    if pre_max_size is not None:
        order = order[:pre_max_size]
    boxes = _xyxyr2xywhr(boxes)[order]
    scores = scores[order]

    keep = nms_rotated(boxes, scores, thresh)[1]
    keep = order[keep]

    if post_max_size is not None:
        keep = keep[:post_max_size]
    return keep


def nms_normal_bev(boxes: Tensor, scores: Tensor, thresh: float) -> Tensor:
    """Normal NMS function GPU implementation (for BEV boxes).

    The overlap of two boxes for IoU calculation is defined as the exact
    overlapping area of the two boxes WITH their yaw angle set to 0.

    Args:
        boxes (torch.Tensor): Input boxes with shape (N, 5)
            ([x1, y1, x2, y2, ry]).
        scores (torch.Tensor): Scores of predicted boxes with shape (N,).
        thresh (float): Overlap threshold of NMS.

    Returns:
        torch.Tensor: Remaining indices with scores in descending order.
    """
    from .nms import nms

    warnings.warn(
        '`iou3d.nms_normal_bev` is deprecated and will be removed in'
        ' the future. Please, use `nms.nms`.', DeprecationWarning)
    assert boxes.shape[1] == 5, 'Input boxes shape should be (N, 5)'

    return nms(boxes[:, :-1], scores, thresh)[1]


def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (torch.Tensor): Rotated boxes in XYWHR format.

    Returns:
        torch.Tensor: Converted boxes in XYXYR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2

    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes


def limit_period(val, offset=0.5, period=np.pi):
    """Limit the value into a period for periodic function.

    Args:
        val (torch.Tensor): The value to be converted.
        offset (float, optional): Offset to set the value range. \
            Defaults to 0.5.
        period ([type], optional): Period of the value. Defaults to np.pi.

    Returns:
        torch.Tensor: Value in the range of \
            [-offset * period, (1-offset) * period]
    """
    return val - torch.floor(val / period + offset) * period


def rotation_3d_in_axis(points, angles, axis=0):
    """Rotate points by angles according to axis.

    Args:
        points (torch.Tensor): Points of shape (N, M, 3).
        angles (torch.Tensor): Vector of angles in shape (N,)
        axis (int, optional): The axis to be rotated. Defaults to 0.

    Raises:
        ValueError: when the axis is not in range [0, 1, 2], it will \
            raise value error.

    Returns:
        torch.Tensor: Rotated points in shape (N, M, 3)
    """
    rot_sin = torch.sin(angles)
    rot_cos = torch.cos(angles)
    ones = torch.ones_like(rot_cos)
    zeros = torch.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, zeros, -rot_sin]),
            torch.stack([zeros, ones, zeros]),
            torch.stack([rot_sin, zeros, rot_cos])
        ])
    elif axis == 2 or axis == -1:
        rot_mat_T = torch.stack([
            torch.stack([rot_cos, -rot_sin, zeros]),
            torch.stack([rot_sin, rot_cos, zeros]),
            torch.stack([zeros, zeros, ones])
        ])
    elif axis == 0:
        rot_mat_T = torch.stack([
            torch.stack([zeros, rot_cos, -rot_sin]),
            torch.stack([zeros, rot_sin, rot_cos]),
            torch.stack([ones, zeros, zeros])
        ])
    else:
        raise ValueError(f'axis should in range [0, 1, 2], got {axis}')

    return torch.einsum('aij,jka->aik', (points, rot_mat_T))


def xywhr2xyxyr(boxes_xywhr):
    """Convert a rotated boxes in XYWHR format to XYXYR format.

    Args:
        boxes_xywhr (torch.Tensor): Rotated boxes in XYWHR format.

    Returns:
        torch.Tensor: Converted boxes in XYXYR format.
    """
    boxes = torch.zeros_like(boxes_xywhr)
    half_w = boxes_xywhr[:, 2] / 2
    half_h = boxes_xywhr[:, 3] / 2

    boxes[:, 0] = boxes_xywhr[:, 0] - half_w
    boxes[:, 1] = boxes_xywhr[:, 1] - half_h
    boxes[:, 2] = boxes_xywhr[:, 0] + half_w
    boxes[:, 3] = boxes_xywhr[:, 1] + half_h
    boxes[:, 4] = boxes_xywhr[:, 4]
    return boxes


def get_box_type(box_type):
    """Get the type and mode of box structure.

    Args:
        box_type (str): The type of box structure.
            The valid value are "LiDAR", "Camera", or "Depth".

    Returns:
        tuple: Box type and box mode.
    """
    from .box_3d_mode import (Box3DMode, CameraInstance3DBoxes,
                              DepthInstance3DBoxes, LiDARInstance3DBoxes)
    box_type_lower = box_type.lower()
    if box_type_lower == 'lidar':
        box_type_3d = LiDARInstance3DBoxes
        box_mode_3d = Box3DMode.LIDAR
    elif box_type_lower == 'camera':
        box_type_3d = CameraInstance3DBoxes
        box_mode_3d = Box3DMode.CAM
    elif box_type_lower == 'depth':
        box_type_3d = DepthInstance3DBoxes
        box_mode_3d = Box3DMode.DEPTH
    else:
        raise ValueError('Only "box_type" of "camera", "lidar", "depth"'
                         f' are supported, got {box_type}')

    return box_type_3d, box_mode_3d


def points_cam2img(points_3d, proj_mat, with_depth=False):
    """Project points from camera coordicates to image coordinates.

    Args:
        points_3d (torch.Tensor): Points in shape (N, 3).
        proj_mat (torch.Tensor): Transformation matrix between coordinates.
        with_depth (bool, optional): Whether to keep depth in the output.
            Defaults to False.

    Returns:
        torch.Tensor: Points in image coordinates with shape [N, 2].
    """
    points_num = list(points_3d.shape)[:-1]

    points_shape = np.concatenate([points_num, [1]], axis=0).tolist()
    assert len(proj_mat.shape) == 2, 'The dimension of the projection'\
        f' matrix should be 2 instead of {len(proj_mat.shape)}.'
    d1, d2 = proj_mat.shape[:2]
    assert (d1 == 3 and d2 == 3) or (d1 == 3 and d2 == 4) or (
        d1 == 4 and d2 == 4), 'The shape of the projection matrix'\
        f' ({d1}*{d2}) is not supported.'
    if d1 == 3:
        proj_mat_expanded = torch.eye(
            4, device=proj_mat.device, dtype=proj_mat.dtype)
        proj_mat_expanded[:d1, :d2] = proj_mat
        proj_mat = proj_mat_expanded

    # previous implementation use new_zeros, new_one yeilds better results
    points_4 = torch.cat(
        [points_3d, points_3d.new_ones(*points_shape)], dim=-1)
    point_2d = torch.matmul(points_4, proj_mat.t())
    point_2d_res = point_2d[..., :2] / point_2d[..., 2:3]

    if with_depth:
        return torch.cat([point_2d_res, point_2d[..., 2:3]], dim=-1)
    return point_2d_res


def mono_cam_box2vis(cam_box):
    """This is a post-processing function on the bboxes from Mono-3D task. If
    we want to perform projection visualization, we need to:

        1. rotate the box along x-axis for np.pi / 2 (roll)
        2. change orientation from local yaw to global yaw
        3. convert yaw by (np.pi / 2 - yaw)

    After applying this function, we can project and draw it on 2D images.

    Args:
        cam_box (:obj:`CameraInstance3DBoxes`): 3D bbox in camera coordinate \
            system before conversion. Could be gt bbox loaded from dataset or \
                network prediction output.

    Returns:
        :obj:`CameraInstance3DBoxes`: Box after conversion.
    """
    warning.warn('DeprecationWarning: The hack of yaw and dimension in the '
                 'monocular 3D detection on nuScenes has been removed. The '
                 'function mono_cam_box2vis will be deprecated.')
    from . import CameraInstance3DBoxes
    assert isinstance(cam_box, CameraInstance3DBoxes), \
        'input bbox should be CameraInstance3DBoxes!'

    loc = cam_box.gravity_center
    dim = cam_box.dims
    yaw = cam_box.yaw
    feats = cam_box.tensor[:, 7:]
    # rotate along x-axis for np.pi / 2
    # see also here: https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/nuscenes_mono_dataset.py#L557  # noqa
    dim[:, [1, 2]] = dim[:, [2, 1]]
    # change local yaw to global yaw for visualization
    # refer to https://github.com/open-mmlab/mmdetection3d/blob/master/mmdet3d/datasets/nuscenes_mono_dataset.py#L164-L166  # noqa
    yaw += torch.atan2(loc[:, 0], loc[:, 2])
    # convert yaw by (-yaw - np.pi / 2)
    # this is because mono 3D box class such as `NuScenesBox` has different
    # definition of rotation with our `CameraInstance3DBoxes`
    yaw = -yaw - np.pi / 2
    cam_box = torch.cat([loc, dim, yaw[:, None], feats], dim=1)
    cam_box = CameraInstance3DBoxes(
        cam_box, box_dim=cam_box.shape[-1], origin=(0.5, 0.5, 0.5))

    return cam_box


def get_proj_mat_by_coord_type(img_meta, coord_type):
    """Obtain image features using points.

    Args:
        img_meta (dict): Meta info.
        coord_type (str): 'DEPTH' or 'CAMERA' or 'LIDAR'.
            Can be case-insensitive.

    Returns:
        torch.Tensor: transformation matrix.
    """
    coord_type = coord_type.upper()
    mapping = {'LIDAR': 'lidar2img', 'DEPTH': 'depth2img', 'CAMERA': 'cam2img'}
    assert coord_type in mapping.keys()
    return img_meta[mapping[coord_type]]
