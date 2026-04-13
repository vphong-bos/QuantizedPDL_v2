import torch 


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
    'iou3d_boxes_overlap_bev_forward'
])


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


