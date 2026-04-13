"""
 Copyright (c) Zhijia Technology. All rights reserved.
 
 Author: Peidong Li (lipeidong@smartxtruck.com / peidongl@outlook.com)
 
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
 
     http://www.apache.org/licenses/LICENSE-2.0
 
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
 
 Modifications:
 - 2025-08-11: Refactored for CPU compatibility - Nhat Nguyen (nhatnguyen@bos-semi.com)
"""
import time
import copy

import torch
from mmdet.models import DETECTORS
from mmcv.runner import force_fp32, auto_fp16
from scipy.optimize import linear_sum_assignment

from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence
from model_wrappers.model.ssr.mmdet3d_plugin.SSR.utils import builder
from model_wrappers.model.ssr.mmdet3d_plugin.SSR.planner.metric_stp3 import PlanningMetric

from .tokenlearner import TokenFuser
import torch.nn.functional as F
import torch.nn as nn

def bbox3d2result(bboxes, scores, labels, attrs=None):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
        labels (torch.Tensor): Labels with shape of (n, ).
        scores (torch.Tensor): Scores with shape of (n, ).
        attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
            Defaults to None.

    Returns:
        dict[str, torch.Tensor]: Bounding box results in cpu mode.

            - boxes_3d (torch.Tensor): 3D boxes.
            - scores (torch.Tensor): Prediction scores.
            - labels_3d (torch.Tensor): Box labels.
            - attrs_3d (torch.Tensor, optional): Box attributes.
    """
    result_dict = dict(
        boxes_3d=bboxes.to("cpu"), scores_3d=scores.cpu(), labels_3d=labels.cpu()
    )

    if attrs is not None:
        result_dict["attrs_3d"] = attrs.cpu()

    return result_dict


@DETECTORS.register_module()
class SSR(nn.Module):
    """SSR model.
    """
    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 latent_world_model=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 fut_ts=6,
                 fut_mode=6,
                 loss_bev=None
                 ):

        super(SSR, self).__init__()
        
        if pts_bbox_head:
            pts_train_cfg = train_cfg.pts if train_cfg else None
            pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = test_cfg.pts if test_cfg else None
            pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head = builder.build_head(pts_bbox_head)
        if img_backbone:
            self.img_backbone = builder.build_backbone(img_backbone)
        if img_neck is not None:
            self.img_neck = builder.build_neck(img_neck)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
           
        # NOTE: Grid Mask is off during inference 
        # self.grid_mask = GridMask(
        #     True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        # self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode
        self.valid_fut_ts = pts_bbox_head['valid_fut_ts']

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }

        self.planning_metric = None
        self.embed_dims = 256
        self.latent_world_model = latent_world_model
        self.tokenfuser = TokenFuser(16, 256)

        if self.latent_world_model is not None:
            self.latent_world_model = build_transformer_layer_sequence(self.latent_world_model)
            for p in self.latent_world_model.parameters():
                if p.dim() > 1:
                    torch.nn.init.xavier_uniform_(p)
            # self.loss_bev = build_loss(loss_bev)

    @property
    def with_img_neck(self):
        """bool: Whether the detector has a neck in image branch."""
        return hasattr(self, 'img_neck') and self.img_neck is not None
    
    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        
        if img.dim() == 6 and img.size(0) == 1:
            img.squeeze_()
        if img.dim() == 5 and img.size(0) == 1:
            img.squeeze_()
        elif img.dim() == 5 and img.size(0) > 1:
            B, N, C, H, W = img.size()
            img = img.reshape(B * N, C, H, W)
        # if self.use_grid_mask:
        #     img = self.grid_mask(img)
        
        img_feats = self.img_backbone(img)
        # op.save_tensor(img_feats[0], "pts/img_resnet.pt")
        if isinstance(img_feats, dict):
            img_feats = list(img_feats.values())
            
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
            # op.save_tensor(img_feats[0], "pts/img_fpn.pt")

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(
                    img_feat.view(int(B / len_queue), len_queue, int(BN / B), C, H, W)
                )
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'), out_fp32=True)
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats
    
    def forward(
        self,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        map_gt_bboxes_3d,
        map_gt_labels_3d,
        img=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img
        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        img_feats = self.extract_feat(img=img[0])
        # op.save_tensor(img_feats[0], "pts/img_feats.pt")
        outs = self.pts_bbox_head(
            mlvl_feats=img_feats,
            img_metas=img_metas[0],
            prev_bev=self.prev_frame_info["prev_bev"],
            cmd=ego_fut_cmd,
            ego_his_trajs=ego_his_trajs[0],
            ego_lcf_feat=ego_lcf_feat[0],
        )
        # op.save_tensor(outs['bev_embed'], "pts/bev_embed.pt")
        
        new_prev_bev, bbox_results = self.simple_test(
            outs=outs,
            img_metas=img_metas[0],
            prev_bev=self.prev_frame_info['prev_bev'],
            gt_bboxes_3d=gt_bboxes_3d,
            gt_labels_3d=gt_labels_3d,
            map_gt_bboxes_3d=map_gt_bboxes_3d,
            map_gt_labels_3d=map_gt_labels_3d,
            ego_his_trajs=ego_his_trajs[0],
            ego_fut_trajs=ego_fut_trajs[0],
            ego_fut_cmd=ego_fut_cmd[0],
            ego_lcf_feat=ego_lcf_feat[0],
            gt_attr_labels=gt_attr_labels,
            **kwargs
        )
        
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_bev'] = new_prev_bev
        self.prev_frame_info['prev_angle'] = tmp_angle

        return bbox_results

    def forward_trace(
        self,
        img_metas,
        img=None,
        ego_fut_cmd=None,
        ego_his_trajs=None,
        ego_lcf_feat=None,
    ):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError(f'{name} must be a list, but got {type(var)}')

        img = [img] if img is None else img

        img_feats = self.extract_feat(img=img[0])

        outs = self.pts_bbox_head(
            mlvl_feats=img_feats,
            img_metas=img_metas[0],
            prev_bev=None,
            cmd=ego_fut_cmd,
            ego_his_trajs=ego_his_trajs[0],
            ego_lcf_feat=ego_lcf_feat[0],
        )

        # Return only tensors / tuples of tensors
        tensors = []
        for k in sorted(outs.keys()):
            v = outs[k]
            if torch.is_tensor(v):
                tensors.append(v)
            elif isinstance(v, (list, tuple)):
                for item in v:
                    if torch.is_tensor(item):
                        tensors.append(item)
                    else:
                        raise TypeError(f"Unsupported non-tensor item in outs[{k}]: {type(item)}")
            else:
                raise TypeError(f"Unsupported output type in outs[{k}]: {type(v)}")

        return tuple(tensors)
    
    def simple_test(
        self,
        outs,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        map_gt_bboxes_3d,
        map_gt_labels_3d,
        img=None,
        prev_bev=None,
        points=None,
        fut_valid_flag=None,
        rescale=False,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
        **kwargs
    ):
        """Test function without augmentaiton."""
        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts, metric_dict = self.simple_test_pts(
            outs,
            img_metas,
            gt_bboxes_3d,
            gt_labels_3d,
            map_gt_bboxes_3d,
            map_gt_labels_3d,
            prev_bev,
            fut_valid_flag=fut_valid_flag,
            rescale=rescale,
            start=None,
            ego_his_trajs=ego_his_trajs,
            ego_fut_trajs=ego_fut_trajs,
            ego_fut_cmd=ego_fut_cmd,
            ego_lcf_feat=ego_lcf_feat,
            gt_attr_labels=gt_attr_labels,
        )
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
            result_dict['metric_results'] = metric_dict

        return new_prev_bev, bbox_list

    def simple_test_pts(
        self,
        outs,
        img_metas,
        gt_bboxes_3d,
        gt_labels_3d,
        map_gt_bboxes_3d,
        map_gt_labels_3d,
        prev_bev=None,
        fut_valid_flag=None,
        rescale=False,
        start=None,
        ego_his_trajs=None,
        ego_fut_trajs=None,
        ego_fut_cmd=None,
        ego_lcf_feat=None,
        gt_attr_labels=None,
    ):
        """Test function"""
        mapped_class_names = [
            'car', 'truck', 'construction_vehicle', 'bus',
            'trailer', 'barrier', 'motorcycle', 'bicycle', 
            'pedestrian', 'traffic_cone'
        ]

        bbox_results = []
        for i in range(len(outs['ego_fut_preds'])):
            bbox_result=dict()
            bbox_result['ego_fut_preds'] = outs['ego_fut_preds'][i].cpu()
            bbox_result['ego_fut_cmd'] = ego_fut_cmd.cpu()
            bbox_results.append(bbox_result)

        assert len(bbox_results) == 1, 'only support batch_size=1 now'
        # score_threshold = 0.6
        with torch.no_grad():
            gt_bbox = gt_bboxes_3d[0][0]
            gt_map_bbox = map_gt_bboxes_3d[0]
            gt_label = gt_labels_3d[0][0].to('cpu')
            gt_map_label = map_gt_labels_3d[0].to('cpu')
            gt_attr_label = gt_attr_labels[0][0].to('cpu')
            fut_valid_flag = bool(fut_valid_flag[0])
      
            metric_dict={}
            # ego planning metric
            assert ego_fut_trajs.shape[0] == 1, 'only support batch_size=1 for testing'
            ego_fut_preds = bbox_result['ego_fut_preds']
            ego_fut_trajs = ego_fut_trajs[0, 0]
            ego_fut_cmd = ego_fut_cmd[0, 0, 0]
            ego_fut_cmd_idx = torch.nonzero(ego_fut_cmd)[0, 0]

            ego_fut_pred = ego_fut_preds[ego_fut_cmd_idx]
            ego_fut_pred = ego_fut_pred.cumsum(dim=-2)
            ego_fut_trajs = ego_fut_trajs.cumsum(dim=-2)
            
            metric_dict_planner_stp3 = self.compute_planner_metric_stp3(
                pred_ego_fut_trajs = ego_fut_pred[None],
                gt_ego_fut_trajs = ego_fut_trajs[None],
                gt_agent_boxes = gt_bbox,
                gt_agent_feats = gt_attr_label.unsqueeze(0),
                gt_map_boxes = gt_map_bbox,
                gt_map_labels = gt_map_label,
                fut_valid_flag = fut_valid_flag
            )
            metric_dict.update(metric_dict_planner_stp3)

        return outs['bev_embed'], bbox_results, metric_dict

    def map_pred2result(self, bboxes, scores, labels, pts, attrs=None):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (torch.Tensor): Bounding boxes with shape of (n, 5).
            labels (torch.Tensor): Labels with shape of (n, ).
            scores (torch.Tensor): Scores with shape of (n, ).
            attrs (torch.Tensor, optional): Attributes with shape of (n, ). \
                Defaults to None.

        Returns:
            dict[str, torch.Tensor]: Bounding box results in cpu mode.

                - boxes_3d (torch.Tensor): 3D boxes.
                - scores (torch.Tensor): Prediction scores.
                - labels_3d (torch.Tensor): Box labels.
                - attrs_3d (torch.Tensor, optional): Box attributes.
        """
        result_dict = dict(
            map_boxes_3d=bboxes.to('cpu'),
            map_scores_3d=scores.cpu(),
            map_labels_3d=labels.cpu(),
            map_pts_3d=pts.to('cpu'))

        if attrs is not None:
            result_dict['map_attrs_3d'] = attrs.cpu()

        return result_dict

    ### same planning metric as stp3
    def compute_planner_metric_stp3(
        self,
        pred_ego_fut_trajs,
        gt_ego_fut_trajs,
        gt_agent_boxes,
        gt_agent_feats,
        gt_map_boxes,
        gt_map_labels,
        fut_valid_flag
    ):
        """Compute planner metric for one sample same as stp3."""
        metric_dict = {
            'plan_L2_1s':0,
            'plan_L2_2s':0,
            'plan_L2_3s':0,
            'plan_obj_col_1s':0,
            'plan_obj_col_2s':0,
            'plan_obj_col_3s':0,
            'plan_obj_box_col_1s':0,
            'plan_obj_box_col_2s':0,
            'plan_obj_box_col_3s':0,
            # 'plan_obj_col_plus_1s':0,
            # 'plan_obj_col_plus_2s':0,
            # 'plan_obj_col_plus_3s':0,
            # 'plan_obj_box_col_plus_1s':0,
            # 'plan_obj_box_col_plus_2s':0,
            # 'plan_obj_box_col_plus_3s':0,
        }
        metric_dict['fut_valid_flag'] = fut_valid_flag
        future_second = 3
        assert pred_ego_fut_trajs.shape[0] == 1, 'only support bs=1'
        if self.planning_metric is None:
            self.planning_metric = PlanningMetric()
        segmentation, pedestrian, segmentation_plus = self.planning_metric.get_label(
            gt_agent_boxes, gt_agent_feats, gt_map_boxes, gt_map_labels)
        occupancy = torch.logical_or(segmentation, pedestrian)

        for i in range(future_second):
            if fut_valid_flag:
                cur_time = (i+1)*2
                traj_L2 = self.planning_metric.compute_L2(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                traj_L2_stp3 = self.planning_metric.compute_L2_stp3(
                    pred_ego_fut_trajs[0, :cur_time].detach().to(gt_ego_fut_trajs.device),
                    gt_ego_fut_trajs[0, :cur_time]
                )
                obj_coll, obj_box_coll = self.planning_metric.evaluate_coll(
                    pred_ego_fut_trajs[:, :cur_time].detach(),
                    gt_ego_fut_trajs[:, :cur_time],
                    occupancy)
                # obj_coll_plus, obj_box_coll_plus = self.planning_metric.evaluate_coll(
                #     pred_ego_fut_trajs[:, :cur_time].detach(),
                #     gt_ego_fut_trajs[:, :cur_time],
                #     segmentation_plus)
                metric_dict['plan_L2_{}s'.format(i+1)] = traj_L2
                metric_dict['plan_obj_col_{}s'.format(i + 1)] = obj_coll.mean().item()
                metric_dict['plan_obj_box_col_{}s'.format(i + 1)] = obj_box_coll.mean().item()
                # metric_dict['plan_obj_col_plus_{}s'.format(i + 1)] = obj_coll_plus.mean().item()
                # metric_dict['plan_obj_box_col_plus_{}s'.format(i + 1)] = obj_box_coll_plus.mean().item()
                metric_dict['plan_L2_stp3_{}s'.format(i+1)] = traj_L2_stp3
                metric_dict['plan_obj_col_stp3_{}s'.format(i + 1)] = obj_coll[-1].item()
                metric_dict['plan_obj_box_col_stp3_{}s'.format(i + 1)] = obj_box_coll[-1].item()
                # metric_dict['plan_obj_col_stp3_plus_{}s'.format(i + 1)] = obj_coll_plus[-1].item()
                # metric_dict['plan_obj_box_col_stp3_plus_{}s'.format(i + 1)] = obj_box_coll_plus[-1].item()
                # if (i == 0):
                #     metric_dict['plan_1'] = obj_box_coll[0].item()
                #     metric_dict['plan_2'] = obj_box_coll[1].item()
                # if (i == 1):
                #     metric_dict['plan_3'] = obj_box_coll[2].item()
                #     metric_dict['plan_4'] = obj_box_coll[3].item()
                # if (i == 2):
                #     metric_dict['plan_5'] = obj_box_coll[4].item()
                #     metric_dict['plan_6'] = obj_box_coll[5].item()
            else:
                metric_dict['plan_L2_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_obj_box_col_{}s'.format(i+1)] = 0.0
                metric_dict['plan_L2_stp3_{}s'.format(i + 1)] = 0.0
            
        return metric_dict

    def set_epoch(self, epoch): 
        self.pts_bbox_head.epoch = epoch
