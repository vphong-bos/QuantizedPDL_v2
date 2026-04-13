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
 - 2025-08-11: Refactored to use ttnn for tensor operations - Nhat Nguyen (nhatnguyen@bos-semi.com)
"""
import copy
from math import pi, cos, sin

import torch
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from mmcv.cnn.bricks import build_activation_layer
from mmcv.cnn.bricks.transformer import build_positional_encoding
from mmdet.models.utils import build_transformer
from mmdet.models import HEADS, build_loss 
from mmdet.models.dense_heads import DETRHead
from mmcv.runner import force_fp32
from mmcv.utils import TORCH_VERSION, digit_version
from mmdet.core import build_assigner, build_sampler
from model_wrappers.model.ssr.mmdet3d_plugin.core.builder import build_bbox_coder
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init
from mmdet.core import (multi_apply, multi_apply, reduce_mean)
from mmcv.cnn.bricks.transformer import build_transformer_layer_sequence

from model_wrappers.model.ssr.mmdet3d_plugin.core.bbox.structures.utils import normalize_bbox
from model_wrappers.model.ssr.mmdet3d_plugin.SSR.utils.map_utils import (
    normalize_2d_pts, normalize_2d_bbox, denormalize_2d_pts, denormalize_2d_bbox
)
from .tokenlearner import *
from mmdet.models.utils import LearnedPositionalEncoding

class MLN(nn.Module):
    ''' 
    from "https://github.com/exiawsh/StreamPETR"
    Args:
        c_dim (int): dimension of latent code c
        f_dim (int): feature dimension
    '''

    def __init__(self, c_dim, f_dim=256, use_ln=True):
        super().__init__()
        self.c_dim = c_dim
        self.f_dim = f_dim
        self.use_ln = use_ln

        self.reduce = nn.Sequential(
            nn.Linear(c_dim, f_dim),
            nn.ReLU(),
        )
        self.gamma = nn.Linear(f_dim, f_dim)
        self.beta = nn.Linear(f_dim, f_dim)
        if self.use_ln:
            self.ln = nn.LayerNorm(f_dim, elementwise_affine=False)
        self.init_weight()

    def init_weight(self):
        nn.init.zeros_(self.gamma.weight)
        nn.init.zeros_(self.beta.weight)
        nn.init.ones_(self.gamma.bias)
        nn.init.zeros_(self.beta.bias)

    def forward(self, x, c):
        if self.use_ln:
            x = self.ln(x)
        c = self.reduce(c)
        gamma = self.gamma(c)
        beta = self.beta(c)
        out = gamma * x + beta

        return out

class SELayer(nn.Module):

    def __init__(self, channels, act_layer=nn.ReLU, gate_layer=nn.Sigmoid):
        super().__init__()
        self.mlp_reduce = nn.Linear(channels, channels)
        self.act1 = act_layer()
        self.mlp_expand = nn.Linear(channels, channels)
        self.gate = gate_layer()

    def forward(self, x, x_se):
        x_se = self.mlp_reduce(x_se)
        x_se = self.act1(x_se)
        x_se = self.mlp_expand(x_se)
        return x * self.gate(x_se)
    
@HEADS.register_module()
class SSRHead(nn.Module):
    """Head of SSR model.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """
    def __init__(self,
                 *args,
                 with_box_refine=False,
                 as_two_stage=False,
                 transformer=None,
                 bbox_coder=None,
                 num_cls_fcs=2,
                 code_weights=None,
                 bev_h=30,
                 bev_w=30,
                 fut_ts=6,
                 fut_mode=6,
                 loss_traj=dict(type='L1Loss', loss_weight=0.25),
                 loss_traj_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.8),
                 map_bbox_coder=None,
                 map_num_query=900,
                 map_num_classes=3,
                 map_num_vec=20,
                 map_num_pts_per_vec=2,
                 map_num_pts_per_gt_vec=2,
                 map_query_embed_type='all_pts',
                 map_transform_method='minmax',
                 map_gt_shift_pts_pattern='v0',
                 map_dir_interval=1,
                 map_code_size=None,
                 map_code_weights=None,
                 loss_map_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0),
                 loss_map_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_map_iou=dict(type='GIoULoss', loss_weight=2.0),
                 loss_map_pts=dict(
                    type='ChamferDistance',loss_src_weight=1.0,loss_dst_weight=1.0
                 ),
                 loss_map_dir=dict(type='PtsDirCosLoss', loss_weight=2.0),
                  num_scenes=16,
                  latent_decoder=None,
                  way_decoder=None,
                 ego_fut_mode=3,
                 loss_plan_reg=dict(type='L1Loss', loss_weight=0.25),
                 ego_lcf_feat_idx=None,
                 valid_fut_ts=6,
                 num_classes,
                 in_channels,
                 num_query=100,
                 num_reg_fcs=2,
                 sync_cls_avg_factor=False,
                 positional_encoding=dict(
                     type='SinePositionalEncoding',
                     num_feats=128,
                     normalize=True,
                 ),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     bg_cls_weight=0.1,
                     use_sigmoid=False,
                     loss_weight=1.0,
                     class_weight=1.0,
                 ),
                 loss_bbox=dict(type='L1Loss', loss_weight=5.0),
                 loss_iou=dict(type='GIoULoss', loss_weight=2.0),
                 train_cfg=dict(
                    assigner=dict(
                        type='HungarianAssigner',
                        cls_cost=dict(type='ClassificationCost', weight=1.),
                        reg_cost=dict(type='BBoxL1Cost', weight=5.0),
                        iou_cost=dict(
                            type='IoUCost', iou_mode='giou', weight=2.0))),
                 test_cfg=dict(max_per_img=100),
                 init_cfg=None,
                 **kwargs):

        super(SSRHead, self).__init__()
            
        self.bev_h = bev_h
        self.bev_w = bev_w
        self.fp16_enabled = False
        self.fut_ts = fut_ts
        self.fut_mode = fut_mode

        self.latent_decoder = latent_decoder
        self.way_decoder = way_decoder

        self.ego_fut_mode = ego_fut_mode
        self.ego_lcf_feat_idx = ego_lcf_feat_idx
        self.valid_fut_ts = valid_fut_ts
        self.num_scenes = num_scenes

        if loss_traj_cls['use_sigmoid'] == True:
            self.traj_num_cls = 1
        else:
          self.traj_num_cls = 2

        self.with_box_refine = with_box_refine
        self.as_two_stage = as_two_stage
        if self.as_two_stage:
            transformer['as_two_stage'] = self.as_two_stage
        if 'code_size' in kwargs:
            self.code_size = kwargs['code_size']
        else:
            self.code_size = 10
        if code_weights is not None:
            self.code_weights = code_weights
        else:
            self.code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]
        if map_code_size is not None:
            self.map_code_size = map_code_size
        else:
            self.map_code_size = 10
        if map_code_weights is not None:
            self.map_code_weights = map_code_weights
        else:
            self.map_code_weights = [1.0, 1.0, 1.0,
                                 1.0, 1.0, 1.0, 1.0, 1.0, 0.2, 0.2]

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.pc_range = self.bbox_coder.pc_range
        self.real_w = self.pc_range[3] - self.pc_range[0]
        self.real_h = self.pc_range[4] - self.pc_range[1]

        self.map_query_embed_type = map_query_embed_type
        self.map_num_vec = map_num_vec
        self.map_num_pts_per_vec = map_num_pts_per_vec

        if loss_map_cls['use_sigmoid'] == True:
            self.map_cls_out_channels = map_num_classes
        else:
            self.map_cls_out_channels = map_num_classes + 1
            
        self.num_query = num_query
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.num_reg_fcs = num_reg_fcs
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        
        if self.loss_cls.use_sigmoid:
            self.cls_out_channels = num_classes
        else:
            self.cls_out_channels = num_classes + 1
        self.act_cfg = transformer.get('act_cfg',
                                       dict(type='ReLU', inplace=True))
        self.activate = build_activation_layer(self.act_cfg)
        self.positional_encoding = build_positional_encoding(positional_encoding)
        self.transformer = build_transformer(transformer)
        self.embed_dims = self.transformer.embed_dims
        assert 'num_feats' in positional_encoding
        num_feats = positional_encoding['num_feats']
        
        assert num_feats * 2 == self.embed_dims, 'embed_dims should' \
            f' be exactly 2 times of num_feats. Found {self.embed_dims}' \
            f' and {num_feats}.'
        self._init_layers()
        

        self.code_weights = nn.Parameter(torch.tensor(
            self.code_weights, requires_grad=False), requires_grad=False)
        self.map_code_weights = nn.Parameter(torch.tensor(
            self.map_code_weights, requires_grad=False), requires_grad=False)

        self.loss_plan_reg = build_loss(loss_plan_reg)

    def _init_layers(self):
        """Initialize classification branch and regression branch of head."""
        cls_branch = []
        for _ in range(self.num_reg_fcs):
            cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            cls_branch.append(nn.LayerNorm(self.embed_dims))
            cls_branch.append(nn.ReLU(inplace=True))
        cls_branch.append(Linear(self.embed_dims, self.cls_out_channels))
        cls_branch = nn.Sequential(*cls_branch)

        reg_branch = []
        for _ in range(self.num_reg_fcs):
            reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            reg_branch.append(nn.ReLU())
        reg_branch.append(Linear(self.embed_dims, self.code_size))
        reg_branch = nn.Sequential(*reg_branch)

        traj_branch = []
        for _ in range(self.num_reg_fcs):
            traj_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            traj_branch.append(nn.ReLU())
        traj_branch.append(Linear(self.embed_dims*2, self.fut_ts*2))
        traj_branch = nn.Sequential(*traj_branch)

        traj_cls_branch = []
        for _ in range(self.num_reg_fcs):
            traj_cls_branch.append(Linear(self.embed_dims*2, self.embed_dims*2))
            traj_cls_branch.append(nn.LayerNorm(self.embed_dims*2))
            traj_cls_branch.append(nn.ReLU(inplace=True))
        traj_cls_branch.append(Linear(self.embed_dims*2, self.traj_num_cls))
        traj_cls_branch = nn.Sequential(*traj_cls_branch)

        map_cls_branch = []
        for _ in range(self.num_reg_fcs):
            map_cls_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_cls_branch.append(nn.LayerNorm(self.embed_dims))
            map_cls_branch.append(nn.ReLU(inplace=True))
        map_cls_branch.append(Linear(self.embed_dims, self.map_cls_out_channels))
        map_cls_branch = nn.Sequential(*map_cls_branch)

        map_reg_branch = []
        for _ in range(self.num_reg_fcs):
            map_reg_branch.append(Linear(self.embed_dims, self.embed_dims))
            map_reg_branch.append(nn.ReLU())
        map_reg_branch.append(Linear(self.embed_dims, self.map_code_size))
        map_reg_branch = nn.Sequential(*map_reg_branch)


        def _get_clones(module, N):
            return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

        # last reg_branch is used to generate proposal from
        # encode feature map when as_two_stage is True.
        num_decoder_layers = 1
        num_map_decoder_layers = 1
        if self.transformer.decoder is not None:
            num_decoder_layers = self.transformer.decoder.num_layers
        if self.transformer.map_decoder is not None:
            num_map_decoder_layers = self.transformer.map_decoder.num_layers
        num_motion_decoder_layers = 1
        num_pred = (num_decoder_layers + 1) if \
            self.as_two_stage else num_decoder_layers
        motion_num_pred = (num_motion_decoder_layers + 1) if \
            self.as_two_stage else num_motion_decoder_layers
        map_num_pred = (num_map_decoder_layers + 1) if \
            self.as_two_stage else num_map_decoder_layers

        if self.with_box_refine:
            self.cls_branches = _get_clones(cls_branch, num_pred)
            self.reg_branches = _get_clones(reg_branch, num_pred)
            self.traj_branches = _get_clones(traj_branch, motion_num_pred)
            self.traj_cls_branches = _get_clones(traj_cls_branch, motion_num_pred)
            self.map_cls_branches = _get_clones(map_cls_branch, map_num_pred)
            self.map_reg_branches = _get_clones(map_reg_branch, map_num_pred)
        else:
            self.cls_branches = nn.ModuleList(
                [cls_branch for _ in range(num_pred)])
            self.reg_branches = nn.ModuleList(
                [reg_branch for _ in range(num_pred)])
            self.traj_branches = nn.ModuleList(
                [traj_branch for _ in range(motion_num_pred)])
            self.traj_cls_branches = nn.ModuleList(
                [traj_cls_branch for _ in range(motion_num_pred)])
            self.map_cls_branches = nn.ModuleList(
                [map_cls_branch for _ in range(map_num_pred)])
            self.map_reg_branches = nn.ModuleList(
                [map_reg_branch for _ in range(map_num_pred)])

        if not self.as_two_stage:
            self.bev_embedding = nn.Embedding(
                self.bev_h * self.bev_w, self.embed_dims)
            self.query_embedding = nn.Embedding(self.num_query,
                                                self.embed_dims * 2)
            if self.map_query_embed_type == 'all_pts':
                self.map_query_embedding = nn.Embedding(self.map_num_query,
                                                    self.embed_dims * 2)
            elif self.map_query_embed_type == 'instance_pts':
                self.map_query_embedding = None
                self.map_instance_embedding = nn.Embedding(self.map_num_vec, self.embed_dims * 2)
                self.map_pts_embedding = nn.Embedding(self.map_num_pts_per_vec, self.embed_dims * 2)
        
        self.ego_query = nn.Embedding(1, self.embed_dims)	

        ego_fut_decoder = []
        ego_fut_dec_in_dim = self.embed_dims + len(self.ego_lcf_feat_idx) \
            if self.ego_lcf_feat_idx is not None else self.embed_dims
        for _ in range(self.num_reg_fcs):
            ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, ego_fut_dec_in_dim))
            ego_fut_decoder.append(nn.ReLU())
        ego_fut_decoder.append(Linear(ego_fut_dec_in_dim, 2))
        self.ego_fut_decoder = nn.Sequential(*ego_fut_decoder)
        self.navi_embedding = nn.Embedding(3, self.embed_dims)
        self.navi_se = SELayer(self.embed_dims)

        self.way_point = nn.Embedding(self.ego_fut_mode*self.fut_ts, self.embed_dims * 2)
        self.tokenlearner = TokenLearnerV11(self.num_scenes, self.embed_dims * 2)

        self.latent_decoder = build_transformer_layer_sequence(self.latent_decoder)
        self.way_decoder = build_transformer_layer_sequence(self.way_decoder)

        self.action_mln = MLN(self.fut_ts*2)
        self.pos_mln = MLN(self.fut_ts*2)

    def init_weights(self):
        """Initialize weights of the DeformDETR head."""
        self.transformer.init_weights()
        if self.latent_decoder is not None:
            for p in self.latent_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p) 
        if self.way_decoder is not None:
            for p in self.way_decoder.parameters():
                if p.dim() > 1:
                    nn.init.xavier_uniform_(p)

    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self,
                mlvl_feats,
                img_metas,
                prev_bev=None,
                only_bev=False,
                ego_his_trajs=None,
                ego_lcf_feat=None,
                cmd=None
            ):
        """Forward function.
        Args:
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        
        bs, num_cam, _, _, _ = mlvl_feats[0].shape
        dtype = mlvl_feats[0].dtype


        bev_queries = self.bev_embedding.weight.to(dtype)

        bev_mask = torch.zeros((bs, self.bev_h, self.bev_w),
                               device=bev_queries.device).to(dtype)
        bev_pos = self.positional_encoding(bev_mask).to(dtype)
        bev_embed = self.transformer.get_bev_features(
                mlvl_feats,
                bev_queries,
                self.bev_h,
                self.bev_w,
                grid_length=(self.real_h / self.bev_h,
                             self.real_w / self.bev_w),
                bev_pos=bev_pos,
                img_metas=img_metas,
                prev_bev=prev_bev,
            )
        # op.save_tensor(bev_embed, "pts/bev_embed_.pt")
        if only_bev:
            return bev_embed
        pos_embd = bev_pos.flatten(2).permute(0, 2, 1)
        
        cmd = cmd[0][0, 0, 0]
        cmd_idx = torch.nonzero(cmd)[0, 0]

        navi_embed = self.navi_embedding.weight[cmd_idx][None, None]
        # NOTE Save navi_embedding for pre-loading in ttnn model
        # op.save_tensor(self.navi_embedding.weight[:, None], "embeddings/navi_embd.pt")
        bev_navi_embed = self.navi_se(bev_embed, navi_embed)
        # op.save_tensor(bev_navi_embed, "pts/bev_navi_embed.pt")

        bev_query = torch.cat((bev_navi_embed, pos_embd), -1)
        # op.save_tensor(bev_query, "pts/bev_query.pt")

        learned_latent_query, selected = self.tokenlearner(bev_query)
        # op.save_tensor(learned_latent_query, "pts/learned_latent_query.pt")

        learned_latent_query=learned_latent_query.permute(1, 0, 2)
        latent_query, latent_pos = torch.split(
            learned_latent_query, self.embed_dims, dim=2)
        # op.save_tensor(latent_query, "pts/latent_query.pt")
        # op.save_tensor(latent_pos, "pts/latent_pos.pt")

        latent_query = self.latent_decoder(
                query=latent_query,
                key=latent_query,
                value=latent_query,
                query_pos=latent_pos,
                key_pos=latent_pos)
        # op.save_tensor(latent_query, "pts/latent_query_decoder.pt")

        way_point = self.way_point.weight.to(dtype)
        wp_pos, way_point = torch.split(
            way_point, self.embed_dims, dim=1)
        # NOTE Save embeddings for pre-loading in ttnn model
        # op.save_tensor(wp_pos, "embeddings/wp_pos.pt")
        # op.save_tensor(way_point, "embeddings/way_point.pt")

        wp_pos = wp_pos.unsqueeze(0).expand(bs, -1, -1)
        way_point = way_point.unsqueeze(0).expand(bs, -1, -1)
        wp_pos = wp_pos.permute(1, 0, 2)
        way_point = way_point.permute(1, 0, 2)

        way_point = self.way_decoder(
                query=way_point,
                key=latent_query,
                value=latent_query,
                query_pos=wp_pos,
                key_pos=latent_pos)
        # op.save_tensor(way_point, "pts/way_point.pt")

        outputs_ego_trajs = self.ego_fut_decoder(way_point)
        # op.save_tensor(outputs_ego_trajs, "pts/outputs_ego_trajs.pt")
        outputs_ego_trajs = outputs_ego_trajs.permute(1, 0, 2). view(bs, 
                                                      self.ego_fut_mode, self.fut_ts, 2)
        # op.save_tensor(outputs_ego_trajs, "pts/outputs_ego_trajs_reshape.pt")
        outputs_ego_trajs_fut=outputs_ego_trajs[:,cmd_idx,...]
        # op.save_tensor(outputs_ego_trajs_fut, "pts/outputs_ego_trajs_fut.pt")
        wp_vector = outputs_ego_trajs_fut.reshape(-1)

        wp_vector = wp_vector.unsqueeze(0).unsqueeze(0)

        # act_query = self.action_mln(latent_query, wp_vector)
        # act_pos = self.pos_mln(latent_pos[:self.num_scenes, ...], wp_vector)

        outs = {
            'bev_embed': bev_embed,
            'scene_query': latent_query,
            # 'act_query': act_query,
            # 'act_pos': act_pos,
            'ego_fut_preds': outputs_ego_trajs,
        }

        return outs

    @force_fp32(apply_to=('preds_dicts'))
    def loss(self,
             gt_bboxes_list,
             gt_labels_list,
             map_gt_bboxes_list,
             map_gt_labels_list,
             preds_dicts,
             ego_fut_gt,
             ego_fut_masks,
             ego_fut_cmd,
             gt_attr_labels,
             gt_bboxes_ignore=None,
             map_gt_bboxes_ignore=None,
             img_metas=None):
        """"Loss function.
        Args:

            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            preds_dicts:
                all_cls_scores (Tensor): Classification score of all
                    decoder layers, has shape
                    [nb_dec, bs, num_query, cls_out_channels].
                all_bbox_preds (Tensor): Sigmoid regression
                    outputs of all decode layers. Each is a 4D-tensor with
                    normalized coordinate format (cx, cy, w, h) and shape
                    [nb_dec, bs, num_query, 4].
                enc_cls_scores (Tensor): Classification scores of
                    points on encode feature map , has shape
                    (N, h*w, num_classes). Only be passed when as_two_stage is
                    True, otherwise is None.
                enc_bbox_preds (Tensor): Regression results of each points
                    on the encode feature map, has shape (N, h*w, 4). Only be
                    passed when as_two_stage is True, otherwise is None.
            gt_bboxes_ignore (list[Tensor], optional): Bounding boxes
                which can be ignored for each image. Default None.
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        ego_fut_preds = preds_dicts['ego_fut_preds']

        loss_dict = dict()

        # Planning Loss
        ego_fut_gt = ego_fut_gt.squeeze(1)
        ego_fut_masks = ego_fut_masks.squeeze(1).squeeze(1)
        ego_fut_cmd = ego_fut_cmd.squeeze(1).squeeze(1)

        ego_fut_gt = ego_fut_gt.unsqueeze(1).repeat(1, self.ego_fut_mode, 1, 1)
        loss_plan_l1_weight = ego_fut_cmd[..., None, None] * ego_fut_masks[:, None, :, None]
        loss_plan_l1_weight = loss_plan_l1_weight.repeat(1, 1, 1, 2)

        loss_plan_l1 = self.loss_plan_reg(
            ego_fut_preds,
            ego_fut_gt,
            loss_plan_l1_weight
        )
        loss_dict['loss_plan_reg'] = loss_plan_l1

        return loss_dict
