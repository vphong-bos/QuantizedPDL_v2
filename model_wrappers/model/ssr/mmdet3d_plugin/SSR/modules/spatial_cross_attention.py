import math
import time
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import constant_init, xavier_init
from mmcv.cnn.bricks.registry import (ATTENTION, TRANSFORMER_LAYER,
                                      TRANSFORMER_LAYER_SEQUENCE)
from mmcv.cnn.bricks.transformer import build_attention
from mmcv.ops.multi_scale_deform_attn import multi_scale_deformable_attn_pytorch
from mmcv.runner import auto_fp16, force_fp32

try:
    import matplotlib

    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - optional dependency
    matplotlib = None
    plt = None

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None

try:
    import imageio.v2 as imageio
except ImportError:  # pragma: no cover - optional dependency
    try:
        import imageio  # type: ignore
    except ImportError:
        imageio = None

from mmcv.runner.base_module import BaseModule, ModuleList, Sequential

from mmcv.utils import ext_loader
from .multi_scale_deformable_attn_function import MultiScaleDeformableAttnFunction_fp32


def save_indices(index_query_per_img: torch.Tensor, save_path: Path) -> None:
    """Persist the provided indices tensor for offline analysis."""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(index_query_per_img.detach().cpu(), save_path)


def build_heatmap_array(indexes: torch.Tensor, bev_size, heatmap=None) -> np.ndarray:
    """Compute normalized heatmap values for provided BEV indices."""
    height, width = bev_size
    if heatmap is None:
        heatmap = np.zeros((height, width), dtype=np.float32)

    if indexes is not None and indexes.numel() > 0:
        flat_idx = indexes.detach().to(torch.long).cpu().numpy()
        flat_idx = flat_idx[(flat_idx >= 0) & (flat_idx < height * width)]
        if flat_idx.size > 0:
            row_idx = flat_idx // width
            col_idx = flat_idx % width
            np.add.at(heatmap, (row_idx, col_idx), 1)

    if heatmap.max() > 0:
        heatmap_ = heatmap / (heatmap.max() + 1e-8)

    return np.clip(heatmap_, 0.0, 1.0)


def heatmap_to_frame(heatmap: np.ndarray, color=None) -> np.ndarray:
    """Render a heatmap array into an RGB image."""
    if color is not None:
        color_array = np.asarray(color, dtype=np.float32)
        if color_array.max() > 1.0:
            color_array = color_array / 255.0
        frame = (heatmap[..., None] * color_array[None, None, :]) * 255.0
        frame = np.clip(frame, 0, 255).astype(np.uint8)
        return frame

    if plt is not None:
        cmap = plt.get_cmap('gray')
        colored = cmap(heatmap)[..., :3]
        frame = (colored * 255).astype(np.uint8)
        return frame

    heatmap_uint8 = (heatmap * 255).astype(np.uint8)
    return np.stack([heatmap_uint8] * 3, axis=-1)


def build_heatmap_frame(indexes: torch.Tensor, bev_size, color=None) -> np.ndarray:
    """Convert active BEV indices into a colored heatmap frame."""
    heatmap = build_heatmap_array(indexes, bev_size)
    return heatmap_to_frame(heatmap, color)


@ATTENTION.register_module()
class SpatialCrossAttention(BaseModule):
    """An attention module used in BEVFormer.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_cams (int): The number of cameras
        dropout (float): A Dropout layer on `inp_residual`.
            Default: 0..
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        deformable_attention: (dict): The config for the deformable attention used in SCA.
    """

    def __init__(self,
                 embed_dims=256,
                 num_cams=6,
                 pc_range=None,
                 dropout=0.1,
                 init_cfg=None,
                 batch_first=False,
                 deformable_attention=dict(
                     type='MSDeformableAttention3D',
                     embed_dims=256,
                     num_levels=4),
                 bev_size=(100, 100),
                 debug_save=True,
                 debug_log_dir='logs',
                 debug_video_fps=10,
                 debug_video_format='mp4',
                 debug_color_labels=True,
                 **kwargs
                 ):
        super(SpatialCrossAttention, self).__init__(init_cfg)

        self.init_cfg = init_cfg
        self.dropout = nn.Dropout(dropout)
        self.pc_range = pc_range
        self.fp16_enabled = False
        self.deformable_attention = build_attention(deformable_attention)
        self.embed_dims = embed_dims
        self.num_cams = num_cams
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.batch_first = batch_first
        self.bev_size = bev_size
        self.debug_save = debug_save
        self.debug_log_dir = Path(debug_log_dir)
        self.debug_video_fps = debug_video_fps
        self.debug_video_format = debug_video_format
        self.debug_color_labels = debug_color_labels
        self._run_timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        self._run_timestamp_ms = int(time.time() * 1000)
        self._camera_labels = {
            0: "CAM 0 (BACK)",
            1: "CAM 1 (BACK-RIGHT)",
            2: "CAM 2 (BACK-LEFT)",
            3: "CAM 3 (FRONT)",
            4: "CAM 4 (FRONT-LEFT)",
            5: "CAM 5 (FRONT-RIGHT)",
        }
        if debug_color_labels:
            base_palette = (
                (0.9, 0.1, 0.1),
                (0.9, 0.5, 0.1),
                (0.1, 0.6, 0.2),
                (0.2, 0.4, 0.9),
                (0.7, 0.1, 0.7),
                (0.1, 0.8, 0.8),
            )
            self._camera_heatmap_colors = {
                cam_idx: base_palette[cam_idx % len(base_palette)] for cam_idx in range(self.num_cams)
            }
            self._camera_text_colors = {
                cam_idx: self._camera_heatmap_colors[cam_idx] for cam_idx in range(self.num_cams)
            }
        else:
            self._camera_heatmap_colors = {cam_idx: None for cam_idx in range(self.num_cams)}
            self._camera_text_colors = {cam_idx: (1.0, 1.0, 1.0) for cam_idx in range(self.num_cams)}
        self._combined_layout = [
            [4, 3, 5],
            [2, 0, 1],
        ]
        self._video_backend = None
        if cv2 is not None:
            self._video_backend = 'cv2'
        elif imageio is not None:
            self._video_backend = 'imageio'
        self._debug_iter = 0
        self._debug_session_id = f"session_{self._run_timestamp_ms}"
        self._debug_dirs_ready = False
        self._video_writers = {}
        self._combined_video_writers = {}
        self._integrated_video_writers = {}
        self._label_patches = {}
        self._heatmaps_list = [np.zeros(bev_size) for _ in range(num_cams)]
        self.init_weight()

    def init_weight(self):
        """Default initialization for Parameters of Module."""
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
    
    @force_fp32(apply_to=('query', 'key', 'value', 'query_pos', 'reference_points_cam'))
    def forward(self,
                query,
                key,
                value,
                residual=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                reference_points_cam=None,
                bev_mask=None,
                level_start_index=None,
                flag='encoder',
                **kwargs):
        """Forward Function of Detr3DCrossAtten.
        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`. (B, N, C, H, W)
            residual (Tensor): The tensor used for addition, with the
                same shape as `x`. Default None. If None, `x` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for  `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, 4),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different level. With shape  (num_levels, 2),
                last dimension represent (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape (num_levels) and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if key is None:
            key = query
        if value is None:
            value = key

        if residual is None:
            inp_residual = query
            slots = torch.zeros_like(query)
        if query_pos is not None:
            query = query + query_pos

        bs, num_query, _ = query.size()

        log_suffix = None
        combined_frames = None
        indices_dir = None
        # if self.debug_save:
        #     if self._video_backend is None:
        #         raise RuntimeError(
        #             "Video logging requires either OpenCV (cv2) or imageio; neither dependency is available.")
        #     if not self._debug_dirs_ready:
        #         (self.debug_log_dir / 'heatmaps').mkdir(parents=True, exist_ok=True)
        #         (self.debug_log_dir / 'index_queries').mkdir(parents=True, exist_ok=True)
        #         self._debug_dirs_ready = True
        #     self._debug_iter += 1
        #     log_suffix = f"{self._run_timestamp}_{self._run_timestamp_ms}_iter_{self._debug_iter:06d}"
        #     combined_frames = {}
        #     integrated_accumulators = {}
        #     indices_dir = self.debug_log_dir / 'index_queries'
        # elif self._video_writers or self._combined_video_writers:
        self.close_debug_writers()

        D = reference_points_cam.size(3)
        indexes = []
        for cam_idx, mask_per_img in enumerate(bev_mask):
            index_query_per_img = mask_per_img[0].sum(-1).nonzero().squeeze(-1)
            indexes.append(index_query_per_img)
            if self.debug_save:
                mask_sum = mask_per_img.detach().sum(-1)
                if mask_sum.dim() == 1:
                    mask_sum = mask_sum.unsqueeze(0)
                for batch_idx in range(mask_sum.size(0)):
                    idx_tensor = mask_sum[batch_idx].nonzero(as_tuple=False).view(-1)
                    if indices_dir is not None and log_suffix is not None:
                        indices_path = indices_dir / f"sample_{batch_idx}_cam_{cam_idx}_{log_suffix}.pt"
                        save_indices(idx_tensor, indices_path)
                    heatmap_arr = build_heatmap_array(idx_tensor, self.bev_size, self._heatmaps_list[cam_idx])
                    color = self._camera_heatmap_colors.get(cam_idx)
                    frame = heatmap_to_frame(heatmap_arr, color)
                    self._append_heatmap_frame(cam_idx, batch_idx, frame)
                    if combined_frames is not None:
                        combined_frames.setdefault(batch_idx, {})[cam_idx] = frame
                        acc = integrated_accumulators.setdefault(
                            batch_idx,
                            np.zeros((self.bev_size[0], self.bev_size[1], 3), dtype=np.float32))
                        color_vec = np.asarray(color if color is not None else (1.0, 1.0, 1.0), dtype=np.float32)
                        if color_vec.max() > 1.0:
                            color_vec = color_vec / 255.0
                        acc += heatmap_arr[..., None] * color_vec[None, None, :] * 255.0
        max_len = max([len(each) for each in indexes])

        if self.debug_save and combined_frames:
            for batch_idx, cam_frames in combined_frames.items():
                combined_frame = self._build_combined_frame(cam_frames)
                self._append_combined_frame(batch_idx, combined_frame)
                integrated_tensor = integrated_accumulators.get(batch_idx)
                if integrated_tensor is not None:
                    integrated_frame = self._build_integrated_frame(integrated_tensor)
                    self._append_integrated_frame(batch_idx, integrated_frame)

        # each camera only interacts with its corresponding BEV queries. This step can  greatly save GPU memory.
        queries_rebatch = query.new_zeros(
            [bs, self.num_cams, max_len, self.embed_dims])
        reference_points_rebatch = reference_points_cam.new_zeros(
            [bs, self.num_cams, max_len, D, 2])
        
        for j in range(bs):
            for i, reference_points_per_img in enumerate(reference_points_cam):   
                index_query_per_img = indexes[i]
                
                queries_rebatch[j, i, :len(index_query_per_img)] = query[j, index_query_per_img]
                reference_points_rebatch[j, i, :len(index_query_per_img)] = reference_points_per_img[j, index_query_per_img]

        num_cams, l, bs, embed_dims = key.shape

        key = key.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)
        value = value.permute(2, 0, 1, 3).reshape(
            bs * self.num_cams, l, self.embed_dims)

        queries = self.deformable_attention(query=queries_rebatch.view(bs*self.num_cams, max_len, self.embed_dims), key=key, value=value,
                                            reference_points=reference_points_rebatch.view(bs*self.num_cams, max_len, D, 2), spatial_shapes=spatial_shapes,
                                            level_start_index=level_start_index).view(bs, self.num_cams, max_len, self.embed_dims)
        for j in range(bs):
            for i, index_query_per_img in enumerate(indexes):
                slots[j, index_query_per_img] += queries[j, i, :len(index_query_per_img)]

        count = bev_mask.sum(-1) > 0
        count = count.permute(1, 2, 0).sum(-1)
        count = torch.clamp(count, min=1.0)
        slots = slots / count[..., None]
        slots = self.output_proj(slots)

        return self.dropout(slots) + inp_residual

    def _build_combined_frame(self, cam_frames: dict) -> np.ndarray:
        if not cam_frames:
            height, width = self.bev_size
            return np.zeros((height, width, 3), dtype=np.uint8)

        sample_frame = next(iter(cam_frames.values()))
        frame_height, frame_width = sample_frame.shape[0], sample_frame.shape[1]
        spacing = max(4, frame_height // 20)
        rows = len(self._combined_layout)
        cols = len(self._combined_layout[0]) if rows > 0 else 0
        combined_height = rows * frame_height + (rows + 1) * spacing
        combined_width = cols * frame_width + (cols + 1) * spacing
        canvas = np.zeros((combined_height, combined_width, 3), dtype=np.uint8)

        for row_idx, row in enumerate(self._combined_layout):
            for col_idx, cam_idx in enumerate(row):
                y = spacing + row_idx * (frame_height + spacing)
                x = spacing + col_idx * (frame_width + spacing)
                frame = cam_frames.get(cam_idx)
                if frame is None:
                    frame = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                elif frame.shape[0] != frame_height or frame.shape[1] != frame_width:
                    frame = self._resize_frame(frame, frame_width, frame_height)
                canvas[y:y + frame_height, x:x + frame_width] = frame
                self._overlay_label(canvas, cam_idx, x, y, frame_width)

        return canvas

    def _resize_frame(self, frame: np.ndarray, width: int, height: int) -> np.ndarray:
        if frame.shape[0] == height and frame.shape[1] == width:
            return frame
        if cv2 is not None:
            return cv2.resize(frame, (width, height), interpolation=cv2.INTER_NEAREST)
        scale_y = height / frame.shape[0]
        scale_x = width / frame.shape[1]
        if abs(scale_y - round(scale_y)) < 1e-6 and abs(scale_x - round(scale_x)) < 1e-6:
            repeat_y = max(1, int(round(scale_y)))
            repeat_x = max(1, int(round(scale_x)))
            return np.repeat(np.repeat(frame, repeat_y, axis=0), repeat_x, axis=1)[:height, :width]
        resized = np.zeros((height, width, 3), dtype=np.uint8)
        min_h = min(height, frame.shape[0])
        min_w = min(width, frame.shape[1])
        resized[:min_h, :min_w] = frame[:min_h, :min_w]
        return resized

    def _overlay_label(self, canvas: np.ndarray, cam_idx: int, x: int, y: int, frame_width: int) -> None:
        patch = self._get_label_patch(cam_idx, frame_width)
        if patch is None:
            return
        patch_h, patch_w, _ = patch.shape
        offset = max(2, patch_h // 10)
        y_start = min(canvas.shape[0] - patch_h, max(0, y + offset))
        x_start = min(canvas.shape[1] - patch_w, max(0, x + offset))
        y_end = y_start + min(patch_h, canvas.shape[0] - y_start)
        x_end = x_start + min(patch_w, canvas.shape[1] - x_start)
        canvas[y_start:y_end, x_start:x_end] = patch[:y_end - y_start, :x_end - x_start]

    def _get_label_patch(self, cam_idx: int, frame_width: int) -> np.ndarray:
        cache_key = (cam_idx, frame_width)
        if cache_key in self._label_patches:
            return self._label_patches[cache_key]
        dpi = 100
        label_width = max(60, min(frame_width - 10, int(frame_width * 0.75)))
        label_height = max(14, min(int(frame_width * 0.12), self.bev_size[0] // 4 if self.bev_size[0] else 40))
        text_color = self._camera_text_colors.get(cam_idx, (1.0, 1.0, 1.0))
        label_text = self._camera_labels.get(cam_idx, f"CAM {cam_idx}")

        if cv2 is not None:
            patch_bgr = np.zeros((label_height, label_width, 3), dtype=np.uint8)
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = max(0.25, min(label_height / 64.0, 0.8))
            thickness = max(1, int(round(label_height / 28.0)))
            text_dims, baseline = cv2.getTextSize(label_text, font, font_scale, thickness)
            text_width, text_height = text_dims
            text_x = max(0, (label_width - text_width) // 2)
            text_y = max(text_height + baseline, (label_height + text_height) // 2)
            text_color_bgr = np.clip(np.array(text_color[::-1]) * 255.0, 0, 255).astype(np.uint8).tolist()
            cv2.putText(
                patch_bgr,
                label_text,
                (text_x, min(text_y, label_height - baseline)),
                font,
                font_scale,
                tuple(int(c) for c in text_color_bgr),
                thickness,
                cv2.LINE_AA,
            )
            patch_rgb = patch_bgr[:, :, ::-1]
            self._label_patches[cache_key] = patch_rgb
            return patch_rgb

        if plt is None:
            return None

        fig, ax = plt.subplots(figsize=(label_width / dpi, label_height / dpi), dpi=dpi)
        fig.patch.set_facecolor('black')
        fig.patch.set_alpha(1.0)
        ax.set_facecolor('black')
        ax.text(
            0.5,
            0.5,
            label_text,
            ha='center',
            va='center',
            color=text_color,
            fontsize=max(3, int(label_height * 0.25)),
            fontweight='bold',
        )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.axis('off')
        fig.subplots_adjust(0, 0, 1, 1)
        canvas = fig.canvas
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        image = buf.reshape(canvas.get_width_height()[::-1] + (4,))[:, :, :3].copy()
        plt.close(fig)
        self._label_patches[cache_key] = image
        return image

    def _resolve_fourcc(self) -> str:
        format_lower = self.debug_video_format.lower()
        if format_lower in ('mp4', 'm4v', 'mov'):
            return 'mp4v'
        if format_lower in ('avi', 'xvid'):
            return 'XVID'
        return 'mp4v'

    def _get_video_writer(self, cam_idx: int, batch_idx: int):
        # if not self.debug_save:
        #     return None
        
        key = (cam_idx, batch_idx)
        writer = self._video_writers.get(key)
        if writer is not None:
            return writer

        cam_dir = self.debug_log_dir / 'heatmaps' / f'CAM{cam_idx}'
        cam_dir.mkdir(parents=True, exist_ok=True)
        filename = f"sample_{batch_idx}_{self._debug_session_id}.{self.debug_video_format}"
        video_path = cam_dir / filename

        if self._video_backend == 'cv2':
            fourcc = cv2.VideoWriter_fourcc(*self._resolve_fourcc())
            writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                self.debug_video_fps,
                (self.bev_size[1], self.bev_size[0]))
            if not writer.isOpened():
                raise RuntimeError(f'Failed to open video writer at {video_path}.')
        elif self._video_backend == 'imageio':
            writer = imageio.get_writer(
                str(video_path),
                mode='I',
                fps=self.debug_video_fps)
        else:  # pragma: no cover - defensive
            raise RuntimeError('Unsupported video backend configuration.')

        self._video_writers[key] = writer
        return writer

    def _append_heatmap_frame(self, cam_idx: int, batch_idx: int, frame: np.ndarray) -> None:        
        pass
        # if frame.ndim == 2:
        #     frame = np.stack([frame] * 3, axis=-1)
        # if frame.dtype != np.uint8:
        #     frame = np.clip(frame, 0, 255).astype(np.uint8)

        # writer = self._get_video_writer(cam_idx, batch_idx)

        # # if writer is None:
        # #     return

        # if self._video_backend == 'cv2':
        #     if frame.shape[0] != self.bev_size[0] or frame.shape[1] != self.bev_size[1]:
        #         frame = cv2.resize(frame, (self.bev_size[1], self.bev_size[0]), interpolation=cv2.INTER_NEAREST)
        #     frame_bgr = np.ascontiguousarray(frame[:, :, ::-1])
        #     writer.write(frame_bgr)
        # else:
        #     writer.append_data(np.ascontiguousarray(frame))

    def _get_combined_writer(self, batch_idx: int, width: int, height: int):
        # if not self.debug_save:
        #     return None     
        
        writer = self._combined_video_writers.get(batch_idx)
        if writer is not None:
            return writer
        combined_dir = self.debug_log_dir / 'heatmaps'
        combined_dir.mkdir(parents=True, exist_ok=True)
        filename = f"combine_sample_{batch_idx}_{self._debug_session_id}.{self.debug_video_format}"
        video_path = combined_dir / filename
        if self._video_backend == 'cv2':
            fourcc = cv2.VideoWriter_fourcc(*self._resolve_fourcc())
            writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                self.debug_video_fps,
                (width, height))
            if not writer.isOpened():
                raise RuntimeError(f'Failed to open combined video writer at {video_path}.')
        elif self._video_backend == 'imageio':
            writer = imageio.get_writer(
                str(video_path),
                mode='I',
                fps=self.debug_video_fps)
        else:  # pragma: no cover - defensive
            raise RuntimeError('Unsupported video backend configuration.')
        self._combined_video_writers[batch_idx] = writer
        return writer

    def _append_combined_frame(self, batch_idx: int, frame: np.ndarray) -> None:
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        height, width = frame.shape[0], frame.shape[1]
        writer = self._get_combined_writer(batch_idx, width, height)

        # if writer is None:
        #     return
        
        if self._video_backend == 'cv2':
            frame_bgr = np.ascontiguousarray(frame[:, :, ::-1])
            writer.write(frame_bgr)
        else:
            writer.append_data(np.ascontiguousarray(frame))

    def _build_integrated_frame(self, accumulator: np.ndarray) -> np.ndarray:
        return np.clip(accumulator, 0, 255).astype(np.uint8)

    def _get_integrated_writer(self, batch_idx: int):
        # if not self.debug_save:
        #     return None
        
        writer = self._integrated_video_writers.get(batch_idx)
        if writer is not None:
            return writer
        integrated_dir = self.debug_log_dir / 'heatmaps'
        integrated_dir.mkdir(parents=True, exist_ok=True)
        filename = f"integrated_sample_{batch_idx}_{self._debug_session_id}.{self.debug_video_format}"
        video_path = integrated_dir / filename
        if self._video_backend == 'cv2':
            fourcc = cv2.VideoWriter_fourcc(*self._resolve_fourcc())
            writer = cv2.VideoWriter(
                str(video_path),
                fourcc,
                self.debug_video_fps,
                (self.bev_size[1], self.bev_size[0]),
            )
            if not writer.isOpened():
                raise RuntimeError(f'Failed to open integrated video writer at {video_path}.')
        elif self._video_backend == 'imageio':
            writer = imageio.get_writer(
                str(video_path),
                mode='I',
                fps=self.debug_video_fps,
            )
        else:  # pragma: no cover - defensive
            raise RuntimeError('Unsupported video backend configuration.')
        self._integrated_video_writers[batch_idx] = writer
        return writer

    def _append_integrated_frame(self, batch_idx: int, frame: np.ndarray) -> None:        
        if frame.ndim == 2:
            frame = np.stack([frame] * 3, axis=-1)
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        writer = self._get_integrated_writer(batch_idx)

        # if writer is None:
        #     return
        
        if self._video_backend == 'cv2':
            frame_bgr = np.ascontiguousarray(frame[:, :, ::-1])
            writer.write(frame_bgr)
        else:
            writer.append_data(np.ascontiguousarray(frame))

    def close_debug_writers(self) -> None:
        if self._video_writers:
            for writer in self._video_writers.values():
                try:
                    if self._video_backend == 'cv2':
                        writer.release()
                    else:
                        writer.close()
                except Exception:
                    continue
            self._video_writers.clear()
        if self._combined_video_writers:
            for writer in self._combined_video_writers.values():
                try:
                    if self._video_backend == 'cv2':
                        writer.release()
                    else:
                        writer.close()
                except Exception:
                    continue
            self._combined_video_writers.clear()
        if self._integrated_video_writers:
            for writer in self._integrated_video_writers.values():
                try:
                    if self._video_backend == 'cv2':
                        writer.release()
                    else:
                        writer.close()
                except Exception:
                    continue
            self._integrated_video_writers.clear()

    def __del__(self):
        try:
            self.close_debug_writers()
        except Exception:
            pass


@ATTENTION.register_module()
class MSDeformableAttention3D(BaseModule):
    """An attention module used in BEVFormer based on Deformable-Detr.
    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.
    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to False.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=8,
                 im2col_step=64,
                 dropout=0.1,
                 batch_first=True,
                 norm_cfg=None,
                 init_cfg=None):
        super().__init__(init_cfg)
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads
        self.norm_cfg = norm_cfg
        self.batch_first = batch_first
        self.output_proj = None
        self.fp16_enabled = False

        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.sampling_offsets = nn.Linear(
            embed_dims, num_heads * num_levels * num_points * 2)
        self.attention_weights = nn.Linear(embed_dims,
                                           num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)

        self.init_weights()

    def init_weights(self):
        """Default initialization for Parameters of Module."""
        constant_init(self.sampling_offsets, 0.)
        thetas = torch.arange(
            self.num_heads,
            dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init /
                     grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1,
            2).repeat(1, self.num_levels, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        constant_init(self.attention_weights, val=0., bias=0.)
        xavier_init(self.value_proj, distribution='uniform', bias=0.)
        xavier_init(self.output_proj, distribution='uniform', bias=0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None,
                **kwargs):
        """Forward Function of MultiScaleDeformAttention.
        Args:
            query (Tensor): Query of Transformer with shape
                ( bs, num_query, embed_dims).
            key (Tensor): The key tensor with shape
                `(bs, num_key,  embed_dims)`.
            value (Tensor): The value tensor with shape
                `(bs, num_key,  embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].
        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """

        if value is None:
            value = query
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos

        if not self.batch_first:
            # change to (bs, num_query ,embed_dims)
            query = query.permute(1, 0, 2)
            value = value.permute(1, 0, 2)

        bs, num_query, _ = query.shape
        bs, num_value, _ = value.shape
        assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value

        value = self.value_proj(value)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], 0.0)
        value = value.view(bs, num_value, self.num_heads, -1)
        sampling_offsets = self.sampling_offsets(query).view(
            bs, num_query, self.num_heads, self.num_levels, self.num_points, 2)
        attention_weights = self.attention_weights(query).view(
            bs, num_query, self.num_heads, self.num_levels * self.num_points)

        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bs, num_query,
                                                   self.num_heads,
                                                   self.num_levels,
                                                   self.num_points)

        if reference_points.shape[-1] == 2:
            """
            For each BEV query, it owns `num_Z_anchors` in 3D space that having different heights.
            After proejcting, each BEV query has `num_Z_anchors` reference points in each 2D image.
            For each referent point, we sample `num_points` sampling points.
            For `num_Z_anchors` reference points,  it has overall `num_points * num_Z_anchors` sampling points.
            """
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], -1)

            bs, num_query, num_Z_anchors, xy = reference_points.shape
            reference_points = reference_points[:, :, None, None, None, :, :]
            sampling_offsets = sampling_offsets / \
                offset_normalizer[None, None, None, :, None, :]
            bs, num_query, num_heads, num_levels, num_all_points, xy = sampling_offsets.shape
            sampling_offsets = sampling_offsets.view(
                bs, num_query, num_heads, num_levels, num_all_points // num_Z_anchors, num_Z_anchors, xy)
            sampling_locations = reference_points + sampling_offsets
            bs, num_query, num_heads, num_levels, num_points, num_Z_anchors, xy = sampling_locations.shape
            assert num_all_points == num_points * num_Z_anchors

            sampling_locations = sampling_locations.view(
                bs, num_query, num_heads, num_levels, num_all_points, xy)

        elif reference_points.shape[-1] == 4:
            assert False
        else:
            raise ValueError(
                f'Last dim of reference_points must be'
                f' 2 or 4, but get {reference_points.shape[-1]} instead.')

        #  sampling_locations.shape: bs, num_query, num_heads, num_levels, num_all_points, 2
        #  attention_weights.shape: bs, num_query, num_heads, num_levels, num_all_points
        #

        if torch.cuda.is_available() and value.is_cuda:
            if value.dtype == torch.float16:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            else:
                MultiScaleDeformableAttnFunction = MultiScaleDeformableAttnFunction_fp32
            output = MultiScaleDeformableAttnFunction.apply(
                value, spatial_shapes, level_start_index, sampling_locations,
                attention_weights, self.im2col_step)
        else:
            output = multi_scale_deformable_attn_pytorch(
                value, spatial_shapes, sampling_locations, attention_weights)
        if not self.batch_first:
            output = output.permute(1, 0, 2)

        return output
