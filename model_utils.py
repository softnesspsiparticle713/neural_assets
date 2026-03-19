# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""
PyTorch implementation of model utilities for Neural Assets.
Converted from the official Jax implementation: https://github.com/google-deepmind/neural_assets/blob/main/model_utils.py.
"""

from __future__ import annotations

import numpy as np
import torch


def flatten_batch_axes(
    x: torch.Tensor, num_data_axes: int
) -> tuple[torch.Tensor, tuple[int, ...]]:
  """Flatten all leading batch dim to only one batch dim."""
  assert num_data_axes >= 1, f'{num_data_axes=} must be >= 1'
  batch_shape = x.shape[:-num_data_axes]
  if len(batch_shape) > 1:
    x = x.reshape((int(np.prod(batch_shape)),) + x.shape[-num_data_axes:])
  return x, batch_shape


def unflatten_batch_axes(
    x: torch.Tensor, batch_shape: tuple[int, ...]
) -> torch.Tensor:
  """Undo flatten_batch_axes."""
  if len(batch_shape) > 1:
    x = x.reshape(batch_shape + x.shape[1:])
  return x


def roi_align(
    feature_map: torch.Tensor, bbox: torch.Tensor, output_size: int
) -> torch.Tensor:
  """Extracts a fixed-size feature map for an ROI from a larger feature map.

  Function adapted from
  https://github.com/google-research/scenic/blob/main/scenic/projects/owl_vit/layers.py

  Compared to the original code, we use a different bbox format here.

  Args:
    feature_map: [h, w, c] map of features from which to crop a region of
      interest.
    bbox: [y0, x0, y1, x1] normalized to (0, 1), bbox defining the region of
      interest.
    output_size: Size of the output feature map.

  Returns:
    Crop of size [output_size, output_size, c] taken from feature_map.
  """
  input_height, input_width, c = feature_map.shape
  output_height = output_width = output_size

  y0, x0, y1, x1 = torch.split(bbox, 1, dim=-1)
  w = x1 - x0
  h = y1 - y0
  eps = torch.tensor(1e-6, dtype=feature_map.dtype, device=feature_map.device)
  w = torch.maximum(w, eps)
  h = torch.maximum(h, eps)

  # Match JAX scale_and_translate's inverse warp and half-centered pixels.
  x_scale = output_width / (w * input_width)
  y_scale = output_height / (h * input_height)
  translation_y = -y0 * output_height / h
  translation_x = -x0 * output_width / w

  out_y = torch.arange(
      output_height, dtype=feature_map.dtype, device=feature_map.device
  ) + 0.5
  out_x = torch.arange(
      output_width, dtype=feature_map.dtype, device=feature_map.device
  ) + 0.5

  sample_y = (out_y[:, None] - translation_y) / y_scale
  sample_x = (out_x[None, :] - translation_x) / x_scale

  # grid_sample with align_corners=False also uses half-centered pixels.
  sample_y = 2.0 * sample_y / input_height - 1.0
  sample_x = 2.0 * sample_x / input_width - 1.0

  grid = torch.stack(
      [
          sample_x.expand(output_height, output_width),
          sample_y.expand(output_height, output_width),
      ],
      dim=-1,
  )

  # Reshape feature_map for grid_sample: [h, w, c] -> [1, c, h, w]
  feature_map_reshaped = feature_map.permute(2, 0, 1).unsqueeze(0)

  # Apply grid sampling with bilinear interpolation
  sampled = torch.nn.functional.grid_sample(
      feature_map_reshaped,
      grid.unsqueeze(0),  # Add batch dimension
      mode='bilinear',
      padding_mode='zeros',
      align_corners=False,
  )

  # Reshape back to [output_height, output_width, c]
  result = sampled.squeeze(0).permute(1, 2, 0)

  return result


def get_roi_align_features(
    bboxes: torch.Tensor,  # [B, n, 4]
    feature_maps: torch.Tensor,  # [B, h, w, c]
    size: int,
) -> torch.Tensor:  # [B, n, s, s, c]
  """Apply RoIAlign to extract fix-sized per-bbox feature maps."""
  # Handle arbitrary leading batch dim.
  bboxes, batch_shape = flatten_batch_axes(bboxes, num_data_axes=2)
  feature_maps, _ = flatten_batch_axes(
      feature_maps, num_data_axes=feature_maps.ndim - len(batch_shape)
  )
  # Apply RoIAlign.
  assert feature_maps.ndim == 4, f'Unsupported {feature_maps.shape=}'

  # Apply roi_align for each batch and bbox
  B, n = bboxes.shape[:2]
  h, w, c = feature_maps.shape[1:]

  # Preallocate output tensor
  roi_features = torch.zeros(B, n, size, size, c, device=bboxes.device, dtype=feature_maps.dtype)

  # Loop over batch
  for b in range(B):
    # Loop over bboxes
    for i in range(n):
      roi_features[b, i] = roi_align(feature_maps[b], bboxes[b, i], size)

  return unflatten_batch_axes(roi_features, batch_shape=batch_shape)


def boxes_to_sparse_segmentations(
    boxes: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
  """Converts bounding boxes into sparse segmentations.

  Args:
    boxes: A bounding box tensor of shape [..., N, 4] in TF format (i.e.,
      normalized coordinates [ymin, xmin, ymax, xmax] in [0, 1]^4.
    height: The frame height.
    width: The frame width.

  Returns:
    A sparse segmentations tensor of shape [..., N, H, W].
  """
  batch_shape = boxes.shape[:-2]
  n = boxes.shape[-2]
  boxes = boxes.reshape(-1, n, 4)
  # Convert the normalized into absolute coordinates.
  scale = torch.tensor([height, width, height, width],
                       dtype=torch.float32,
                       device=boxes.device)
  boxes_absolute = torch.round(boxes * scale[None, None, :]).to(torch.int32)
  ymin, xmin, ymax, xmax = torch.split(boxes_absolute, 1, dim=-1)
  # Generate yx-grid and mask the boxes.
  grid_y, grid_x = torch.meshgrid(
      torch.arange(0, height, device=boxes.device),
      torch.arange(0, width, device=boxes.device),
      indexing='ij'
  )
  sparse_segmentations = torch.logical_and(
      torch.logical_and(
          grid_y >= ymin[..., None],
          grid_y < ymax[..., None],
      ),
      torch.logical_and(
          grid_x >= xmin[..., None],
          grid_x < xmax[..., None],
      ),
  )
  sparse_segmentations = sparse_segmentations.reshape(
      *batch_shape, n, height, width
  ).to(torch.uint8)
  return sparse_segmentations
