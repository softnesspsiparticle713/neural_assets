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
PyTorch implementation of modules for Neural Assets.
Converted from the official Jax implementation: https://github.com/google-deepmind/neural_assets/blob/main/modules.py.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, TypedDict

import diffusion
import einops
import torch
import torch.nn as nn
import torch.nn.functional as F
import model_utils
import numpy as np
import preprocessing
import timm


class ControllableGeneratorReturn(TypedDict):
    """Output shapes of ControllableGenerator."""

    diff: torch.Tensor  # [B, h, w, c], GT added noise
    pred_diff: torch.Tensor  # [B, h, w, c], predicted noise

    # Inputs used to extract conditioning tokens, mostly for visualization
    src_bboxes: Optional[torch.Tensor]  # [B, n, 4]
    conditioning_object_poses: Optional[torch.Tensor]  # [B, n, co]


class ControllableGenerator(nn.Module):
    """Wrapper for a conditional generative model."""

    def __init__(
        self,
        generator: diffusion.DiffuserDiffusionWrapper,
        conditioning_encoder: nn.Module,
        conditioning_neck: nn.Module,
    ):
        super().__init__()
        self.generator = generator
        self.conditioning_encoder = conditioning_encoder
        self.conditioning_neck = conditioning_neck

    def _get_conditioning_tokens(
        self,
        tgt_object_poses: torch.Tensor,  # [B, n, co]
        src_images: torch.Tensor,  # [B, H, W, C]
        src_bboxes: torch.Tensor,  # [B, n, 4]
        src_bg_images: Optional[torch.Tensor] = None,  # [B, H, W, C]
    ) -> tuple[torch.Tensor, Dict]:
        # Encode conditioning inputs to tokens
        cond_dict = self.conditioning_encoder(
            tgt_object_poses=tgt_object_poses,
            src_images=src_images,
            src_bboxes=src_bboxes,
            src_bg_images=src_bg_images,
        )

        # Process encoded tokens as the model's conditioning inputs
        conditioning_tokens = self.conditioning_neck(conditioning_dict=cond_dict)

        return conditioning_tokens, cond_dict

    def forward(
        self,
        tgt_images: torch.Tensor,  # [B, H, W, C]
        tgt_object_poses: torch.Tensor,  # [B, n, co]
        src_images: torch.Tensor,  # [B, H, W, C]
        src_bboxes: torch.Tensor,  # [B, n, 4]
        src_bg_images: Optional[torch.Tensor] = None,  # [B, H, W, C]
    ) -> ControllableGeneratorReturn:
        # Encode inputs to conditioning_tokens
        conditioning_tokens, cond_dict = self._get_conditioning_tokens(
            tgt_object_poses=tgt_object_poses,
            src_images=src_images,
            src_bboxes=src_bboxes,
            src_bg_images=src_bg_images,
        )

        # Condition the generator on conditioning_tokens to reconstruct tgt_images
        result_dict = self.generator(
            images=tgt_images,
            conditioning_tokens=conditioning_tokens,
        )

        return self.postprocess_model_output(result_dict, cond_dict)

    def postprocess_model_output(
        self,
        generator_output: Dict[str, Any],
        cond_dict: Dict,
    ) -> ControllableGeneratorReturn:
        """Convert model output to desired format."""
        result_dict = {
            # For loss computation
            'diff': generator_output['diff'],
            'pred_diff': generator_output['pred_diff'],
            # For visualization
            'src_bboxes': cond_dict['src_bboxes'],
            'conditioning_object_poses': cond_dict['conditioning_object_poses'],
        }

        return result_dict


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_hidden_layers: int = 1,
        input_size: int = None,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size or input_size
        self.num_hidden_layers = num_hidden_layers
        self.input_size = input_size

        self.layers = nn.ModuleList()
        current_size = input_size

        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(current_size, hidden_size))
            current_size = hidden_size

        self.layers.append(nn.Linear(current_size, self.output_size))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        x = inputs
        for layer in self.layers[:-1]:
            x = layer(x)
            x = F.gelu(x)
        x = self.layers[-1](x)
        return x


class ConditioningEncoderReturn(TypedDict):
    """Output shapes of ConditioningEncoder."""

    appearance_tokens: torch.Tensor  # [B, _n, da]
    object_pose_tokens: Optional[torch.Tensor]  # [B, _n, do]

    # Inputs used to extract conditioning tokens
    src_bboxes: Optional[torch.Tensor]  # [B, n, 4]
    conditioning_object_poses: Optional[torch.Tensor]  # [B, n, co]


class ConditioningEncoder(nn.Module):
    """Wrapper for different conditioning encoders."""

    def __init__(
        self,
        appearance_encoder: nn.Module,
        object_pose_encoder: nn.Module,
        mask_out_bg_for_appearance: bool = True,
        background_value: float = 0.5,
        background_pos_enc_type: Optional[str] = None,
        bg_pos_mlp_input_size: Optional[int] = None,
    ):
        super().__init__()
        self.appearance_encoder = appearance_encoder
        self.object_pose_encoder = object_pose_encoder
        self.mask_out_bg_for_appearance = mask_out_bg_for_appearance
        self.background_value = background_value
        self.background_pos_enc_type = background_pos_enc_type

        if self.background_pos_enc_type == 'mlp':
            self.bg_pos_mlp = MLP(
                hidden_size=bg_pos_mlp_input_size * 2,
                output_size=bg_pos_mlp_input_size,
                num_hidden_layers=1,
                input_size=bg_pos_mlp_input_size,
            )
        elif self.background_pos_enc_type is not None:
            raise ValueError(
                f'Unknown {self.background_pos_enc_type=}'
            )

    def _prepare_bg_pos_enc(self, pos_input: torch.Tensor) -> torch.Tensor:
        """We may need a special positional token for background modeling."""
        if self.background_pos_enc_type is None:
            return pos_input
        elif self.background_pos_enc_type == 'mlp':
            bg_pos = pos_input[..., -1:, :]  # [*B, 1, c]
            bg_pos = self.bg_pos_mlp(bg_pos)
            pos_input = torch.cat([pos_input[..., :-1, :], bg_pos], dim=-2)
        else:
            raise ValueError(f'Unknown {self.background_pos_enc_type=}')
        return pos_input

    def forward(
        self,
        tgt_object_poses: torch.Tensor,  # [B, n, co]
        src_images: torch.Tensor,  # [B, H, W, C]
        src_bboxes: torch.Tensor,  # [B, n, 4]
        src_bg_images: Optional[torch.Tensor] = None,  # [B, H, W, C]
    ) -> ConditioningEncoderReturn:
        result_dict = {
            'src_bboxes': src_bboxes,
            'conditioning_object_poses': tgt_object_poses,
        }

        # Extract conditioning inputs from the image
        appearance_token_src = src_images

        # Encode object appearance information.
        if self.mask_out_bg_for_appearance:
            h, w = src_images.shape[-3:-1]
            # Get coarse object masks from bboxes
            fg_bboxes = (
                src_bboxes if src_bg_images is None else src_bboxes[..., :-1, :]
            )
            fg_masks = model_utils.boxes_to_sparse_segmentations(fg_bboxes, h, w)
            # [*B, n, H, W] -> [*B, H, W, 1]
            fg_masks = torch.any(fg_masks > 0, dim=-3).unsqueeze(-1)
            appearance_token_src = torch.where(
                fg_masks,
                src_images,
                torch.ones_like(src_images) * self.background_value,
            )

        result_dict['appearance_tokens'] = self.appearance_encoder(
            images=appearance_token_src,
            bboxes=src_bboxes,
            bg_images=src_bg_images,
        )

        # Encode object pose token.
        tgt_object_poses = self._prepare_bg_pos_enc(tgt_object_poses)
        result_dict['object_pose_tokens'] = self.object_pose_encoder(
            tgt_object_poses
        )

        return result_dict


class RoIAlignAppearanceEncoder(nn.Module):
    """An appearance encoder that uses the RoIAligned features of object bbox."""

    def __init__(
        self,
        shape: tuple[int, int],  # [max_num_objects, feature_dim]
        image_backbone: nn.Module,
        roi_align_size: int = 7,
        aggregate_method: str = 'mean',
    ):
        super().__init__()
        self.shape = shape
        self.image_backbone = image_backbone
        self.roi_align_size = roi_align_size
        self.aggregate_method = aggregate_method
        
        self.output_projection = nn.Linear(
            self._get_feature_dim(),
            shape[1]
        )

    def _get_feature_dim(self):
        """Get feature dimension after aggregation."""
        # This will be set properly after first forward pass
        return 768  # DINO default, will be updated

    def _aggregate_obj_features(
        self,
        bboxes: torch.Tensor,  # [B, n, 4]
        roi_features: torch.Tensor,  # [B, n, s, s, c]
    ) -> torch.Tensor:  # [B, _n, _c]
        """Aggregate per-bbox feature maps to get object appearance tokens."""
        if self.aggregate_method == 'mean':
            obj_features = torch.mean(roi_features, dim=(-3, -2))
        elif self.aggregate_method == 'max':
            obj_features = torch.amax(roi_features, dim=(-3, -2))
        elif self.aggregate_method == 'flatten':
            obj_features = einops.rearrange(
                roi_features, '... n s1 s2 d -> ... (n s1 s2) d'
            )
        else:
            raise ValueError(f'Unknown pooling method: {self.aggregate_method}.')

        # Special case: Empty bboxes result in taking image_features[..., 0, 0, :]
        # We set them to zeros here
        is_non_empty = torch.any(
            bboxes != torch.tensor(preprocessing.NOTRACK_BOX, device=bboxes.device),
            dim=-1,
            keepdim=True,
        )  # [*B, n, 1]
        
        # Duplicate it to [*B, (n*s*s), 1] when token number is more than 1
        if self.aggregate_method == 'flatten':
            is_non_empty = einops.repeat(
                is_non_empty,
                '... n 1 -> ... (n repeat) 1',
                repeat=self.roi_align_size**2,
            )
        obj_features = torch.where(
            is_non_empty, obj_features, torch.zeros_like(obj_features)
        )
        return obj_features

    def _extract_obj_features(
        self,
        images: torch.Tensor,  # [B, H, W, C]
        bboxes: torch.Tensor,  # [B, n, 4]
    ) -> torch.Tensor:  # [B, _n, _c]
        """Extract object-centric features via RoIAlign using 2D bboxes."""
        # Extract image features
        image_features = self.image_backbone(images)
        # Shape: [*B, h', w', c]

        # Apply RoIAlign to get per-bbox feature maps
        roi_features = model_utils.get_roi_align_features(
            bboxes, image_features, size=self.roi_align_size
        )
        # Shape: [*B, n, s, s, c]

        # Aggregate per-bbox feature maps to get object appearance tokens
        obj_features = self._aggregate_obj_features(bboxes, roi_features)
        # Shape: [*B, n or (n*s*s), c]
        return obj_features

    def forward(
        self,
        images: torch.Tensor,  # [B, H, W, C]
        bboxes: torch.Tensor,  # [B, n, 4]
        bg_images: Optional[torch.Tensor] = None,  # [B, H, W, C]
    ) -> torch.Tensor:  # [B, _n, d]
        bboxes = bboxes.detach()
        assert (
            bboxes.shape[-2] == self.shape[-2]
        ), f'Expected {self.shape[-2]} bboxes, but got {bboxes.shape[-2]}'

        if bg_images is None:
            app_features = self._extract_obj_features(images, bboxes)
        else:
            # We assume the first (n-1) bboxes are for foreground objects
            # The last bbox is a global bbox for background features
            obj_features = self._extract_obj_features(images, bboxes[..., :-1, :])
            bg_features = self._extract_obj_features(bg_images, bboxes[..., -1:, :])
            app_features = torch.cat([obj_features, bg_features], dim=-2)
        # Shape: [*B, n or (n*s*s), c]

        # Project to the desired feature dim
        return self.output_projection(app_features)


class MLPPoseEncoder(nn.Module):
    """A module that encodes object pose with MLP as conditioning inputs."""

    def __init__(
        self,
        mlp_module: nn.Module,
        duplicate_factor: int = 1,
    ):
        super().__init__()
        self.mlp_module = mlp_module
        self.duplicate_factor = duplicate_factor

    def forward(self, bboxes: torch.Tensor) -> torch.Tensor:
        valid_mask = torch.any(
            bboxes
            != torch.tensor(
                preprocessing.NOTRACK_PROJ_4_CORNER,
                dtype=bboxes.dtype,
                device=bboxes.device,
            ),
            dim=-1,
            keepdim=True,
        )
        pose_features = self.mlp_module(bboxes)
        if self.duplicate_factor > 1:
            pose_features = einops.repeat(
                pose_features,
                '... n d -> ... (n repeat) d',
                repeat=self.duplicate_factor,
            )
            valid_mask = einops.repeat(
                valid_mask,
                '... n 1 -> ... (n repeat) 1',
                repeat=self.duplicate_factor,
            )
        pose_features = torch.where(
            valid_mask, pose_features, torch.zeros_like(pose_features)
        )
        return pose_features


class FeedForwardNeck(nn.Module):
    """A simple module for converting encoded conditioning to generator inputs."""

    def __init__(self, feed_forward_module: nn.Module):
        super().__init__()
        self.feed_forward_module = feed_forward_module

    def forward(self, conditioning_dict: ConditioningEncoderReturn) -> torch.Tensor:
        # Process encoded tokens into one feature tensor
        conds = [
            conditioning_dict['appearance_tokens'],
            conditioning_dict['object_pose_tokens'],
        ]
        # Each entry has shape [*B, n, c]

        # Concatenate conditioning tokens along the channel dimension
        conditioning_tokens = torch.cat(conds, dim=-1)

        # Fuse the appearance and the pose tokens
        conditioning_tokens = self.feed_forward_module(conditioning_tokens)

        return conditioning_tokens  # [B, num_tokens, token_dim]


class DINOViT(nn.Module):
    """DINO v1 ViT model using timm library."""

    def __init__(
        self,
        patch_size: Sequence[int] = (16, 16),
        width: int = 768,
        depth: int = 12,
        mlp_dim: int = 3072,
        num_heads: int = 12,
        in_vrange: tuple[float, float] = (0.0, 1.0),
        use_imagenet_value_range: bool = True,
        frozen_model: bool = True,
        pretrained_dino_path = None,
    ):
        super().__init__()
        self.patch_size = patch_size
        self.width = width
        self.depth = depth
        self.mlp_dim = mlp_dim
        self.num_heads = num_heads
        self.in_vrange = in_vrange
        self.use_imagenet_value_range = use_imagenet_value_range
        self.frozen_model = frozen_model
        self.pretrained_dino_path = pretrained_dino_path
        
        # Map patch size to model name
        patch_str = f"{patch_size[0]}"
        model_name = f'vit_base_patch{patch_str}_224.dino'
        try:
            self.vit = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,  # Remove classification head
                pretrained_cfg_overlay={'file': self.pretrained_dino_path} if self.pretrained_dino_path is not None else None,
            )
        except:
            # Fallback to a similar model if exact match not available
            print(f"Warning: Could not load {model_name}, using vit_base_patch16_224 instead")
            self.vit = timm.create_model(
                'vit_base_patch16_224',
                pretrained=True,
                num_classes=0,
                pretrained_cfg_overlay={'file': self.pretrained_dino_path} if self.pretrained_dino_path is not None else None,
            )
        
        if frozen_model:
            for param in self.vit.parameters():
                param.requires_grad = False

    def _preprocess(self, image: torch.Tensor) -> torch.Tensor:
        """Preprocess input image."""
        # Resize to 224x224
        # Input is [B, H, W, C], need [B, C, H, W]
        image = image.permute(0, 3, 1, 2)
        image = F.interpolate(image, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Normalize to [0, 1]
        vmin, vmax = self.in_vrange
        image = (image - vmin) / (vmax - vmin)

        # Optionally adjust to ImageNet pre-trained value range
        if self.use_imagenet_value_range:
            mean = torch.tensor([0.485, 0.456, 0.406], device=image.device).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225], device=image.device).view(1, 3, 1, 1)
            image = (image - mean) / std

        return image

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Forward pass through DINO ViT.
        
        Args:
            image: [B, H, W, C] input images
            
        Returns:
            [B, h, w, c] feature maps
        """
        image = self._preprocess(image)  # [B, C, H, W]

        # Get features from ViT
        if self.frozen_model:
            with torch.no_grad():
                features = self.vit.forward_features(image)
        else:
            features = self.vit.forward_features(image)
        
        # features shape depends on model, typically [B, N+1, C] where N = (H/P)*(W/P)
        # Remove CLS token and reshape to spatial
        if features.dim() == 3:  # [B, N+1, C]
            features = features[:, 1:, :]  # Remove CLS token
            B, N, C = features.shape
            # Calculate spatial dimensions
            h = w = int(np.sqrt(N))
            features = features.reshape(B, h, w, C)
        
        return features  # [B, h, w, c]

    @classmethod
    def from_variant_str(
        cls,
        version: str,
        variant: str,
        **kwargs
    ) -> 'DINOViT':
        """Create DINO model from version and variant string."""
        if version == 'v1':
            assert variant in ('B/16', 'B/8'), 'DINO v1 only supports B/16 and B/8.'
        else:
            raise ValueError(
                f'Unknown version: {version}, version should be either v1 or v2.'
            )

        v, patch = variant.split('/')
        patch = (int(patch), int(patch))

        if v == 'B':
            kwargs.update({
                'width': 768,
                'depth': 12,
                'mlp_dim': 3072,
                'num_heads': 12,
            })
        else:
            raise ValueError('Only supports DINO with ViT-B.')

        return cls(patch_size=patch, **kwargs)
