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
PyTorch implementation of Stable Diffusion from Diffusers for Neural Assets.
Converted from the official Jax implementation: https://github.com/google-deepmind/neural_assets/blob/main/diffusion.py.
"""

from __future__ import annotations

from typing import TypedDict

import diffusers
import torch
import torch.nn as nn


class DiffuserDiffusionLossReturn(TypedDict):
    """Output shapes of DiffuserDiffusion's training forward pass."""

    diff: torch.Tensor  # [B, h, w, c], GT added noise
    pred_diff: torch.Tensor  # [B, h, w, c], predicted noise


class DiffuserDiffusionWrapper(nn.Module):
    """Interface between diffuser models and our modules."""

    def __init__(self, model_name: str = 'stable_diffusion_v2_1', device='cpu', pretrain_vae_folder='./sd/vae/', pretrain_unet_folder='./sd/unet/'):
        super().__init__()
        self.model_name = model_name
        # Use MPS if available (Apple Silicon), otherwise CPU
        if device == 'cuda' and not torch.cuda.is_available():
            if torch.backends.mps.is_available():
                device = 'mps'
            else:
                device = 'cpu'
        self.device = device
        
        if self.model_name != 'stable_diffusion_v2_1':
            raise ValueError(f'Unknown model name {self.model_name}')
        
        # Load VAE
        self.vae = diffusers.AutoencoderKL.from_pretrained(
            pretrain_vae_folder
        ).to(device)

        # Freeze VAE parameters
        for param in self.vae.parameters():
            param.requires_grad = False
        
        # Load UNet
        self.unet = diffusers.UNet2DConditionModel.from_pretrained(
            pretrain_unet_folder
        ).to(device)
        
        # Setup noise scheduler
        self.noise_scheduler = diffusers.DDPMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
            num_train_timesteps=1000,
        )

    def forward(
        self,
        images: torch.Tensor,  # [B, H, W, C]
        conditioning_tokens: torch.Tensor,  # [B, n, d]
    ) -> DiffuserDiffusionLossReturn:
        """Training pass for diffusion model.
        
        Args:
            images: Target images in [B, H, W, C] format
            conditioning_tokens: Conditioning tokens from Neural Assets encoder
            
        Returns:
            Dictionary with ground truth noise and predicted noise
        """
        # Convert images to latent space
        # PyTorch VAE expects [B, C, H, W], so convert from [B, H, W, C]
        images = images.permute(0, 3, 1, 2)

        images = 2.0 * images - 1.0
        
        with torch.no_grad():
            latents = self.vae.encode(images).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor

        # Sample noise that we'll add to the latents
        noise = torch.randn_like(latents)
        
        # Sample a random timestep for each image
        bsz = latents.shape[0]
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (bsz,),
            device=latents.device,
        ).long()

        # Add noise to the latents according to the noise magnitude at each timestep
        # (this is the forward diffusion process)
        noisy_latents = self.noise_scheduler.add_noise(latents, noise, timesteps)

        # Replace the text embedding with our Neural Assets for conditioning
        encoder_hidden_states = conditioning_tokens

        # Predict the noise residual
        model_pred = self.unet(
            noisy_latents,
            timesteps,
            encoder_hidden_states=encoder_hidden_states,
            return_dict=False,
        )[0]

        # Get the target for loss depending on the prediction type
        pred_type = self.noise_scheduler.config.prediction_type
        if pred_type == 'epsilon':
            target = noise
        elif pred_type == 'v_prediction':
            target = self.noise_scheduler.get_velocity(latents, noise, timesteps)
        else:
            raise ValueError(f'Unknown prediction type {pred_type}')

        loss_dict = {
            'diff': target,
            'pred_diff': model_pred,
        }

        return loss_dict
