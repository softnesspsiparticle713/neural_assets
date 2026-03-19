"""
Test file to load a neural asset checkpoint and perform image synthesis.
"""

import os
import warnings
import numpy as np
import argparse
from diffusers import DDIMScheduler

# Suppress TensorFlow warnings and cleanup errors
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TF logging
warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

import tensorflow as tf
import tensorflow_datasets as tfds
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pathlib import Path

# Disable GPU for TensorFlow (we only use it for data loading)
tf.config.set_visible_devices([], 'GPU')

# Import the necessary modules
import preprocessing
from train_movi import NeuralAssetsLightningModule


def _get_max_obj_num(variant):
    """Max number of objects in the dataset."""
    if variant in ['a', 'b', 'c']:
        return 10
    elif variant in ['e', 'f']:
        return 23
    else:
        raise ValueError(f'Invalid MOVi variant: {variant}')


def load_movi(args):
    """Build the MOVi dataset."""
    ds_name = f'movi_{args.variant}/{args.resolution}x{args.resolution}:1.0.0'
    ds_builder = tfds.builder(ds_name, data_dir=args.data_dir)
    ds = ds_builder.as_dataset(split='test', shuffle_files=True)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    
    print(f"Loaded dataset: {ds_name}")
    return ds


def generate_image(args, model, batch, device, num_inference_steps=50):
    """
    Generate an image from the source image and target object poses.

    Args:
        model: The trained neural assets model
        batch: A batch of data containing source and target information
        device: The device to run inference on
        num_inference_steps: Number of denoising steps (default 50)

    Returns:
        generated_image: The reconstructed target image [B, H, W, 3]
    """
    model.eval()

    with torch.no_grad():
        # Move batch to device
        batch = {k: torch.tensor(v).to(device) if isinstance(v, np.ndarray) else v
                 for k, v in batch.items()}

        # Get conditioning tokens from the model
        conditioning_tokens, _ = model.ns_model._get_conditioning_tokens(
            tgt_object_poses=batch['tgt_bboxes_3d'],
            src_images=batch['src_image'],
            src_bboxes=batch['src_bboxes'],
            src_bg_images=batch['src_bg_image'],
        )

        uncond_tokens, _ = model.ns_model._get_conditioning_tokens(
            tgt_object_poses=torch.zeros_like(batch['tgt_bboxes_3d']),
            src_images=batch['src_image'],
            src_bboxes=torch.zeros_like(batch['src_bboxes']),
            src_bg_images=batch['src_bg_image'],
        )

        # Get the shape for the latent space
        # Stable Diffusion VAE downsamples by 8x, and has 4 latent channels
        bsz = batch['src_image'].shape[0]
        height = batch['tgt_image'].shape[1] // 8
        width = batch['tgt_image'].shape[2] // 8
        latent_channels = 4

        # Start from random noise in latent space
        latents = torch.randn(
            bsz, latent_channels, height, width,
            device=device,
            dtype=conditioning_tokens.dtype
        )

        # Set up the scheduler for inference
        scheduler = DDIMScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
            num_train_timesteps=1000,
            clip_sample=False,
        )
        scheduler.set_timesteps(num_inference_steps)

        # Scale the initial noise by the scheduler's init noise sigma
        latents = latents * scheduler.init_noise_sigma

        # Denoising loop
        for t in scheduler.timesteps:
            latent_model_input = scheduler.scale_model_input(latents, t)

            # Predict the noise residual using the U-Net
            noise_pred_cond = model.ns_model.generator.unet(
                latent_model_input,
                t,
                encoder_hidden_states=conditioning_tokens,
                return_dict=False,
            )[0]

            noise_pred_uncond = model.ns_model.generator.unet(
                latent_model_input,
                t,
                encoder_hidden_states=uncond_tokens,
                return_dict=False,
            )[0]

            noise_pred = noise_pred_uncond + args.guidance_strength * (noise_pred_cond - noise_pred_uncond)

            # Compute the previous noisy sample x_t -> x_t-1
            latents = scheduler.step(noise_pred, t, latents, return_dict=False)[0]

        # Decode the latents to pixel space
        latents = latents / model.ns_model.generator.vae.config.scaling_factor
        images = model.ns_model.generator.vae.decode(latents, return_dict=False)[0]

        # Convert from [B, C, H, W] to [B, H, W, C]
        images = images.permute(0, 2, 3, 1)

        # Clip to [0, 1] range
        images = torch.clamp((images + 1.0) / 2.0, 0.0, 1.0)

        # Convert to numpy and move to CPU
        generated_image = images.cpu().numpy()

    return generated_image


def save_visualization(src_image, tgt_image, generated_image, output_path):
    """
    Save a visualization comparing source, target, and generated images.

    Args:
        src_image: Source image [H, W, 3] in range [0, 1]
        tgt_image: Target image [H, W, 3] in range [0, 1]
        generated_image: Generated image [H, W, 3] in range [0, 1]
        output_path: Path to save the visualization
    """
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Source image
    axes[0].imshow(np.clip(src_image, 0, 1))
    axes[0].set_title('Source Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')

    # Target image (ground truth)
    axes[1].imshow(np.clip(tgt_image, 0, 1))
    axes[1].set_title('Target Image (Ground Truth)', fontsize=14, fontweight='bold')
    axes[1].axis('off')

    # Generated image
    axes[2].imshow(np.clip(generated_image, 0, 1))
    axes[2].set_title('Reconstructed Image', fontsize=14, fontweight='bold')
    axes[2].axis('off')

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved visualization to {output_path}")
    plt.close()


def main():
    """Main test function."""
    print("=" * 60)
    print("Neural Assets - Image Reconstruction Test")
    print("=" * 60)
    print()

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_dir", type=str, default='~/checkpoints/step-step=200000.ckpt')
    parser.add_argument("--num_steps", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default='~/datasets/')
    parser.add_argument("--save_every", type=int, default=1000)
    parser.add_argument("--log_every", type=int, default=1)
    parser.add_argument("--gradient_clip", type=float, default=1.0)
    parser.add_argument("--variant", type=str, default='e')
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--hidden_size", type=int, default=1024)
    parser.add_argument("--roi_align_size", type=int, default=2)
    parser.add_argument("--model_name", type=str, default='stable_diffusion_v2_1')
    parser.add_argument("--pretrain_vae_folder", type=str, default='~/pretrained_models/vae/')
    parser.add_argument("--pretrain_unet_folder", type=str, default='~/pretrained_models/unet/')
    parser.add_argument("--pretrained_dino_path", type=str, default=None)
    parser.add_argument("--dino_version", type=str, default='v1')
    parser.add_argument("--dino_variant", type=str, default='B/8')
    parser.add_argument("--lr_generator_encoder", type=float, default=5e-5)
    parser.add_argument("--lr_mlp_projection", type=float, default=5e-4)
    parser.add_argument("--warmup_steps", type=int, default=1000)
    parser.add_argument("--wandb_project", type=str, default="neural_assets")
    parser.add_argument("--wandb_entity", type=str, default="neural_assets", help="Wandb team/entity name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Custom run name for wandb")
    parser.add_argument("--guidance_strength", type=float, default=2.0)

    args = parser.parse_args()

    # Set device
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    print(f"Using device: {device}")
    print()

    # Build model
    model = NeuralAssetsLightningModule.load_from_checkpoint(args.checkpoint_dir, args=args)
    model = model.to(device)

    # Load checkpoint
    print()

    # Load dataset
    print("Loading dataset...")
    test_ds = load_movi(args)

    # Preprocess dataset
    preproc_fn = lambda x: preprocessing.preprocess_gv_movi_example(
        x,
        max_instances=_get_max_obj_num(variant=args.variant),
        resolution=args.resolution,
        drop_cond_prob=0.0,  # No dropout during testing
    )
    test_loader = test_ds.map(preproc_fn).batch(batch_size=args.batch_size).prefetch(tf.data.AUTOTUNE)

    # Get a test batch
    print("Getting test batch...")
    test_iter = iter(test_loader)
    batch = next(test_iter)

    # Convert TensorFlow tensors to NumPy
    batch = {k: v.numpy() if hasattr(v, 'numpy') else v for k, v in batch.items()}
    print()

    # Extract images from the batch (first sample in the batch)
    src_image = batch['src_image'][0]  # [H, W, 3]
    tgt_image = batch['tgt_image'][0]  # [H, W, 3]

    print(f"Source image shape: {src_image.shape}")
    print(f"Target image shape: {tgt_image.shape}")
    print()

    # Generate reconstructed image
    print("Generating reconstructed image...")
    print("This may take a few moments...")
    generated_image = generate_image(args, model, batch, device, num_inference_steps=50)

    # Extract the first sample from the batch
    generated_image = generated_image[0]  # [H, W, 3]
    print(f"Generated image shape: {generated_image.shape}")
    print()

    # Create output directory
    output_dir = Path('./test_outputs')
    output_dir.mkdir(exist_ok=True)

    # Save visualization
    import time
    output_path = output_dir / f'reconstruction_test_{int(time.time())}.png'
    save_visualization(src_image, tgt_image, generated_image, output_path)

    # Calculate reconstruction error (MSE)
    mse = np.mean((tgt_image - generated_image) ** 2)
    print()
    print("=" * 60)
    print(f"Mean Squared Error (MSE): {mse:.6f}")
    print("=" * 60)
    print()
    print("Test complete!")
    print(f"Results saved to: {output_path}")


if __name__ == '__main__':
    try:
        main()
    finally:
        # Clean up TensorFlow resources to avoid shutdown errors
        import gc
        gc.collect()

        # Suppress any remaining TensorFlow cleanup errors
        import atexit
        def suppress_tf_errors():
            try:
                pass
            except:
                pass
        atexit.register(suppress_tf_errors)
