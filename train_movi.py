import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

import diffusion
import modules
import preprocessing
import numpy as np
import tensorflow as tf

# Keep TF strictly on CPU
import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
try:
    tf.config.set_visible_devices([], 'GPU')
except Exception as e:
    print("TF GPU visibility config failed:", e)

import tensorflow_datasets as tfds
import torch
import torch.nn as nn
import torch.optim as optim
import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader, IterableDataset

def _get_max_obj_num(args):
  """Max number of objects in the dataset."""
  if args.variant in ['a', 'b', 'c']:
    return 10
  elif args.variant in ['e', 'f']:
    return 23
  else:
    raise ValueError(f'Invalid MOVi variant: {args.variant}')


def load_movi(args):
  """Build the MOVi dataset."""
  ds_name = f'movi_{args.variant}/{args.resolution}x{args.resolution}:1.0.0'
  ds_builder = tfds.builder(ds_name, data_dir=args.data_dir)
  ds = ds_builder.as_dataset(split='train', shuffle_files=True)

  print(f"Loaded dataset: {ds_name}")
  return ds

class TFDSIterableDataset(IterableDataset):
    """Wrap a tf.data.Dataset so that PyTorch Lightning can iterate over it."""

    def __init__(self, tf_dataset):
        super().__init__()
        self.tf_dataset = tf_dataset

    def __iter__(self):
        for batch in self.tf_dataset:
            # Convert TF tensors to NumPy (will be converted to PyTorch in the model)
            yield {k: (v.numpy() if hasattr(v, 'numpy') else v) for k, v in batch.items()}

class MoviDataModule(pl.LightningDataModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.data_dir = args.data_dir
        self.train_dataset = None

    def setup(self, stage=None):
        # Build tf.data pipeline exactly as before
        train_ds = load_movi(args=self.args)

        # DDP: shard dataset so each GPU sees disjoint data
        world_size = self.trainer.world_size if self.trainer else 1
        rank = self.trainer.global_rank if self.trainer else 0
        if world_size > 1:
            train_ds = train_ds.shard(num_shards=world_size, index=rank)
        per_gpu_batch_size = self.args.batch_size // world_size

        preproc_fn = lambda x: preprocessing.preprocess_gv_movi_example(
            x,
            max_instances=_get_max_obj_num(args=self.args),
            resolution=self.args.resolution,
            drop_cond_prob=0.1,  # 10% dropout for classifier-free guidance (CFG)
        )

        tf_train = (
            train_ds.map(preproc_fn)
            .batch(batch_size=per_gpu_batch_size)
            .prefetch(tf.data.AUTOTUNE)
            .repeat() # uncomment this to repeat the dataset indefinitely
        )
        self.train_dataset = TFDSIterableDataset(tf_train)

    def train_dataloader(self):
        # The TF pipeline already yields batched examples, so batch_size=None
        return DataLoader(
            self.train_dataset,
            batch_size=None,
            num_workers=0,
        )

class NeuralAssetsLightningModule(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        generator = diffusion.DiffuserDiffusionWrapper(model_name=args.model_name, 
                                                       pretrain_vae_folder=args.pretrain_vae_folder, 
                                                       pretrain_unet_folder=args.pretrain_unet_folder)
        # Learnable appearance tokens + pose tokens
        token_dim = args.hidden_size // 2
        # We will do RoIAlign to extract 2x2 feature maps as object appearance tokens
        roi_align_size = args.roi_align_size
        # We use DINO as our visual encoder
        dino_version, dino_variant = args.dino_version, args.dino_variant
        conditioning_encoder = modules.ConditioningEncoder(
            appearance_encoder=modules.RoIAlignAppearanceEncoder(
                # +1 because we add a global background bbox
                shape=(_get_max_obj_num(args=args) + 1, token_dim),
                image_backbone=modules.DINOViT.from_variant_str(
                    version=dino_version,
                    variant=dino_variant,
                    in_vrange=(0, 1),
                    use_imagenet_value_range=True,
                    frozen_model=False,  # we fine-tune DINO
                    pretrained_dino_path=args.pretrained_dino_path,
                ),
                roi_align_size=roi_align_size,  # feature map size: (28, 28)
                aggregate_method='flatten',  # flatten the 2x2 feature map
            ),
            # Treat 3D bbox as object pose
            object_pose_encoder=modules.MLPPoseEncoder(
                mlp_module=modules.MLP(
                    hidden_size=token_dim * 2,
                    output_size=token_dim,
                    num_hidden_layers=1,
                    input_size=12,  # 3D bbox in projected 4-corner format: (u,v,d)*4 = 12
                ),
                # Duplicate bbox tokens to match the length of appearance tokens
                duplicate_factor=roi_align_size**2,
            ),
            # Mask out background pixels when encoding object tokens
            mask_out_bg_for_appearance=True,
            background_value=0.5,
            # Map the relative camera pose with a MLP
            # This serves as the pose token for the background
            background_pos_enc_type='mlp',
            bg_pos_mlp_input_size=12,  # Input is 3D bbox in projected 4-corner format
        )
        # Fuse appearance and pose tokens with a neck module
        conditioning_neck = modules.FeedForwardNeck(
            feed_forward_module=nn.Linear(token_dim * 2, args.hidden_size),
        )

        # Full Model
        self.ns_model = modules.ControllableGenerator(
            generator=generator,
            conditioning_encoder=conditioning_encoder,
            conditioning_neck=conditioning_neck,
        )

        # Verify VAE is frozen
        num_trainable_vae = sum(p.numel() for p in self.ns_model.generator.vae.parameters() if p.requires_grad)
        print(f"Number of trainable VAE parameters: {num_trainable_vae}")

        # Save hparams for reproducibility (no need to save full model).
        self.save_hyperparameters(
            {
                "VARIANT": args.variant,
                "RESOLUTION":args.resolution,
                "BATCH_SIZE": args.batch_size,
                "hidden_size": args.hidden_size,
                "lr_generator_encoder": args.lr_generator_encoder,
                "lr_mlp_projection": args.lr_mlp_projection,
                "warmup_steps": args.warmup_steps,
                "gradient_clip": args.gradient_clip,
            }
        )
        
    def forward(self, **kwargs):
        return self.ns_model(**kwargs)
    
    def _to_device_tensor(self, v):
        if isinstance(v, torch.Tensor):
            return v.to(self.device)
        if isinstance(v, np.ndarray):
            return torch.from_numpy(v).to(self.device)
        return v
    
    # Define loss function for gradient computation
    def compute_loss(self, batch):
        """Compute the denoising loss."""
        # Move batch to device
        batch = {k: self._to_device_tensor(v) for k, v in batch.items()}

        output_dict = self.ns_model(
            tgt_images=batch['tgt_image'],
            tgt_object_poses=batch['tgt_bboxes_3d'],
            src_images=batch['src_image'],
            src_bboxes=batch['src_bboxes'],
            src_bg_images=batch['src_bg_image'],
        )
        loss = ((output_dict['diff'] - output_dict['pred_diff']) ** 2).mean()
        return loss, output_dict
    
    def training_step(self, batch, batch_idx):
        loss, _ = self.compute_loss(batch)

        # Log loss; Lightning handles printing/progress bar
        self.log("train_loss", loss, on_step=True, on_epoch=False, prog_bar=True, batch_size=self.args.batch_size)

        # Optionally log current learning rates
        if self.trainer is not None and self.trainer.optimizers:
            opt = self.trainer.optimizers[0]
            lr_gen = opt.param_groups[0]['lr']
            if len(opt.param_groups) > 1:
                lr_mlp = opt.param_groups[1]['lr']
            else:
                lr_mlp = lr_gen
            self.log("lr_gen", lr_gen, on_step=True, on_epoch=False, prog_bar=False)
            self.log("lr_mlp", lr_mlp, on_step=True, on_epoch=False, prog_bar=False)

        return loss

    def configure_optimizers(self):
        # Define optimizer with separate parameter groups for different learning rates
        # Paper uses 5e-5 for generator+encoder, 1e-3 for MLPs+projections
        vae_params = set(self.ns_model.generator.vae.parameters())
        generator_encoder_params = [p for p in self.ns_model.generator.parameters() if p not in vae_params] + list(self.ns_model.conditioning_encoder.appearance_encoder.image_backbone.parameters())
        mlp_projection_params = [
            p for n, p in self.ns_model.named_parameters() 
            if p not in set(generator_encoder_params)
        ]
        self.num_generator_params = sum(p.numel() for p in generator_encoder_params)
        self.num_mlp_params = sum(p.numel() for p in mlp_projection_params)
        self.num_total_params = sum(p.numel() for p in self.ns_model.parameters() if p.requires_grad)

        optimizer = optim.AdamW([
            {'params': generator_encoder_params, 'lr': self.args.lr_generator_encoder},
            {'params': mlp_projection_params, 'lr': self.args.lr_mlp_projection}
        ], weight_decay=0.01)

        # Learning-rate warmup scheduler (step-based)
        scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / self.args.warmup_steps,           # start at 0 × base LR
            end_factor=1.0,             # end at 1 × base LR
            total_iters=self.args.warmup_steps,  # number of warmup steps
        )
        # scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=get_lr_multiplier)
        print("Optimizer and training functions initialized")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1,
            },
        }

    def on_fit_start(self):
        print(f"Number of generator params: {self.num_generator_params}")
        print(f"Number of MLP params: {self.num_mlp_params}")
        print(f"Number of total params: {self.num_total_params}")


def main(args):
    print(f"Starting training for {args.max_steps} steps...")
    print(f"Batch size: {args.batch_size}")
    print("=" * 60)

    # Set up model and data module
    model = NeuralAssetsLightningModule(args=args)
    data_module = MoviDataModule(args=args)

    # Select accelerator and strategy (DDP for multi-GPU)
    if torch.cuda.is_available():
        accelerator = "gpu"
        devices = "auto"  # use all available GPUs
        strategy = "ddp" if torch.cuda.device_count() > 1 else "auto"
    elif torch.backends.mps.is_available():
        accelerator = "mps"
        devices = 1
        strategy = "auto"
    else:
        accelerator = "cpu"
        devices = 1
        strategy = "auto"

    # Set up loggers
    if args.report_to == 'wandb':
        # Construct wandb run name
        run_name = getattr(args, 'wandb_run_name', None) or f"movi_{args.variant}_bs{args.batch_size}"
        
        wandb_logger = WandbLogger(
            project=getattr(args, 'wandb_project', 'neural_assets'),
            entity=getattr(args, 'wandb_entity', None),
            name=run_name,
            save_dir="logs",
            log_model=False,  # Don't upload model checkpoints to wandb (they're large)
        )
        # Log hyperparameters to wandb
        wandb_logger.experiment.config.update(vars(args))
    elif args.report_to == 'tensorboard':
        tb_logger = TensorBoardLogger("logs", name="neural_assets")
    else:
        raise ValueError(f"Unsupported report_to option: {args.report_to}")
    
    callbacks = []
    # Step-based checkpointing
    if args.save_every and args.save_every > 0:
        callbacks.append(
            ModelCheckpoint(
                dirpath=args.checkpoint_dir,
                filename="step-{step}",
                save_top_k=-1,
                every_n_train_steps=args.save_every,
                save_on_train_epoch_end=False,
            )
        )

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        strategy=strategy,
        max_steps=args.max_steps,
        logger=tb_logger if args.report_to == 'tensorboard' else wandb_logger,
        callbacks=callbacks,
        gradient_clip_val=args.gradient_clip,
        log_every_n_steps=args.log_every,
    )

    trainer.fit(model, datamodule=data_module)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_dir", type=str, default='~/checkpoints/')
    parser.add_argument("--max_steps", type=int, default=200000)
    parser.add_argument("--batch_size", type=int, default=64)
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
    parser.add_argument("--report_to", type=str, default='wandb')
    parser.add_argument("--wandb_project", type=str, default="neural_assets")
    parser.add_argument("--wandb_entity", type=str, default="neural_assets", help="Wandb team/entity name")
    parser.add_argument("--wandb_run_name", type=str, default=None, help="Custom run name for wandb")

    args = parser.parse_args()
    main(args)