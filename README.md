# Unofficial PyTorch Implementation of Neural Assets

This code repository contains an unofficial PyTorch implementation of the Neural Assets paper from Google DeepMind. Our implementation follows the details as described in the [original Neural Assets paper](https://arxiv.org/abs/2406.09292) and inspired by the [official JAX implementation of Neural Assets](https://github.com/google-deepmind/neural_assets).


To reproduce the results of neural assets on the MoVi dataset, please follow the following guide:

1. Create a Python 3.11 virtual environment.
2. Install the required packages in `requirements.txt`.
3. Download the required model checkpoints and datasets from the following links:
    - Stable Diffusion 2.1 pytorch checkpoint: https://huggingface.co/Manojb/stable-diffusion-2-1-base/tree/main
    - MoVi dataset: https://console.cloud.google.com/storage/browser/kubric-public/tfds
3. Run the training script `train_movi.py` and inference script `inference_movi.py`. Note that you may need to modify the arguments in these scripts depending on your setup.

## Authors
This code base is created by
- [Wenlin Chen](https://wenlin-chen.github.io)
- [Xi Ye](https://github.com/XiYe20)
- [Mingtian Zhang](https://mingtian.ai)