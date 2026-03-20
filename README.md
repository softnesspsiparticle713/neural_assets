# Neural Assets — PyTorch Reimplementation

This code repository contains an unofficial PyTorch implementation of Neural Assets from Google DeepMind. Our implementation follows the details as described in the [original paper](https://arxiv.org/abs/2406.09292) and is adapted from the [official JAX implementation](https://github.com/google-deepmind/neural_assets).

<img width="722" height="403" alt="neural_assets" src="https://github.com/user-attachments/assets/f3e38fb0-67aa-4e67-ae63-12457e5a1839" />

## Setup

To reproduce the results of neural assets on the MoVi dataset, please follow the following guide:

1. Create a Python 3.11 virtual environment.
2. Install the required packages in `requirements.txt`.
3. Download the required model checkpoints and datasets from the following links:
    - Stable Diffusion 2.1 pytorch checkpoint: https://huggingface.co/Manojb/stable-diffusion-2-1-base/tree/main
    - MoVi dataset: https://console.cloud.google.com/storage/browser/kubric-public/tfds
3. Run the training script `train_movi.py` and then the inference script `inference_movi.py` (you may need to modify the arguments in these scripts depending on your setup).


## Citing this work
If you use this work, please cite the original paper of Neural Assets:

```
@inproceedings{neuralassets_2024,
  title = {{Neural Assets}: 3D-Aware Multi-Object Scene Synthesis with Image Diffusion Models},
  author = {Ziyi Wu and
            Yulia Rubanova and
            Rishabh Kabra and
            Drew A. Hudson and
            Igor Gilitschenski and
            Yusuf Aytar and
            Sjoerd van Steenkiste and
            Kelsey Allen and
            Thomas Kipf},
  booktitle = {Advances in Neural Information Processing Systems},
  year = {2024}
}
```

## Authors of this codebase
This code base is created by
- [Wenlin Chen](https://wenlin-chen.github.io)
- [Xi Ye](https://github.com/XiYe20)
- [Mingtian Zhang](https://mingtian.ai)
