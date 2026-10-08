# Cascade HQP-DETR

Official implementation for the paper:

High-Quality Proposal Encoding and Cascade Denoising for Imaginary-Supervised Object Detection


## Getting Started

### 1. Installation

This project is built on the [Detrex](https://github.com/IDEA-Research/detrex) toolbox.

1. Follow the official [Detrex Getting Started Guide](https://detrex.readthedocs.io/en/latest/tutorials/Getting_Started.html) to set up the Detectron2, and Detrex.
2. Clone this repository into the `projects/` directory of your Detrex workspace.

### 2. Training

To train Cascade HQP-DETR with the provided configuration:

```bash
python tools/train_net.py \
  --config-file projects/CascadeHQP_DETR/configs/hqp-detr.py
```

### 3. Dataset

The datasets have been released at:

[Download](https://drive.google.com/file/d/17Aze4Qu1luJ2ExwiSl39Gj-7B8a9-G7d/view?usp=sharing)

## Acknowledgments

This work is built upon the excellent research and open-source codebases from the following projects. We sincerely thank their authors for making these resources available.

- [Detrex](https://github.com/IDEA-Research/detrex): A unified open-source object detection toolbox based on Detectron2.
- [DAB-DETR](https://github.com/IDEA-Research/DAB-DETR): Dynamic Anchor Boxes are Better Queries for DETR.
- [DN-DETR](https://github.com/IDEA-Research/DN-DETR): Accelerate DETR Training by Introducing Query DeNoising.
- [StableDINO](https://github.com/IDEA-Research/Stable-DINO): Detection Transformer with Stable Matching.